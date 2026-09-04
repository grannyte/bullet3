// Port of Bullet3OpenCL satClipHullContacts.cl - reference/incident face clipping.
// Runs SAT inline rather than reading a prior pass's result, so the axis and the faces it selects
// can never disagree; the standalone SAT kernel stays for testing the axis in isolation.

// Included by B3NarrowphaseClip.hlsl (f32) and B3NarrowphaseClipDS.hlsl (df64).

// rsqrt/normalize are APPROXIMATE instructions with vendor-defined precision; sqrt and divide
// are correctly rounded, so these forms are bit-identical across GPUs. Needed for lockstep.
#include "B3Precision.hlsli"

#define WG_SIZE 64
#define SAT_EPSILON 1e-6f
// Clipping a quad against 4 side planes yields at most 8 vertices. This bound is also the UNROLL
// factor - FXC must fully unroll every loop that indexes a local array, and 16 blew up (X3511).
#define MAX_POLY 8

#define SHAPE_CONVEX_HULL 3

struct b3Collidable
{
	int numChildShapes;
	float radius;
	int shapeType;
	int shapeIndex;
};

#ifdef B3_LEAF_PAIRS
// Layout-identical to b3GpuChildShape (48 bytes); the two unions are read as ints here.
struct b3GpuChildShape
{
	float4 childPosition;
	float4 childOrientation;
	int shapeIndex;
	int radiusOrNumChildren;
	int collidableShapeIndex;
	int shapeType;
};
#endif

struct b3ConvexPolyhedronData
{
	float4 localCenter;
	float4 extents;
	float4 mC;
	float4 mE;
	float radius;
	int faceOffset;
	int numFaces;
	int numVertices;
	int vertexOffset;
	int uniqueEdgesOffset;
	int numUniqueEdges;
	int unused;
};

struct b3GpuFace
{
	float4 plane;
	int indexOffset;
	int numIndices;
	int unusedPadding1;
	int unusedPadding2;
};

// relPosA maps onto b3Contact4Data::m_worldPosB; xyz is relative to body A's origin, w the separation.
struct b3Contact4Data
{
	float4 relPosA[4];
	float4 worldNormalOnB;
	uint restitutionAndFriction;
	int batchIdx;
	int bodyAPtrAndSignBit;
	int bodyBPtrAndSignBit;
	int childIndexA;
	int childIndexB;
	int unused1;
	int unused2;
};

struct ClipParams
{
	uint numPairs;
	float collisionMargin;
	float speculativeDt;        // > 0: emit a speculative contact for a pair that meets within dt
	float ccdMotionThreshold;   // closing motion per step below which no speculative contact is made
};

// Marks b3Contact4Data.unused1: the point's w is a POSITIVE gap the solver may close this step.
#define B3_CONTACT_SPECULATIVE 1

StructuredBuffer<ClipParams> Params : register(t0);
StructuredBuffer<b3RigidBodyData> Bodies : register(t1);
StructuredBuffer<b3Collidable> Collidables : register(t2);
StructuredBuffer<b3ConvexPolyhedronData> ConvexShapes : register(t3);
StructuredBuffer<float4> Vertices : register(t4);
StructuredBuffer<b3GpuFace> Faces : register(t5);
StructuredBuffer<float4> UniqueEdges : register(t6);
#ifdef B3_LEAF_PAIRS
StructuredBuffer<int4> Pairs : register(t7);   // (bodyA, bodyB, childA, childB), -1 = own shape
#else
StructuredBuffer<uint2> Pairs : register(t7);
#endif
StructuredBuffer<int> FaceIndices : register(t8);
#ifdef B3_LEAF_PAIRS
StructuredBuffer<b3GpuChildShape> ChildShapes : register(t9);
#endif

AppendStructuredBuffer<b3Contact4Data> OutContacts : register(u0);

//! 16-bit fixed point, matching b3Contact4Data's packed coefficient fields.
uint packCoeff(float restitution, float friction)
{
	const uint r = (uint)(saturate(restitution) * 65535.f + 0.5f);
	const uint f = (uint)(saturate(friction) * 65535.f + 0.5f);
	return (f << 16) | (r & 0xffffu);
}

float3 quatRotate(float4 q, float3 v)
{
	const float3 qv = q.xyz;
	return v + 2.f * cross(qv, cross(qv, v) + q.w * v);
}

#ifdef B3_LEAF_PAIRS
//! Hamilton product: quatRotate(quatMul(p, c), v) == quatRotate(p, quatRotate(c, v)).
float4 quatMul(float4 p, float4 c)
{
	return float4(p.w * c.xyz + c.w * p.xyz + cross(p.xyz, c.xyz), p.w * c.w - dot(p.xyz, c.xyz));
}
#endif

void projectHull(b3ConvexPolyhedronData hull, float3 pos, float4 quat, float3 axis,
				 out float minProj, out float maxProj)
{
	minProj = 1e30f;
	maxProj = -1e30f;
	for (int i = 0; i < hull.numVertices; ++i)
	{
		const float d = dot(pos + quatRotate(quat, Vertices[hull.vertexOffset + i].xyz), axis);
		minProj = min(minProj, d);
		maxProj = max(maxProj, d);
	}
}

// Linear sweep along one axis: the interval over which the two projections overlap, given B's
// velocity relative to A. Feeds the speculative-contact decision; the discrete SAT result is unchanged.
struct SweepState
{
	float tEnter;       // max over axes of the entry time; <= 0 while already overlapping
	float tExit;        // min over axes of the exit time
	float3 enterAxis;   // A->B axis of the last entry
	float enterGap;     // separation along it now
	float enterSpeed;   // closing speed along it
	float enterExtB;    // B's half-reach along -enterAxis from its origin
};

bool testAxis(b3ConvexPolyhedronData hullA, float3 posA, float4 quatA,
			  b3ConvexPolyhedronData hullB, float3 posB, float4 quatB,
			  float3 axis, inout float3 bestAxis, inout float bestDepth,
			  float3 relVelBA, inout SweepState sweep)
{
	const float len2 = dot(axis, axis);
	if (len2 < SAT_EPSILON)
		return true;

	const float3 n = axis / sqrt(len2);
	float minA, maxA, minB, maxB;
	projectHull(hullA, posA, quatA, n, minA, maxA);
	projectHull(hullB, posB, quatB, n, minB, maxB);

	const float depth = min(maxA - minB, maxB - minA);
	const float3 delta = posB - posA;
	const float3 nAB = (dot(delta, n) < 0.f) ? -n : n;

	// Entry/exit of B's interval into A's along n; a parallel-moving separated pair never enters.
	const float u = dot(relVelBA, n);
	float tIn, tOut;
	if (abs(u) < 1e-9f)
	{
		tIn = (depth < 0.f) ? 1e30f : -1e30f;
		tOut = 1e30f;
	}
	else
	{
		tIn = ((u > 0.f) ? (minA - maxB) : (maxA - minB)) / u;
		tOut = ((u > 0.f) ? (maxA - minB) : (minA - maxB)) / u;
	}
	sweep.tExit = min(sweep.tExit, tOut);
	if (tIn > sweep.tEnter)
	{
		sweep.tEnter = tIn;
		sweep.enterAxis = nAB;
		sweep.enterGap = -depth;
		sweep.enterSpeed = abs(u);
		sweep.enterExtB = (dot(nAB, n) > 0.f) ? (dot(posB, n) - minB) : (maxB - dot(posB, n));
	}

	if (depth < 0.f)
		return false;

	if (depth < bestDepth)
	{
		bestDepth = depth;
		bestAxis = nAB;
	}
	return true;
}

//! Face of `hull` most aligned with `dir`. Reference face uses +axis, incident uses -axis, which
//! is what makes the two faces oppose each other.
int selectFace(b3ConvexPolyhedronData hull, float4 quat, float3 dir)
{
	int best = 0;
	float bestDot = -1e30f;
	for (int f = 0; f < hull.numFaces; ++f)
	{
		const float d = dot(quatRotate(quat, Faces[hull.faceOffset + f].plane.xyz), dir);
		if (d > bestDot)
		{
			bestDot = d;
			best = f;
		}
	}
	return best;
}

[numthreads(WG_SIZE, 1, 1)]
void CSClipContacts(uint3 tid : SV_DispatchThreadID)
{
	const uint pairIndex = tid.x;
	if (pairIndex >= Params[0].numPairs)
		return;

	const uint bodyA = (uint)Pairs[pairIndex].x;
	const uint bodyB = (uint)Pairs[pairIndex].y;
	b3RigidBodyData a = Bodies[bodyA];
	b3RigidBodyData b = Bodies[bodyB];
	b3Collidable ca = Collidables[a.collidableIdx];
	b3Collidable cb = Collidables[b.collidableIdx];

	int childIndexA = -1;
	int childIndexB = -1;
#ifdef B3_LEAF_PAIRS
	// A compound child collides as its own collidable, placed by composing the child transform
	// onto the body's; contacts still name the parent body and stay relative to ITS origin.
	float3 childOffA = float3(0.f, 0.f, 0.f);
	float3 childOffB = float3(0.f, 0.f, 0.f);
	childIndexA = Pairs[pairIndex].z;
	childIndexB = Pairs[pairIndex].w;
	if (childIndexA >= 0)
	{
		const b3GpuChildShape ch = ChildShapes[childIndexA];
		ca = Collidables[ch.collidableShapeIndex];
		childOffA = quatRotate(a.quat, ch.childPosition.xyz);
		a.quat = quatMul(a.quat, ch.childOrientation);
	}
	if (childIndexB >= 0)
	{
		const b3GpuChildShape ch = ChildShapes[childIndexB];
		cb = Collidables[ch.collidableShapeIndex];
		childOffB = quatRotate(b.quat, ch.childPosition.xyz);
		b.quat = quatMul(b.quat, ch.childOrientation);
	}
#endif

	// A compound's shapeIndex addresses the child table, not ConvexShapes - never dereference it here.
	if (ca.shapeType != SHAPE_CONVEX_HULL || cb.shapeType != SHAPE_CONVEX_HULL)
		return;
	if (ca.shapeIndex < 0 || cb.shapeIndex < 0)
		return;

	const b3ConvexPolyhedronData hullA = ConvexShapes[ca.shapeIndex];
	const b3ConvexPolyhedronData hullB = ConvexShapes[cb.shapeIndex];

	// The one place absolute position is read: both hulls are moved into a shared frame here, so
	// every quantity below is a difference and stays float-exact however far the pair is from 0.
	float3 pA, pB;
	osPairFrame(a, b, pA, pB);
#ifdef B3_LEAF_PAIRS
	pA += childOffA;
	pB += childOffB;
#endif

	float3 axis = float3(0.f, 0.f, 0.f);
	float depth = 1e30f;

	// Speculative mode keeps scanning a separated pair: the swept-interval test over ALL axes
	// decides whether it meets within the step. Translation only; spin is covered by the AABB margin.
	const float specDt = Params[0].speculativeDt;
	const float3 relVelBA = b.linVel.xyz - a.linVel.xyz;
	SweepState sweep;
	sweep.tEnter = -1e30f;
	sweep.tExit = 1e30f;
	sweep.enterAxis = float3(0.f, 1.f, 0.f);
	sweep.enterGap = 0.f;
	sweep.enterSpeed = 0.f;
	sweep.enterExtB = 0.f;
	bool separated = false;

	for (int fa = 0; fa < hullA.numFaces; ++fa)
		if (!testAxis(hullA, pA, a.quat, hullB, pB, b.quat,
					  quatRotate(a.quat, Faces[hullA.faceOffset + fa].plane.xyz), axis, depth,
					  relVelBA, sweep))
		{
			if (specDt <= 0.f || sweep.tEnter > sweep.tExit || sweep.tEnter > specDt)
				return;
			separated = true;
		}

	for (int fb = 0; fb < hullB.numFaces; ++fb)
		if (!testAxis(hullA, pA, a.quat, hullB, pB, b.quat,
					  quatRotate(b.quat, Faces[hullB.faceOffset + fb].plane.xyz), axis, depth,
					  relVelBA, sweep))
		{
			if (specDt <= 0.f || sweep.tEnter > sweep.tExit || sweep.tEnter > specDt)
				return;
			separated = true;
		}

	for (int ea = 0; ea < hullA.numUniqueEdges; ++ea)
	{
		const float3 edgeA = quatRotate(a.quat, UniqueEdges[hullA.uniqueEdgesOffset + ea].xyz);
		for (int eb = 0; eb < hullB.numUniqueEdges; ++eb)
		{
			const float3 edgeB = quatRotate(b.quat, UniqueEdges[hullB.uniqueEdgesOffset + eb].xyz);
			if (!testAxis(hullA, pA, a.quat, hullB, pB, b.quat, cross(edgeA, edgeB),
						  axis, depth, relVelBA, sweep))
			{
				if (specDt <= 0.f || sweep.tEnter > sweep.tExit || sweep.tEnter > specDt)
					return;
				separated = true;
			}
		}
	}

	if (separated)
	{
		// One Bullet-style predictive point: B's origin pushed to its surface facing A, so the
		// constraint acts through B's centre and cannot spin a flat-falling box on a corner.
		if (sweep.tEnter > sweep.tExit || sweep.tEnter > specDt
			|| sweep.enterSpeed * specDt < Params[0].ccdMotionThreshold
			|| isnan(sweep.tEnter) || isinf(sweep.tEnter))
			return;

		const float3 nAB = sweep.enterAxis;
		float3 pointRelA = (pB - pA) - nAB * sweep.enterExtB;
#ifdef B3_LEAF_PAIRS
		pointRelA += childOffA;   // back to the parent body's origin
#endif

		b3Contact4Data spec;
		// Floored at a positive epsilon: a float-noise "gap" of exactly 0 breaks the speculative
		// contract (positive gap) while changing the solver's target by nothing measurable.
		spec.relPosA[0] = float4(pointRelA, max(sweep.enterGap, 1e-6f));
		spec.relPosA[1] = float4(0.f, 0.f, 0.f, 0.f);
		spec.relPosA[2] = float4(0.f, 0.f, 0.f, 0.f);
		spec.relPosA[3] = float4(0.f, 0.f, 0.f, 0.f);
		spec.worldNormalOnB = float4(nAB, 1.f);
		spec.restitutionAndFriction = packCoeff(a.restituitionCoeff * b.restituitionCoeff,
												a.frictionCoeff * b.frictionCoeff);
		spec.batchIdx = 0;
		spec.bodyAPtrAndSignBit = (a.invMass == 0.f) ? -(int)bodyA : (int)bodyA;
		spec.bodyBPtrAndSignBit = (b.invMass == 0.f) ? -(int)bodyB : (int)bodyB;
		spec.childIndexA = childIndexA;
		spec.childIndexB = childIndexB;
		spec.unused1 = B3_CONTACT_SPECULATIVE;
		spec.unused2 = 0;
		OutContacts.Append(spec);
		return;
	}

	// Reference face on A faces along the axis; incident face on B faces back against it.
	const int refFace = selectFace(hullA, a.quat, axis);
	const int incFace = selectFace(hullB, b.quat, -axis);

	const b3GpuFace rf = Faces[hullA.faceOffset + refFace];
	const b3GpuFace inf = Faces[hullB.faceOffset + incFace];

	const float3 refNormal = quatRotate(a.quat, rf.plane.xyz);

	// The clip runs in A-relative space; every quantity below is a difference, so results are identical.
	const float3 bRelA = pB - pA;

	// Every loop below is bounded by the COMPILE-TIME MAX_POLY with a runtime break: FXC must unroll
	// any loop indexing a local array, and cannot unroll a runtime trip count (X3511).
	float3 refVerts[MAX_POLY];
	const int refCount = min(rf.numIndices, MAX_POLY);
	[unroll] for (int ri = 0; ri < MAX_POLY; ++ri)
	{
		if (ri >= refCount) break;
		refVerts[ri] = quatRotate(a.quat, Vertices[hullA.vertexOffset + FaceIndices[rf.indexOffset + ri]].xyz);
	}

	float3 poly[MAX_POLY];
	int polyCount = min(inf.numIndices, MAX_POLY);
	[unroll] for (int ii = 0; ii < MAX_POLY; ++ii)
	{
		if (ii >= polyCount) break;
		poly[ii] = bRelA + quatRotate(b.quat, Vertices[hullB.vertexOffset + FaceIndices[inf.indexOffset + ii]].xyz);
	}

	// Centroid decides each side plane's inward direction, so winding order does not matter -
	// the box faces are generated by bit pattern and are not guaranteed consistently wound.
	float3 refCenter = float3(0.f, 0.f, 0.f);
	[unroll] for (int ci = 0; ci < MAX_POLY; ++ci)
	{
		if (ci >= refCount) break;
		refCenter += refVerts[ci];
	}
	refCenter /= (float)refCount;

	// Sutherland-Hodgman against each side plane of the reference face.
	[unroll] for (int e = 0; e < MAX_POLY; ++e)
	{
		if (e >= refCount) break;
		const float3 v0 = refVerts[e];
		const float3 v1 = refVerts[(e + 1) % refCount];
		float3 sideNormal = cross(v1 - v0, refNormal);
		const float sideLen2 = dot(sideNormal, sideNormal);
		if (sideLen2 < SAT_EPSILON)
			continue;
		sideNormal /= sqrt(sideLen2);
		if (dot(sideNormal, refCenter - v0) < 0.f)
			sideNormal = -sideNormal;
		const float sideOffset = dot(sideNormal, v0);

		float3 clipped[MAX_POLY];
		int clippedCount = 0;
		[unroll] for (int i = 0; i < MAX_POLY; ++i)
		{
			if (i >= polyCount) break;
			const float3 cur = poly[i];
			const float3 nxt = poly[(i + 1) % polyCount];
			const float dCur = dot(sideNormal, cur) - sideOffset;
			const float dNxt = dot(sideNormal, nxt) - sideOffset;

			if (dCur >= 0.f && clippedCount < MAX_POLY)
				clipped[clippedCount++] = cur;
			if (dCur * dNxt < 0.f && clippedCount < MAX_POLY)
				clipped[clippedCount++] = cur + (nxt - cur) * (dCur / (dCur - dNxt));
		}

		polyCount = clippedCount;
		[unroll] for (int k = 0; k < MAX_POLY; ++k)
		{
			if (k >= clippedCount) break;
			poly[k] = clipped[k];
		}

		if (polyCount == 0)
			return;
	}

	// Keep only points at or below the reference plane; those are the ones actually penetrating.
	const float refOffset = dot(refNormal, refVerts[0]);

	// Flagged in place rather than compacted: poly[] is only ever indexed by an unrolled induction
	// variable that way, which is what keeps FXC from having to unroll on a runtime index (X3511).
	float sep[MAX_POLY];
	bool valid[MAX_POLY];
	int numValid = 0;
	float3 firstValid = float3(0.f, 0.f, 0.f);
	[unroll] for (int vi = 0; vi < MAX_POLY; ++vi)
	{
		sep[vi] = 0.f;
		valid[vi] = false;
		if (vi >= polyCount)
			continue;
		sep[vi] = dot(refNormal, poly[vi]) - refOffset;
		if (sep[vi] <= Params[0].collisionMargin)
		{
			if (numValid == 0)
				firstValid = poly[vi];
			valid[vi] = true;
			++numValid;
		}
	}

	if (numValid == 0)
		return;

	b3Contact4Data contact;
	contact.worldNormalOnB = float4(axis, 0.f);
	contact.restitutionAndFriction = packCoeff(a.restituitionCoeff * b.restituitionCoeff,
											   a.frictionCoeff * b.frictionCoeff);
	contact.batchIdx = 0;
	contact.bodyAPtrAndSignBit = (a.invMass == 0.f) ? -(int)bodyA : (int)bodyA;
	contact.bodyBPtrAndSignBit = (b.invMass == 0.f) ? -(int)bodyB : (int)bodyB;
	contact.childIndexA = childIndexA;
	contact.childIndexB = childIndexB;
	contact.unused1 = 0;
	contact.unused2 = 0;
	contact.relPosA[0] = float4(0.f, 0.f, 0.f, 0.f);
	contact.relPosA[1] = float4(0.f, 0.f, 0.f, 0.f);
	contact.relPosA[2] = float4(0.f, 0.f, 0.f, 0.f);
	contact.relPosA[3] = float4(0.f, 0.f, 0.f, 0.f);

	// Gathered into locals, then written to fixed slots: indexing contact.relPosA by a runtime
	// counter is itself dynamic indexing and forces the same failed unroll (X3511).
	float4 kept[4];
	kept[0] = kept[1] = kept[2] = kept[3] = float4(0.f, 0.f, 0.f, 0.f);
	int numPoints = 0;

	if (numValid <= 4)
	{
		// w carries the signed separation, matching the sphere kernel's convention.
		[unroll] for (int pi = 0; pi < MAX_POLY; ++pi)
		{
			if (pi >= polyCount) break;
			if (!valid[pi]) continue;
			const float4 v = float4(poly[pi], sep[pi]);
			if (numPoints == 0) kept[0] = v;
			else if (numPoints == 1) kept[1] = v;
			else if (numPoints == 2) kept[2] = v;
			else kept[3] = v;
			++numPoints;
		}
	}
	else
	{
		// b3 newContactReductionKernel: the deepest point plus the extremes along two perpendicular
		// in-plane axes. Winners are carried as VALUES, so no array is ever indexed at runtime.
		float3 center = float3(0.f, 0.f, 0.f);
		[unroll] for (int ai = 0; ai < MAX_POLY; ++ai)
		{
			if (ai >= polyCount) break;
			if (valid[ai]) center += poly[ai];
		}
		center /= (float)numValid;

		float3 uAxis = cross(refNormal, firstValid - center);
		if (dot(uAxis, uAxis) < SAT_EPSILON)
			uAxis = cross(refNormal, (abs(refNormal.y) > 0.9f) ? float3(1.f, 0.f, 0.f)
															   : float3(0.f, 1.f, 0.f));
		float3 vAxis = cross(refNormal, uAxis);
		uAxis = uAxis / sqrt(dot(uAxis, uAxis));
		vAxis = vAxis / sqrt(dot(vAxis, vAxis));

		float4 deepest = float4(0.f, 0.f, 0.f, 0.f);
		float4 sel0 = deepest, sel1 = deepest, sel2 = deepest, sel3 = deepest;
		int deepIdx = -1, i0 = -1, i1 = -1, i2 = -1, i3 = -1;
		float minSep = 1e30f, d0 = 1e30f, d1 = 1e30f, d2 = 1e30f, d3 = 1e30f;

		[unroll] for (int si = 0; si < MAX_POLY; ++si)
		{
			if (si >= polyCount) break;
			if (!valid[si]) continue;
			const float4 pw = float4(poly[si], sep[si]);
			if (pw.w < minSep) { minSep = pw.w; deepest = pw; deepIdx = si; }

			const float3 r = poly[si] - center;
			float f = dot(uAxis, r);
			if (f < d0) { d0 = f; sel0 = pw; i0 = si; }
			if (-f < d1) { d1 = -f; sel1 = pw; i1 = si; }
			f = dot(vAxis, r);
			if (f < d2) { d2 = f; sel2 = pw; i2 = si; }
			if (-f < d3) { d3 = -f; sel3 = pw; i3 = si; }
		}

		if (deepIdx != i0 && deepIdx != i1 && deepIdx != i2 && deepIdx != i3)
		{
			sel0 = deepest;
			i0 = deepIdx;
		}

		// Dedupe by source index: one point selected twice would double that constraint's impulse.
		kept[0] = sel0;
		numPoints = 1;
		if (i1 != i0)
		{
			kept[1] = sel1;
			numPoints = 2;
		}
		if (i2 != i0 && i2 != i1)
		{
			if (numPoints == 1) kept[1] = sel2;
			else kept[2] = sel2;
			++numPoints;
		}
		if (i3 != i0 && i3 != i1 && i3 != i2)
		{
			if (numPoints == 1) kept[1] = sel3;
			else if (numPoints == 2) kept[2] = sel3;
			else kept[3] = sel3;
			++numPoints;
		}
	}

#ifdef B3_LEAF_PAIRS
	// Clipped relative to the child's origin; the solver's lever arm is from the parent body's.
	kept[0].xyz += childOffA;
	kept[1].xyz += childOffA;
	kept[2].xyz += childOffA;
	kept[3].xyz += childOffA;
#endif

	contact.relPosA[0] = kept[0];
	contact.relPosA[1] = kept[1];
	contact.relPosA[2] = kept[2];
	contact.relPosA[3] = kept[3];

	if (numPoints == 0)
		return;

	contact.worldNormalOnB.w = (float)numPoints;
	OutContacts.Append(contact);
}
