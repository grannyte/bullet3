// PREFER_FLOW_CONTROL - data-dependent loops here must stay real loops; Release flattens
// branches by default, which breaks stack traversal and runtime-bounded clipping.
// Port of Bullet3OpenCL sat.cl findSeparatingAxisKernel - convex/convex separating axis test.
// Emits the minimum-penetration axis per pair; contact clipping is a separate pass, so a wrong
// axis surfaces here rather than as a malformed manifold three kernels later.

#define WG_SIZE 64
#define SAT_EPSILON 1e-6f

struct b3RigidBodyData
{
	float4 pos;
	float4 quat;
	float4 linVel;
	float4 angVel;
	int collidableIdx;
	float invMass;
	float restituitionCoeff;
	float frictionCoeff;
};

struct b3Collidable
{
	int numChildShapes;
	float radius;
	int shapeType;
	int shapeIndex;
};

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

//! One per pair. hasSeparatingAxis is 0/1 so the clipping pass can skip non-colliding pairs
//! without a second predicate.
struct SatResult
{
	float4 axisAndDepth;   // xyz = axis (A->B), w = penetration depth
	int hasSeparatingAxis; // 0 = separated (no contact), 1 = overlapping
	int pad0;
	int pad1;
	int pad2;
};

struct SatParams
{
	uint numPairs;
	uint pad0;
	uint pad1;
	uint pad2;
};

StructuredBuffer<SatParams> Params : register(t0);
StructuredBuffer<b3RigidBodyData> Bodies : register(t1);
StructuredBuffer<b3Collidable> Collidables : register(t2);
StructuredBuffer<b3ConvexPolyhedronData> ConvexShapes : register(t3);
StructuredBuffer<float4> Vertices : register(t4);
StructuredBuffer<b3GpuFace> Faces : register(t5);
StructuredBuffer<float4> UniqueEdges : register(t6);
StructuredBuffer<uint2> Pairs : register(t7);

RWStructuredBuffer<SatResult> OutSat : register(u0);

float3 quatRotate(float4 q, float3 v)
{
	const float3 qv = q.xyz;
	return v + 2.f * cross(qv, cross(qv, v) + q.w * v);
}

//! Interval of a hull along a world-space axis. Loops vertices rather than using the mC/mE
//! shortcut - exact for any hull, and hulls here are small enough that it is not the bottleneck.
void projectHull(b3ConvexPolyhedronData hull, float4 pos, float4 quat, float3 axis,
				 out float minProj, out float maxProj)
{
	minProj = 1e30f;
	maxProj = -1e30f;

	for (int i = 0; i < hull.numVertices; ++i)
	{
		const float3 worldVertex = pos.xyz + quatRotate(quat, Vertices[hull.vertexOffset + i].xyz);
		const float d = dot(worldVertex, axis);
		minProj = min(minProj, d);
		maxProj = max(maxProj, d);
	}
}

//! Overlap along one candidate axis. Negative means the axis separates the hulls, so the caller
//! can stop. The axis is flipped to always point A->B, which the clipping pass relies on.
bool testAxis(b3ConvexPolyhedronData hullA, float4 posA, float4 quatA,
			  b3ConvexPolyhedronData hullB, float4 posB, float4 quatB,
			  float3 axis, inout float3 bestAxis, inout float bestDepth)
{
	const float len2 = dot(axis, axis);
	// A degenerate axis (parallel edges) carries no information; skipping it is correct, not a
	// tolerance fudge - the pair is still covered by the face axes.
	if (len2 < SAT_EPSILON)
		return true;

	const float3 n = axis / sqrt(len2);

	float minA, maxA, minB, maxB;
	projectHull(hullA, posA, quatA, n, minA, maxA);
	projectHull(hullB, posB, quatB, n, minB, maxB);

	const float depth = min(maxA - minB, maxB - minA);
	if (depth < 0.f)
		return false;

	if (depth < bestDepth)
	{
		bestDepth = depth;
		// Point the axis from A to B so downstream code never has to re-derive the sign.
		const float3 delta = posB.xyz - posA.xyz;
		bestAxis = (dot(delta, n) < 0.f) ? -n : n;
	}
	return true;
}

[numthreads(WG_SIZE, 1, 1)]
void CSFindSeparatingAxis(uint3 tid : SV_DispatchThreadID)
{
	const uint pairIndex = tid.x;
	if (pairIndex >= Params[0].numPairs)
		return;

	SatResult result;
	result.axisAndDepth = float4(0.f, 0.f, 0.f, 0.f);
	result.hasSeparatingAxis = 0;
	result.pad0 = result.pad1 = result.pad2 = 0;

	const uint bodyA = Pairs[pairIndex].x;
	const uint bodyB = Pairs[pairIndex].y;
	const b3RigidBodyData a = Bodies[bodyA];
	const b3RigidBodyData b = Bodies[bodyB];
	const b3Collidable ca = Collidables[a.collidableIdx];
	const b3Collidable cb = Collidables[b.collidableIdx];

	// Sphere pairs are handled by their own kernel; anything without hull data cannot be tested.
	// A compound's shapeIndex addresses the child table, so it is excluded too (shapeType 3 = hull).
	if (ca.shapeIndex < 0 || cb.shapeIndex < 0 || ca.shapeType != 3 || cb.shapeType != 3)
	{
		OutSat[pairIndex] = result;
		return;
	}

	const b3ConvexPolyhedronData hullA = ConvexShapes[ca.shapeIndex];
	const b3ConvexPolyhedronData hullB = ConvexShapes[cb.shapeIndex];

	float3 bestAxis = float3(0.f, 0.f, 0.f);
	float bestDepth = 1e30f;

	for (int fa = 0; fa < hullA.numFaces; ++fa)
	{
		const float3 n = quatRotate(a.quat, Faces[hullA.faceOffset + fa].plane.xyz);
		if (!testAxis(hullA, a.pos, a.quat, hullB, b.pos, b.quat, n, bestAxis, bestDepth))
		{
			OutSat[pairIndex] = result;
			return;
		}
	}

	for (int fb = 0; fb < hullB.numFaces; ++fb)
	{
		const float3 n = quatRotate(b.quat, Faces[hullB.faceOffset + fb].plane.xyz);
		if (!testAxis(hullA, a.pos, a.quat, hullB, b.pos, b.quat, n, bestAxis, bestDepth))
		{
			OutSat[pairIndex] = result;
			return;
		}
	}

	// Edge-edge axes catch the cases face normals miss - two boxes crossing like a plus sign
	// separate on no face normal at all.
	for (int ea = 0; ea < hullA.numUniqueEdges; ++ea)
	{
		const float3 edgeA = quatRotate(a.quat, UniqueEdges[hullA.uniqueEdgesOffset + ea].xyz);
		for (int eb = 0; eb < hullB.numUniqueEdges; ++eb)
		{
			const float3 edgeB = quatRotate(b.quat, UniqueEdges[hullB.uniqueEdgesOffset + eb].xyz);
			if (!testAxis(hullA, a.pos, a.quat, hullB, b.pos, b.quat, cross(edgeA, edgeB),
						  bestAxis, bestDepth))
			{
				OutSat[pairIndex] = result;
				return;
			}
		}
	}

	result.axisAndDepth = float4(bestAxis, bestDepth);
	result.hasSeparatingAxis = 1;
	OutSat[pairIndex] = result;
}
