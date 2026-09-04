// Concave triangle-mesh narrowphase: one thread per pair with exactly one mesh side, walking that
// mesh's BVH into ONE manifold per pair - per triangle, the Jacobi solver corrects a seam twice.

// Included by B3TriangleMeshContacts.hlsl (f32) and B3TriangleMeshContactsDS.hlsl (df64).

#include "B3Precision.hlsli"

#define WG_SIZE 64
#define SAT_EPSILON 1e-6f
// Triangle-vs-quad clipping yields at most 7 vertices; 8 matches the hull clip kernel's bound and
// its unroll cost. Hull faces with more than 5 vertices may lose clip points, never corrupt.
#define MAX_POLY 8
#define MESH_STACK_SIZE 64

#define SHAPE_CONVEX_HULL 3
#define SHAPE_CONCAVE_TRIMESH 5
#define SHAPE_SPHERE 7

#define MESH_FLAG_ONE_SIDED 1u

// Two bits per edge in the adjacency record's w, edge e at bits [2e+1:2e].
#define EDGE_BOUNDARY 0u
#define EDGE_CONVEX 1u
#define EDGE_FLAT 2u
#define EDGE_CONCAVE 3u

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

struct ContactParams
{
	uint numPairs;
	float collisionMargin;
	uint pad0;
	uint pad1;
};

// 32 bytes, matching b3IrrMeshHeader on the host. Offsets are into the concatenated mesh buffers.
struct b3IrrMeshHeader
{
	uint triangleOffset;   // first triangle record (2 uint4 per triangle)
	uint numTriangles;
	uint nodeOffset;       // first internal node: ChildNodes[2*(nodeOffset+i)]
	int rootIndex;         // marker-encoded, LOCAL to this mesh
	uint leafOffset;       // first leaf slot in LeafTriangles
	uint aabbOffset;       // internal-node AABBs, followed by one AABB per leaf slot
	uint numLeaves;
	uint flags;
};

StructuredBuffer<ContactParams> Params : register(t0);
StructuredBuffer<b3RigidBodyData> Bodies : register(t1);
StructuredBuffer<b3Collidable> Collidables : register(t2);
StructuredBuffer<b3ConvexPolyhedronData> ConvexShapes : register(t3);
StructuredBuffer<float4> Vertices : register(t4);
StructuredBuffer<b3GpuFace> Faces : register(t5);
StructuredBuffer<float4> UniqueEdges : register(t6);
StructuredBuffer<uint2> Pairs : register(t7);
StructuredBuffer<int> FaceIndices : register(t8);
// A SHAPE_CONCAVE_TRIMESH collidable's shapeIndex is its index here (Bullet's own convention).
StructuredBuffer<b3IrrMeshHeader> Meshes : register(t9);
StructuredBuffer<float4> MeshVertices : register(t10);          // local space, GLOBAL indices
// Two records per triangle: [2t] = (v0, v1, v2, edgeClassBits), [2t+1] = (n0, n1, n2, 0).
StructuredBuffer<uint4> MeshTriangles : register(t11);
StructuredBuffer<int> MeshChildNodes : register(t12);           // 2 per internal node, local encoding
StructuredBuffer<uint> MeshLeafTriangles : register(t13);       // leaf slot -> GLOBAL triangle index
StructuredBuffer<b3AabbF> MeshAabbs : register(t14);            // local space; see aabbOffset

// u0 deliberately: an append buffer bound elsewhere produced a silent zero count.
AppendStructuredBuffer<b3Contact4Data> OutContacts : register(u0);

int isLeafNode(int index) { return (index >> 31) == 0; }
int indexMarkerRemoved(int index) { return index & (~0x80000000); }

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

float3 quatRotateInv(float4 q, float3 v)
{
	return quatRotate(float4(-q.xyz, q.w), v);
}

bool aabbOverlapF(float3 aMin, float3 aMax, b3AabbF b)
{
	if (aMin.x > b.maxVec.x || aMax.x < b.minVec.x) return false;
	if (aMin.y > b.maxVec.y || aMax.y < b.minVec.y) return false;
	if (aMin.z > b.maxVec.z || aMax.z < b.minVec.z) return false;
	return true;
}

uint edgeClass(uint bits, int e) { return (bits >> (2 * e)) & 3u; }
bool edgeSnaps(uint bits, int e)
{
	const uint c = edgeClass(bits, e);
	return c == EDGE_FLAT || c == EDGE_CONCAVE;
}

// ---------------------------------------------------------------------------------------------
// Streaming 4-point manifold. Slots are separate values so nothing is ever indexed at runtime.
// ---------------------------------------------------------------------------------------------

//! Spread of 4 unordered points: sum of squared pairwise distances. cross(a-b, c-d) pseudo-area
//! depends on slot order and greedy-swaps into local optima that drop true footprint corners.
float quadArea(float3 a, float3 b, float3 c, float3 d)
{
	const float3 ab = a - b, ac = a - c, ad = a - d, bc = b - c, bd = b - d, cd = c - d;
	return dot(ab, ab) + dot(ac, ac) + dot(ad, ad) + dot(bc, bc) + dot(bd, bd) + dot(cd, cd);
}

//! Inserts p, keeping the deepest point and otherwise the 4 that span the largest area.
void manifoldInsert(float4 p, inout float4 k0, inout float4 k1, inout float4 k2, inout float4 k3,
					inout int count)
{
	const float dupe2 = 1e-6f;
	if (count > 0 && dot(p.xyz - k0.xyz, p.xyz - k0.xyz) < dupe2) { if (p.w < k0.w) k0 = p; return; }
	if (count > 1 && dot(p.xyz - k1.xyz, p.xyz - k1.xyz) < dupe2) { if (p.w < k1.w) k1 = p; return; }
	if (count > 2 && dot(p.xyz - k2.xyz, p.xyz - k2.xyz) < dupe2) { if (p.w < k2.w) k2 = p; return; }
	if (count > 3 && dot(p.xyz - k3.xyz, p.xyz - k3.xyz) < dupe2) { if (p.w < k3.w) k3 = p; return; }

	if (count == 0) { k0 = p; count = 1; return; }
	if (count == 1) { k1 = p; count = 2; return; }
	if (count == 2) { k2 = p; count = 3; return; }
	if (count == 3) { k3 = p; count = 4; return; }

	int deepest = 4;
	float minW = p.w;
	if (k0.w < minW) { minW = k0.w; deepest = 0; }
	if (k1.w < minW) { minW = k1.w; deepest = 1; }
	if (k2.w < minW) { minW = k2.w; deepest = 2; }
	if (k3.w < minW) { minW = k3.w; deepest = 3; }

	// Depth tie (a flat rest: every point equal): the deepest slot stays a candidate, or an
	// arbitrary first-arrived point pins its slot and a true extreme corner can never enter.
	if (deepest != 4 && p.w <= minW + 1e-5f)
		deepest = 4;

	// Area with slot i removed and p inserted; the deepest slot is never a candidate.
	const float a0 = (deepest == 0) ? -1.f : quadArea(p.xyz, k1.xyz, k2.xyz, k3.xyz);
	const float a1 = (deepest == 1) ? -1.f : quadArea(p.xyz, k0.xyz, k2.xyz, k3.xyz);
	const float a2 = (deepest == 2) ? -1.f : quadArea(p.xyz, k0.xyz, k1.xyz, k3.xyz);
	const float a3 = (deepest == 3) ? -1.f : quadArea(p.xyz, k0.xyz, k1.xyz, k2.xyz);

	int best = 0;
	float bestArea = a0;
	if (a1 > bestArea) { bestArea = a1; best = 1; }
	if (a2 > bestArea) { bestArea = a2; best = 2; }
	if (a3 > bestArea) { bestArea = a3; best = 3; }

	// A non-deepest point only enters if it spreads the manifold; the deepest always enters.
	if (deepest != 4 && bestArea <= quadArea(k0.xyz, k1.xyz, k2.xyz, k3.xyz))
		return;

	if (best == 0) k0 = p;
	else if (best == 1) k1 = p;
	else if (best == 2) k2 = p;
	else k3 = p;
}

// ---------------------------------------------------------------------------------------------
// Sphere vs triangle
// ---------------------------------------------------------------------------------------------

//! Ericson's closest point on a triangle. region: 0 = face, 1..3 = edge (e-1), 4..6 = vertex (v-4).
float3 closestPointOnTriangle(float3 p, float3 a, float3 b, float3 c, out int region)
{
	const float3 ab = b - a;
	const float3 ac = c - a;
	const float3 ap = p - a;
	const float d1 = dot(ab, ap);
	const float d2 = dot(ac, ap);
	if (d1 <= 0.f && d2 <= 0.f) { region = 4; return a; }

	const float3 bp = p - b;
	const float d3 = dot(ab, bp);
	const float d4 = dot(ac, bp);
	if (d3 >= 0.f && d4 <= d3) { region = 5; return b; }

	const float vc = d1 * d4 - d3 * d2;
	if (vc <= 0.f && d1 >= 0.f && d3 <= 0.f)
	{
		region = 1;
		const float v = d1 / (d1 - d3);
		return a + v * ab;
	}

	const float3 cp = p - c;
	const float d5 = dot(ab, cp);
	const float d6 = dot(ac, cp);
	if (d6 >= 0.f && d5 <= d6) { region = 6; return c; }

	const float vb = d5 * d2 - d1 * d6;
	if (vb <= 0.f && d2 >= 0.f && d6 <= 0.f)
	{
		region = 3;
		const float w = d2 / (d2 - d6);
		return a + w * ac;
	}

	const float va = d3 * d6 - d5 * d4;
	if (va <= 0.f && (d4 - d3) >= 0.f && (d5 - d6) >= 0.f)
	{
		region = 2;
		const float w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
		return b + w * (c - b);
	}

	region = 0;
	const float denom = 1.f / (va + vb + vc);
	const float v = vb * denom;
	const float w = vc * denom;
	return a + ab * v + ac * w;
}

//! Whether an edge/vertex region's contact should be snapped onto the face: every edge the
//! closest feature borders must be flat or concave, so the neighbour continues the surface.
bool regionSnaps(int region, uint bits)
{
	if (region >= 1 && region <= 3)
		return edgeSnaps(bits, region - 1);
	if (region == 4) return edgeSnaps(bits, 0) && edgeSnaps(bits, 2);
	if (region == 5) return edgeSnaps(bits, 0) && edgeSnaps(bits, 1);
	if (region == 6) return edgeSnaps(bits, 1) && edgeSnaps(bits, 2);
	return false;
}

// ---------------------------------------------------------------------------------------------
// Hull vs triangle SAT + clipping
// ---------------------------------------------------------------------------------------------

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

void projectTri(float3 t0, float3 t1, float3 t2, float3 axis, out float minProj, out float maxProj)
{
	const float d0 = dot(t0, axis);
	const float d1 = dot(t1, axis);
	const float d2 = dot(t2, axis);
	minProj = min(d0, min(d1, d2));
	maxProj = max(d0, max(d1, d2));
}

//! Returns false on a separating axis. bestAxis is kept oriented hull -> triangle (A -> B).
bool testAxisTri(b3ConvexPolyhedronData hull, float3 pA, float4 qA, float3 hullCenter,
				 float3 t0, float3 t1, float3 t2, float3 triCenter, float3 axis,
				 inout float3 bestAxis, inout float bestDepth)
{
	const float len2 = dot(axis, axis);
	if (len2 < SAT_EPSILON)
		return true;

	const float3 n = axis / sqrt(len2);
	float minA, maxA, minB, maxB;
	projectHull(hull, pA, qA, n, minA, maxA);
	projectTri(t0, t1, t2, n, minB, maxB);

	const float depth = min(maxA - minB, maxB - minA);
	if (depth < 0.f)
		return false;

	if (depth < bestDepth)
	{
		bestDepth = depth;
		bestAxis = (dot(triCenter - hullCenter, n) < 0.f) ? -n : n;
	}
	return true;
}

//! Hull face most aligned with dir; returns the alignment in bestDot.
int selectHullFace(b3ConvexPolyhedronData hull, float4 quat, float3 dir, out float bestDot)
{
	int best = 0;
	bestDot = -1e30f;
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

//! Sutherland-Hodgman of poly against the reference polygon's side planes. [loop] + runtime indexing
//! is load-bearing: unrolling these nested loops made FXC eat gigabytes and never finish.
int clipAgainstReference(float3 refVerts[MAX_POLY], int refCount, float3 refNormal,
						 inout float3 poly[MAX_POLY], int polyCount)
{
	float3 refCenter = float3(0.f, 0.f, 0.f);
	[loop] for (int ci = 0; ci < refCount; ++ci)
		refCenter += refVerts[ci];
	refCenter /= (float)max(refCount, 1);

	[loop] for (int e = 0; e < refCount; ++e)
	{
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
		[loop] for (int i = 0; i < polyCount; ++i)
		{
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
		[loop] for (int k = 0; k < clippedCount; ++k)
			poly[k] = clipped[k];

		if (polyCount == 0)
			return 0;
	}
	return polyCount;
}

// ---------------------------------------------------------------------------------------------

[numthreads(WG_SIZE, 1, 1)]
void CSTriangleMeshContacts(uint3 tid : SV_DispatchThreadID)
{
	const uint pairIndex = tid.x;
	if (pairIndex >= Params[0].numPairs)
		return;

	const uint body0 = Pairs[pairIndex].x;
	const uint body1 = Pairs[pairIndex].y;
	const int col0 = Bodies[body0].collidableIdx;
	const int col1 = Bodies[body1].collidableIdx;
	if (col0 < 0 || col1 < 0)
		return;
	const b3Collidable c0 = Collidables[col0];
	const b3Collidable c1 = Collidables[col1];
	const bool isMesh0 = c0.shapeType == SHAPE_CONCAVE_TRIMESH && c0.shapeIndex >= 0;
	const bool isMesh1 = c1.shapeType == SHAPE_CONCAVE_TRIMESH && c1.shapeIndex >= 0;

	// Exactly one mesh side; mesh vs mesh is not supported (both static in practice).
	if (isMesh0 == isMesh1)
		return;

	// A is the convex body, B the mesh body - contact points land on B's surface.
	const uint bodyA = isMesh0 ? body1 : body0;
	const uint bodyB = isMesh0 ? body0 : body1;
	const int meshIndex = isMesh0 ? c0.shapeIndex : c1.shapeIndex;

	const b3RigidBodyData a = Bodies[bodyA];
	const b3RigidBodyData b = Bodies[bodyB];
	// HLSL cannot select a whole struct with ?: (X3012/X3020).
	b3Collidable ca;
	if (isMesh0)
		ca = c1;
	else
		ca = c0;
	const b3IrrMeshHeader mesh = Meshes[meshIndex];
	if (mesh.numLeaves < 2)
		return;

	const bool isSphere = ca.shapeType == SHAPE_SPHERE;
	if (!isSphere && (ca.shapeType != SHAPE_CONVEX_HULL || ca.shapeIndex < 0))
		return;

	// The one place absolute position is read; everything below is a difference in this frame.
	float3 pA, pB;
	osPairFrame(a, b, pA, pB);

	b3ConvexPolyhedronData hull = ConvexShapes[max(ca.shapeIndex, 0)];
	float boundRadius;
	float3 hullCenter;
	if (isSphere)
	{
		boundRadius = ca.radius;
		hullCenter = pA;
	}
	else
	{
		boundRadius = hull.radius;
		hullCenter = pA + quatRotate(a.quat, hull.localCenter.xyz);
	}

	const float margin = Params[0].collisionMargin;
	if (isnan(boundRadius) || isinf(boundRadius))
		return;

	// Body A's bounding sphere in mesh-local space; triangle AABBs are exact so this is enough.
	const float3 localCenter = quatRotateInv(b.quat, hullCenter - pB);
	const float reach = boundRadius + margin + 1e-3f;
	const float3 qMin = localCenter - reach.xxx;
	const float3 qMax = localCenter + reach.xxx;

	const bool oneSided = (mesh.flags & MESH_FLAG_ONE_SIDED) != 0u;

	float4 k0 = float4(0.f, 0.f, 0.f, 0.f);
	float4 k1 = k0, k2 = k0, k3 = k0;
	int numPoints = 0;
	float3 manifoldNormal = float3(0.f, 1.f, 0.f);
	float deepestSep = 1e30f;

	int stack[MESH_STACK_SIZE];
	int stackSize = 1;
	stack[0] = mesh.rootIndex;

	// Bounded by the leaf count: a malformed tree cannot hang the GPU.
	int visited = 0;
	const int visitLimit = 4 * (int)mesh.numLeaves + 4;

	while (stackSize > 0 && visited++ < visitLimit)
	{
		const int encoded = stack[--stackSize];
		const bool isLeaf = isLeafNode(encoded) != 0;
		const int nodeIndex = indexMarkerRemoved(encoded);

		b3AabbF nodeAabb;
		uint tri = 0;
		if (isLeaf)
		{
			if ((uint)nodeIndex >= mesh.numLeaves)
				continue;
			tri = MeshLeafTriangles[mesh.leafOffset + nodeIndex];
			nodeAabb = MeshAabbs[mesh.aabbOffset + (mesh.numLeaves - 1) + nodeIndex];
		}
		else
		{
			if ((uint)nodeIndex >= mesh.numLeaves - 1)
				continue;
			nodeAabb = MeshAabbs[mesh.aabbOffset + nodeIndex];
		}

		if (!aabbOverlapF(qMin, qMax, nodeAabb))
			continue;

		if (!isLeaf)
		{
			if (stackSize + 2 <= MESH_STACK_SIZE)
			{
				stack[stackSize++] = MeshChildNodes[2 * (mesh.nodeOffset + nodeIndex) + 0];
				stack[stackSize++] = MeshChildNodes[2 * (mesh.nodeOffset + nodeIndex) + 1];
			}
			continue;
		}

		// ---- leaf: one triangle ----
		const uint4 triRec = MeshTriangles[2 * tri];
		const uint edgeBits = triRec.w;
		const float3 t0 = pB + quatRotate(b.quat, MeshVertices[triRec.x].xyz);
		const float3 t1 = pB + quatRotate(b.quat, MeshVertices[triRec.y].xyz);
		const float3 t2 = pB + quatRotate(b.quat, MeshVertices[triRec.z].xyz);

		float3 nf = cross(t1 - t0, t2 - t0);
		const float nfLen2 = dot(nf, nf);
		if (nfLen2 < SAT_EPSILON)
			continue;
		nf /= sqrt(nfLen2);

		// nf faces body A: one-sided meshes reject a body behind the surface, two-sided flip.
		const float side = dot(hullCenter - t0, nf);
		if (side < 0.f)
		{
			if (oneSided)
				continue;
			nf = -nf;
		}
		const float3 triCenter = (t0 + t1 + t2) * (1.f / 3.f);

		if (isSphere)
		{
			int region;
			const float3 cp = closestPointOnTriangle(pA, t0, t1, t2, region);
			const float3 d = cp - pA;
			const float dist2 = dot(d, d);
			const float r = ca.radius;
			if (dist2 > (r + margin) * (r + margin))
				continue;

			float3 normal;
			float sep;
			float3 pointOnB;
			if (region == 0 || regionSnaps(region, edgeBits))
			{
				const float h = dot(pA - t0, nf);   // >= 0 by the flip above
				sep = h - r;
				if (sep > margin)
					continue;
				normal = -nf;
				pointOnB = pA - nf * h;
			}
			else
			{
				const float dist = sqrt(dist2);
				sep = dist - r;
				normal = (dist > 1e-6f) ? d / dist : -nf;
				pointOnB = cp;
			}
			if (isnan(sep) || isinf(sep))
				continue;

			// A sphere keeps one point: the deepest. Its normal is the manifold's.
			if (sep < deepestSep)
			{
				deepestSep = sep;
				manifoldNormal = normal;
				k0 = float4(pointOnB - pA, sep);
				numPoints = 1;
			}
			continue;
		}

		// ---- hull vs triangle SAT ----
		float3 axis = float3(0.f, 0.f, 0.f);
		float depth = 1e30f;
		bool separated = false;

		if (!testAxisTri(hull, pA, a.quat, hullCenter, t0, t1, t2, triCenter, nf, axis, depth))
			continue;

		for (int fa = 0; fa < hull.numFaces && !separated; ++fa)
			if (!testAxisTri(hull, pA, a.quat, hullCenter, t0, t1, t2, triCenter,
							 quatRotate(a.quat, Faces[hull.faceOffset + fa].plane.xyz), axis, depth))
				separated = true;
		if (separated)
			continue;

		const float3 e0 = t1 - t0;
		const float3 e1 = t2 - t1;
		const float3 e2 = t0 - t2;
		for (int ea = 0; ea < hull.numUniqueEdges && !separated; ++ea)
		{
			const float3 edgeA = quatRotate(a.quat, UniqueEdges[hull.uniqueEdgesOffset + ea].xyz);
			if (!testAxisTri(hull, pA, a.quat, hullCenter, t0, t1, t2, triCenter, cross(edgeA, e0), axis, depth) ||
				!testAxisTri(hull, pA, a.quat, hullCenter, t0, t1, t2, triCenter, cross(edgeA, e1), axis, depth) ||
				!testAxisTri(hull, pA, a.quat, hullCenter, t0, t1, t2, triCenter, cross(edgeA, e2), axis, depth))
				separated = true;
		}
		if (separated)
			continue;

		// Internal-edge fix: an axis tilted off the face normal pushes A across some triangle
		// edge. If every edge it crosses continues flat or concave, the tilt is a seam artefact.
		const float3 faceAxis = -nf;
		if (dot(axis, faceAxis) < 0.999f)
		{
			const float3 tangent = axis - faceAxis * dot(axis, faceAxis);
			const float3 pushDir = -tangent;   // direction A is pushed, in the triangle plane
			bool crossesReal = false;
			bool crossesAny = false;
			float3 en;
			en = cross(e0, nf); if (dot(en, triCenter - t0) > 0.f) en = -en;
			if (dot(pushDir, en) > SAT_EPSILON) { crossesAny = true; if (!edgeSnaps(edgeBits, 0)) crossesReal = true; }
			en = cross(e1, nf); if (dot(en, triCenter - t1) > 0.f) en = -en;
			if (dot(pushDir, en) > SAT_EPSILON) { crossesAny = true; if (!edgeSnaps(edgeBits, 1)) crossesReal = true; }
			en = cross(e2, nf); if (dot(en, triCenter - t2) > 0.f) en = -en;
			if (dot(pushDir, en) > SAT_EPSILON) { crossesAny = true; if (!edgeSnaps(edgeBits, 2)) crossesReal = true; }

			if (crossesAny && !crossesReal)
			{
				float minA, maxA;
				projectHull(hull, pA, a.quat, nf, minA, maxA);
				const float planeSep = minA - dot(t0, nf);
				if (planeSep > margin)
					continue;
				axis = faceAxis;
				depth = -planeSep;
			}
		}

		// Reference side: the triangle if it aligns with the axis at least as well as any hull face.
		float hullFaceDot;
		const int incFace = selectHullFace(hull, a.quat, axis, hullFaceDot);
		const bool triIsReference = dot(faceAxis, axis) >= hullFaceDot;

		float3 refVerts[MAX_POLY];
		float3 poly[MAX_POLY];
		[loop] for (int zi = 0; zi < MAX_POLY; ++zi)
		{
			refVerts[zi] = float3(0.f, 0.f, 0.f);
			poly[zi] = float3(0.f, 0.f, 0.f);
		}
		int refCount;
		int polyCount;
		float3 refNormal;
		float refOffset;

		if (triIsReference)
		{
			refVerts[0] = t0; refVerts[1] = t1; refVerts[2] = t2;
			refCount = 3;
			refNormal = nf;
			refOffset = dot(nf, t0);

			const b3GpuFace inf = Faces[hull.faceOffset + incFace];
			polyCount = min(inf.numIndices, MAX_POLY);
			[loop] for (int ii = 0; ii < polyCount; ++ii)
				poly[ii] = pA + quatRotate(a.quat, Vertices[hull.vertexOffset + FaceIndices[inf.indexOffset + ii]].xyz);
		}
		else
		{
			const b3GpuFace rf = Faces[hull.faceOffset + incFace];
			refCount = min(rf.numIndices, MAX_POLY);
			[loop] for (int ri = 0; ri < refCount; ++ri)
				refVerts[ri] = pA + quatRotate(a.quat, Vertices[hull.vertexOffset + FaceIndices[rf.indexOffset + ri]].xyz);
			refNormal = quatRotate(a.quat, rf.plane.xyz);
			refOffset = dot(refNormal, refVerts[0]);

			poly[0] = t0; poly[1] = t1; poly[2] = t2;
			polyCount = 3;
		}

		polyCount = clipAgainstReference(refVerts, refCount, refNormal, poly, polyCount);
		if (polyCount == 0)
			continue;

		[loop] for (int pi = 0; pi < polyCount; ++pi)
		{
			const float sep = dot(refNormal, poly[pi]) - refOffset;
			if (sep > margin || isnan(sep) || isinf(sep))
				continue;
			// Points live on B: a hull-face point is projected down onto the triangle plane.
			const float3 onB = triIsReference ? (poly[pi] - nf * sep) : poly[pi];
			manifoldInsert(float4(onB - pA, sep), k0, k1, k2, k3, numPoints);
			if (sep < deepestSep)
			{
				deepestSep = sep;
				manifoldNormal = axis;
			}
		}
	}

	if (numPoints == 0)
		return;

	b3Contact4Data contact;
	contact.relPosA[0] = k0;
	contact.relPosA[1] = k1;
	contact.relPosA[2] = k2;
	contact.relPosA[3] = k3;
	contact.worldNormalOnB = float4(manifoldNormal, (float)numPoints);
	contact.restitutionAndFriction = packCoeff(a.restituitionCoeff * b.restituitionCoeff,
											   a.frictionCoeff * b.frictionCoeff);
	contact.batchIdx = 0;
	contact.bodyAPtrAndSignBit = (a.invMass == 0.f) ? -(int)bodyA : (int)bodyA;
	contact.bodyBPtrAndSignBit = (b.invMass == 0.f) ? -(int)bodyB : (int)bodyB;
	contact.childIndexA = -1;
	contact.childIndexB = -1;
	contact.unused1 = 0;
	contact.unused2 = 0;
	OutContacts.Append(contact);
}
