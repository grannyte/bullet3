// Batched scene queries over the resident LBVH: closest-hit rays and sphere sweeps, one thread
// per query, with an owner filter replacing the CPU IgnoreIdsRayCallback. Phase 5h.
//
// Exact vs conservative, per shape:
//   ray     vs sphere       exact (quadratic)
//   ray     vs convex hull  exact (half-space clip of the face planes)
//   sphere  vs sphere       exact (ray vs sphere of summed radius)
//   sphere  vs convex hull  CONSERVATIVE: face planes pushed out by the radius, i.e. the hull is
//                           expanded without rounding its edges/corners, so a sweep grazing an
//                           edge or corner reports a hit up to radius*(sqrt(3)-1) early. Exact on
//                           face regions. Tighten with a per-edge/per-vertex pass if a caller needs it.
//   anything else           world-AABB slab hit, normal = entered slab axis (conservative)
// A query starting inside a shape reports no hit on it, matching Bullet's convex ray test.
//
// Included by B3Queries.hlsl (f32) and B3QueriesDS.hlsl (df64) - see B3Precision.hlsli. Under
// OS_DS only the hi halves are used: a query is authored in float world space anyway.

#include "B3Precision.hlsli"

#define WG_SIZE 64
#define TRAVERSE_MAX_STACK_SIZE 128
#define QUERY_MAX_FACES 64

#define QUERY_KIND_RAY 0u
#define QUERY_KIND_SPHERE 1u

#define SHAPE_CONVEX_HULL 3
#define SHAPE_SPHERE 7

struct QueryParams
{
	uint numQueries;
	int rootIndex;      // marker-encoded, as written by CSBuildInternalNodes
	uint useOwnerFilter;
	uint numBodies;
};

struct b3Query
{
	float4 from;
	float4 to;
	float radius;   // 0 for a ray
	int ownerRoot;  // -1 filters nothing
	uint kind;
	uint pad;
};

struct b3QueryHit
{
	float4 pointAndFraction;   // xyz = world hit point, w = fraction along from->to
	float4 normal;             // xyz = world normal at the hit
	int bodyIndex;             // -1 when nothing was hit
	int hit;                   // 0/1
	int pad0;
	int pad1;
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
	float4 plane;   // xyz = outward normal in shape space, w: dot(n, x) + w <= 0 inside
	int indexOffset;
	int numIndices;
	int unusedPadding1;
	int unusedPadding2;
};

StructuredBuffer<QueryParams> Params : register(t0);
StructuredBuffer<int> ChildNodes : register(t1);                  // 2 ints per internal node
StructuredBuffer<uint2> MortonCodes : register(t2);               // (key, leaf row), sorted
StructuredBuffer<b3Aabb> WorldAabbs : register(t3);               // per BODY
StructuredBuffer<b3Aabb> InternalAabbs : register(t4);
StructuredBuffer<b3Query> Queries : register(t5);
StructuredBuffer<uint> LeafToBody : register(t6);                 // leaf row -> body index
StructuredBuffer<b3RigidBodyData> Bodies : register(t7);
StructuredBuffer<b3Collidable> Collidables : register(t8);
StructuredBuffer<b3ConvexPolyhedronData> ConvexShapes : register(t9);
StructuredBuffer<b3GpuFace> Faces : register(t10);
StructuredBuffer<int> BodyOwnerRoots : register(t11);             // per body; only read when useOwnerFilter

RWStructuredBuffer<b3QueryHit> OutHits : register(u0);

int isLeafNode(int index) { return (index >> 31) == 0; }
int indexMarkerRemoved(int index) { return index & (~0x80000000); }

float3 quatRotate(float4 q, float3 v)
{
	const float3 qv = q.xyz;
	return v + 2.f * cross(qv, cross(qv, v) + q.w * v);
}

float3 quatRotateInv(float4 q, float3 v)
{
	return quatRotate(float4(-q.xyz, q.w), v);
}

//! Slab test against an AABB grown by radius; tEntry receives the entry distance so the traversal
//! can skip subtrees that cannot beat the best hit. Same NaN discipline as CSRayTraverse: a
//! component-parallel ray divides by zero and min/max return the non-NaN operand.
bool rayHitsAabb(float3 rayOrigin, float rayLength, float3 rayDir, b3Aabb aabb, float radius,
				 out float tEntry, out float3 entryNormal)
{
	const bool3 isNegative = rayDir < 0.f.xxx;
	const float3 lo = osAabbMinF(aabb) - radius.xxx;
	const float3 hi = osAabbMaxF(aabb) + radius.xxx;
	const float3 nearPlane = float3(isNegative.x ? hi.x : lo.x,
									isNegative.y ? hi.y : lo.y,
									isNegative.z ? hi.z : lo.z);
	const float3 farPlane = float3(isNegative.x ? lo.x : hi.x,
								   isNegative.y ? lo.y : hi.y,
								   isNegative.z ? lo.z : hi.z);

	const float3 tMin = (nearPlane - rayOrigin) / rayDir;
	const float3 tMax = (farPlane - rayOrigin) / rayDir;

	float tMinFinal = max(tMin.z, max(tMin.y, max(tMin.x, 0.f)));
	const float tMaxFinal = min(tMax.z, min(tMax.y, min(tMax.x, rayLength)));

	// The slab that entered last is the one the ray crossed into the box through. A NaN tMin
	// never equals tMinFinal, so a parallel axis is never picked.
	entryNormal = float3(0.f, 0.f, 0.f);
	if (tMin.x == tMinFinal) entryNormal = float3(isNegative.x ? 1.f : -1.f, 0.f, 0.f);
	else if (tMin.y == tMinFinal) entryNormal = float3(0.f, isNegative.y ? 1.f : -1.f, 0.f);
	else if (tMin.z == tMinFinal) entryNormal = float3(0.f, 0.f, isNegative.z ? 1.f : -1.f);

	tEntry = tMinFinal;
	return tMinFinal <= tMaxFinal;
}

//! Ray vs sphere of radius R centred at c. Start-inside is a miss. t is a distance along dir.
bool rayHitsSphere(float3 o, float3 dir, float rayLength, float3 c, float R, out float t, out float3 n)
{
	t = 0.f;
	n = float3(0.f, 0.f, 0.f);
	const float3 m = o - c;
	const float b = dot(m, dir);
	const float cc = dot(m, m) - R * R;
	if (cc <= 0.f)
		return false;   // inside
	if (b > 0.f)
		return false;   // pointing away
	const float disc = b * b - cc;
	if (disc < 0.f)
		return false;
	t = -b - sqrt(disc);
	if (t < 0.f || t > rayLength)
		return false;
	n = (o + dir * t - c) / R;
	return true;
}

//! Ray (or swept sphere centre) vs convex hull by clipping against every face half-space, in the
//! shape's local frame. radius > 0 offsets the planes outward - see the header note.
bool rayHitsHull(float3 o, float3 dir, float rayLength, b3RigidBodyData body,
				 b3ConvexPolyhedronData hull, float radius, out float t, out float3 worldNormal)
{
	t = 0.f;
	worldNormal = float3(0.f, 0.f, 0.f);

	const float3 lo = quatRotateInv(body.quat, o - body.pos.xyz);
	const float3 ld = quatRotateInv(body.quat, dir);

	float tEnter = 0.f;
	float tExit = rayLength;
	float3 enterNormal = float3(0.f, 0.f, 0.f);
	bool enteredAny = false;

	// Compile-time bound with a runtime break: FXC sizes code from the bound, not the data.
	[loop]
	for (int f = 0; f < QUERY_MAX_FACES; ++f)
	{
		if (f >= hull.numFaces)
			break;
		const float4 plane = Faces[hull.faceOffset + f].plane;
		const float3 n = plane.xyz;
		const float dist = dot(n, lo) + plane.w - radius;   // >0: origin outside this half-space
		const float denom = dot(n, ld);

		if (abs(denom) < 1e-12f)
		{
			if (dist > 0.f)
				return false;   // parallel and outside: can never enter
			continue;
		}

		const float tp = -dist / denom;
		if (denom < 0.f)
		{
			// Entering this half-space.
			if (tp > tEnter)
			{
				tEnter = tp;
				enterNormal = n;
				enteredAny = true;
			}
		}
		else
		{
			tExit = min(tExit, tp);
		}
		if (tEnter > tExit)
			return false;
	}

	// No entering plane beat t=0 means the origin is already inside: report no hit, as Bullet does.
	if (!enteredAny || tEnter <= 0.f)
		return false;
	if (isnan(tEnter) || isinf(tEnter))
		return false;

	t = tEnter;
	worldNormal = quatRotate(body.quat, enterNormal);
	return true;
}

//! Exact (or documented-conservative) test of one query against one body. t is a distance.
bool queryHitsBody(float3 o, float3 dir, float rayLength, float radius, uint bodyIndex,
				   out float t, out float3 n)
{
	t = 0.f;
	n = float3(0.f, 0.f, 0.f);

	const b3RigidBodyData body = Bodies[bodyIndex];
	if (body.collidableIdx < 0)
		return false;
	const b3Collidable col = Collidables[body.collidableIdx];

	if (col.shapeType == SHAPE_SPHERE)
		return rayHitsSphere(o, dir, rayLength, body.pos.xyz, col.radius + radius, t, n);

	if (col.shapeType == SHAPE_CONVEX_HULL && col.shapeIndex >= 0)
		return rayHitsHull(o, dir, rayLength, body, ConvexShapes[col.shapeIndex], radius, t, n);

	// Shapes without an exact kernel here: the (conservative) world bound stands in for them.
	float3 entryNormal;
	if (!rayHitsAabb(o, rayLength, dir, WorldAabbs[bodyIndex], radius, t, entryNormal))
		return false;
	if (t <= 0.f)
		return false;   // inside
	n = entryNormal;
	return true;
}

[numthreads(WG_SIZE, 1, 1)]
void CSQueryTraverse(uint3 tid : SV_DispatchThreadID)
{
	const uint queryIndex = tid.x;
	if (queryIndex >= Params[0].numQueries)
		return;

	b3QueryHit best;
	best.pointAndFraction = float4(0.f, 0.f, 0.f, 1.f);
	best.normal = float4(0.f, 0.f, 0.f, 0.f);
	best.bodyIndex = -1;
	best.hit = 0;
	best.pad0 = 0;
	best.pad1 = 0;

	const b3Query q = Queries[queryIndex];
	const float3 rayFrom = q.from.xyz;
	const float3 delta = q.to.xyz - rayFrom;
	const float rayLength = length(delta);
	// A zero-length query would produce a NaN direction and match nothing; skip it explicitly.
	if (rayLength <= 0.f || isnan(rayLength) || isinf(rayLength))
	{
		OutHits[queryIndex] = best;
		return;
	}
	const float3 rayDir = delta / rayLength;
	const float radius = (q.kind == QUERY_KIND_SPHERE) ? max(q.radius, 0.f) : 0.f;
	const bool ownerFilter = Params[0].useOwnerFilter != 0 && q.ownerRoot >= 0;

	float bestT = rayLength;

	int stack[TRAVERSE_MAX_STACK_SIZE];
	int stackSize = 1;
	stack[0] = Params[0].rootIndex;

	while (stackSize > 0)
	{
		const int encoded = stack[--stackSize];
		const bool isLeaf = isLeafNode(encoded) != 0;
		const int nodeIndex = indexMarkerRemoved(encoded);

		b3Aabb nodeAabb;
		uint bodyIndex = 0;
		if (isLeaf)
		{
			bodyIndex = LeafToBody[MortonCodes[nodeIndex].y];
			nodeAabb = WorldAabbs[bodyIndex];
		}
		else
		{
			nodeAabb = InternalAabbs[nodeIndex];
		}

		float tEntry;
		float3 entryNormal;
		if (!rayHitsAabb(rayFrom, rayLength, rayDir, nodeAabb, radius, tEntry, entryNormal))
			continue;
		// Closest-hit prune: nothing inside this bound can beat what is already found. The gate
		// below rejects a hit ahead of its own bound, so the result never depends on visit order.
		const float gate = tEntry - 1e-5f * rayLength;
		if (gate > bestT)
			continue;

		if (isLeaf)
		{
			if (ownerFilter && BodyOwnerRoots[bodyIndex] == q.ownerRoot)
				continue;

			float t;
			float3 n;
			if (queryHitsBody(rayFrom, rayDir, rayLength, radius, bodyIndex, t, n) && t >= gate && t < bestT)
			{
				bestT = t;
				best.pointAndFraction = float4(rayFrom + rayDir * t, t / rayLength);
				best.normal = float4(n, 0.f);
				best.bodyIndex = (int)bodyIndex;
				best.hit = 1;
			}
		}
		else if (stackSize + 2 <= TRAVERSE_MAX_STACK_SIZE)
		{
			stack[stackSize++] = ChildNodes[2 * nodeIndex + 0];
			stack[stackSize++] = ChildNodes[2 * nodeIndex + 1];
		}
	}

	OutHits[queryIndex] = best;
}
