// Port of Bullet3OpenCL parallelLinearBvh.cl - leaf index ranges + overlapping pair finding.
// Phase 4c. The pair count is produced GPU-side into an append buffer, which is what
// copyStructureCount + dispatchComputeShaderIndirect were built for: narrowphase can be sized
// from it without a readback stall.

// Included by B3LbvhPairs.hlsl (f32) and B3LbvhPairsDS.hlsl (df64) - see B3Precision.hlsli.

#include "B3Precision.hlsli"

#define WG_SIZE 128
#define TRAVERSE_MAX_STACK_SIZE 128

struct PairParams
{
	uint numLeafNodes;
	uint numInternalNodes;
	int rootIndex;   // marker-encoded, as written by CSBuildInternalNodes
	uint numLargeAabbs;
	uint useSleepGate;
	uint pad0;
	uint pad1;
	uint pad2;
};

StructuredBuffer<PairParams> Params : register(t0);
StructuredBuffer<int> ChildNodes : register(t1);          // 2 ints per internal node
StructuredBuffer<uint2> MortonCodes : register(t2);       // (key, smallIndex), sorted
StructuredBuffer<b3Aabb> RigidAabbs : register(t3);       // the SMALL set, indexed by small index
StructuredBuffer<b3Aabb> InternalAabbs : register(t4);
StructuredBuffer<int2> LeafIndexRanges : register(t5);
// The tree is built from the small set only, so every emitted index is mapped back through these
// to the caller's original, unpartitioned array order. Identity when the scene needed no split.
StructuredBuffer<uint> SmallToOriginal : register(t7);
StructuredBuffer<b3Aabb> LargeAabbs : register(t8);
StructuredBuffer<uint> LargeToOriginal : register(t9);
// Bit 31 = asleep, as written by the sleep kernel. Only read when useSleepGate is set.
StructuredBuffer<uint> SleepStates : register(t10);
#define SLEEP_ASLEEP_BIT 0x80000000u

RWStructuredBuffer<int2> OutLeafIndexRanges : register(u1);
// u0 deliberately: the proven G3 gate path binds its append buffer at slot 0.
AppendStructuredBuffer<uint2> OutPairs : register(u0);

int isLeafNode(int index) { return (index >> 31) == 0; }
int indexMarkerRemoved(int index) { return index & (~0x80000000); }

bool testAabbAgainstAabb(b3Aabb a, b3Aabb b) { return osAabbOverlap(a, b); }

//! Contiguous leaf range each internal node covers. Walking left children gives the lowest leaf,
//! right children the highest - this is what makes the duplicate-pair rejection below possible.
[numthreads(WG_SIZE, 1, 1)]
void CSFindLeafIndexRanges(uint3 tid : SV_DispatchThreadID)
{
	const int numInternalNodes = (int)Params[0].numInternalNodes;
	const int nodeIndex = (int)tid.x;
	if (nodeIndex >= numInternalNodes)
		return;

	int2 range;

	int lowest = ChildNodes[2 * nodeIndex + 0];
	// Bounded by tree depth; the counter stops a malformed tree hanging the GPU.
	int guard = 0;
	while (isLeafNode(lowest) == 0 && guard++ < numInternalNodes)
		lowest = ChildNodes[2 * indexMarkerRemoved(lowest) + 0];
	range.x = lowest;

	int highest = ChildNodes[2 * nodeIndex + 1];
	guard = 0;
	while (isLeafNode(highest) == 0 && guard++ < numInternalNodes)
		highest = ChildNodes[2 * indexMarkerRemoved(highest) + 1];
	range.y = highest;

	OutLeafIndexRanges[nodeIndex] = range;
}

//! One query per leaf, descending the tree. Threads are indexed by SORTED leaf slot rather than
//! rigid index so neighbouring threads walk spatially adjacent subtrees.
[numthreads(WG_SIZE, 1, 1)]
void CSCalculateOverlappingPairs(uint3 tid : SV_DispatchThreadID)
{
	const uint numLeafNodes = Params[0].numLeafNodes;
	const int queryLeafSlot = (int)tid.x;
	if ((uint)queryLeafSlot >= numLeafNodes)
		return;

	const uint querySmallIndex = MortonCodes[queryLeafSlot].y;
	const b3Aabb queryAabb = RigidAabbs[querySmallIndex];

	// A sleeping body has not moved, so it never has to query: every pair it belongs to is emitted
	// by the awake side, which reports sleeping targets regardless of slot order (see below).
	const bool sleepGate = Params[0].useSleepGate != 0;
	if (sleepGate && (SleepStates[SmallToOriginal[querySmallIndex]] & SLEEP_ASLEEP_BIT) != 0)
		return;

	int stack[TRAVERSE_MAX_STACK_SIZE];
	int stackSize = 1;
	stack[0] = Params[0].rootIndex;

	while (stackSize > 0)
	{
		const int encoded = stack[--stackSize];
		const bool isLeaf = isLeafNode(encoded) != 0;
		const int nodeIndex = indexMarkerRemoved(encoded);

		// Dedup + self-pair removal: only report a pair from the lower-slotted leaf's query.
		// Correctness depends only on this leaf test. The internal-node prune below is a pure
		// optimisation using the contiguous leaf range each subtree covers, and is kept
		// separable so a wrong range can never cost pairs, only speed.
		if (isLeaf && nodeIndex <= queryLeafSlot)
		{
			// Slot-order dedup is only valid between two leaves that BOTH query. A sleeping target
			// never does, so its pair has to be claimed here or it is lost outright.
			if (!sleepGate || nodeIndex == queryLeafSlot)
				continue;
			if ((SleepStates[SmallToOriginal[MortonCodes[nodeIndex].y]] & SLEEP_ASLEEP_BIT) == 0)
				continue;
		}
		// Gated, a lower-slotted subtree can still hold sleeping leaves only this query claims, so
		// the ranges must not be read at all - which is what lets the host skip building them.
		if (!isLeaf && !sleepGate)
		{
			const int highestLeaf = LeafIndexRanges[nodeIndex].y;
			if (highestLeaf > 0 && highestLeaf <= queryLeafSlot)
				continue;
		}

		b3Aabb nodeAabb;
		uint smallIndex = 0;
		if (isLeaf)
		{
			smallIndex = MortonCodes[nodeIndex].y;
			nodeAabb = RigidAabbs[smallIndex];
		}
		else
		{
			nodeAabb = InternalAabbs[nodeIndex];
		}

		if (!testAabbAgainstAabb(queryAabb, nodeAabb))
			continue;

		if (isLeaf)
		{
			// Caller-array indices, not leaf slots or small-set slots - both are internal details.
			OutPairs.Append(uint2(SmallToOriginal[querySmallIndex], SmallToOriginal[smallIndex]));
		}
		else if (stackSize + 2 <= TRAVERSE_MAX_STACK_SIZE)
		{
			stack[stackSize++] = ChildNodes[2 * nodeIndex + 0];
			stack[stackSize++] = ChildNodes[2 * nodeIndex + 1];
		}
		// Silent drop on stack overflow matches the original; a tree deep enough to overflow
		// 128 entries would need ~2^128 leaves, so this is unreachable in practice.
	}
}

// ---------------------------------------------------------------------------------------------
// Large-AABB side list. A body whose AABB dwarfs the rest inflates the Morton grid until the
// small bodies' codes stop discriminating, so it is kept out of the tree and tested brute force.
// ---------------------------------------------------------------------------------------------

//! Every small AABB against every large one. Port of plbvhLargeAabbAabbTest.
[numthreads(WG_SIZE, 1, 1)]
void CSLargeAabbAabbTest(uint3 tid : SV_DispatchThreadID)
{
	const uint numSmall = Params[0].numLeafNodes;
	const uint numLarge = Params[0].numLargeAabbs;
	const uint smallIndex = tid.x;
	if (smallIndex >= numSmall)
		return;

	const b3Aabb smallAabb = RigidAabbs[smallIndex];
	const uint originalSmall = SmallToOriginal[smallIndex];

	[loop]
	for (uint i = 0; i < numLarge; ++i)
	{
		if (testAabbAgainstAabb(smallAabb, LargeAabbs[i]))
			OutPairs.Append(uint2(LargeToOriginal[i], originalSmall));
	}
}

//! Large against large, i < j only. Not in the original, which drops these pairs outright; needed
//! here so the split stays a pure performance change and never loses a pair.
[numthreads(WG_SIZE, 1, 1)]
void CSLargeLargeAabbTest(uint3 tid : SV_DispatchThreadID)
{
	const uint numLarge = Params[0].numLargeAabbs;
	const uint i = tid.x;
	if (i >= numLarge)
		return;

	const b3Aabb aabbI = LargeAabbs[i];
	const uint originalI = LargeToOriginal[i];

	[loop]
	for (uint j = i + 1; j < numLarge; ++j)
	{
		if (testAabbAgainstAabb(aabbI, LargeAabbs[j]))
			OutPairs.Append(uint2(originalI, LargeToOriginal[j]));
	}
}

// ---------------------------------------------------------------------------------------------
// Raycast against the same tree. Needs only AABBs + tree - no collidables or convex shapes - so
// it lands well before narrowphase. Output is ray->candidate-body pairs (broadphase raycast);
// exact ray-vs-shape is a later narrowphase concern.
// ---------------------------------------------------------------------------------------------

struct b3RayInfo
{
	float4 from;
	float4 to;
};

StructuredBuffer<b3RayInfo> Rays : register(t6);
AppendStructuredBuffer<uint2> OutRayPairs : register(u2);

//! Slab test. NaN handling matters: a component-parallel ray divides by zero, and the original
//! relies on fmin/fmax returning the non-NaN operand. HLSL min/max do the same for one NaN input.
bool rayIntersectsAabb(float3 rayOrigin, float rayLength, float3 rayDir, b3Aabb aabb)
{
	const bool3 isNegative = rayDir < 0.f.xxx;

	// Hi terms only in df64 mode: a ray is authored in float world space anyway, so a df64 slab
	// test would be precision the query itself does not carry.
	const float3 lo = osAabbMinF(aabb);
	const float3 hi = osAabbMaxF(aabb);
	const float3 nearPlane = float3(isNegative.x ? hi.x : lo.x,
									isNegative.y ? hi.y : lo.y,
									isNegative.z ? hi.z : lo.z);
	const float3 farPlane = float3(isNegative.x ? lo.x : hi.x,
								   isNegative.y ? lo.y : hi.y,
								   isNegative.z ? lo.z : hi.z);

	const float3 tMin = (nearPlane - rayOrigin) / rayDir;
	const float3 tMax = (farPlane - rayOrigin) / rayDir;

	float tMinFinal = max(tMin.z, max(tMin.y, max(tMin.x, 0.f)));
	float tMaxFinal = min(tMax.z, min(tMax.y, min(tMax.x, rayLength)));

	return tMinFinal <= tMaxFinal;
}

[numthreads(WG_SIZE, 1, 1)]
void CSRayTraverse(uint3 tid : SV_DispatchThreadID)
{
	const uint numRays = Params[0].numLeafNodes;  // host reuses this slot for the ray count
	const uint rayIndex = tid.x;
	if (rayIndex >= numRays)
		return;

	const float3 rayFrom = Rays[rayIndex].from.xyz;
	const float3 rayTo = Rays[rayIndex].to.xyz;
	const float3 delta = rayTo - rayFrom;
	const float rayLength = length(delta);
	// A zero-length ray would produce a NaN direction and match nothing; skip it explicitly.
	if (rayLength <= 0.f)
		return;
	const float3 rayDir = delta / rayLength;

	int stack[TRAVERSE_MAX_STACK_SIZE];
	int stackSize = 1;
	stack[0] = Params[0].rootIndex;

	while (stackSize > 0)
	{
		const int encoded = stack[--stackSize];
		const bool isLeaf = isLeafNode(encoded) != 0;
		const int nodeIndex = indexMarkerRemoved(encoded);

		b3Aabb nodeAabb;
		uint smallIndex = 0;
		if (isLeaf)
		{
			smallIndex = MortonCodes[nodeIndex].y;
			nodeAabb = RigidAabbs[smallIndex];
		}
		else
		{
			nodeAabb = InternalAabbs[nodeIndex];
		}

		if (!rayIntersectsAabb(rayFrom, rayLength, rayDir, nodeAabb))
			continue;

		if (isLeaf)
		{
			OutRayPairs.Append(uint2(rayIndex, SmallToOriginal[smallIndex]));
		}
		else if (stackSize + 2 <= TRAVERSE_MAX_STACK_SIZE)
		{
			stack[stackSize++] = ChildNodes[2 * nodeIndex + 0];
			stack[stackSize++] = ChildNodes[2 * nodeIndex + 1];
		}
	}
}

//! Rays against the flat large list. Port of plbvhLargeAabbRayTest.
[numthreads(WG_SIZE, 1, 1)]
void CSLargeAabbRayTest(uint3 tid : SV_DispatchThreadID)
{
	const uint numRays = Params[0].numLeafNodes;  // host reuses this slot for the ray count
	const uint numLarge = Params[0].numLargeAabbs;
	const uint rayIndex = tid.x;
	if (rayIndex >= numRays)
		return;

	const float3 rayFrom = Rays[rayIndex].from.xyz;
	const float3 delta = Rays[rayIndex].to.xyz - rayFrom;
	const float rayLength = length(delta);
	if (rayLength <= 0.f)
		return;
	const float3 rayDir = delta / rayLength;

	[loop]
	for (uint i = 0; i < numLarge; ++i)
	{
		if (rayIntersectsAabb(rayFrom, rayLength, rayDir, LargeAabbs[i]))
			OutRayPairs.Append(uint2(rayIndex, LargeToOriginal[i]));
	}
}
