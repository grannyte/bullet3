// Port of Bullet3OpenCL parallelLinearBvh.cl - GPU-built linear BVH (Karras-style).
// Phase 4a: merged world AABB, then Morton codes paired with leaf indices for the radix sort.
//
// Chosen over the uniform grid broadphase deliberately: this world is 1u = 1m across a solar
// system with debris-to-station size variance, so a grid has no workable cell size or extent.
// Included by B3Lbvh.hlsl (f32) and B3LbvhDS.hlsl (df64) - see B3Precision.hlsli.

#include "B3Precision.hlsli"

#define WG_SIZE 128
#define MORTON_GRID 1024

struct LbvhParams
{
	uint numAabbs;
	uint pad0;
	uint pad1;
	uint pad2;
};

StructuredBuffer<LbvhParams> Params : register(t0);
StructuredBuffer<b3Aabb> WorldAabbs : register(t1);
StructuredBuffer<b3AabbF> MergedAabb : register(t2);
StructuredBuffer<uint> SeparateIndices : register(t3);
// Leaf slot -> body index. Identity for a whole-world build, the awake/sleeping list for a subset;
// gathered through unconditionally so both cases stay ONE code path.
StructuredBuffer<uint> LeafToBody : register(t4);

RWStructuredBuffer<b3AabbF> MergedAabbOut : register(u0);
RWStructuredBuffer<uint2> MortonCodes : register(u1);
RWStructuredBuffer<b3Aabb> SeparatedAabbs : register(u2);

groupshared float3 ldsMin[WG_SIZE];
groupshared float3 ldsMax[WG_SIZE];

//! Whole-world AABB in one group via a grid-stride loop plus an LDS tree reduce.
//! min/max are exact and order-independent, so this needs no fixed reduction order the way a
//! float sum would (plan rule R1 does not bite here).
[numthreads(WG_SIZE, 1, 1)]
void CSMergeAabbs(uint3 gtid : SV_GroupThreadID)
{
	const uint numAabbs = Params[0].numAabbs;
	const uint lIdx = gtid.x;

	float3 mn = float3(1e30f, 1e30f, 1e30f);
	float3 mx = float3(-1e30f, -1e30f, -1e30f);

	for (uint i = lIdx; i < numAabbs; i += WG_SIZE)
	{
		const b3Aabb a = WorldAabbs[LeafToBody[i]];
		mn = min(mn, osAabbMinF(a));
		mx = max(mx, osAabbMaxF(a));
	}

	ldsMin[lIdx] = mn;
	ldsMax[lIdx] = mx;
	GroupMemoryBarrierWithGroupSync();

	for (uint s = WG_SIZE / 2; s > 0; s >>= 1)
	{
		if (lIdx < s)
		{
			ldsMin[lIdx] = min(ldsMin[lIdx], ldsMin[lIdx + s]);
			ldsMax[lIdx] = max(ldsMax[lIdx], ldsMax[lIdx + s]);
		}
		GroupMemoryBarrierWithGroupSync();
	}

	if (lIdx == 0)
	{
		b3AabbF merged;
		merged.minVec = float4(ldsMin[0], 0.f);
		merged.maxVec = float4(ldsMax[0], 0.f);
		MergedAabbOut[0] = merged;
	}
}

//! Gathers a subset of the world AABBs into a compact buffer, so the tree can be built from the
//! small set alone. Params[0].numAabbs is the SUBSET size here, not the world AABB count.
[numthreads(WG_SIZE, 1, 1)]
void CSSeparateAabbs(uint3 tid : SV_DispatchThreadID)
{
	const uint outIndex = tid.x;
	if (outIndex >= Params[0].numAabbs)
		return;

	SeparatedAabbs[outIndex] = WorldAabbs[SeparateIndices[outIndex]];
}

//! Spreads the low 10 bits of x so they occupy every third bit.
uint interleaveBits(uint x)
{
	x &= 0x000003FF;
	x = (x ^ (x << 16)) & 0xFF0000FF;
	x = (x ^ (x << 8)) & 0x0300F00F;
	x = (x ^ (x << 4)) & 0x030C30C3;
	x = (x ^ (x << 2)) & 0x09249249;
	return x;
}

uint getMortonCode(uint x, uint y, uint z)
{
	return (interleaveBits(x) << 0) | (interleaveBits(y) << 1) | (interleaveBits(z) << 2);
}

//! One 30-bit Morton code per leaf, paired with its AABB index for the sort.
[numthreads(WG_SIZE, 1, 1)]
void CSAssignMortonCodes(uint3 tid : SV_DispatchThreadID)
{
	const uint numAabbs = Params[0].numAabbs;
	const uint leaf = tid.x;
	if (leaf >= numAabbs)
		return;

	const b3AabbF merged = MergedAabb[0];
	const float3 gridCenter = (merged.minVec.xyz + merged.maxVec.xyz) * 0.5f;
	const float3 gridCellSize = (merged.maxVec.xyz - merged.minVec.xyz) / (float)MORTON_GRID;

	// pair.y stays the LEAF slot, not the body: the pairs kernel maps it back via SmallToOriginal.
	const b3Aabb aabb = WorldAabbs[LeafToBody[leaf]];
	const float3 aabbCenter = (osAabbMinF(aabb) + osAabbMaxF(aabb)) * 0.5f;
	const float3 rel = aabbCenter - gridCenter;

	// A degenerate axis (all bodies coplanar) would divide by zero; collapse it to cell 0.
	float3 gridPosition;
	gridPosition.x = (gridCellSize.x > 0.f) ? rel.x / gridCellSize.x : 0.f;
	gridPosition.y = (gridCellSize.y > 0.f) ? rel.y / gridCellSize.y : 0.f;
	gridPosition.z = (gridCellSize.z > 0.f) ? rel.z / gridCellSize.z : 0.f;

	// floor() on the negative side only, matching the original: without it the centre cell at
	// (0,0,0) would be twice the width of every other cell.
	int3 discretePosition;
	discretePosition.x = (int)((gridPosition.x >= 0.f) ? gridPosition.x : floor(gridPosition.x));
	discretePosition.y = (int)((gridPosition.y >= 0.f) ? gridPosition.y : floor(gridPosition.y));
	discretePosition.z = (int)((gridPosition.z >= 0.f) ? gridPosition.z : floor(gridPosition.z));

	discretePosition = max(-512, min(discretePosition, 511)) + 512;

	uint2 pair;
	pair.x = getMortonCode((uint)discretePosition.x, (uint)discretePosition.y, (uint)discretePosition.z);
	pair.y = leaf;
	MortonCodes[leaf] = pair;
}
