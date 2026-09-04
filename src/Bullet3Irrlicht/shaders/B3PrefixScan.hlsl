// PREFER_FLOW_CONTROL - data-dependent loops here must stay real loops; Release flattens
// branches by default, which breaks stack traversal and runtime-bounded clipping.
// Port of Bullet3OpenCL PrefixScanKernels.cl - work-efficient (Blelloch) exclusive scan.
// SM5.0 groupshared+barriers, NOT wave intrinsics: those need SM6.0/DXIL, which the D3D11
// runtime cannot consume at any feature level. A wave fast path can be added for D3D12 later.

#define WG_SIZE 128
#define ELEMS_PER_BLOCK (WG_SIZE * 2)

struct ScanParams
{
	uint numElems;
	uint numBlocks;
	uint pad0;
	uint pad1;
};

StructuredBuffer<ScanParams> Params : register(t0);
StructuredBuffer<uint> Src : register(t1);
RWStructuredBuffer<uint> Dst : register(u0);
RWStructuredBuffer<uint> BlockSums : register(u1);

groupshared uint lds[ELEMS_PER_BLOCK];
groupshared uint ldsBlockSum;

// Every barrier below sits in group-uniform control flow: the loop bounds depend only on n,
// never on the thread index, so all threads reach each barrier the same number of times.
void ScanExclusiveLDS(int n, int lIdx, int lSize)
{
	int offset = 1;

	for (int nActive = n >> 1; nActive > 0; nActive >>= 1, offset <<= 1)
	{
		GroupMemoryBarrierWithGroupSync();
		for (int iIdx = lIdx; iIdx < nActive; iIdx += lSize)
		{
			int ai = offset * (2 * iIdx + 1) - 1;
			int bi = offset * (2 * iIdx + 2) - 1;
			lds[bi] += lds[ai];
		}
	}

	GroupMemoryBarrierWithGroupSync();
	if (lIdx == 0)
	{
		ldsBlockSum = lds[n - 1];
		lds[n - 1] = 0;
	}

	offset >>= 1;
	for (int nActive2 = 1; nActive2 < n; nActive2 <<= 1, offset >>= 1)
	{
		GroupMemoryBarrierWithGroupSync();
		for (int iIdx = lIdx; iIdx < nActive2; iIdx += lSize)
		{
			int ai = offset * (2 * iIdx + 1) - 1;
			int bi = offset * (2 * iIdx + 2) - 1;
			uint temp = lds[ai];
			lds[ai] = lds[bi];
			lds[bi] += temp;
		}
	}
	GroupMemoryBarrierWithGroupSync();
}

//! Per-block exclusive scan; each block's total lands in BlockSums[groupID].
[numthreads(WG_SIZE, 1, 1)]
void CSLocalScan(uint3 tid : SV_DispatchThreadID, uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID)
{
	const uint numElems = Params[0].numElems;
	const int lIdx = (int)gtid.x;
	const uint gIdx = tid.x;

	lds[2 * lIdx] = (2 * gIdx < numElems) ? Src[2 * gIdx] : 0;
	lds[2 * lIdx + 1] = (2 * gIdx + 1 < numElems) ? Src[2 * gIdx + 1] : 0;

	ScanExclusiveLDS(ELEMS_PER_BLOCK, lIdx, WG_SIZE);

	if (lIdx == 0)
		BlockSums[gid.x] = ldsBlockSum;

	if (2 * gIdx < numElems)
		Dst[2 * gIdx] = lds[2 * lIdx];
	if (2 * gIdx + 1 < numElems)
		Dst[2 * gIdx + 1] = lds[2 * lIdx + 1];
}

//! Scans BlockSums in place, in a single group. numBlocks must be <= ELEMS_PER_BLOCK.
[numthreads(WG_SIZE, 1, 1)]
void CSTopLevelScan(uint3 gtid : SV_GroupThreadID)
{
	const uint numBlocks = Params[0].numBlocks;
	const int lIdx = (int)gtid.x;

	lds[2 * lIdx] = (2 * (uint)lIdx < numBlocks) ? BlockSums[2 * lIdx] : 0;
	lds[2 * lIdx + 1] = (2 * (uint)lIdx + 1 < numBlocks) ? BlockSums[2 * lIdx + 1] : 0;

	ScanExclusiveLDS(ELEMS_PER_BLOCK, lIdx, WG_SIZE);

	if (2 * (uint)lIdx < numBlocks)
		BlockSums[2 * lIdx] = lds[2 * lIdx];
	if (2 * (uint)lIdx + 1 < numBlocks)
		BlockSums[2 * lIdx + 1] = lds[2 * lIdx + 1];
}

//! Adds each block's scanned offset back onto its elements.
[numthreads(WG_SIZE, 1, 1)]
void CSAddOffset(uint3 tid : SV_DispatchThreadID, uint3 gid : SV_GroupID)
{
	const uint numElems = Params[0].numElems;
	const uint offset = BlockSums[gid.x];
	const uint gIdx = tid.x;

	if (2 * gIdx < numElems)
		Dst[2 * gIdx] += offset;
	if (2 * gIdx + 1 < numElems)
		Dst[2 * gIdx + 1] += offset;
}
