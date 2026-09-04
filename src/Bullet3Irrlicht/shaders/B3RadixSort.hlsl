// PREFER_FLOW_CONTROL - data-dependent loops here must stay real loops; Release flattens
// branches by default, which breaks stack traversal and runtime-bounded clipping.
// Port of Bullet3OpenCL RadixSort32Kernels.cl - stable LSD radix sort over (key,value) pairs.
// 4 bits per pass / 16 buckets / 256 elements per block, matching the original.
// SM5.0 groupshared+barriers; see B3PrefixScan.hlsl for why not wave intrinsics.
//
// Stability is not optional: multi-pass LSD only works if each pass is stable, and Bullet's
// contact-determinism pass relies on that same property across four sorts.

#define WG_SIZE 128
#define ELEMS_PER_BLOCK (WG_SIZE * 2)
#define BITS_PER_PASS 4
#define NUM_BUCKETS (1 << BITS_PER_PASS)

struct RadixParams
{
	uint numElems;      // element count, or the capacity when useLiveCount is set
	uint numBlocks;
	uint bitShift;
	uint useLiveCount;
};

StructuredBuffer<RadixParams> Params : register(t0);
StructuredBuffer<uint2> SrcPairs : register(t1);
StructuredBuffer<uint> ScannedHistogram : register(t2);

RWStructuredBuffer<uint> Histogram : register(u0);
RWStructuredBuffer<uint2> DstPairs : register(u1);
// Append counter landed by CopyStructureCount. Raw, not structured: an indirect-args buffer has
// no SRV, so it is read through its UAV. Only read when useLiveCount is set.
RWByteAddressBuffer LiveCount : register(u2);

// Blocks past the live count still run (the dispatch covers the capacity) and write zero
// histogram entries, which is what keeps the scan layout valid without a CPU-side count.
uint liveNumElems()
{
	const uint cap = Params[0].numElems;
	if (Params[0].useLiveCount == 0)
		return cap;
	return min(LiveCount.Load(0), cap);
}

groupshared uint ldsHist[NUM_BUCKETS];
groupshared uint2 ldsPairs[ELEMS_PER_BLOCK];
groupshared uint2 ldsSwap[ELEMS_PER_BLOCK];
groupshared uint ldsScan[ELEMS_PER_BLOCK];
groupshared uint ldsTotalZeros;
groupshared uint ldsDigitStart[NUM_BUCKETS];

uint digitOf(uint key, uint shift)
{
	return (key >> shift) & (NUM_BUCKETS - 1);
}

// Blelloch exclusive scan over ldsScan. Barriers stay in group-uniform control flow.
void scanLdsExclusive(int lIdx)
{
	int n = ELEMS_PER_BLOCK;
	int offset = 1;

	for (int nActive = n >> 1; nActive > 0; nActive >>= 1, offset <<= 1)
	{
		GroupMemoryBarrierWithGroupSync();
		for (int i = lIdx; i < nActive; i += WG_SIZE)
		{
			int ai = offset * (2 * i + 1) - 1;
			int bi = offset * (2 * i + 2) - 1;
			ldsScan[bi] += ldsScan[ai];
		}
	}

	GroupMemoryBarrierWithGroupSync();
	if (lIdx == 0)
		ldsScan[n - 1] = 0;

	offset >>= 1;
	for (int nActive2 = 1; nActive2 < n; nActive2 <<= 1, offset >>= 1)
	{
		GroupMemoryBarrierWithGroupSync();
		for (int j = lIdx; j < nActive2; j += WG_SIZE)
		{
			int ai = offset * (2 * j + 1) - 1;
			int bi = offset * (2 * j + 2) - 1;
			uint t = ldsScan[ai];
			ldsScan[ai] = ldsScan[bi];
			ldsScan[bi] += t;
		}
	}
	GroupMemoryBarrierWithGroupSync();
}

//! Per-block digit histogram. Laid out digit-major (digit * numBlocks + block) so one plain
//! exclusive scan of the whole array yields each block's global bucket offset directly.
[numthreads(WG_SIZE, 1, 1)]
void CSStreamCount(uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID)
{
	const uint numElems = liveNumElems();
	const uint numBlocks = Params[0].numBlocks;
	const uint shift = Params[0].bitShift;
	const int lIdx = (int)gtid.x;

	if (lIdx < NUM_BUCKETS)
		ldsHist[lIdx] = 0;
	GroupMemoryBarrierWithGroupSync();

	const uint base = gid.x * ELEMS_PER_BLOCK;
	for (uint e = 0; e < 2; ++e)
	{
		const uint idx = base + lIdx + e * WG_SIZE;
		if (idx < numElems)
			InterlockedAdd(ldsHist[digitOf(SrcPairs[idx].x, shift)], 1);
	}
	GroupMemoryBarrierWithGroupSync();

	if (lIdx < NUM_BUCKETS)
		Histogram[lIdx * numBlocks + gid.x] = ldsHist[lIdx];
}

//! Locally sorts the block by the current digit (four stable 1-bit splits), then scatters each
//! element to ScannedHistogram[digit][block] + its rank within that digit.
[numthreads(WG_SIZE, 1, 1)]
void CSSortAndScatter(uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID)
{
	const uint numElems = liveNumElems();
	const uint numBlocks = Params[0].numBlocks;
	const uint shift = Params[0].bitShift;
	const int lIdx = (int)gtid.x;
	const uint base = gid.x * ELEMS_PER_BLOCK;

	// Out-of-range slots get a key that sorts last and a sentinel value; they are never written.
	for (uint e = 0; e < 2; ++e)
	{
		const uint local = lIdx + e * WG_SIZE;
		const uint idx = base + local;
		ldsPairs[local] = (idx < numElems) ? SrcPairs[idx] : uint2(0xFFFFFFFFu, 0xFFFFFFFFu);
	}
	GroupMemoryBarrierWithGroupSync();

	// Four 1-bit stable splits => the block ends up stably sorted by the 4-bit digit.
	for (uint bit = 0; bit < BITS_PER_PASS; ++bit)
	{
		const uint mask = 1u << (shift + bit);

		for (uint e2 = 0; e2 < 2; ++e2)
		{
			const uint local = lIdx + e2 * WG_SIZE;
			// Count zeros: those keep relative order at the front.
			ldsScan[local] = (ldsPairs[local].x & mask) ? 0u : 1u;
		}
		GroupMemoryBarrierWithGroupSync();

		if (lIdx == 0)
			ldsTotalZeros = ldsScan[ELEMS_PER_BLOCK - 1];
		GroupMemoryBarrierWithGroupSync();

		// ldsScan[last] is consumed above before the scan overwrites it.
		uint zeroFlagLast = ldsTotalZeros;
		scanLdsExclusive(lIdx);

		if (lIdx == 0)
			ldsTotalZeros = ldsScan[ELEMS_PER_BLOCK - 1] + zeroFlagLast;
		GroupMemoryBarrierWithGroupSync();

		const uint totalZeros = ldsTotalZeros;
		for (uint e3 = 0; e3 < 2; ++e3)
		{
			const uint local = lIdx + e3 * WG_SIZE;
			const uint2 pair = ldsPairs[local];
			const uint rankOfZeros = ldsScan[local];
			const uint dest = (pair.x & mask) ? (totalZeros + local - rankOfZeros) : rankOfZeros;
			ldsSwap[dest] = pair;
		}
		GroupMemoryBarrierWithGroupSync();

		for (uint e4 = 0; e4 < 2; ++e4)
		{
			const uint local = lIdx + e4 * WG_SIZE;
			ldsPairs[local] = ldsSwap[local];
		}
		GroupMemoryBarrierWithGroupSync();
	}

	// Where each digit run begins inside this now-locally-sorted block.
	if (lIdx < NUM_BUCKETS)
		ldsDigitStart[lIdx] = ELEMS_PER_BLOCK;
	GroupMemoryBarrierWithGroupSync();

	for (uint e5 = 0; e5 < 2; ++e5)
	{
		const uint local = lIdx + e5 * WG_SIZE;
		const uint d = digitOf(ldsPairs[local].x, shift);
		if (local == 0 || digitOf(ldsPairs[local - 1].x, shift) != d)
			InterlockedMin(ldsDigitStart[d], local);
	}
	GroupMemoryBarrierWithGroupSync();

	for (uint e6 = 0; e6 < 2; ++e6)
	{
		const uint local = lIdx + e6 * WG_SIZE;
		const uint2 pair = ldsPairs[local];
		if (pair.x == 0xFFFFFFFFu && pair.y == 0xFFFFFFFFu)
			continue;  // padding slot

		const uint d = digitOf(pair.x, shift);
		const uint dest = ScannedHistogram[d * numBlocks + gid.x] + (local - ldsDigitStart[d]);
		if (dest < numElems)
			DstPairs[dest] = pair;
	}
}
