// PREFER_FLOW_CONTROL - data-dependent loops here must stay real loops; Release flattens
// branches by default, which breaks stack traversal and runtime-bounded clipping.
// Canonical pair order for the GPU-resident broadphase. An append buffer hands out slots in
// thread-scheduling order, so the pair SET is stable but its ORDER is not. These two kernels
// bracket two stable LSD radix passes (high index first, then low) - the same discipline
// b3IrrlichtLbvh::sortPairsForDeterminism uses on the CPU, so both paths yield the same bytes.
// Precision-agnostic: one file serves the f32 and df64 kernel sets.

#define WG_SIZE 128

struct PairSortParams
{
	uint capacity;   // pair buffer capacity; the live count is clamped to it
	uint pad0;
	uint pad1;
	uint pad2;
};

StructuredBuffer<PairSortParams> Params : register(t0);
StructuredBuffer<uint2> InPairs : register(t1);

// Append counter landed by CopyStructureCount. Raw, not structured: an indirect-args buffer has
// no SRV, so it is read through its UAV.
RWByteAddressBuffer LiveCount : register(u0);
RWStructuredBuffer<uint2> OutPairs : register(u1);

uint liveCount()
{
	return min(LiveCount.Load(0), Params[0].capacity);
}

//! (query, other) -> (key = high, value = low): the input of the first, least-significant pass.
[numthreads(WG_SIZE, 1, 1)]
void CSNormalisePairs(uint3 tid : SV_DispatchThreadID)
{
	if (tid.x >= liveCount())
		return;
	const uint2 p = InPairs[tid.x];
	OutPairs[tid.x] = uint2(max(p.x, p.y), min(p.x, p.y));
}

//! (key, value) -> (value, key): re-keys the first pass's output on the low index for the second.
[numthreads(WG_SIZE, 1, 1)]
void CSSwapKeyValue(uint3 tid : SV_DispatchThreadID)
{
	if (tid.x >= liveCount())
		return;
	const uint2 p = InPairs[tid.x];
	OutPairs[tid.x] = uint2(p.y, p.x);
}
