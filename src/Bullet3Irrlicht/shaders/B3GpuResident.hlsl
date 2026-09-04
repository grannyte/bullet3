// PREFER_FLOW_CONTROL - data-dependent loops here must stay real loops; Release flattens
// branches by default, which breaks stack traversal and runtime-bounded clipping.
// Glue kernels for the GPU-resident pipeline: turning a GPU-side append count into dispatch
// arguments, and reducing an int array to one value.

#define WG_SIZE 64
#define REDUCE_SIZE 256

// x = threads per group for the indirect dispatch, y = element cap, z = reduce element count.
// One struct serves both kernels; each reads only the fields it needs.
StructuredBuffer<uint4> PatchParams : register(t0);
StructuredBuffer<int> ReduceSrc : register(t1);

// Hidden append counter, landed here by CopyStructureCount. Raw, not structured: an indirect
// args buffer cannot carry an SRV, so it is read through its own UAV.
RWByteAddressBuffer CountSrc : register(u0);
// Any 16-byte params struct whose first field is the element count.
RWStructuredBuffer<uint4> TargetParams : register(u1);
RWByteAddressBuffer DispatchArgs : register(u2);
RWStructuredBuffer<int> ReduceDst : register(u3);

/**
 * Publishes a GPU-decided count as both a params field and a dispatch argument, so the phase
 * consuming it never has to read the count back to size its own dispatch.
 */
[numthreads(1, 1, 1)]
void CSPatchCountParams(uint3 tid : SV_DispatchThreadID)
{
	const uint raw = CountSrc.Load(0);
	const uint maxCount = PatchParams[0].y;
	const uint count = min(raw, maxCount);

	uint4 p = TargetParams[0];
	p.x = count;
	TargetParams[0] = p;

	const uint threads = max(PatchParams[0].x, 1u);
	DispatchArgs.Store3(0, uint3((count + threads - 1) / threads, 1, 1));
	// Unclamped count kept for an optional overflow check off the hot path.
	DispatchArgs.Store(12, raw);
}

groupshared int gsMax[REDUCE_SIZE];

//! Single-group max over ReduceSrc, so a host loop bound costs 4 bytes of readback, not 4N.
[numthreads(REDUCE_SIZE, 1, 1)]
void CSMaxInt(uint3 tid : SV_GroupThreadID)
{
	const uint count = PatchParams[0].z;
	int best = 0;
	for (uint i = tid.x; i < count; i += REDUCE_SIZE)
		best = max(best, ReduceSrc[i]);

	gsMax[tid.x] = best;
	GroupMemoryBarrierWithGroupSync();

	for (uint s = REDUCE_SIZE / 2; s > 0; s >>= 1)
	{
		if (tid.x < s)
			gsMax[tid.x] = max(gsMax[tid.x], gsMax[tid.x + s]);
		GroupMemoryBarrierWithGroupSync();
	}

	if (tid.x == 0)
		ReduceDst[0] = gsMax[0];
}
