// GPU-side fingerprint of the body buffer, so a determinism check reads back 8 bytes instead of the
// whole buffer. Included by B3StateHash.hlsl (f32) and B3StateHashDS.hlsl (df64).

#include "B3Precision.hlsli"

#define WG_SIZE 256

struct HashParams
{
	uint numBodies;
	uint bucketCount;
	uint bodiesPerBucket;
	uint pad0;
};

StructuredBuffer<HashParams> Params : register(t0);
StructuredBuffer<b3RigidBodyData> Bodies : register(t1);
// bucketCount uint pairs. Buckets are ranges of BODY INDEX, never BVH subtrees: after a divergence
// the peers' Morton order can differ, so subtree N would not name the same bodies on both sides.
RWStructuredBuffer<uint> OutHash : register(u0);

//! One FNV-1a round per byte of v, low byte first - matching the host's byte order so the two
//! implementations can be compared directly during bring-up.
uint fnvFold(uint h, uint v)
{
	[unroll]
	for (int b = 0; b < 4; ++b)
	{
		h ^= (v >> (b * 8)) & 0xFFu;
		h *= 16777619u;
	}
	return h;
}

[numthreads(WG_SIZE, 1, 1)]
void CSClearHash(uint3 tid : SV_DispatchThreadID)
{
	if (tid.x >= Params[0].bucketCount * 2u)
		return;
	OutHash[tid.x] = 0;
}

//! Per-body hash XORed into its bucket's accumulator pair. XOR is associative and commutative, so
//! the result is independent of thread order - which is what makes this reproducible at all.
[numthreads(WG_SIZE, 1, 1)]
void CSHashBodies(uint3 tid : SV_DispatchThreadID)
{
	const uint i = tid.x;
	if (i >= Params[0].numBodies)
		return;

	const b3RigidBodyData body = Bodies[i];

	// The index is folded in first so two identical bodies cannot cancel each other under XOR.
	uint h0 = fnvFold(2166136261u, i);
	uint h1 = fnvFold(2654435761u, i);

	uint words[16];   // 13 in f32, +3 for df64's low words
	int n = 0;
	words[n++] = asuint(body.pos.x);
	words[n++] = asuint(body.pos.y);
	words[n++] = asuint(body.pos.z);
#ifdef OS_DS
	words[n++] = asuint(body.posLo.x);
	words[n++] = asuint(body.posLo.y);
	words[n++] = asuint(body.posLo.z);
#endif
	words[n++] = asuint(body.quat.x);
	words[n++] = asuint(body.quat.y);
	words[n++] = asuint(body.quat.z);
	words[n++] = asuint(body.quat.w);
	words[n++] = asuint(body.linVel.x);
	words[n++] = asuint(body.linVel.y);
	words[n++] = asuint(body.linVel.z);
	words[n++] = asuint(body.angVel.x);
	words[n++] = asuint(body.angVel.y);
	words[n++] = asuint(body.angVel.z);

	[loop]
	for (int w = 0; w < n; ++w)
	{
		h0 = fnvFold(h0, words[w]);
		h1 = fnvFold(h1, words[w] ^ 0x9E3779B9u);
	}

	const uint perBucket = max(Params[0].bodiesPerBucket, 1u);
	const uint bucket = min(i / perBucket, max(Params[0].bucketCount, 1u) - 1u);

	uint ignored;
	InterlockedXor(OutHash[bucket * 2u], h0, ignored);
	InterlockedXor(OutHash[bucket * 2u + 1u], h1, ignored);
}
