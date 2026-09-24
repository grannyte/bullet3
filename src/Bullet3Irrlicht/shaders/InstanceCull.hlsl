// GPU instance culling, sorting keys and matrix build. Input is a packed {float3 pos; float4 quat;}
// transform buffer; nothing here knows what produced it.

struct CullParams
{
	float4x4 ViewProj;
	float4 Planes[6];      // xyz = normal, w = d; inside when dot(n,p) + d >= 0
	float4 CameraPos;
	float4 AabbMin;        // mesh-local bounds, w unused
	float4 AabbMax;
	uint ObjectCount;      // threads this bucket iterates: MemberCount when set, else the object count
	uint FilterMask;       // 0 accepts every key
	uint FilterValue;
	uint Flags;            // bit0 sort descending
	uint Capacity;         // instances the output buffer can hold
	uint IndexCount;       // indices per instance, stamped into the draw arguments
	uint PaddedCount;      // ObjectCount rounded up to the sort's block size
	uint PairBase;         // batched: first thread, survivor slot and instance record of this bucket
	uint MemberBase;       // offset into MemberIndices this bucket's slice starts at
	uint MemberCount;      // 0 walks tid directly as the object index
	float LodMinDist;      // distance-from-camera window
	float LodMaxDist;
	uint Layout;           // 0 = Matrix64, 1 = InstID80 (matrix + Scalars float4)
	uint BatchCount;       // batched: buckets in Params, read from Params[0]
	uint BatchThreads;     // batched: sum of every bucket's ObjectCount, read from Params[0]
	uint Pad5;
};

struct PackedTransform
{
	float3 Position;
	float4 Orientation;   // xyzw
};

#define CULL_GROUP 64
#define FLAG_SORT_DESCENDING 1
#define LAYOUT_INSTID80 1
// asuint(+inf): above every finite distance key, and never sets bit 31, which keeps the radix
// sort's top pass out of the picture entirely.
#define KEY_CULLED 0x7F800000
#define KEY_MAX 0x7F7FFFFF

StructuredBuffer<CullParams> Params : register(t0);
StructuredBuffer<PackedTransform> Transforms : register(t1);
StructuredBuffer<uint> FilterKeys : register(t2);
StructuredBuffer<uint2> SortedPairs : register(t3);
StructuredBuffer<uint> MemberIndices : register(t4);
StructuredBuffer<float4> Scalars : register(t5);
StructuredBuffer<float4> ObjectScales : register(t6);   // xyz per-object scale, always bound

RWStructuredBuffer<uint2> SurvivorPairs : register(u1);
RWStructuredBuffer<uint> CountBuffer : register(u2);   // [0] raw count, [1] clamped
RWByteAddressBuffer DrawArgs : register(u3);
RWByteAddressBuffer DispatchArgs : register(u4);
RWByteAddressBuffer InstanceOut : register(u5);

/// Row-major world matrix from a position and a unit quaternion, matching Irrlicht's row layout.
float4x4 MatrixFromPosQuat(float3 p, float4 q)
{
	const float xx = q.x * q.x, yy = q.y * q.y, zz = q.z * q.z;
	const float xy = q.x * q.y, xz = q.x * q.z, yz = q.y * q.z;
	const float wx = q.w * q.x, wy = q.w * q.y, wz = q.w * q.z;

	float4x4 m;
	m[0] = float4(1.0f - 2.0f * (yy + zz), 2.0f * (xy + wz), 2.0f * (xz - wy), 0.0f);
	m[1] = float4(2.0f * (xy - wz), 1.0f - 2.0f * (xx + zz), 2.0f * (yz + wx), 0.0f);
	m[2] = float4(2.0f * (xz + wy), 2.0f * (yz - wx), 1.0f - 2.0f * (xx + yy), 0.0f);
	m[3] = float4(p, 1.0f);
	return m;
}

/// Rows scaled the way ISceneNode::getRelativeTransformation's trailing `mat *= scaleMatrix` does,
/// so a scaled instance's world AABB below comes out scaled for free.
float4x4 MatrixFromPosQuatScale(float3 p, float4 q, float3 s)
{
	float4x4 m = MatrixFromPosQuat(p, q);
	m[0].xyz *= s.x;
	m[1].xyz *= s.y;
	m[2].xyz *= s.z;
	return m;
}

/// True when the transformed mesh AABB is fully behind any one plane.
bool FrustumRejects(float4x4 world, float3 lo, float3 hi, CullParams prm)
{
	// Rotating the extent by the matrix's absolute value gives the world AABB of the rotated box.
	const float3 centre = 0.5f * (lo + hi);
	const float3 extent = 0.5f * (hi - lo);

	const float3 worldCentre = mul(float4(centre, 1.0f), world).xyz;
	const float3 worldExtent = float3(
		dot(extent, abs(float3(world[0].x, world[1].x, world[2].x))),
		dot(extent, abs(float3(world[0].y, world[1].y, world[2].y))),
		dot(extent, abs(float3(world[0].z, world[1].z, world[2].z))));

	[unroll]
	for (uint i = 0; i < 6; ++i)
	{
		const float4 plane = prm.Planes[i];
		const float dist = dot(plane.xyz, worldCentre) + plane.w;
		const float radius = dot(abs(plane.xyz), worldExtent);
		if (dist + radius < 0.0f)
			return true;
	}
	return false;
}

/// IEEE bit patterns of non-negative floats compare as unsigned ints, so the distance IS the key.
uint DepthKeyFromDistance(float dist, CullParams prm)
{
	uint key = min(asuint(max(dist, 0.0f)), KEY_MAX);
	if (prm.Flags & FLAG_SORT_DESCENDING)
		key = KEY_MAX - key;
	return key;
}

/// tid is this bucket's own iteration index; resolves to an object slot via MemberIndices when set.
uint ObjectIndexFor(uint tid, CullParams prm)
{
	return prm.MemberCount > 0 ? MemberIndices[prm.MemberBase + tid] : tid;
}

/// Shared test over a resolved object index; returns false when filtered, culled or outside the LOD window.
bool Survives(uint i, CullParams prm, out uint key)
{
	key = KEY_CULLED;

	const uint filterKey = FilterKeys[i];
	if ((filterKey & prm.FilterMask) != prm.FilterValue)
		return false;

	// Same zero-scale drop as InstancedRenderer::AcceptsCpuInstance, or a collapsed instance would
	// reach the stream as a degenerate matrix instead of being handed back.
	const float3 scale = ObjectScales[i].xyz;
	if (dot(scale, scale) <= 0.0f)
		return false;

	const PackedTransform t = Transforms[i];
	const float4x4 world = MatrixFromPosQuatScale(t.Position, t.Orientation, scale);
	if (FrustumRejects(world, prm.AabbMin.xyz, prm.AabbMax.xyz, prm))
		return false;

	const float dist = length(t.Position - prm.CameraPos.xyz);
	if (dist < prm.LodMinDist || dist > prm.LodMaxDist)
		return false;

	key = DepthKeyFromDistance(dist, prm);
	return true;
}

/// Unsorted bucket: an atomic index compacts survivors directly into SurvivorPairs.
[numthreads(CULL_GROUP, 1, 1)]
void CSCullAppend(uint3 tid : SV_DispatchThreadID)
{
	const CullParams prm = Params[0];
	if (tid.x >= prm.ObjectCount)
		return;

	const uint i = ObjectIndexFor(tid.x, prm);
	uint key;
	if (!Survives(i, prm, key))
		return;

	uint slot;
	InterlockedAdd(CountBuffer[0], 1, slot);
	SurvivorPairs[slot] = uint2(key, i);
}

/// Sorted bucket: every slot is written so the whole array can go through the radix sort, with
/// culled entries keyed to sink past the survivors.
[numthreads(CULL_GROUP, 1, 1)]
void CSCullMark(uint3 tid : SV_DispatchThreadID)
{
	const CullParams prm = Params[0];
	// Every padded slot is written too: an unwritten (zero-key) tail would sort ahead of survivors.
	if (tid.x >= prm.PaddedCount)
		return;

	if (tid.x >= prm.ObjectCount)
	{
		SurvivorPairs[tid.x] = uint2(KEY_CULLED, 0xFFFFFFFF);
		return;
	}

	const uint i = ObjectIndexFor(tid.x, prm);
	uint key;
	const bool alive = Survives(i, prm, key);
	SurvivorPairs[tid.x] = alive ? uint2(key, i) : uint2(KEY_CULLED, 0xFFFFFFFF);

	if (alive)
	{
		uint previous;
		InterlockedAdd(CountBuffer[0], 1, previous);
	}
}

/// One thread: turns the device-side survivor count into draw and dispatch arguments.
[numthreads(1, 1, 1)]
void CSPatchArgs(uint3 tid : SV_DispatchThreadID)
{
	const CullParams prm = Params[0];
	const uint count = min(CountBuffer[0], prm.Capacity);

	DrawArgs.Store(4, count);                                   // InstanceCount
	DispatchArgs.Store(0, (count + CULL_GROUP - 1) / CULL_GROUP);
	DispatchArgs.Store(4, 1);
	DispatchArgs.Store(8, 1);
	CountBuffer[1] = count;                                     // clamped copy the build kernel reads
}

/// Expands surviving indices into the per-instance vertex stream; InstID80 appends the object's scalars.
[numthreads(CULL_GROUP, 1, 1)]
void CSBuildInstances(uint3 tid : SV_DispatchThreadID)
{
	const CullParams prm = Params[0];
	const uint count = CountBuffer[1];
	if (tid.x >= count)
		return;

	const uint index = SortedPairs[tid.x].y;
	const PackedTransform t = Transforms[index];
	const float4x4 world =
		MatrixFromPosQuatScale(t.Position, t.Orientation, ObjectScales[index].xyz);

	const uint stride = prm.Layout == LAYOUT_INSTID80 ? 80 : 64;
	const uint base = tid.x * stride;
	InstanceOut.Store4(base, asuint(world[0]));
	InstanceOut.Store4(base + 16, asuint(world[1]));
	InstanceOut.Store4(base + 32, asuint(world[2]));
	InstanceOut.Store4(base + 48, asuint(world[3]));

	if (prm.Layout == LAYOUT_INSTID80)
		InstanceOut.Store4(base + 64, asuint(Scalars[index]));
}

// Batched buckets: Params holds one CullParams per bucket, ordered by PairBase, and one dispatch of
// each kernel below serves them all. CountBuffer holds two uints and DrawArgs five per bucket.

#define DRAW_ARGS_STRIDE 20

/// The bucket a batched thread belongs to: the last one whose PairBase is at or below it.
uint BatchBucketOf(uint t, uint bucketCount)
{
	uint lo = 0;
	uint hi = bucketCount;
	while (hi - lo > 1)
	{
		const uint mid = (lo + hi) >> 1;
		if (Params[mid].PairBase <= t)
			lo = mid;
		else
			hi = mid;
	}
	return lo;
}

/// One thread per bucket: zeroes its counters and stamps its draw arguments, StartInstance = PairBase.
[numthreads(CULL_GROUP, 1, 1)]
void CSBatchReset(uint3 tid : SV_DispatchThreadID)
{
	if (tid.x >= Params[0].BatchCount)
		return;
	const CullParams prm = Params[tid.x];
	CountBuffer[tid.x * 2] = 0;
	CountBuffer[tid.x * 2 + 1] = 0;
	const uint args = tid.x * DRAW_ARGS_STRIDE;
	DrawArgs.Store(args, prm.IndexCount);
	DrawArgs.Store(args + 4, 0);
	DrawArgs.Store(args + 8, 0);
	DrawArgs.Store(args + 12, 0);
	DrawArgs.Store(args + 16, prm.PairBase);
}

/// CSCullAppend over every bucket at once, each compacting into its own PairBase region.
[numthreads(CULL_GROUP, 1, 1)]
void CSBatchCull(uint3 tid : SV_DispatchThreadID)
{
	if (tid.x >= Params[0].BatchThreads)
		return;
	const uint b = BatchBucketOf(tid.x, Params[0].BatchCount);
	const CullParams prm = Params[b];
	const uint local = tid.x - prm.PairBase;
	if (local >= prm.ObjectCount)
		return;

	const uint i = ObjectIndexFor(local, prm);
	uint key;
	if (!Survives(i, prm, key))
		return;

	uint slot;
	InterlockedAdd(CountBuffer[b * 2], 1, slot);
	if (slot < prm.Capacity)
		SurvivorPairs[prm.PairBase + slot] = uint2(key, i);
}

/// One thread per bucket: CSPatchArgs for each.
[numthreads(CULL_GROUP, 1, 1)]
void CSBatchPatch(uint3 tid : SV_DispatchThreadID)
{
	if (tid.x >= Params[0].BatchCount)
		return;
	const CullParams prm = Params[tid.x];
	const uint count = min(CountBuffer[tid.x * 2], prm.Capacity);
	DrawArgs.Store(tid.x * DRAW_ARGS_STRIDE + 4, count);
	CountBuffer[tid.x * 2 + 1] = count;
}

/// CSBuildInstances over every bucket; a record lands at its bucket's PairBase plus its draw index.
[numthreads(CULL_GROUP, 1, 1)]
void CSBatchBuild(uint3 tid : SV_DispatchThreadID)
{
	if (tid.x >= Params[0].BatchThreads)
		return;
	const uint b = BatchBucketOf(tid.x, Params[0].BatchCount);
	const CullParams prm = Params[b];
	const uint local = tid.x - prm.PairBase;
	if (local >= CountBuffer[b * 2 + 1])
		return;

	const uint index = SurvivorPairs[prm.PairBase + local].y;
	const PackedTransform t = Transforms[index];
	const float4x4 world =
		MatrixFromPosQuatScale(t.Position, t.Orientation, ObjectScales[index].xyz);

	const uint stride = prm.Layout == LAYOUT_INSTID80 ? 80 : 64;
	const uint base = tid.x * stride;
	InstanceOut.Store4(base, asuint(world[0]));
	InstanceOut.Store4(base + 16, asuint(world[1]));
	InstanceOut.Store4(base + 32, asuint(world[2]));
	InstanceOut.Store4(base + 48, asuint(world[3]));

	if (prm.Layout == LAYOUT_INSTID80)
		InstanceOut.Store4(base + 64, asuint(Scalars[index]));
}

/// Zeroes the counter and stamps the constant fields of the draw arguments.
[numthreads(1, 1, 1)]
void CSResetCount(uint3 tid : SV_DispatchThreadID)
{
	const CullParams prm = Params[0];
	CountBuffer[0] = 0;
	CountBuffer[1] = 0;
	DrawArgs.Store(0, prm.IndexCount);
	DrawArgs.Store(4, 0);
	DrawArgs.Store(8, 0);
	DrawArgs.Store(12, 0);
	DrawArgs.Store(16, 0);
}
