/*******************************************************
* Copyright (C) 2026 David Delisle granyte@hotmail.com
*
* This file is part of Outerspace.
*
* Outerspace can not be copied and/or distributed without the express
* permission of David Delisle
*******************************************************/
#include "GpuInstanceCuller.h"

#include "Bullet3Irrlicht/b3IrrlichtRadixSort.h"

#include <irrlicht.h>
#include <ComputeBuffer.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

using namespace irr;

namespace
{
const unsigned int kCullGroup = 64;
const unsigned int kKeyMax = 0x7F7FFFFFu;

/// Mirrors CullParams in InstanceCull.hlsl field for field.
struct CullParamsGpu
{
	float ViewProj[16];
	float Planes[6][4];
	float CameraPos[4];
	float AabbMin[4];
	float AabbMax[4];
	unsigned int ObjectCount;
	unsigned int FilterMask;
	unsigned int FilterValue;
	unsigned int Flags;
	unsigned int Capacity;
	unsigned int IndexCount;
	unsigned int PaddedCount;
	unsigned int PairBase;
	unsigned int MemberBase;
	unsigned int MemberCount;
	float LodMinDist;
	float LodMaxDist;
	unsigned int Layout;
	unsigned int BatchCount;
	unsigned int BatchThreads;
	unsigned int Pad5;
};

const unsigned int kDrawArgsStride = 20;

/// The radix sort works in blocks, so a sorted bucket rounds its element count up to one - an
/// unwritten tail would carry key 0 and sort ahead of every survivor.
unsigned int PadToSortBlock(unsigned int count)
{
	const unsigned int block = 256;
	return ((count + block - 1) / block) * block;
}

struct SortPair
{
	unsigned int Key;
	unsigned int Value;
};

struct InstanceMatrix
{
	float M[16];
};

/// InstID80: the game's InstancedRenderer TEXCOORD1..5 layout, matrix then four scalar slots.
struct InstanceMatrixScalar
{
	float M[16];
	float Scalars[4];
};

/// One object's scale, float4 because that is what the kernel's StructuredBuffer declares.
struct InstanceScale
{
	float Value[4];
};

void DropBuffer(scene::IComputeBuffer*& buffer)
{
	if (buffer)
	{
		buffer->drop();
		buffer = 0;
	}
}

/// Grow-only: a steady-state frame allocates nothing.
template <typename T>
scene::IComputeBuffer* EnsureBuffer(scene::IComputeBuffer*& buffer, u32 count, u32 flags)
{
	if (count == 0)
		return buffer;

	if (buffer && buffer->getStructureStride() == sizeof(T) && buffer->getBufferFlags() == flags
		&& buffer->getStructureCount() >= count)
		return buffer;

	DropBuffer(buffer);
	scene::ComputeBuffer<T>* fresh = new scene::ComputeBuffer<T>();
	fresh->setHardwareMappingHint(scene::EHM_DYNAMIC);
	if (flags)
		fresh->setBufferFlags(flags);
	fresh->set_used(count);
	buffer = fresh;
	return buffer;
}

unsigned int FloatBits(float value)
{
	unsigned int bits;
	memcpy(&bits, &value, sizeof(bits));
	return bits;
}
}  // namespace

GpuInstancingCounters& GpuInstanceCuller::Counters()
{
	static GpuInstancingCounters counters;
	return counters;
}

GpuInstanceCuller::Bucket::Bucket()
	: Capacity(0), Params(0), Pairs(0), Count(0), DrawArgs(0), DispatchArgs(0), Instances(0),
	  Ordered(0), Released(false), BatchSlot(-1), PairBase(0)
{
}

GpuInstanceCuller::GpuInstanceCuller(video::IVideoDriver* driver, io::IFileSystem* fileSystem)
	: m_driver(driver), m_fileSystem(fileSystem), m_cullAppendMaterial(-1), m_cullMarkMaterial(-1),
	  m_patchArgsMaterial(-1), m_buildMaterial(-1), m_resetMaterial(-1), m_batchResetMaterial(-1),
	  m_batchCullMaterial(-1), m_batchPatchMaterial(-1), m_batchBuildMaterial(-1), m_batchParams(0),
	  m_batchPairs(0), m_batchCounts(0), m_batchDrawArgs(0), m_batchInstances(0), m_sort(0),
	  m_defaultKeys(0), m_defaultKeyCount(0), m_defaultScales(0), m_defaultScaleCount(0),
	  m_available(false)
{
}

GpuInstanceCuller::~GpuInstanceCuller()
{
	ClearBuckets();
	DropBuffer(m_batchParams);
	DropBuffer(m_batchPairs);
	DropBuffer(m_batchCounts);
	DropBuffer(m_batchDrawArgs);
	DropBuffer(m_batchInstances);
	DropBuffer(m_defaultKeys);
	DropBuffer(m_defaultScales);
	delete m_sort;
}

void GpuInstanceCuller::MatrixFromPose(const float* pos, const float* q, const float* scale,
									   float* m)
{
	const float xx = q[0] * q[0], yy = q[1] * q[1], zz = q[2] * q[2];
	const float xy = q[0] * q[1], xz = q[0] * q[2], yz = q[1] * q[2];
	const float wx = q[3] * q[0], wy = q[3] * q[1], wz = q[3] * q[2];

	m[0] = 1.f - 2.f * (yy + zz); m[1] = 2.f * (xy + wz);      m[2] = 2.f * (xz - wy);       m[3] = 0.f;
	m[4] = 2.f * (xy - wz);       m[5] = 1.f - 2.f * (xx + zz); m[6] = 2.f * (yz + wx);      m[7] = 0.f;
	m[8] = 2.f * (xz + wy);       m[9] = 2.f * (yz - wx);      m[10] = 1.f - 2.f * (xx + yy); m[11] = 0.f;
	m[12] = pos[0];               m[13] = pos[1];              m[14] = pos[2];                m[15] = 1.f;

	if (!scale)
		return;
	for (int row = 0; row < 3; ++row)
		for (int column = 0; column < 3; ++column)
			m[row * 4 + column] *= scale[row];
}

bool GpuInstanceCuller::Init()
{
	if (!m_driver)
		return false;

	video::IGPUProgrammingServices* gpu = m_driver->getGPUProgrammingServices();
	if (!gpu)
		return false;

	const io::path path = "media/shaders/InstanceCull.hlsl";
	if (m_fileSystem && !m_fileSystem->existFile(path))
		return false;

	m_cullAppendMaterial = gpu->addComputeShaderFromFile(path, "CSCullAppend", video::ECST_CS_5_0, 0);
	m_cullMarkMaterial = gpu->addComputeShaderFromFile(path, "CSCullMark", video::ECST_CS_5_0, 0);
	m_patchArgsMaterial = gpu->addComputeShaderFromFile(path, "CSPatchArgs", video::ECST_CS_5_0, 0);
	m_buildMaterial = gpu->addComputeShaderFromFile(path, "CSBuildInstances", video::ECST_CS_5_0, 0);
	m_resetMaterial = gpu->addComputeShaderFromFile(path, "CSResetCount", video::ECST_CS_5_0, 0);

	m_available = m_cullAppendMaterial >= 0 && m_cullMarkMaterial >= 0 && m_patchArgsMaterial >= 0
				  && m_buildMaterial >= 0 && m_resetMaterial >= 0;
	if (!m_available)
		return false;

	// Optional like the sort: without them every bucket takes the per-bucket path.
	m_batchResetMaterial = gpu->addComputeShaderFromFile(path, "CSBatchReset", video::ECST_CS_5_0, 0);
	m_batchCullMaterial = gpu->addComputeShaderFromFile(path, "CSBatchCull", video::ECST_CS_5_0, 0);
	m_batchPatchMaterial = gpu->addComputeShaderFromFile(path, "CSBatchPatch", video::ECST_CS_5_0, 0);
	m_batchBuildMaterial = gpu->addComputeShaderFromFile(path, "CSBatchBuild", video::ECST_CS_5_0, 0);

	// Sorting is optional per bucket, so a sort that will not initialise only costs sorted buckets.
	m_sort = new b3IrrlichtRadixSort(m_driver);
	if (!m_sort->init(m_fileSystem))
	{
		delete m_sort;
		m_sort = 0;
	}
	return true;
}

int GpuInstanceCuller::AddBucket(const BucketDesc& desc)
{
	if (!m_available)
		return -1;
	if (desc.Sort && !m_sort)
		return -1;

	// A shared culler outlives the managers that come and go inside it, so ids are recycled rather
	// than appended forever.
	for (size_t i = 0; i < m_buckets.size(); ++i)
		if (m_buckets[i].Released)
		{
			m_buckets[i].Desc = desc;
			m_buckets[i].Released = false;
			return (int)i;
		}

	Bucket bucket;
	bucket.Desc = desc;
	m_buckets.push_back(bucket);
	return (int)m_buckets.size() - 1;
}

bool GpuInstanceCuller::UpdateBucket(int bucket, const BucketDesc& desc)
{
	if (bucket < 0 || bucket >= (int)m_buckets.size() || m_buckets[bucket].Released)
		return false;
	if (desc.Sort && !m_sort)
		return false;

	m_buckets[bucket].Desc = desc;
	return true;
}

void GpuInstanceCuller::ReleaseBucket(int bucket)
{
	if (bucket < 0 || bucket >= (int)m_buckets.size())
		return;

	Bucket& b = m_buckets[bucket];
	DropBuffer(b.Params);
	DropBuffer(b.Pairs);
	DropBuffer(b.Count);
	DropBuffer(b.DrawArgs);
	DropBuffer(b.DispatchArgs);
	DropBuffer(b.Instances);
	b.Ordered = 0;
	b.Capacity = 0;
	b.Released = true;
	b.BatchSlot = -1;
}

void GpuInstanceCuller::ClearBuckets()
{
	for (size_t i = 0; i < m_buckets.size(); ++i)
		ReleaseBucket((int)i);
	m_buckets.clear();
}

bool GpuInstanceCuller::EnsureBucket(Bucket& bucket, unsigned int iterCount)
{
	const unsigned int pairCount = bucket.Desc.Sort ? PadToSortBlock(iterCount) : iterCount;

	EnsureBuffer<CullParamsGpu>(bucket.Params, 1, 0);
	EnsureBuffer<SortPair>(bucket.Pairs, pairCount, 0);
	EnsureBuffer<unsigned int>(bucket.Count, 2, 0);
	EnsureBuffer<unsigned int>(bucket.DrawArgs, 5, video::EHBF_DRAW_INDIRECT_ARGS);
	EnsureBuffer<unsigned int>(bucket.DispatchArgs, 4, video::EHBF_DRAW_INDIRECT_ARGS);
	if (bucket.Desc.Layout == RecordLayout::InstID80)
		EnsureBuffer<InstanceMatrixScalar>(bucket.Instances, iterCount, video::EHBF_VERTEX_ADDITIONAL_BIND);
	else
		EnsureBuffer<InstanceMatrix>(bucket.Instances, iterCount, video::EHBF_VERTEX_ADDITIONAL_BIND);

	bucket.Capacity = iterCount;
	return bucket.Params && bucket.Pairs && bucket.Count && bucket.DrawArgs && bucket.DispatchArgs
		   && bucket.Instances;
}

namespace
{
/**
 * @brief One bucket's params block, as InstanceCull.hlsl's CullParams reads it.
 * @param desc Bucket description.
 * @param capacity Records the bucket's output can hold.
 * @param iterCount Objects the cull iterates.
 * @param camera The bucket's camera.
 * @param p Receives the block; the batch fields stay zero.
 */
void FillParams(const GpuInstanceCuller::BucketDesc& desc, unsigned int capacity, unsigned int iterCount,
				const GpuInstanceCuller::CameraState& camera, CullParamsGpu& p)
{
	memset(&p, 0, sizeof(p));
	memcpy(p.ViewProj, camera.ViewProj, sizeof(p.ViewProj));
	memcpy(p.Planes, camera.Planes, sizeof(p.Planes));
	for (int i = 0; i < 3; ++i)
	{
		p.CameraPos[i] = camera.Position[i];
		p.AabbMin[i] = desc.BoundsMin[i];
		p.AabbMax[i] = desc.BoundsMax[i];
	}
	p.ObjectCount = iterCount;
	p.FilterMask = desc.FilterMask;
	p.FilterValue = desc.FilterValue;
	p.Flags = desc.SortDescending ? 1u : 0u;
	p.Capacity = capacity;
	p.IndexCount = desc.IndexCount;
	p.PaddedCount = desc.Sort ? PadToSortBlock(iterCount) : iterCount;
	p.MemberBase = desc.MemberBase;
	p.MemberCount = desc.MemberCount;
	p.LodMinDist = desc.LodMinDist;
	p.LodMaxDist = desc.LodMaxDist;
	p.Layout = desc.Layout == GpuInstanceCuller::RecordLayout::InstID80 ? 1u : 0u;
}
}  // namespace

void GpuInstanceCuller::UploadParams(Bucket& bucket, unsigned int iterCount,
									 const CameraState& camera)
{
	CullParamsGpu p;
	FillParams(bucket.Desc, bucket.Capacity, iterCount, camera, p);
	memcpy(bucket.Params->getBufferPointer(), &p, sizeof(p));
	bucket.Params->setDirty();
	++Counters().Uploads;
}

void GpuInstanceCuller::SetKernel(int material)
{
	video::SMaterial mat;
	mat.MaterialType = (video::E_MATERIAL_TYPE)material;
	m_driver->setMaterial(mat);
}

void GpuInstanceCuller::EnsureDefaults(scene::IComputeBuffer*& filterKeys,
									   scene::IComputeBuffer*& scales, unsigned int objectCount)
{
	if (!filterKeys)
	{
		// A zero mask accepts everything, so the contents never matter - only the binding does.
		if (m_defaultKeyCount < objectCount)
		{
			EnsureBuffer<unsigned int>(m_defaultKeys, objectCount, 0);
			memset(m_defaultKeys->getBufferPointer(), 0, objectCount * sizeof(unsigned int));
			m_defaultKeys->setDirty();
			m_defaultKeyCount = objectCount;
		}
		filterKeys = m_defaultKeys;
	}

	if (!scales)
	{
		// The scale is read unconditionally, and a zero one is a survival test, so an unbound slot
		// reading as zero would cull the whole bucket.
		if (m_defaultScaleCount < objectCount)
		{
			EnsureBuffer<InstanceScale>(m_defaultScales, objectCount, 0);
			InstanceScale* unit = (InstanceScale*)m_defaultScales->getBufferPointer();
			for (unsigned int i = 0; i < objectCount; ++i)
			{
				unit[i].Value[0] = unit[i].Value[1] = unit[i].Value[2] = 1.f;
				unit[i].Value[3] = 0.f;
			}
			m_defaultScales->setDirty();
			m_defaultScaleCount = objectCount;
		}
		scales = m_defaultScales;
	}
}

bool GpuInstanceCuller::DispatchBucket(Bucket& bucket, scene::IComputeBuffer* transforms,
									   scene::IComputeBuffer* filterKeys, unsigned int objectCount,
									   const CameraState& camera,
									   scene::IComputeBuffer* memberIndices,
									   scene::IComputeBuffer* scalars, scene::IComputeBuffer* scales)
{
	const core::vector3d<u32> single(1, 1, 1);
	bucket.BatchSlot = -1;

	{
		const unsigned int iterCount = bucket.Desc.MemberCount > 0 ? bucket.Desc.MemberCount : objectCount;
		if (!EnsureBucket(bucket, iterCount))
			return false;
		UploadParams(bucket, iterCount, camera);

		SetKernel(m_resetMaterial);
		m_driver->bindComputeBuffer(0, bucket.Params, video::EHBT_SHADER_RESOURCE);
		m_driver->bindComputeBuffer(2, bucket.Count, video::EHBT_COMPUTE);
		m_driver->bindComputeBuffer(3, bucket.DrawArgs, video::EHBT_COMPUTE);
		m_driver->dispatchComputeShaderBound(single);
		m_driver->unbindComputeResources();

		if (bucket.Desc.Sort)
		{
			const unsigned int padded = PadToSortBlock(iterCount);
			const core::vector3d<u32> markGroups((padded + kCullGroup - 1) / kCullGroup, 1, 1);

			SetKernel(m_cullMarkMaterial);
			m_driver->bindComputeBuffer(0, bucket.Params, video::EHBT_SHADER_RESOURCE);
			m_driver->bindComputeBuffer(1, transforms, video::EHBT_SHADER_RESOURCE);
			m_driver->bindComputeBuffer(2, filterKeys, video::EHBT_SHADER_RESOURCE);
			m_driver->bindComputeBuffer(4, memberIndices, video::EHBT_SHADER_RESOURCE);
			m_driver->bindComputeBuffer(6, scales, video::EHBT_SHADER_RESOURCE);
			m_driver->bindComputeBuffer(1, bucket.Pairs, video::EHBT_COMPUTE);
			m_driver->bindComputeBuffer(2, bucket.Count, video::EHBT_COMPUTE);
			m_driver->dispatchComputeShaderBound(markGroups);
			m_driver->unbindComputeResources();

			// Culled entries carry the maximum key, so ascending order leaves them past the
			// survivors and the survivor count alone decides how many get drawn.
			if (!m_sort->executeResident(bucket.Pairs, padded, 32, &bucket.Ordered))
				return false;
		}
		else
		{
			const core::vector3d<u32> cullGroups((iterCount + kCullGroup - 1) / kCullGroup, 1, 1);

			// An atomic index into SurvivorPairs, not AppendStructuredBuffer: this driver's
			// CopyStructureCount does not reliably surface an append buffer's hidden counter.
			SetKernel(m_cullAppendMaterial);
			m_driver->bindComputeBuffer(0, bucket.Params, video::EHBT_SHADER_RESOURCE);
			m_driver->bindComputeBuffer(1, transforms, video::EHBT_SHADER_RESOURCE);
			m_driver->bindComputeBuffer(2, filterKeys, video::EHBT_SHADER_RESOURCE);
			m_driver->bindComputeBuffer(4, memberIndices, video::EHBT_SHADER_RESOURCE);
			m_driver->bindComputeBuffer(6, scales, video::EHBT_SHADER_RESOURCE);
			m_driver->bindComputeBuffer(1, bucket.Pairs, video::EHBT_COMPUTE);
			m_driver->bindComputeBuffer(2, bucket.Count, video::EHBT_COMPUTE);
			m_driver->dispatchComputeShaderBound(cullGroups);
			m_driver->unbindComputeResources();

			bucket.Ordered = bucket.Pairs;
		}

		SetKernel(m_patchArgsMaterial);
		m_driver->bindComputeBuffer(0, bucket.Params, video::EHBT_SHADER_RESOURCE);
		m_driver->bindComputeBuffer(2, bucket.Count, video::EHBT_COMPUTE);
		m_driver->bindComputeBuffer(3, bucket.DrawArgs, video::EHBT_COMPUTE);
		m_driver->bindComputeBuffer(4, bucket.DispatchArgs, video::EHBT_COMPUTE);
		m_driver->dispatchComputeShaderBound(single);
		m_driver->unbindComputeResources();

		SetKernel(m_buildMaterial);
		m_driver->bindComputeBuffer(0, bucket.Params, video::EHBT_SHADER_RESOURCE);
		m_driver->bindComputeBuffer(1, transforms, video::EHBT_SHADER_RESOURCE);
		m_driver->bindComputeBuffer(3, bucket.Ordered, video::EHBT_SHADER_RESOURCE);
		m_driver->bindComputeBuffer(5, scalars, video::EHBT_SHADER_RESOURCE);
		m_driver->bindComputeBuffer(6, scales, video::EHBT_SHADER_RESOURCE);
		m_driver->bindComputeBuffer(2, bucket.Count, video::EHBT_COMPUTE);
		m_driver->bindComputeBuffer(5, bucket.Instances, video::EHBT_COMPUTE);
		m_driver->dispatchComputeShaderIndirect(bucket.DispatchArgs, 0);
		m_driver->unbindComputeResources();
	}

	// Reset, cull or mark, patch, build.
	Counters().Dispatches += 4;
	Counters().SortRuns += bucket.Desc.Sort ? 1 : 0;
	return true;
}

bool GpuInstanceCuller::Run(scene::IComputeBuffer* transforms, scene::IComputeBuffer* filterKeys,
							unsigned int objectCount, const CameraState& camera,
							scene::IComputeBuffer* memberIndices, scene::IComputeBuffer* scalars,
							scene::IComputeBuffer* scales)
{
	if (!m_available || !transforms || objectCount == 0)
		return false;

	EnsureDefaults(filterKeys, scales, objectCount);

	for (size_t b = 0; b < m_buckets.size(); ++b)
		if (!m_buckets[b].Released
			&& !DispatchBucket(m_buckets[b], transforms, filterKeys, objectCount, camera,
							   memberIndices, scalars, scales))
			return false;

	return true;
}

bool GpuInstanceCuller::RunBucket(int bucket, scene::IComputeBuffer* transforms,
								  scene::IComputeBuffer* filterKeys, unsigned int objectCount,
								  const CameraState& camera, scene::IComputeBuffer* memberIndices,
								  scene::IComputeBuffer* scalars, scene::IComputeBuffer* scales)
{
	if (!m_available || !transforms || objectCount == 0)
		return false;
	if (bucket < 0 || bucket >= (int)m_buckets.size() || m_buckets[bucket].Released)
		return false;

	EnsureDefaults(filterKeys, scales, objectCount);
	return DispatchBucket(m_buckets[bucket], transforms, filterKeys, objectCount, camera,
						  memberIndices, scalars, scales);
}

bool GpuInstanceCuller::Batchable(int bucket) const
{
	if (bucket < 0 || bucket >= (int)m_buckets.size() || m_buckets[bucket].Released)
		return false;
	const BucketDesc& desc = m_buckets[bucket].Desc;
	return m_batchResetMaterial >= 0 && m_batchCullMaterial >= 0 && m_batchPatchMaterial >= 0
		   && m_batchBuildMaterial >= 0 && !desc.Sort && desc.MemberCount > 0
		   && desc.Layout == RecordLayout::InstID80;
}

void GpuInstanceCuller::ClearBatch()
{
	for (Bucket& bucket : m_buckets)
		bucket.BatchSlot = -1;
}

bool GpuInstanceCuller::Batched(int bucket) const
{
	return bucket >= 0 && bucket < (int)m_buckets.size() && m_buckets[bucket].BatchSlot >= 0;
}

bool GpuInstanceCuller::RunBatch(const std::vector<BatchEntry>& entries, scene::IComputeBuffer* transforms,
								 scene::IComputeBuffer* filterKeys, unsigned int objectCount,
								 scene::IComputeBuffer* memberIndices, scene::IComputeBuffer* scalars,
								 scene::IComputeBuffer* scales)
{
	if (!m_available || !transforms || !memberIndices || objectCount == 0 || entries.empty())
		return false;
	for (const BatchEntry& entry : entries)
		if (!Batchable(entry.Bucket))
			return false;

	EnsureDefaults(filterKeys, scales, objectCount);
	ClearBatch();

	const unsigned int count = (unsigned int)entries.size();
	unsigned int threads = 0;
	for (const BatchEntry& entry : entries)
		threads += m_buckets[entry.Bucket].Desc.MemberCount;

	EnsureBuffer<CullParamsGpu>(m_batchParams, count, 0);
	EnsureBuffer<SortPair>(m_batchPairs, threads, 0);
	EnsureBuffer<unsigned int>(m_batchCounts, count * 2, 0);
	EnsureBuffer<unsigned int>(m_batchDrawArgs, count * (kDrawArgsStride / 4), video::EHBF_DRAW_INDIRECT_ARGS);
	EnsureBuffer<InstanceMatrixScalar>(m_batchInstances, threads, video::EHBF_VERTEX_ADDITIONAL_BIND);
	if (!m_batchParams || !m_batchPairs || !m_batchCounts || !m_batchDrawArgs || !m_batchInstances)
		return false;

	CullParamsGpu* params = (CullParamsGpu*)m_batchParams->getBufferPointer();
	unsigned int base = 0;
	for (unsigned int slot = 0; slot < count; ++slot)
	{
		Bucket& bucket = m_buckets[entries[slot].Bucket];
		const unsigned int members = bucket.Desc.MemberCount;
		FillParams(bucket.Desc, members, members, entries[slot].Camera, params[slot]);
		params[slot].PairBase = base;
		params[slot].BatchCount = count;
		params[slot].BatchThreads = threads;
		bucket.BatchSlot = (int)slot;
		bucket.PairBase = base;
		base += members;
	}
	m_batchParams->setDirty();
	++Counters().Uploads;

	const core::vector3d<u32> perBucket((count + kCullGroup - 1) / kCullGroup, 1, 1);
	const core::vector3d<u32> perThread((threads + kCullGroup - 1) / kCullGroup, 1, 1);

	SetKernel(m_batchResetMaterial);
	m_driver->bindComputeBuffer(0, m_batchParams, video::EHBT_SHADER_RESOURCE);
	m_driver->bindComputeBuffer(2, m_batchCounts, video::EHBT_COMPUTE);
	m_driver->bindComputeBuffer(3, m_batchDrawArgs, video::EHBT_COMPUTE);
	m_driver->dispatchComputeShaderBound(perBucket);
	m_driver->unbindComputeResources();

	SetKernel(m_batchCullMaterial);
	m_driver->bindComputeBuffer(0, m_batchParams, video::EHBT_SHADER_RESOURCE);
	m_driver->bindComputeBuffer(1, transforms, video::EHBT_SHADER_RESOURCE);
	m_driver->bindComputeBuffer(2, filterKeys, video::EHBT_SHADER_RESOURCE);
	m_driver->bindComputeBuffer(4, memberIndices, video::EHBT_SHADER_RESOURCE);
	m_driver->bindComputeBuffer(6, scales, video::EHBT_SHADER_RESOURCE);
	m_driver->bindComputeBuffer(1, m_batchPairs, video::EHBT_COMPUTE);
	m_driver->bindComputeBuffer(2, m_batchCounts, video::EHBT_COMPUTE);
	m_driver->dispatchComputeShaderBound(perThread);
	m_driver->unbindComputeResources();

	SetKernel(m_batchPatchMaterial);
	m_driver->bindComputeBuffer(0, m_batchParams, video::EHBT_SHADER_RESOURCE);
	m_driver->bindComputeBuffer(2, m_batchCounts, video::EHBT_COMPUTE);
	m_driver->bindComputeBuffer(3, m_batchDrawArgs, video::EHBT_COMPUTE);
	m_driver->dispatchComputeShaderBound(perBucket);
	m_driver->unbindComputeResources();

	SetKernel(m_batchBuildMaterial);
	m_driver->bindComputeBuffer(0, m_batchParams, video::EHBT_SHADER_RESOURCE);
	m_driver->bindComputeBuffer(1, transforms, video::EHBT_SHADER_RESOURCE);
	m_driver->bindComputeBuffer(5, scalars, video::EHBT_SHADER_RESOURCE);
	m_driver->bindComputeBuffer(6, scales, video::EHBT_SHADER_RESOURCE);
	m_driver->bindComputeBuffer(1, m_batchPairs, video::EHBT_COMPUTE);
	m_driver->bindComputeBuffer(2, m_batchCounts, video::EHBT_COMPUTE);
	m_driver->bindComputeBuffer(5, m_batchInstances, video::EHBT_COMPUTE);
	m_driver->dispatchComputeShaderBound(perThread);
	m_driver->unbindComputeResources();

	Counters().Dispatches += 4;
	return true;
}

scene::IComputeBuffer* GpuInstanceCuller::InstanceBuffer(int bucket) const
{
	if (bucket < 0 || bucket >= (int)m_buckets.size())
		return 0;
	return m_buckets[bucket].BatchSlot >= 0 ? m_batchInstances : m_buckets[bucket].Instances;
}

scene::IComputeBuffer* GpuInstanceCuller::DrawArgsBuffer(int bucket) const
{
	if (bucket < 0 || bucket >= (int)m_buckets.size())
		return 0;
	return m_buckets[bucket].BatchSlot >= 0 ? m_batchDrawArgs : m_buckets[bucket].DrawArgs;
}

unsigned int GpuInstanceCuller::DrawArgsOffset(int bucket) const
{
	if (bucket < 0 || bucket >= (int)m_buckets.size() || m_buckets[bucket].BatchSlot < 0)
		return 0;
	return (unsigned int)m_buckets[bucket].BatchSlot * kDrawArgsStride;
}

unsigned int GpuInstanceCuller::InstanceStride(int bucket) const
{
	if (bucket < 0 || bucket >= (int)m_buckets.size())
		return 64;
	return m_buckets[bucket].Desc.Layout == RecordLayout::InstID80 ? 80 : 64;
}

bool GpuInstanceCuller::DownloadOrder(int bucket, std::vector<unsigned int>& out)
{
	out.clear();
	if (bucket < 0 || bucket >= (int)m_buckets.size())
		return false;

	Bucket& b = m_buckets[bucket];
	if (b.BatchSlot >= 0)
	{
		m_batchCounts->downloadFromGPU();
		m_batchPairs->downloadFromGPU();
		Counters().Readbacks += 2;
		const unsigned int count = ((const unsigned int*)m_batchCounts->getBufferPointer())[b.BatchSlot * 2 + 1];
		const SortPair* pairs = (const SortPair*)m_batchPairs->getBufferPointer() + b.PairBase;
		for (unsigned int i = 0; i < count && b.PairBase + i < m_batchPairs->getStructureCount(); ++i)
			out.push_back(pairs[i].Value);
		return true;
	}
	if (!b.Count || !b.Ordered)
		return false;

	b.Count->downloadFromGPU();
	++Counters().Readbacks;
	const unsigned int count = ((const unsigned int*)b.Count->getBufferPointer())[1];
	if (count == 0)
		return true;

	b.Ordered->downloadFromGPU();
	++Counters().Readbacks;
	const SortPair* pairs = (const SortPair*)b.Ordered->getBufferPointer();

	if (getenv("OSDEMO_CULL_DUMP"))
	{
		b.Pairs->downloadFromGPU();
		const SortPair* raw = (const SortPair*)b.Pairs->getBufferPointer();
		printf("[cull-dump] count=%u pairs=%u ordered=%u\n", count, b.Pairs->getStructureCount(),
			   b.Ordered->getStructureCount());
		for (unsigned int i = 0; i < 8; ++i)
			printf("[cull-dump]  %u: pre(%08x,%u) post(%08x,%u)\n", i, raw[i].Key, raw[i].Value,
				   pairs[i].Key, pairs[i].Value);
		fflush(stdout);
	}

	const unsigned int available = b.Ordered->getStructureCount();
	for (unsigned int i = 0; i < count && i < available; ++i)
		out.push_back(pairs[i].Value);
	return true;
}

bool GpuInstanceCuller::DownloadInstances(int bucket, std::vector<float>& outFloats)
{
	outFloats.clear();
	if (bucket < 0 || bucket >= (int)m_buckets.size())
		return false;

	Bucket& b = m_buckets[bucket];
	if (b.BatchSlot >= 0)
	{
		m_batchCounts->downloadFromGPU();
		m_batchInstances->downloadFromGPU();
		Counters().Readbacks += 2;
		const unsigned int count = ((const unsigned int*)m_batchCounts->getBufferPointer())[b.BatchSlot * 2 + 1];
		const unsigned int strideFloats = InstanceStride(bucket) / (unsigned int)sizeof(float);
		const unsigned int available = m_batchInstances->getStructureCount() - b.PairBase;
		const float* data = (const float*)m_batchInstances->getBufferPointer() + (size_t)b.PairBase * strideFloats;
		outFloats.assign(data, data + (size_t)(count < available ? count : available) * strideFloats);
		return true;
	}
	if (!b.Count || !b.Instances)
		return false;

	b.Count->downloadFromGPU();
	const unsigned int count = ((const unsigned int*)b.Count->getBufferPointer())[1];
	if (count == 0)
		return true;

	b.Instances->downloadFromGPU();
	const unsigned int strideFloats = InstanceStride(bucket) / (unsigned int)sizeof(float);
	const unsigned int available = b.Instances->getStructureCount();
	const float* data = (const float*)b.Instances->getBufferPointer();

	const unsigned int n = count < available ? count : available;
	outFloats.assign(data, data + (size_t)n * strideFloats);
	return true;
}

void GpuInstanceCuller::CpuReference(const float* transforms, const unsigned int* filterKeys,
									 unsigned int objectCount, const BucketDesc& desc,
									 const CameraState& camera, std::vector<unsigned int>& out,
									 const unsigned int* memberIndices, const float* scales)
{
	out.clear();
	std::vector<SortPair> survivors;

	const unsigned int iterCount = desc.MemberCount > 0 ? desc.MemberCount : objectCount;
	for (unsigned int t = 0; t < iterCount; ++t)
	{
		const unsigned int i = (desc.MemberCount > 0 && memberIndices) ? memberIndices[desc.MemberBase + t] : t;

		const unsigned int key = filterKeys ? filterKeys[i] : 0u;
		if ((key & desc.FilterMask) != desc.FilterValue)
			continue;

		const float* scale = scales ? scales + i * 3 : 0;
		if (scale && scale[0] * scale[0] + scale[1] * scale[1] + scale[2] * scale[2] <= 0.f)
			continue;

		const float* pos = transforms + i * 7;
		const float* quat = pos + 3;
		float m[16];
		MatrixFromPose(pos, quat, scale, m);

		float centre[3], extent[3];
		for (int a = 0; a < 3; ++a)
		{
			centre[a] = 0.5f * (desc.BoundsMin[a] + desc.BoundsMax[a]);
			extent[a] = 0.5f * (desc.BoundsMax[a] - desc.BoundsMin[a]);
		}

		float worldCentre[3], worldExtent[3];
		for (int a = 0; a < 3; ++a)
		{
			worldCentre[a] = centre[0] * m[a] + centre[1] * m[4 + a] + centre[2] * m[8 + a] + m[12 + a];
			worldExtent[a] = extent[0] * std::fabs(m[a]) + extent[1] * std::fabs(m[4 + a])
							 + extent[2] * std::fabs(m[8 + a]);
		}

		bool rejected = false;
		for (int p = 0; p < 6 && !rejected; ++p)
		{
			const float* plane = camera.Planes[p];
			const float dist = plane[0] * worldCentre[0] + plane[1] * worldCentre[1]
							   + plane[2] * worldCentre[2] + plane[3];
			const float radius = std::fabs(plane[0]) * worldExtent[0]
								 + std::fabs(plane[1]) * worldExtent[1]
								 + std::fabs(plane[2]) * worldExtent[2];
			if (dist + radius < 0.f)
				rejected = true;
		}
		if (rejected)
			continue;

		const float dx = pos[0] - camera.Position[0];
		const float dy = pos[1] - camera.Position[1];
		const float dz = pos[2] - camera.Position[2];
		const float distance = std::sqrt(dx * dx + dy * dy + dz * dz);
		if (distance < desc.LodMinDist || distance > desc.LodMaxDist)
			continue;

		SortPair pair;
		pair.Key = FloatBits(distance > 0.f ? distance : 0.f);
		if (pair.Key > kKeyMax)
			pair.Key = kKeyMax;
		if (desc.SortDescending)
			pair.Key = kKeyMax - pair.Key;
		pair.Value = i;
		survivors.push_back(pair);
	}

	if (desc.Sort)
	{
		// Stable, to match the radix sort's own stability on equal keys.
		std::stable_sort(survivors.begin(), survivors.end(),
						 [](const SortPair& a, const SortPair& b) { return a.Key < b.Key; });
	}

	out.reserve(survivors.size());
	for (size_t i = 0; i < survivors.size(); ++i)
		out.push_back(survivors[i].Value);
}

void GpuInstanceCuller::MakeCameraState(const float* viewProj, float eyeX, float eyeY, float eyeZ,
										CameraState& out)
{
	memcpy(out.ViewProj, viewProj, sizeof(out.ViewProj));

	out.Position[0] = eyeX;
	out.Position[1] = eyeY;
	out.Position[2] = eyeZ;

	// Gribb/Hartmann, D3D depth range: the z pair is {col2, col3 - col2}, which stays correct
	// under this engine's reversed-Z projection because only the two planes' names swap.
	const float* m = viewProj;
	const float wWeight[6] = {1.f, 1.f, 1.f, 1.f, 0.f, 1.f};
	const int axis[6] = {0, 0, 1, 1, 2, 2};
	const float axisWeight[6] = {1.f, -1.f, 1.f, -1.f, 1.f, -1.f};
	for (int p = 0; p < 6; ++p)
	{
		for (int c = 0; c < 4; ++c)
		{
			// Irrlicht matrix4 is row-major with the translation in row 3, so element (r, c) is
			// m[r * 4 + c] and a frustum plane is a combination of two of its columns.
			out.Planes[p][c] = wWeight[p] * m[c * 4 + 3] + axisWeight[p] * m[c * 4 + axis[p]];
		}
		const float len = std::sqrt(out.Planes[p][0] * out.Planes[p][0]
									+ out.Planes[p][1] * out.Planes[p][1]
									+ out.Planes[p][2] * out.Planes[p][2]);
		if (len > 0.f)
			for (int c = 0; c < 4; ++c)
				out.Planes[p][c] /= len;
	}
}
