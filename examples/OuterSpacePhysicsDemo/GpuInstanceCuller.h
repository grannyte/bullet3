/*******************************************************
* Copyright (C) 2026 David Delisle granyte@hotmail.com
*
* This file is part of Outerspace.
*
* Outerspace can not be copied and/or distributed without the express
* permission of David Delisle
*******************************************************/
#pragma once

#include <atomic>
#include <vector>

namespace irr
{
namespace video
{
class IVideoDriver;
}
namespace io
{
class IFileSystem;
}
namespace scene
{
class IComputeBuffer;
}
}  // namespace irr

class b3IrrlichtRadixSort;

/// Process-wide running totals of the GPU instancing path's device work; diagnostics only.
struct GpuInstancingCounters
{
	std::atomic<unsigned long long> Dispatches{0};    ///< compute dispatches, direct or indirect
	std::atomic<unsigned long long> SortRuns{0};      ///< radix sorts, each several dispatches
	std::atomic<unsigned long long> Uploads{0};       ///< compute buffers marked dirty for upload
	std::atomic<unsigned long long> Readbacks{0};     ///< blocking downloads
	std::atomic<unsigned long long> IndirectDraws{0};
	std::atomic<unsigned long long> CpuFillNs{0};     ///< InstancedRenderer's CPU instance-buffer fill
	std::atomic<unsigned long long> GpuDrawNs{0};     ///< InstancedRenderer's indirect draw calls
	std::atomic<unsigned long long> BodyReadNs{0};    ///< GpuInstanceBuild's Physic-pool pass, lock wait included
	std::atomic<unsigned long long> BodyReadPasses{0};
	std::atomic<unsigned long long> DetailReadNs{0};  ///< its Detail*-pool pass, lock wait included
	std::atomic<unsigned long long> BatchDispatchNs{0}; ///< its batched cull dispatch
};

/**
 * @brief GPU frustum culling, depth sorting and instance-matrix build over a device-resident transform buffer.
 *
 * Renderer-agnostic: the only input is {float3 pos; float4 quat;} plus a per-object filter key
 * whose meaning the caller chooses, one dispatch per bucket via a plain mask/value pair.
 */
class GpuInstanceCuller
{
public:
	/// Per-frame camera state; planes are xyz normal + w distance, inside when dot(n,p)+w >= 0.
	struct CameraState
	{
		float ViewProj[16];
		float Planes[6][4];
		float Position[3];
	};

	/// Per-instance record a bucket's build kernel writes; InstID80 adds the game renderer's
	/// four TEXCOORD scalar slots after the matrix.
	enum class RecordLayout
	{
		Matrix64,
		InstID80
	};

	/// One output stream: a filter, a mesh, and whether the survivors get depth-sorted.
	/// Default member initializers, not a declared ctor: a nested type's own methods aren't exported.
	struct BucketDesc
	{
		unsigned int FilterMask = 0;    ///< 0 accepts every key
		unsigned int FilterValue = 0;
		bool Sort = false;                  ///< depth sort the survivors before drawing
		bool SortDescending = false;        ///< back-to-front, for transparent geometry
		unsigned int IndexCount = 0;    ///< indices per instance, stamped into the draw arguments
		float BoundsMin[3] = {-0.5f, -0.5f, -0.5f};         ///< mesh-local AABB
		float BoundsMax[3] = {0.5f, 0.5f, 0.5f};

		/// Slice of the shared MemberIndices buffer (Run()'s memberIndices) this bucket dispatches
		/// over; MemberCount == 0 keeps the whole-object-space walk gated by FilterMask/FilterValue.
		unsigned int MemberBase = 0;
		unsigned int MemberCount = 0;

		float LodMinDist = 0.f;    ///< distance-from-camera window; default [0, ~FLT_MAX] accepts all
		float LodMaxDist = 3.4e38f;

		RecordLayout Layout = RecordLayout::Matrix64;
	};

	GpuInstanceCuller(irr::video::IVideoDriver* driver, irr::io::IFileSystem* fileSystem);
	~GpuInstanceCuller();

	/**
	 * @brief Compiles the kernels and the sort.
	 * @return False when the pass is unavailable and the caller must fall back.
	 */
	bool Init();

	bool IsAvailable() const { return m_available; }

	/// False once Init() gave up on the radix sort, which is exactly when AddBucket refuses Sort.
	bool SupportsSorting() const { return m_sort != 0; }

	/**
	 * @brief Registers an output bucket, reusing a released slot when one is free.
	 * @param desc Filter, mesh bounds and sort selection.
	 * @return Bucket id, or -1 if the pass is unavailable.
	 */
	int AddBucket(const BucketDesc& desc);

	/**
	 * @brief Replaces a registered bucket's description in place.
	 * @param bucket Bucket id.
	 * @param desc New filter, bounds, member slice and sort selection.
	 * @return False when the id is unknown or a sort was asked for with none available.
	 */
	bool UpdateBucket(int bucket, const BucketDesc& desc);

	/**
	 * @brief Drops one bucket's device buffers and frees its id for the next AddBucket.
	 * @param bucket Bucket id; out-of-range ids are ignored.
	 */
	void ReleaseBucket(int bucket);

	/// Drops every bucket and its device buffers.
	void ClearBuckets();

	/**
	 * @brief Runs cull, optional sort and matrix build for every bucket.
	 * @param transforms Device-resident packed transforms, 28-byte stride.
	 * @param filterKeys One uint per object, or 0 to accept everything.
	 * @param objectCount Objects in those buffers.
	 * @param camera This frame's view-projection, frustum and eye position.
	 * @param memberIndices Shared object-index array a bucket's MemberBase/MemberCount slices
	 *        into; null when no bucket restricts membership.
	 * @param scalars Per-object float4, appended by an InstID80 bucket's build kernel; null reads
	 *        as zero.
	 * @param scales Per-object float4 whose xyz is the instance scale; null is unit scale.
	 * @return False if a dispatch could not be issued; the caller should fall back.
	 */
	bool Run(irr::scene::IComputeBuffer* transforms, irr::scene::IComputeBuffer* filterKeys,
			 unsigned int objectCount, const CameraState& camera,
			 irr::scene::IComputeBuffer* memberIndices = nullptr,
			 irr::scene::IComputeBuffer* scalars = nullptr,
			 irr::scene::IComputeBuffer* scales = nullptr);

	/**
	 * @brief Runs one bucket against its own pose source and camera.
	 * @param bucket Bucket id.
	 * @param transforms Device-resident packed transforms for THIS bucket, 28-byte stride.
	 * @param filterKeys One uint per object, or 0 to accept everything.
	 * @param objectCount Objects in those buffers.
	 * @param camera The camera of the scene THIS bucket draws in.
	 * @param memberIndices Shared object-index array the bucket's MemberBase/MemberCount slices into.
	 * @param scalars Per-object float4 an InstID80 bucket appends; null reads as zero.
	 * @param scales Per-object float4 whose xyz is the instance scale; null is unit scale.
	 * @return False if the dispatch could not be issued; the caller should fall back.
	 *
	 * Composed poses are per world and Background/Main have different cameras, so one culler is
	 * shared per world and each manager's bucket brings its own pair.
	 */
	bool RunBucket(int bucket, irr::scene::IComputeBuffer* transforms,
				   irr::scene::IComputeBuffer* filterKeys, unsigned int objectCount,
				   const CameraState& camera, irr::scene::IComputeBuffer* memberIndices = nullptr,
				   irr::scene::IComputeBuffer* scalars = nullptr,
				   irr::scene::IComputeBuffer* scales = nullptr);

	/// One bucket of a batch and the camera of the scene it draws in.
	struct BatchEntry
	{
		int Bucket = -1;
		CameraState Camera;
	};

	/**
	 * @brief Culls and builds every listed bucket in one dispatch per kernel over shared buffers.
	 * @param entries Unsorted InstID80 buckets with a member slice, each with its own camera.
	 * @param transforms Device-resident packed transforms shared by every entry, 28-byte stride.
	 * @param filterKeys One uint per object, or 0 to accept everything.
	 * @param objectCount Objects in those buffers.
	 * @param memberIndices Shared object-index array every entry's member slice indexes.
	 * @param scalars Per-object float4 the records append; null reads as zero.
	 * @param scales Per-object float4 whose xyz is the instance scale; null is unit scale.
	 * @return False when an entry cannot be batched or a dispatch could not be issued.
	 *
	 * Until the next RunBucket of a bucket, InstanceBuffer/DrawArgsBuffer/DrawArgsOffset and the
	 * downloads read that bucket's region of the shared buffers.
	 */
	bool RunBatch(const std::vector<BatchEntry>& entries, irr::scene::IComputeBuffer* transforms,
				  irr::scene::IComputeBuffer* filterKeys, unsigned int objectCount,
				  irr::scene::IComputeBuffer* memberIndices, irr::scene::IComputeBuffer* scalars = nullptr,
				  irr::scene::IComputeBuffer* scales = nullptr);

	/// Returns every batched bucket to its own buffers, so a failed RunBatch never serves stale regions.
	void ClearBatch();

	/**
	 * @brief Whether a bucket can ride RunBatch.
	 * @param bucket Bucket id.
	 * @return True for a live, unsorted InstID80 bucket with a member slice.
	 */
	bool Batchable(int bucket) const;

	/**
	 * @brief Whether a bucket's last run was a RunBatch.
	 * @param bucket Bucket id.
	 * @return True when its stream lives in the shared batch buffers.
	 */
	bool Batched(int bucket) const;

	/// Per-instance vertex stream a bucket's draw reads, 64-byte row-major matrices.
	irr::scene::IComputeBuffer* InstanceBuffer(int bucket) const;
	/// DrawIndexedInstancedIndirect arguments for a bucket.
	irr::scene::IComputeBuffer* DrawArgsBuffer(int bucket) const;
	/// Byte offset of a bucket's arguments in DrawArgsBuffer(bucket); 0 unless batched.
	unsigned int DrawArgsOffset(int bucket) const;
	static unsigned int InstanceStride() { return 64; }
	/// Per-bucket stride: 64 for Matrix64, 80 for InstID80.
	unsigned int InstanceStride(int bucket) const;

	/**
	 * @brief Reads a bucket's surviving object indices back, in draw order. Stalls; verification only.
	 * @param bucket Bucket id.
	 * @param out Receives the object indices the GPU would draw, in order.
	 * @return False if the bucket has not run.
	 */
	bool DownloadOrder(int bucket, std::vector<unsigned int>& out);

	/**
	 * @brief Reads a bucket's built instance records back, in draw order. Stalls; verification only.
	 * @param bucket Bucket id.
	 * @param outFloats Receives the records, InstanceStride(bucket) bytes each, packed contiguously.
	 * @return False if the bucket has not run.
	 */
	bool DownloadInstances(int bucket, std::vector<float>& outFloats);

	/**
	 * @brief The same cull and sort on the CPU, as the reference the GPU result is checked against.
	 * @param transforms Packed transforms, 7 floats per object.
	 * @param filterKeys One uint per object, or 0.
	 * @param objectCount Objects to consider.
	 * @param desc Bucket filter, bounds and sort selection.
	 * @param camera Same camera the GPU pass was given.
	 * @param out Receives surviving object indices in draw order.
	 * @param memberIndices Same meaning as Run()'s, sliced by desc.MemberBase/MemberCount.
	 * @param scales Three floats per object; null is unit scale.
	 */
	static void CpuReference(const float* transforms, const unsigned int* filterKeys,
							 unsigned int objectCount, const BucketDesc& desc,
							 const CameraState& camera, std::vector<unsigned int>& out,
							 const unsigned int* memberIndices = nullptr,
							 const float* scales = nullptr);

	/**
	 * @brief The instance matrix a bucket's build kernel writes, in Irrlicht's row layout.
	 * @param pos Three floats, the instance position.
	 * @param quat Four floats, xyzw.
	 * @param scale Three floats; null is unit scale.
	 * @param out16 Receives the sixteen matrix floats.
	 */
	static void MatrixFromPose(const float* pos, const float* quat, const float* scale, float* out16);

	/**
	 * @brief Extracts the frustum planes from a combined view-projection matrix.
	 * @param viewProj 16 floats, Irrlicht row-major with the translation in row 3.
	 * @param eyeX Eye position x.
	 * @param eyeY Eye position y.
	 * @param eyeZ Eye position z.
	 * @param out Receives the packed camera state.
	 */
	static void MakeCameraState(const float* viewProj, float eyeX, float eyeY, float eyeZ,
								CameraState& out);

	/**
	 * @brief The process-wide instancing counters.
	 * @return The one counter block every culler, compose pass, feed and renderer adds to.
	 */
	static GpuInstancingCounters& Counters();

private:
	GpuInstanceCuller(const GpuInstanceCuller&);
	GpuInstanceCuller& operator=(const GpuInstanceCuller&);

	struct Bucket
	{
		Bucket();

		BucketDesc Desc;
		unsigned int Capacity;
		irr::scene::IComputeBuffer* Params;
		irr::scene::IComputeBuffer* Pairs;       ///< compacted survivors (unsorted) or full array (sorted)
		irr::scene::IComputeBuffer* Count;
		irr::scene::IComputeBuffer* DrawArgs;
		irr::scene::IComputeBuffer* DispatchArgs;
		irr::scene::IComputeBuffer* Instances;
		irr::scene::IComputeBuffer* Ordered;     ///< sort result, or Pairs when unsorted
		bool Released;                           ///< id kept so live buckets never shift; reusable
		int BatchSlot;                           ///< index in the last RunBatch, -1 when not batched
		unsigned int PairBase;                   ///< first survivor slot and record in the batch buffers
	};

	/**
	 * @brief Binds the shared defaults for a run and dispatches one bucket.
	 * @param bucket Bucket to dispatch.
	 * @param transforms Pose source for this bucket.
	 * @param filterKeys Filter keys, already defaulted.
	 * @param objectCount Objects in the pose source.
	 * @param camera Camera for this bucket.
	 * @param memberIndices Shared member-index array, or null.
	 * @param scalars Per-object scalars, or null.
	 * @param scales Per-object scale, already defaulted.
	 * @return False if a dispatch could not be issued.
	 */
	bool DispatchBucket(Bucket& bucket, irr::scene::IComputeBuffer* transforms,
						irr::scene::IComputeBuffer* filterKeys, unsigned int objectCount,
						const CameraState& camera, irr::scene::IComputeBuffer* memberIndices,
						irr::scene::IComputeBuffer* scalars, irr::scene::IComputeBuffer* scales);

	/**
	 * @brief Substitutes the accept-everything key buffer and the unit-scale buffer when unbound.
	 * @param filterKeys Replaced with the default keys when null.
	 * @param scales Replaced with the unit scales when null.
	 * @param objectCount Objects the defaults must cover.
	 */
	void EnsureDefaults(irr::scene::IComputeBuffer*& filterKeys,
						irr::scene::IComputeBuffer*& scales, unsigned int objectCount);

	/**
	 * @brief Grows a bucket's device buffers to cover iterCount objects.
	 * @param bucket Bucket to size.
	 * @param iterCount Objects the cull will iterate (MemberCount when set, else objectCount).
	 * @return False if a buffer could not be created.
	 */
	bool EnsureBucket(Bucket& bucket, unsigned int iterCount);

	/**
	 * @brief Uploads this frame's params block for one bucket.
	 * @param bucket Bucket to update.
	 * @param iterCount Objects the cull will iterate (MemberCount when set, else objectCount).
	 * @param camera This frame's camera state.
	 */
	void UploadParams(Bucket& bucket, unsigned int iterCount, const CameraState& camera);

	/// Selects a compiled compute kernel as the current material.
	void SetKernel(int material);

	irr::video::IVideoDriver* m_driver;
	irr::io::IFileSystem* m_fileSystem;

	int m_cullAppendMaterial;
	int m_cullMarkMaterial;
	int m_patchArgsMaterial;
	int m_buildMaterial;
	int m_resetMaterial;
	int m_batchResetMaterial;
	int m_batchCullMaterial;
	int m_batchPatchMaterial;
	int m_batchBuildMaterial;

	/// Shared by every bucket of the last RunBatch.
	irr::scene::IComputeBuffer* m_batchParams;
	irr::scene::IComputeBuffer* m_batchPairs;
	irr::scene::IComputeBuffer* m_batchCounts;
	irr::scene::IComputeBuffer* m_batchDrawArgs;
	irr::scene::IComputeBuffer* m_batchInstances;

	b3IrrlichtRadixSort* m_sort;
	std::vector<Bucket> m_buckets;
	/// Accepts everything; bound when the caller supplies no filter keys.
	irr::scene::IComputeBuffer* m_defaultKeys;
	unsigned int m_defaultKeyCount;
	/// Unit scale; bound when the caller supplies none, so the kernel never reads an unbound slot.
	irr::scene::IComputeBuffer* m_defaultScales;
	unsigned int m_defaultScaleCount;
	bool m_available;
};
