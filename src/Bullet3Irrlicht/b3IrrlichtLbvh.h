#ifndef B3_IRRLICHT_LBVH_H
#define B3_IRRLICHT_LBVH_H

#include <vector>
#include <utility>

namespace irr
{
namespace video
{
class IVideoDriver;
}
namespace scene
{
class IComputeBuffer;
}
namespace io
{
class IFileSystem;
}
}  // namespace irr

class b3IrrlichtRadixSort;
struct b3IrrSortData;

namespace b3IrrGpu
{
class DispatchHelper;
}

/// Layout-identical to b3Aabb (2 x float4, 32 bytes) and to the HLSL struct.
struct b3IrrAabb
{
	float minVec[4];
	float maxVec[4];
};

/// Emulated-double world AABB, 64 bytes, matching B3Precision.hlsli under OS_DS.
struct b3IrrAabbDS
{
	float minVec[4];
	float maxVec[4];
	float minLo[4];
	float maxLo[4];
};

/// Irrlicht-compute analog of b3GpuParallelLinearBvh. Phase 4a: merged world AABB and
/// Morton-code assignment, sorted by the GPU radix sort. Tree construction is Phase 4b.
class b3IrrlichtLbvh
{
public:
	b3IrrlichtLbvh(irr::video::IVideoDriver* driver);
	~b3IrrlichtLbvh();

	/**
	 * @brief Compiles the broadphase kernels.
	 * @param fileSystem Device filesystem; without it a missing shader binds an unrelated material.
	 * @param doubleSingle Compile the emulated-double (df64) kernel set instead of the f32 one.
	 * @return False if a required shader is missing or failed to compile.
	 */
	bool init(irr::io::IFileSystem* fileSystem = 0, bool doubleSingle = false);

	/// Whether this instance runs the emulated-double kernels. The std::vector entry points are
	/// f32-only and refuse in that mode; the resident path is the df64 one.
	bool isDoubleSingle() const { return m_doubleSingle; }

	/// Reduces every AABB to one. Exact (min/max), so no ordering concerns.
	bool computeMergedAabb(const std::vector<b3IrrAabb>& aabbs, b3IrrAabb& merged);

	/// Morton code + leaf index per AABB, sorted ascending by code.
	/// Morton codes are 30-bit, so only the passes that matter are run.
	bool buildSortedMortonCodes(const std::vector<b3IrrAabb>& aabbs,
								std::vector<b3IrrSortData>& sortedCodes);

	/// Full Karras tree build. childNodes holds 2 ints per internal node (left, right) with the
	/// high bit marking an internal child; internalAabbs is the fitted AABB per internal node.
	struct TreeResult
	{
		int rootIndex;                          // marker-encoded
		std::vector<int> childNodes;            // 2 * numInternalNodes
		std::vector<int> leafParents;           // numLeafNodes
		std::vector<int> internalParents;       // numInternalNodes
		std::vector<int> distanceFromRoot;      // numInternalNodes
		std::vector<b3IrrAabb> internalAabbs;   // numInternalNodes
		std::vector<b3IrrSortData> sortedCodes; // numLeafNodes
	};

	bool buildTree(const std::vector<b3IrrAabb>& aabbs, TreeResult& out);

	/// Broadphase result: every overlapping AABB pair, as indices into the CALLER's aabbs array,
	/// each pair reported once. Builds the tree internally, over the small set only - AABBs that
	/// dwarf the scene go to a flat side list tested brute force, which never changes which pairs
	/// are found. overflowed is set when the append buffer hit maxPairs - the pair list is then
	/// TRUNCATED, never silently short, so callers can grow and retry.
	bool calculateOverlappingPairs(const std::vector<b3IrrAabb>& aabbs,
								   std::vector<std::pair<unsigned int, unsigned int> >& pairs,
								   unsigned int maxPairs = 65536,
								   bool* overflowed = 0,
								   bool sortForDeterminism = true);

	/// Ray segment, layout-identical to the HLSL b3RayInfo (2 x float4).
	struct b3IrrRay
	{
		float from[4];
		float to[4];
	};

	/// Broadphase raycast: ray index -> candidate body index for every AABB the ray crosses, the
	/// body index being into the CALLER's aabbs array. Exact ray-vs-shape is narrowphase's job.
	bool castRays(const std::vector<b3IrrAabb>& aabbs, const std::vector<b3IrrRay>& rays,
				  std::vector<std::pair<unsigned int, unsigned int> >& rayBodyPairs,
				  unsigned int maxPairs = 65536, bool* overflowed = 0);

	/// Canonical ordering for a pair list, so float accumulation downstream is reproducible.
	/// Public so narrowphase can re-apply it to any pair list it produces itself.
	static void sortPairsForDeterminism(std::vector<std::pair<unsigned int, unsigned int> >& pairs);

	/**
	 * @brief Whether the device-buffer entry point below can run.
	 * @return True once B3GpuResident.hlsl compiled.
	 */
	bool isResidentPathAvailable() const;

	/**
	 * @brief Broadphase over AABBs that never leave the device; the pair list stays there too.
	 *
	 * Every AABB goes into the tree - the large-AABB side list needs a CPU-side classification, so
	 * a scene mixing scene-spanning AABBs with small ones should stay on the std::vector path.
	 *
	 * @param aabbs Device-resident world AABBs, one per body.
	 * @param numAabbs AABBs in that buffer; at least 2.
	 * @param maxPairs Append capacity; the count is clamped to it downstream. Above
	 *        maxSortablePairs() while resident sorting is on, the call refuses (see setResidentPairSorting).
	 * @param subset Optional body-index list to build the tree over instead of all of them.
	 * @param subsetCount Entries in subset; ignored when subset is null.
	 * @return False if the kernels are unavailable, the tree could not be built, the AABB stride
	 *         belongs to the other precision, or the pair sort was required and refused.
	 *
	 * @param refit Reuse the previous topology and only re-fit AABBs. Conservative, so no pair can
	 *        be missed, but quality decays as bodies drift; ignored if the leaf count changed.
	 * @param sleepStates Optional per-body sleep bits; when given, sleeping leaves stop querying and
	 *        the slot-order dedup yields to them. Drops both-asleep pairs, which the pipeline
	 *        discards before narrowphase anyway.
	 * @param refitSleepStates Same bits, used instead to re-fit only the root-paths of leaves that
	 *        moved. Must be the buffer that gated this step's world-AABB update, or bounds go stale.
	 */
	bool calculateOverlappingPairsResident(irr::scene::IComputeBuffer* aabbs, unsigned int numAabbs,
										   unsigned int maxPairs = 65536,
										   irr::scene::IComputeBuffer* subset = 0,
										   unsigned int subsetCount = 0, bool refit = false,
										   irr::scene::IComputeBuffer* sleepStates = 0,
										   irr::scene::IComputeBuffer* refitSleepStates = 0);

	/// Whether the sparse-refit kernels compiled; without them a refit falls back to the full fit.
	bool isSparseRefitAvailable() const;

	/// Deepest internal node of the last full build - the AABB fit costs one dispatch per level.
	int getLastMaxDistance() const { return m_cachedMaxDistance; }

	/// Reuse of the per-internal-node leaf ranges across refits, and skipping them entirely when
	/// the sleep-gated query cannot read them. Off restores a rebuild of them every step.
	void setLeafRangeCaching(bool enable) { m_leafRangeCaching = enable; }

	/// Pairs from the last broadphase call - in canonical (low, high) order when the resident sort
	/// ran (see setResidentPairSorting), otherwise in append order.
	irr::scene::IComputeBuffer* getPairBuffer() const
	{
		return m_sortedPairBuffer ? m_sortedPairBuffer : m_pairBuffer;
	}
	/// Appended pair count, as EHBF_DRAW_INDIRECT_ARGS - feeds an indirect narrowphase dispatch.
	irr::scene::IComputeBuffer* getPairCountBuffer() const { return m_pairCountBuffer; }

	/// Whether B3PairSort.hlsl compiled, i.e. the resident path can order its pairs on the GPU.
	bool isResidentPairSortAvailable() const;

	/**
	 * @brief Sorts the resident pair list into sortPairsForDeterminism's order, on the GPU.
	 *
	 * On by default. With it on, calculateOverlappingPairsResident REFUSES (returns false) a
	 * maxPairs above maxSortablePairs() rather than sorting a truncated prefix.
	 *
	 * @param enable False restores append order, which varies run to run.
	 */
	void setResidentPairSorting(bool enable) { m_sortResidentPairs = enable; }
	bool getResidentPairSorting() const { return m_sortResidentPairs; }

	/// Largest maxPairs the resident sort accepts - the radix sort's histogram-scan limit.
	static unsigned int maxSortablePairs();

	/// Whether the last resident broadphase call left its pairs sorted.
	bool lastPairsWereSorted() const { return m_sortedPairBuffer != 0; }

	/// Tree left by the last calculateOverlappingPairsResident call, for kernels that traverse it
	/// (b3IrrlichtQueries). Only the resident path keeps these consistent with each other.
	irr::scene::IComputeBuffer* getResidentChildNodeBuffer() const { return m_childNodeBuffer; }
	irr::scene::IComputeBuffer* getResidentInternalAabbBuffer() const { return m_internalAabbBuffer; }
	irr::scene::IComputeBuffer* getResidentSortedCodeBuffer() const { return m_cachedSortedCodes; }
	/// Leaf row -> body index of that call: the caller's subset buffer, else the identity map.
	irr::scene::IComputeBuffer* getResidentLeafToBodyBuffer() const { return m_residentLeafToBody; }
	int getResidentRootIndex() const { return m_cachedRootIndex; }
	unsigned int getResidentLeafCount() const { return m_cachedLeafCount; }

	static bool isLeafIndex(int index) { return (index >> 31) == 0; }
	static int stripMarker(int index) { return index & (~0x80000000); }

	/**
	 * @brief Original-array indices the last broadphase call routed through the large side list.
	 * @return Ascending indices; empty when the scene needed no split.
	 */
	const std::vector<unsigned int>& getLastLargeAabbIndices() const { return m_largeIndices; }

private:
	b3IrrlichtLbvh(const b3IrrlichtLbvh&);
	b3IrrlichtLbvh& operator=(const b3IrrlichtLbvh&);

	bool uploadAabbs(const std::vector<b3IrrAabb>& aabbs);
	void releaseBuffers();

	/**
	 * @brief Splits aabbs into the small set (tree) and the large side list, by extent relative
	 *        to the whole-scene extent.
	 * @param aabbs Caller's unpartitioned AABB array.
	 */
	void classifyAabbs(const std::vector<b3IrrAabb>& aabbs);

	/**
	 * @brief Classifies, gathers the large set on the GPU and uploads both index maps.
	 * @param aabbs Caller's unpartitioned AABB array.
	 * @return false when the GPU-side separation could not be set up.
	 */
	bool prepareAabbSets(const std::vector<b3IrrAabb>& aabbs);

	/**
	 * @brief Karras tree build over buffers already on the device.
	 * @param leafAabbs Leaf AABBs, indexed by each sorted code's value field.
	 * @param sortedCodes Morton codes sorted ascending.
	 * @param numLeaf Leaf count; at least 2.
	 * @param rootIndex Receives the marker-encoded root.
	 * @param distanceOut When set, the per-node depths are read back into it and the fitting
	 *        loop is bounded from there; otherwise the bound is reduced on the GPU.
	 * @return False if a kernel is unavailable.
	 */
	/// refitOnly reuses the existing topology and only re-fits AABBs - bounds stay conservative so
	/// no pair can be missed, but tree quality decays as bodies drift from their sorted order.
	bool buildTreeCore(irr::scene::IComputeBuffer* leafAabbs, irr::scene::IComputeBuffer* sortedCodes,
					   unsigned int numLeaf, int& rootIndex, std::vector<int>* distanceOut, bool refitOnly = false,
					   irr::scene::IComputeBuffer* refitSleepStates = 0,
					   irr::scene::IComputeBuffer* leafToBody = 0);

	/**
	 * @brief Re-fits only the internal nodes above leaves whose AABB could have changed.
	 * @param leafAabbs Leaf AABBs, indexed by each sorted code's value field.
	 * @param sortedCodes Morton codes sorted ascending.
	 * @param numLeaf Leaf count.
	 * @param sleepStates Per-body sleep bits that gated this step's world-AABB update.
	 * @param leafToBody Leaf row -> body index, so the sleep bit of a leaf can be found.
	 * @return False when the kernels are unavailable, leaving the caller to fit everything.
	 */
	bool refitDirtySubtrees(irr::scene::IComputeBuffer* leafAabbs,
							irr::scene::IComputeBuffer* sortedCodes, unsigned int numLeaf,
							irr::scene::IComputeBuffer* sleepStates,
							irr::scene::IComputeBuffer* leafToBody);

	/// Identity leaf->body map for the resident path, which never splits off large AABBs.
	bool ensureIdentityIndexMap(unsigned int numAabbs);

	/**
	 * @brief Identity leaf->body map for callers that already gathered their AABBs.
	 * @param count Leaves needing an entry.
	 * @return The cached buffer, resized and refilled only when count changes.
	 */
	irr::scene::IComputeBuffer* ensureLeafIdentity(unsigned int count);

	irr::video::IVideoDriver* m_driver;
	bool m_doubleSingle;
	b3IrrlichtRadixSort* m_sort;
	b3IrrGpu::DispatchHelper* m_dispatch;
	unsigned int m_identityIndexCount;

	irr::scene::IComputeBuffer* m_paramBuffer;
	irr::scene::IComputeBuffer* m_aabbBuffer;
	irr::scene::IComputeBuffer* m_mergedBuffer;
	irr::scene::IComputeBuffer* m_mortonBuffer;

	int m_mergeMaterial;
	int m_mortonMaterial;

	// Tree construction (B3LbvhTree.hlsl)
	int m_prefixMaterial;
	int m_leafMaterial;
	int m_internalMaterial;
	int m_distanceMaterial;
	int m_treeAabbMaterial;

	// Sparse refit (B3LbvhRefit.hlsl)
	int m_markDirtyMaterial;
	int m_refitDirtyMaterial;
	irr::scene::IComputeBuffer* m_refitParamBuffer;
	irr::scene::IComputeBuffer* m_dirtyStampBuffer;
	irr::scene::IComputeBuffer* m_dirtyNodeBuffer;
	irr::scene::IComputeBuffer* m_dirtyNodeCountBuffer;
	irr::scene::IComputeBuffer* m_dirtyCountParamBuffer;
	unsigned int m_dirtyStampCapacity;
	unsigned int m_refitFrameStamp;

	irr::scene::IComputeBuffer* m_treeParamBuffer;
	irr::scene::IComputeBuffer* m_sortedMortonBuffer;
	irr::scene::IComputeBuffer* m_commonPrefixBuffer;
	irr::scene::IComputeBuffer* m_commonPrefixLenBuffer;
	irr::scene::IComputeBuffer* m_childNodeBuffer;
	irr::scene::IComputeBuffer* m_leafParentBuffer;
	irr::scene::IComputeBuffer* m_internalParentBuffer;
	irr::scene::IComputeBuffer* m_rootIndexBuffer;
	irr::scene::IComputeBuffer* m_distanceBuffer;
	irr::scene::IComputeBuffer* m_internalAabbBuffer;

	// Phase 4c (B3LbvhPairs.hlsl)
	int m_leafRangeMaterial;
	int m_pairsMaterial;
	irr::scene::IComputeBuffer* m_pairParamBuffer;
	irr::scene::IComputeBuffer* m_leafRangeBuffer;
	irr::scene::IComputeBuffer* m_pairBuffer;
	irr::scene::IComputeBuffer* m_pairCountBuffer;
	int m_rayMaterial;
	irr::scene::IComputeBuffer* m_rayBuffer;
	irr::scene::IComputeBuffer* m_rayPairBuffer;

	// Resident pair ordering (B3PairSort.hlsl). Its own radix sort: the shared one's ping-pong
	// result is the cached Morton order a refit still reads.
	int m_normalisePairsMaterial;
	int m_swapKeyValueMaterial;
	b3IrrlichtRadixSort* m_pairSort;
	irr::scene::IComputeBuffer* m_pairSortParamBuffer;
	irr::scene::IComputeBuffer* m_pairSortInputBuffer;
	/// Non-owning: m_pairSort's result buffer from the last resident call, 0 when unsorted.
	irr::scene::IComputeBuffer* m_sortedPairBuffer;
	bool m_sortResidentPairs;

	/**
	 * @brief Orders m_pairBuffer's live prefix into m_sortedPairBuffer, entirely on the device.
	 * @param maxPairs Pair buffer capacity; the live count comes from m_pairCountBuffer.
	 * @param numBodies Bound on every body index, which sets how many radix passes run.
	 * @return False if the kernels are unavailable or maxPairs exceeds maxSortablePairs().
	 */
	bool sortResidentPairs(unsigned int maxPairs, unsigned int numBodies);

	// Large-AABB side list. m_smallIndices/m_largeIndices map back to the caller's array order.
	int m_separateMaterial;
	int m_largePairMaterial;
	int m_largeLargePairMaterial;
	int m_largeRayMaterial;

	std::vector<unsigned int> m_smallIndices;
	std::vector<unsigned int> m_largeIndices;
	std::vector<b3IrrAabb> m_smallAabbs;

	irr::scene::IComputeBuffer* m_separateParamBuffer;
	irr::scene::IComputeBuffer* m_smallIndexBuffer;
	/// Cached from the last FULL build so a refit needs neither the depth reduction nor the
	/// root-index readback (both are stalls, and neither can change while topology is reused).
	int m_cachedRootIndex;
	int m_cachedMaxDistance;
	irr::scene::IComputeBuffer* m_cachedSortedCodes;
	unsigned int m_cachedLeafCount;
	/// Non-owning: the subset buffer or m_smallIndexBuffer the last resident call traversed with.
	irr::scene::IComputeBuffer* m_residentLeafToBody;
	/// Internal-node count the cached leaf ranges describe; 0 when they must be recomputed.
	unsigned int m_cachedLeafRangeCount;
	bool m_leafRangeCaching;
	irr::scene::IComputeBuffer* m_leafIdentityBuffer;
	unsigned int m_leafIdentityCount;
	irr::scene::IComputeBuffer* m_largeIndexBuffer;
	irr::scene::IComputeBuffer* m_largeAabbBuffer;
};

#endif  //B3_IRRLICHT_LBVH_H
