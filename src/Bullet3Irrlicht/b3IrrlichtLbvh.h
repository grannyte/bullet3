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
	 * @param maxPairs Append capacity; the count is clamped to it downstream.
	 * @return False if the kernels are unavailable or the tree could not be built.
	 */
	bool calculateOverlappingPairsResident(irr::scene::IComputeBuffer* aabbs, unsigned int numAabbs,
										   unsigned int maxPairs = 65536);

	/// Pairs appended by the last broadphase call.
	irr::scene::IComputeBuffer* getPairBuffer() const { return m_pairBuffer; }
	/// Appended pair count, as EHBF_DRAW_INDIRECT_ARGS - feeds an indirect narrowphase dispatch.
	irr::scene::IComputeBuffer* getPairCountBuffer() const { return m_pairCountBuffer; }

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
	bool buildTreeCore(irr::scene::IComputeBuffer* leafAabbs, irr::scene::IComputeBuffer* sortedCodes,
					   unsigned int numLeaf, int& rootIndex, std::vector<int>* distanceOut);

	/// Identity leaf->body map for the resident path, which never splits off large AABBs.
	bool ensureIdentityIndexMap(unsigned int numAabbs);

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
	irr::scene::IComputeBuffer* m_largeIndexBuffer;
	irr::scene::IComputeBuffer* m_largeAabbBuffer;
};

#endif  //B3_IRRLICHT_LBVH_H
