#include "b3IrrlichtLbvh.h"
#include "b3IrrlichtRadixSort.h"
#include "b3IrrlichtGpuBuffers.h"

#include <irrlicht.h>
#include "ComputeBuffer.h"

#include <cstring>
#include <algorithm>

namespace
{
struct LbvhParams
{
	unsigned int numAabbs;
	unsigned int pad0;
	unsigned int pad1;
	unsigned int pad2;
};

static_assert(sizeof(LbvhParams) == 16, "LbvhParams must match the HLSL struct stride");
static_assert(sizeof(b3IrrAabb) == 32, "b3IrrAabb must match b3Aabb and the HLSL struct");
static_assert(sizeof(b3IrrAabbDS) == 64, "b3IrrAabbDS must match the OS_DS HLSL b3Aabb stride");

const unsigned int MORTON_BITS = 30;
}  // namespace

b3IrrlichtLbvh::b3IrrlichtLbvh(irr::video::IVideoDriver* driver)
	: m_driver(driver), m_doubleSingle(false), m_sort(0), m_dispatch(0), m_identityIndexCount(0), m_paramBuffer(0),
	  m_aabbBuffer(0), m_mergedBuffer(0),
	  m_mortonBuffer(0), m_mergeMaterial(-1), m_mortonMaterial(-1),
	  m_prefixMaterial(-1), m_leafMaterial(-1), m_internalMaterial(-1), m_distanceMaterial(-1),
	  m_treeAabbMaterial(-1), m_markDirtyMaterial(-1), m_refitDirtyMaterial(-1),
	  m_refitParamBuffer(0), m_dirtyStampBuffer(0), m_dirtyNodeBuffer(0), m_dirtyNodeCountBuffer(0),
	  m_dirtyCountParamBuffer(0), m_dirtyStampCapacity(0), m_refitFrameStamp(0),
	  m_treeParamBuffer(0), m_sortedMortonBuffer(0), m_commonPrefixBuffer(0),
	  m_commonPrefixLenBuffer(0), m_childNodeBuffer(0), m_leafParentBuffer(0), m_internalParentBuffer(0),
	  m_rootIndexBuffer(0), m_distanceBuffer(0), m_internalAabbBuffer(0),
	  m_leafRangeMaterial(-1), m_pairsMaterial(-1), m_pairParamBuffer(0),
	  m_leafRangeBuffer(0), m_pairBuffer(0), m_pairCountBuffer(0),
	  m_rayMaterial(-1), m_rayBuffer(0), m_rayPairBuffer(0),
	  m_separateMaterial(-1), m_largePairMaterial(-1), m_largeLargePairMaterial(-1),
	  m_largeRayMaterial(-1), m_separateParamBuffer(0), m_smallIndexBuffer(0),
	  m_largeIndexBuffer(0), m_largeAabbBuffer(0),
	  m_leafIdentityBuffer(0), m_leafIdentityCount(0),
	  m_cachedRootIndex(0), m_cachedMaxDistance(0), m_cachedSortedCodes(0), m_cachedLeafCount(0),
	  m_cachedLeafRangeCount(0), m_leafRangeCaching(true)
{
}

b3IrrlichtLbvh::~b3IrrlichtLbvh()
{
	releaseBuffers();
	if (m_paramBuffer) m_paramBuffer->drop();
	if (m_treeParamBuffer) m_treeParamBuffer->drop();
	if (m_sortedMortonBuffer) m_sortedMortonBuffer->drop();
	if (m_commonPrefixBuffer) m_commonPrefixBuffer->drop();
	if (m_commonPrefixLenBuffer) m_commonPrefixLenBuffer->drop();
	if (m_childNodeBuffer) m_childNodeBuffer->drop();
	if (m_leafParentBuffer) m_leafParentBuffer->drop();
	if (m_internalParentBuffer) m_internalParentBuffer->drop();
	if (m_rootIndexBuffer) m_rootIndexBuffer->drop();
	if (m_distanceBuffer) m_distanceBuffer->drop();
	if (m_internalAabbBuffer) m_internalAabbBuffer->drop();
	if (m_refitParamBuffer) m_refitParamBuffer->drop();
	if (m_dirtyStampBuffer) m_dirtyStampBuffer->drop();
	if (m_dirtyNodeBuffer) m_dirtyNodeBuffer->drop();
	if (m_dirtyNodeCountBuffer) m_dirtyNodeCountBuffer->drop();
	if (m_dirtyCountParamBuffer) m_dirtyCountParamBuffer->drop();
	if (m_pairParamBuffer) m_pairParamBuffer->drop();
	if (m_leafRangeBuffer) m_leafRangeBuffer->drop();
	if (m_pairBuffer) m_pairBuffer->drop();
	if (m_pairCountBuffer) m_pairCountBuffer->drop();
	if (m_rayBuffer) m_rayBuffer->drop();
	if (m_rayPairBuffer) m_rayPairBuffer->drop();
	if (m_separateParamBuffer) m_separateParamBuffer->drop();
	if (m_smallIndexBuffer) m_smallIndexBuffer->drop();
	if (m_largeIndexBuffer) m_largeIndexBuffer->drop();
	if (m_largeAabbBuffer) m_largeAabbBuffer->drop();
	if (m_leafIdentityBuffer) m_leafIdentityBuffer->drop();
	delete m_sort;
	delete m_dispatch;
}

void b3IrrlichtLbvh::releaseBuffers()
{
	if (m_aabbBuffer) { m_aabbBuffer->drop(); m_aabbBuffer = 0; }
	if (m_mergedBuffer) { m_mergedBuffer->drop(); m_mergedBuffer = 0; }
	if (m_mortonBuffer) { m_mortonBuffer->drop(); m_mortonBuffer = 0; }
}

bool b3IrrlichtLbvh::init(irr::io::IFileSystem* fileSystem, bool doubleSingle)
{
	m_doubleSingle = doubleSingle;
	if (!m_driver)
		return false;

	irr::video::IGPUProgrammingServices* gpu = m_driver->getGPUProgrammingServices();
	if (!gpu)
		return false;

	const irr::io::path path = m_doubleSingle ? "media/shaders/B3LbvhDS.hlsl" : "media/shaders/B3Lbvh.hlsl";
	if (fileSystem && !fileSystem->existFile(path))
		return false;

	m_mergeMaterial = gpu->addComputeShaderFromFile(path, "CSMergeAabbs", irr::video::ECST_CS_5_0, 0);
	m_mortonMaterial = gpu->addComputeShaderFromFile(path, "CSAssignMortonCodes", irr::video::ECST_CS_5_0, 0);
	m_separateMaterial = gpu->addComputeShaderFromFile(path, "CSSeparateAabbs", irr::video::ECST_CS_5_0, 0);

	const irr::io::path treePath = m_doubleSingle ? "media/shaders/B3LbvhTreeDS.hlsl" : "media/shaders/B3LbvhTree.hlsl";
	if (fileSystem && !fileSystem->existFile(treePath))
		return false;

	m_prefixMaterial = gpu->addComputeShaderFromFile(treePath, "CSComputeAdjacentPairCommonPrefix", irr::video::ECST_CS_5_0, 0);
	m_leafMaterial = gpu->addComputeShaderFromFile(treePath, "CSBuildLeafNodes", irr::video::ECST_CS_5_0, 0);
	m_internalMaterial = gpu->addComputeShaderFromFile(treePath, "CSBuildInternalNodes", irr::video::ECST_CS_5_0, 0);
	m_distanceMaterial = gpu->addComputeShaderFromFile(treePath, "CSFindDistanceFromRoot", irr::video::ECST_CS_5_0, 0);
	m_treeAabbMaterial = gpu->addComputeShaderFromFile(treePath, "CSBuildTreeAabbs", irr::video::ECST_CS_5_0, 0);

	// Optional: a missing/broken refit shader just leaves every refit fitting the whole tree.
	const irr::io::path refitPath = m_doubleSingle ? "media/shaders/B3LbvhRefitDS.hlsl" : "media/shaders/B3LbvhRefit.hlsl";
	if (!fileSystem || fileSystem->existFile(refitPath))
	{
		m_markDirtyMaterial = gpu->addComputeShaderFromFile(refitPath, "CSMarkMovedLeafAncestors", irr::video::ECST_CS_5_0, 0);
		m_refitDirtyMaterial = gpu->addComputeShaderFromFile(refitPath, "CSRefitDirtyAabbs", irr::video::ECST_CS_5_0, 0);
	}

	const irr::io::path pairPath = m_doubleSingle ? "media/shaders/B3LbvhPairsDS.hlsl" : "media/shaders/B3LbvhPairs.hlsl";
	if (fileSystem && !fileSystem->existFile(pairPath))
		return false;
	m_leafRangeMaterial = gpu->addComputeShaderFromFile(pairPath, "CSFindLeafIndexRanges", irr::video::ECST_CS_5_0, 0);
	m_pairsMaterial = gpu->addComputeShaderFromFile(pairPath, "CSCalculateOverlappingPairs", irr::video::ECST_CS_5_0, 0);
	m_rayMaterial = gpu->addComputeShaderFromFile(pairPath, "CSRayTraverse", irr::video::ECST_CS_5_0, 0);
	m_largePairMaterial = gpu->addComputeShaderFromFile(pairPath, "CSLargeAabbAabbTest", irr::video::ECST_CS_5_0, 0);
	m_largeLargePairMaterial = gpu->addComputeShaderFromFile(pairPath, "CSLargeLargeAabbTest", irr::video::ECST_CS_5_0, 0);
	m_largeRayMaterial = gpu->addComputeShaderFromFile(pairPath, "CSLargeAabbRayTest", irr::video::ECST_CS_5_0, 0);

	m_sort = new b3IrrlichtRadixSort(m_driver);
	if (!m_sort->init(fileSystem))
		return false;

	// Optional: without it only the std::vector entry points work.
	if (!m_dispatch)
		m_dispatch = new b3IrrGpu::DispatchHelper(m_driver);
	m_dispatch->init(fileSystem);

	return m_mergeMaterial >= 0 && m_mortonMaterial >= 0 && m_prefixMaterial >= 0 &&
		   m_leafMaterial >= 0 && m_internalMaterial >= 0 && m_distanceMaterial >= 0 &&
		   m_treeAabbMaterial >= 0 && m_leafRangeMaterial >= 0 && m_pairsMaterial >= 0 &&
		   m_rayMaterial >= 0 && m_separateMaterial >= 0 && m_largePairMaterial >= 0 &&
		   m_largeLargePairMaterial >= 0 && m_largeRayMaterial >= 0;
}

bool b3IrrlichtLbvh::uploadAabbs(const std::vector<b3IrrAabb>& aabbs)
{
	// The std::vector entry points move 32-byte AABBs; the df64 kernels index a 64-byte stride.
	if (m_doubleSingle)
		return false;

	b3IrrGpu::uploadBuffer<b3IrrAabb>(m_aabbBuffer, &aabbs[0], (irr::u32)aabbs.size());
	b3IrrGpu::ensureBuffer<b3IrrAabb>(m_mergedBuffer, 1);
	b3IrrGpu::ensureBuffer<b3IrrSortData>(m_mortonBuffer, (irr::u32)aabbs.size());
	b3IrrGpu::ensureBuffer<LbvhParams>(m_paramBuffer, 1);

	LbvhParams params;
	params.numAabbs = (unsigned int)aabbs.size();
	params.pad0 = params.pad1 = params.pad2 = 0;
	memcpy(m_paramBuffer->getBufferPointer(), &params, sizeof(params));
	m_paramBuffer->setDirty();

	return true;
}

bool b3IrrlichtLbvh::computeMergedAabb(const std::vector<b3IrrAabb>& aabbs, b3IrrAabb& merged)
{
	if (m_mergeMaterial < 0 || aabbs.empty())
		return false;

	if (!uploadAabbs(aabbs))
		return false;

	irr::video::SMaterial mat;
	mat.MaterialType = (irr::video::E_MATERIAL_TYPE)m_mergeMaterial;
	m_driver->setMaterial(mat);
	m_driver->bindComputeBuffer(0, m_paramBuffer, irr::video::EHBT_SHADER_RESOURCE);
	m_driver->bindComputeBuffer(1, m_aabbBuffer, irr::video::EHBT_SHADER_RESOURCE);
	m_driver->bindComputeBuffer(4, ensureLeafIdentity((unsigned int)aabbs.size()), irr::video::EHBT_SHADER_RESOURCE);
	m_driver->bindComputeBuffer(0, m_mergedBuffer, irr::video::EHBT_COMPUTE);
	m_driver->dispatchComputeShaderBound(irr::core::vector3d<irr::u32>(1, 1, 1));
	m_driver->unbindComputeResources();

	m_mergedBuffer->downloadFromGPU();
	memcpy(&merged, m_mergedBuffer->getBufferPointer(), sizeof(b3IrrAabb));
	return true;
}

bool b3IrrlichtLbvh::buildSortedMortonCodes(const std::vector<b3IrrAabb>& aabbs,
											std::vector<b3IrrSortData>& sortedCodes)
{
	if (m_mortonMaterial < 0 || aabbs.empty())
		return false;

	if (!uploadAabbs(aabbs))
		return false;

	irr::video::SMaterial mat;

	mat.MaterialType = (irr::video::E_MATERIAL_TYPE)m_mergeMaterial;
	m_driver->setMaterial(mat);
	m_driver->bindComputeBuffer(0, m_paramBuffer, irr::video::EHBT_SHADER_RESOURCE);
	m_driver->bindComputeBuffer(1, m_aabbBuffer, irr::video::EHBT_SHADER_RESOURCE);
	m_driver->bindComputeBuffer(4, ensureLeafIdentity((unsigned int)aabbs.size()), irr::video::EHBT_SHADER_RESOURCE);
	m_driver->bindComputeBuffer(0, m_mergedBuffer, irr::video::EHBT_COMPUTE);
	m_driver->dispatchComputeShaderBound(irr::core::vector3d<irr::u32>(1, 1, 1));
	m_driver->unbindComputeResources();

	// The merged AABB is written as a UAV above and read as an SRV here, so it must be
	// rebound rather than left in the UAV slot - computeBarrier makes that explicit.
	m_driver->computeBarrier(m_mergedBuffer);

	const irr::u32 groups = ((irr::u32)aabbs.size() + 127) / 128;
	mat.MaterialType = (irr::video::E_MATERIAL_TYPE)m_mortonMaterial;
	m_driver->setMaterial(mat);
	m_driver->bindComputeBuffer(0, m_paramBuffer, irr::video::EHBT_SHADER_RESOURCE);
	m_driver->bindComputeBuffer(1, m_aabbBuffer, irr::video::EHBT_SHADER_RESOURCE);
	m_driver->bindComputeBuffer(4, ensureLeafIdentity((unsigned int)aabbs.size()), irr::video::EHBT_SHADER_RESOURCE);
	m_driver->bindComputeBuffer(2, m_mergedBuffer, irr::video::EHBT_SHADER_RESOURCE);
	m_driver->bindComputeBuffer(1, m_mortonBuffer, irr::video::EHBT_COMPUTE);
	m_driver->dispatchComputeShaderBound(irr::core::vector3d<irr::u32>(groups, 1, 1));
	m_driver->unbindComputeResources();

	m_mortonBuffer->downloadFromGPU();
	sortedCodes.resize(aabbs.size());
	memcpy(&sortedCodes[0], m_mortonBuffer->getBufferPointer(), aabbs.size() * sizeof(b3IrrSortData));

	// 30-bit codes, so the top two 4-bit passes would be no-ops.
	return m_sort->execute(sortedCodes, MORTON_BITS);
}

// ---------------------------------------------------------------------------------------------
// Phase 4b - Karras binary radix tree construction (B3LbvhTree.hlsl)
// ---------------------------------------------------------------------------------------------

namespace
{
struct TreeParams
{
	unsigned int numLeafNodes;
	unsigned int numInternalNodes;
	int processedDistance;
	int pad0;
};

static_assert(sizeof(TreeParams) == 16, "TreeParams must match the HLSL struct stride");

struct RefitParams
{
	unsigned int numLeafNodes;
	unsigned int numInternalNodes;
	int processedDistance;
	unsigned int frameStamp;
};

static_assert(sizeof(RefitParams) == 16, "RefitParams must match the HLSL struct stride");

/// DispatchHelper patches the dirty-node count into the first field of a 16-byte struct.
struct DirtyCountParams
{
	unsigned int count;
	unsigned int pad0;
	unsigned int pad1;
	unsigned int pad2;
};

static_assert(sizeof(DirtyCountParams) == 16, "DirtyCountParams must match the HLSL uint4 stride");

using b3IrrGpu::dropBuffer;
using b3IrrGpu::ensureBuffer;
}  // namespace

bool b3IrrlichtLbvh::isSparseRefitAvailable() const
{
	return m_markDirtyMaterial >= 0 && m_refitDirtyMaterial >= 0 && m_dispatch && m_dispatch->isAvailable();
}

bool b3IrrlichtLbvh::refitDirtySubtrees(irr::scene::IComputeBuffer* leafAabbs,
										irr::scene::IComputeBuffer* sortedCodes, unsigned int numLeaf,
										irr::scene::IComputeBuffer* sleepStates,
										irr::scene::IComputeBuffer* leafToBody)
{
	if (!isSparseRefitAvailable() || !sleepStates || !leafToBody || numLeaf < 2)
		return false;

	const irr::u32 numInternal = numLeaf - 1;

	// Stamps must start at a value no frame can match; ensureBuffer leaves a grown one undefined.
	if (!m_dirtyStampBuffer || m_dirtyStampCapacity < numInternal)
	{
		const std::vector<unsigned int> zeros(numInternal, 0u);
		b3IrrGpu::uploadBuffer<unsigned int>(m_dirtyStampBuffer, &zeros[0], numInternal);
		if (!m_dirtyStampBuffer)
			return false;
		m_dirtyStampCapacity = numInternal;
		m_refitFrameStamp = 0;
	}

	// Each node is appended at most once, so numInternal cannot overflow.
	ensureBuffer<unsigned int>(m_dirtyNodeBuffer, numInternal, irr::video::EHBF_COMPUTE_APPEND);
	ensureBuffer<unsigned int>(m_dirtyNodeCountBuffer, 4, irr::video::EHBF_DRAW_INDIRECT_ARGS);
	ensureBuffer<RefitParams>(m_refitParamBuffer, 1);
	ensureBuffer<DirtyCountParams>(m_dirtyCountParamBuffer, 1);
	if (!m_dirtyNodeBuffer || !m_dirtyNodeCountBuffer || !m_refitParamBuffer || !m_dirtyCountParamBuffer)
		return false;

	if (++m_refitFrameStamp == 0)
		m_refitFrameStamp = 1;

	RefitParams rp;
	rp.numLeafNodes = numLeaf;
	rp.numInternalNodes = numInternal;
	rp.processedDistance = 0;
	rp.frameStamp = m_refitFrameStamp;
	memcpy(m_refitParamBuffer->getBufferPointer(), &rp, sizeof(rp));
	m_refitParamBuffer->setDirty();

	irr::video::SMaterial mat;
	mat.MaterialType = (irr::video::E_MATERIAL_TYPE)m_markDirtyMaterial;
	m_driver->setMaterial(mat);
	m_driver->bindComputeBuffer(0, m_refitParamBuffer, irr::video::EHBT_SHADER_RESOURCE);
	m_driver->bindComputeBuffer(1, sortedCodes, irr::video::EHBT_SHADER_RESOURCE);
	m_driver->bindComputeBuffer(2, m_leafParentBuffer, irr::video::EHBT_SHADER_RESOURCE);
	m_driver->bindComputeBuffer(3, m_internalParentBuffer, irr::video::EHBT_SHADER_RESOURCE);
	m_driver->bindComputeBuffer(4, sleepStates, irr::video::EHBT_SHADER_RESOURCE);
	m_driver->bindComputeBuffer(5, leafToBody, irr::video::EHBT_SHADER_RESOURCE);
	m_driver->bindComputeBuffer(0, m_dirtyStampBuffer, irr::video::EHBT_COMPUTE);
	m_driver->bindComputeBuffer(1, m_dirtyNodeBuffer, irr::video::EHBT_COMPUTE);
	m_driver->resetStructureCount(m_dirtyNodeBuffer, 0);
	m_driver->dispatchComputeShaderBound(irr::core::vector3d<irr::u32>((numLeaf + 127) / 128, 1, 1));
	m_driver->copyStructureCount(m_dirtyNodeCountBuffer, 0, m_dirtyNodeBuffer);
	m_driver->unbindComputeResources();
	m_driver->computeBarrier(m_dirtyNodeBuffer);

	if (!m_dispatch->prepareIndirect(m_dirtyNodeCountBuffer, m_dirtyCountParamBuffer, 128, numInternal))
		return false;
	irr::scene::IComputeBuffer* args = m_dispatch->getArgsBuffer();

	for (int d = m_cachedMaxDistance; d >= 0; --d)
	{
		rp.processedDistance = d;
		memcpy(m_refitParamBuffer->getBufferPointer(), &rp, sizeof(rp));
		m_refitParamBuffer->setDirty();

		mat.MaterialType = (irr::video::E_MATERIAL_TYPE)m_refitDirtyMaterial;
		m_driver->setMaterial(mat);
		m_driver->bindComputeBuffer(0, m_refitParamBuffer, irr::video::EHBT_SHADER_RESOURCE);
		m_driver->bindComputeBuffer(1, sortedCodes, irr::video::EHBT_SHADER_RESOURCE);
		m_driver->bindComputeBuffer(6, m_dirtyCountParamBuffer, irr::video::EHBT_SHADER_RESOURCE);
		m_driver->bindComputeBuffer(7, m_dirtyNodeBuffer, irr::video::EHBT_SHADER_RESOURCE);
		m_driver->bindComputeBuffer(8, m_childNodeBuffer, irr::video::EHBT_SHADER_RESOURCE);
		m_driver->bindComputeBuffer(9, m_distanceBuffer, irr::video::EHBT_SHADER_RESOURCE);
		m_driver->bindComputeBuffer(10, leafAabbs, irr::video::EHBT_SHADER_RESOURCE);
		m_driver->bindComputeBuffer(2, m_internalAabbBuffer, irr::video::EHBT_COMPUTE);
		m_driver->dispatchComputeShaderIndirect(args, 0);
		m_driver->unbindComputeResources();
	}

	return true;
}

bool b3IrrlichtLbvh::buildTreeCore(irr::scene::IComputeBuffer* leafAabbs,
								   irr::scene::IComputeBuffer* sortedCodes, unsigned int numLeaf,
								   int& rootIndex, std::vector<int>* distanceOut, bool refitOnly,
								   irr::scene::IComputeBuffer* refitSleepStates,
								   irr::scene::IComputeBuffer* leafToBody)
{
	if (m_prefixMaterial < 0 || !leafAabbs || !sortedCodes || numLeaf < 2)
		return false;

	const irr::u32 numInternal = numLeaf - 1;

	ensureBuffer<b3IrrSortData>(m_commonPrefixBuffer, numInternal);  // uint2, same 8-byte stride
	ensureBuffer<int>(m_commonPrefixLenBuffer, numInternal);
	ensureBuffer<int>(m_childNodeBuffer, numInternal * 2);
	ensureBuffer<int>(m_leafParentBuffer, numLeaf);
	ensureBuffer<int>(m_internalParentBuffer, numInternal);
	ensureBuffer<int>(m_rootIndexBuffer, 1);
	ensureBuffer<int>(m_distanceBuffer, numInternal);
	if (m_doubleSingle)
		ensureBuffer<b3IrrAabbDS>(m_internalAabbBuffer, numInternal);
	else
		ensureBuffer<b3IrrAabb>(m_internalAabbBuffer, numInternal);
	ensureBuffer<TreeParams>(m_treeParamBuffer, 1);

	TreeParams tp;
	tp.numLeafNodes = numLeaf;
	tp.numInternalNodes = numInternal;
	tp.processedDistance = 0;
	tp.pad0 = 0;
	memcpy(m_treeParamBuffer->getBufferPointer(), &tp, sizeof(tp));
	m_treeParamBuffer->setDirty();

	const irr::u32 internalGroups = (numInternal + 127) / 128;
	const irr::u32 leafGroups = (numLeaf + 127) / 128;
	irr::video::SMaterial mat;

	int maxDistance = 0;

	// Stages 1-4 build the TOPOLOGY. A refit keeps it and only re-fits AABBs below, which is
	// correct for any leaf movement - bounds stay conservative - but degrades tree quality.
	if (!refitOnly)
	{
		// 1) adjacent-pair common prefixes
		mat.MaterialType = (irr::video::E_MATERIAL_TYPE)m_prefixMaterial;
		m_driver->setMaterial(mat);
		m_driver->bindComputeBuffer(0, m_treeParamBuffer, irr::video::EHBT_SHADER_RESOURCE);
		m_driver->bindComputeBuffer(1, sortedCodes, irr::video::EHBT_SHADER_RESOURCE);
		m_driver->bindComputeBuffer(0, m_commonPrefixBuffer, irr::video::EHBT_COMPUTE);
		m_driver->bindComputeBuffer(1, m_commonPrefixLenBuffer, irr::video::EHBT_COMPUTE);
		m_driver->dispatchComputeShaderBound(irr::core::vector3d<irr::u32>(internalGroups, 1, 1));
		m_driver->unbindComputeResources();

		// 2) leaves attach to their split
		mat.MaterialType = (irr::video::E_MATERIAL_TYPE)m_leafMaterial;
		m_driver->setMaterial(mat);
		m_driver->bindComputeBuffer(0, m_treeParamBuffer, irr::video::EHBT_SHADER_RESOURCE);
		m_driver->bindComputeBuffer(3, m_commonPrefixLenBuffer, irr::video::EHBT_SHADER_RESOURCE);
		m_driver->bindComputeBuffer(2, m_childNodeBuffer, irr::video::EHBT_COMPUTE);
		m_driver->bindComputeBuffer(3, m_leafParentBuffer, irr::video::EHBT_COMPUTE);
		m_driver->dispatchComputeShaderBound(irr::core::vector3d<irr::u32>(leafGroups, 1, 1));
		m_driver->unbindComputeResources();

		// 3) internal node linking + root discovery
		mat.MaterialType = (irr::video::E_MATERIAL_TYPE)m_internalMaterial;
		m_driver->setMaterial(mat);
		m_driver->bindComputeBuffer(0, m_treeParamBuffer, irr::video::EHBT_SHADER_RESOURCE);
		m_driver->bindComputeBuffer(2, m_commonPrefixBuffer, irr::video::EHBT_SHADER_RESOURCE);
		m_driver->bindComputeBuffer(3, m_commonPrefixLenBuffer, irr::video::EHBT_SHADER_RESOURCE);
		m_driver->bindComputeBuffer(2, m_childNodeBuffer, irr::video::EHBT_COMPUTE);
		m_driver->bindComputeBuffer(4, m_internalParentBuffer, irr::video::EHBT_COMPUTE);
		m_driver->bindComputeBuffer(5, m_rootIndexBuffer, irr::video::EHBT_COMPUTE);
		m_driver->dispatchComputeShaderBound(irr::core::vector3d<irr::u32>(internalGroups, 1, 1));
		m_driver->unbindComputeResources();

		// 4) depth per internal node
		mat.MaterialType = (irr::video::E_MATERIAL_TYPE)m_distanceMaterial;
		m_driver->setMaterial(mat);
		m_driver->bindComputeBuffer(0, m_treeParamBuffer, irr::video::EHBT_SHADER_RESOURCE);
		m_driver->bindComputeBuffer(5, m_internalParentBuffer, irr::video::EHBT_SHADER_RESOURCE);
		m_driver->bindComputeBuffer(6, m_distanceBuffer, irr::video::EHBT_COMPUTE);
		m_driver->dispatchComputeShaderBound(irr::core::vector3d<irr::u32>(internalGroups, 1, 1));
		m_driver->unbindComputeResources();

		// The depth range drives how many fitting passes are needed, so the bound has to reach the
		// host: as the whole array when the caller wants it anyway, otherwise as a 4-byte reduction.
		maxDistance = 0;
		if (distanceOut)
		{
			m_distanceBuffer->downloadFromGPU();
			distanceOut->resize(numInternal);
			memcpy(&(*distanceOut)[0], m_distanceBuffer->getBufferPointer(), numInternal * sizeof(int));
			for (irr::u32 i = 0; i < numInternal; ++i)
				if ((*distanceOut)[i] > maxDistance)
					maxDistance = (*distanceOut)[i];
		}
		else if (!m_dispatch || !m_dispatch->reduceMax(m_distanceBuffer, numInternal, maxDistance))
		{
			return false;
		}

	}
	else
	{
		maxDistance = m_cachedMaxDistance;
	}

	// 5) fit AABBs deepest-first so children are always ready before their parent. A refit can do
	// this over the moved leaves' ancestors alone - every other node's bound is still exact.
	const bool sparse = refitOnly && refitSleepStates &&
						refitDirtySubtrees(leafAabbs, sortedCodes, numLeaf, refitSleepStates, leafToBody);
	for (int d = sparse ? -1 : maxDistance; d >= 0; --d)
	{
		tp.processedDistance = d;
		memcpy(m_treeParamBuffer->getBufferPointer(), &tp, sizeof(tp));
		m_treeParamBuffer->setDirty();

		mat.MaterialType = (irr::video::E_MATERIAL_TYPE)m_treeAabbMaterial;
		m_driver->setMaterial(mat);
		m_driver->bindComputeBuffer(0, m_treeParamBuffer, irr::video::EHBT_SHADER_RESOURCE);
		m_driver->bindComputeBuffer(1, sortedCodes, irr::video::EHBT_SHADER_RESOURCE);
		m_driver->bindComputeBuffer(4, m_childNodeBuffer, irr::video::EHBT_SHADER_RESOURCE);
		m_driver->bindComputeBuffer(6, m_distanceBuffer, irr::video::EHBT_SHADER_RESOURCE);
		m_driver->bindComputeBuffer(7, leafAabbs, irr::video::EHBT_SHADER_RESOURCE);
		m_driver->bindComputeBuffer(7, m_internalAabbBuffer, irr::video::EHBT_COMPUTE);
		m_driver->dispatchComputeShaderBound(irr::core::vector3d<irr::u32>(internalGroups, 1, 1));
		m_driver->unbindComputeResources();
	}

	if (refitOnly)
	{
		rootIndex = m_cachedRootIndex;
		return true;
	}

	m_rootIndexBuffer->downloadFromGPU();
	memcpy(&rootIndex, m_rootIndexBuffer->getBufferPointer(), sizeof(int));
	m_cachedRootIndex = rootIndex;
	m_cachedMaxDistance = maxDistance;
	return true;
}

bool b3IrrlichtLbvh::buildTree(const std::vector<b3IrrAabb>& aabbs, TreeResult& out)
{
	if (m_prefixMaterial < 0 || aabbs.size() < 2)
		return false;  // a 1-leaf tree has no internal nodes

	if (!buildSortedMortonCodes(aabbs, out.sortedCodes))
		return false;

	const irr::u32 numLeaf = (irr::u32)aabbs.size();
	const irr::u32 numInternal = numLeaf - 1;

	// The sort round-trips through the CPU on this path, so the sorted codes go back up here.
	b3IrrGpu::uploadBuffer<b3IrrSortData>(m_sortedMortonBuffer, &out.sortedCodes[0], numLeaf);

	if (!buildTreeCore(m_aabbBuffer, m_sortedMortonBuffer, numLeaf, out.rootIndex,
					   &out.distanceFromRoot))
		return false;

	m_childNodeBuffer->downloadFromGPU();
	out.childNodes.resize(numInternal * 2);
	memcpy(&out.childNodes[0], m_childNodeBuffer->getBufferPointer(), numInternal * 2 * sizeof(int));

	m_leafParentBuffer->downloadFromGPU();
	out.leafParents.resize(numLeaf);
	memcpy(&out.leafParents[0], m_leafParentBuffer->getBufferPointer(), numLeaf * sizeof(int));

	m_internalParentBuffer->downloadFromGPU();
	out.internalParents.resize(numInternal);
	memcpy(&out.internalParents[0], m_internalParentBuffer->getBufferPointer(), numInternal * sizeof(int));

	m_internalAabbBuffer->downloadFromGPU();
	out.internalAabbs.resize(numInternal);
	memcpy(&out.internalAabbs[0], m_internalAabbBuffer->getBufferPointer(), numInternal * sizeof(b3IrrAabb));

	return true;
}

// ---------------------------------------------------------------------------------------------
// Phase 4c - leaf index ranges + overlapping pair finding (B3LbvhPairs.hlsl)
// ---------------------------------------------------------------------------------------------

namespace
{
struct PairParams
{
	unsigned int numLeafNodes;
	unsigned int numInternalNodes;
	int rootIndex;
	unsigned int numLargeAabbs;
	unsigned int useSleepGate;
	unsigned int pad0;
	unsigned int pad1;
	unsigned int pad2;
};

static_assert(sizeof(PairParams) == 32, "PairParams must match the HLSL struct stride");

struct IrrInt2
{
	int x;
	int y;
};

struct IrrUint2
{
	unsigned int x;
	unsigned int y;
};

// An AABB spanning this much of the scene on any axis contributes no discriminating Morton bits.
const float LARGE_AABB_EXTENT_FRACTION = 0.25f;
// Brute force is O(large * small), so the side list is capped rather than left scene-dependent.
const size_t MAX_LARGE_AABBS = 32;
}  // namespace

void b3IrrlichtLbvh::classifyAabbs(const std::vector<b3IrrAabb>& aabbs)
{
	m_smallIndices.clear();
	m_largeIndices.clear();

	float worldMin[3];
	float worldMax[3];
	for (int a = 0; a < 3; ++a)
	{
		worldMin[a] = aabbs[0].minVec[a];
		worldMax[a] = aabbs[0].maxVec[a];
	}
	for (size_t i = 1; i < aabbs.size(); ++i)
		for (int a = 0; a < 3; ++a)
		{
			if (aabbs[i].minVec[a] < worldMin[a]) worldMin[a] = aabbs[i].minVec[a];
			if (aabbs[i].maxVec[a] > worldMax[a]) worldMax[a] = aabbs[i].maxVec[a];
		}

	float threshold[3];
	for (int a = 0; a < 3; ++a)
		threshold[a] = (worldMax[a] - worldMin[a]) * LARGE_AABB_EXTENT_FRACTION;

	// A NaN extent fails every > test and therefore stays small - the tree already tolerates it.
	std::vector<std::pair<float, unsigned int> > candidates;
	for (size_t i = 0; i < aabbs.size(); ++i)
	{
		bool large = false;
		float biggest = 0.f;
		for (int a = 0; a < 3; ++a)
		{
			const float e = aabbs[i].maxVec[a] - aabbs[i].minVec[a];
			if (e > threshold[a]) large = true;
			if (e > biggest) biggest = e;
		}
		if (large)
			candidates.push_back(std::make_pair(-biggest, (unsigned int)i));
	}

	if (candidates.size() > MAX_LARGE_AABBS)
	{
		std::stable_sort(candidates.begin(), candidates.end());  // negated extent: biggest first
		candidates.resize(MAX_LARGE_AABBS);
	}

	std::vector<unsigned char> isLarge(aabbs.size(), 0);
	for (size_t c = 0; c < candidates.size(); ++c)
		isLarge[candidates[c].second] = 1;

	for (size_t i = 0; i < aabbs.size(); ++i)
	{
		if (isLarge[i])
			m_largeIndices.push_back((unsigned int)i);
		else
			m_smallIndices.push_back((unsigned int)i);
	}

	// A tree needs two leaves, and a split that leaves fewer than that buys nothing anyway.
	if (m_smallIndices.size() < 2)
	{
		m_largeIndices.clear();
		m_smallIndices.resize(aabbs.size());
		for (size_t i = 0; i < aabbs.size(); ++i)
			m_smallIndices[i] = (unsigned int)i;
	}
}

bool b3IrrlichtLbvh::prepareAabbSets(const std::vector<b3IrrAabb>& aabbs)
{
	classifyAabbs(aabbs);

	m_smallAabbs.resize(m_smallIndices.size());
	for (size_t i = 0; i < m_smallIndices.size(); ++i)
		m_smallAabbs[i] = aabbs[m_smallIndices[i]];

	b3IrrGpu::uploadBuffer<unsigned int>(m_smallIndexBuffer, &m_smallIndices[0],
										(irr::u32)m_smallIndices.size());
	// The identity map the resident path installs would otherwise look reusable here.
	m_identityIndexCount = 0;

	if (m_largeIndices.empty())
		return true;

	if (m_separateMaterial < 0)
		return false;

	const irr::u32 numLarge = (irr::u32)m_largeIndices.size();
	b3IrrGpu::uploadBuffer<unsigned int>(m_largeIndexBuffer, &m_largeIndices[0], numLarge);
	ensureBuffer<b3IrrAabb>(m_largeAabbBuffer, numLarge);

	// The gather reads the UNPARTITIONED array, so it has to run before buildTree() re-uploads
	// m_aabbBuffer with the small set.
	if (!uploadAabbs(aabbs))
		return false;

	ensureBuffer<LbvhParams>(m_separateParamBuffer, 1);

	LbvhParams sp;
	sp.numAabbs = numLarge;
	sp.pad0 = sp.pad1 = sp.pad2 = 0;
	memcpy(m_separateParamBuffer->getBufferPointer(), &sp, sizeof(sp));
	m_separateParamBuffer->setDirty();

	irr::video::SMaterial mat;
	mat.MaterialType = (irr::video::E_MATERIAL_TYPE)m_separateMaterial;
	m_driver->setMaterial(mat);
	m_driver->bindComputeBuffer(0, m_separateParamBuffer, irr::video::EHBT_SHADER_RESOURCE);
	m_driver->bindComputeBuffer(1, m_aabbBuffer, irr::video::EHBT_SHADER_RESOURCE);
	m_driver->bindComputeBuffer(3, m_largeIndexBuffer, irr::video::EHBT_SHADER_RESOURCE);
	m_driver->bindComputeBuffer(2, m_largeAabbBuffer, irr::video::EHBT_COMPUTE);
	m_driver->dispatchComputeShaderBound(irr::core::vector3d<irr::u32>((numLarge + 127) / 128, 1, 1));
	m_driver->unbindComputeResources();

	// Written as a UAV here, read as an SRV by the brute-force passes.
	m_driver->computeBarrier(m_largeAabbBuffer);
	return true;
}

bool b3IrrlichtLbvh::calculateOverlappingPairs(const std::vector<b3IrrAabb>& aabbs,
											   std::vector<std::pair<unsigned int, unsigned int> >& pairs,
											   unsigned int maxPairs, bool* overflowed,
											   bool sortForDeterminism)
{
	if (overflowed)
		*overflowed = false;
	pairs.clear();

	if (m_pairsMaterial < 0 || aabbs.size() < 2)
		return false;

	if (!prepareAabbSets(aabbs))
		return false;

	TreeResult tree;
	if (!buildTree(m_smallAabbs, tree))
		return false;

	const irr::u32 numLeaf = (irr::u32)m_smallAabbs.size();
	const irr::u32 numInternal = numLeaf - 1;
	const irr::u32 numLarge = (irr::u32)m_largeIndices.size();

	ensureBuffer<IrrInt2>(m_leafRangeBuffer, numInternal);

	// Append buffer: the pair count is decided on the GPU and read back via copyStructureCount,
	// which is the whole point of G3 - narrowphase can be sized from it without a readback.
	ensureBuffer<IrrUint2>(m_pairBuffer, maxPairs, irr::video::EHBF_COMPUTE_APPEND);
	// DRAW_INDIRECT_ARGS destination, matching the proven G3 gate path - and it is the
	// shape narrowphase will want anyway, to indirect-dispatch straight off this count.
	ensureBuffer<unsigned int>(m_pairCountBuffer, 4, irr::video::EHBF_DRAW_INDIRECT_ARGS);

	ensureBuffer<PairParams>(m_pairParamBuffer, 1);

	PairParams pp;
	memset(&pp, 0, sizeof(pp));
	pp.numLeafNodes = numLeaf;
	pp.numInternalNodes = numInternal;
	pp.rootIndex = tree.rootIndex;
	pp.numLargeAabbs = numLarge;
	memcpy(m_pairParamBuffer->getBufferPointer(), &pp, sizeof(pp));
	m_pairParamBuffer->setDirty();

	const irr::u32 internalGroups = (numInternal + 127) / 128;
	const irr::u32 leafGroups = (numLeaf + 127) / 128;
	const irr::u32 largeGroups = (numLarge + 127) / 128;
	irr::video::SMaterial mat;

	// 1) contiguous leaf range per internal node - what makes the duplicate-pair rejection work
	m_cachedLeafRangeCount = 0;  // this path shares the buffer but not the resident tree
	mat.MaterialType = (irr::video::E_MATERIAL_TYPE)m_leafRangeMaterial;
	m_driver->setMaterial(mat);
	m_driver->bindComputeBuffer(0, m_pairParamBuffer, irr::video::EHBT_SHADER_RESOURCE);
	m_driver->bindComputeBuffer(1, m_childNodeBuffer, irr::video::EHBT_SHADER_RESOURCE);
	m_driver->bindComputeBuffer(1, m_leafRangeBuffer, irr::video::EHBT_COMPUTE);
	m_driver->dispatchComputeShaderBound(irr::core::vector3d<irr::u32>(internalGroups, 1, 1));
	m_driver->unbindComputeResources();

	// 2) traverse per leaf, appending overlapping pairs (small vs small)
	mat.MaterialType = (irr::video::E_MATERIAL_TYPE)m_pairsMaterial;
	m_driver->setMaterial(mat);
	m_driver->bindComputeBuffer(0, m_pairParamBuffer, irr::video::EHBT_SHADER_RESOURCE);
	m_driver->bindComputeBuffer(1, m_childNodeBuffer, irr::video::EHBT_SHADER_RESOURCE);
	m_driver->bindComputeBuffer(2, m_sortedMortonBuffer, irr::video::EHBT_SHADER_RESOURCE);
	m_driver->bindComputeBuffer(3, m_aabbBuffer, irr::video::EHBT_SHADER_RESOURCE);
	m_driver->bindComputeBuffer(4, m_internalAabbBuffer, irr::video::EHBT_SHADER_RESOURCE);
	m_driver->bindComputeBuffer(5, m_leafRangeBuffer, irr::video::EHBT_SHADER_RESOURCE);
	m_driver->bindComputeBuffer(7, m_smallIndexBuffer, irr::video::EHBT_SHADER_RESOURCE);
	m_driver->bindComputeBuffer(0, m_pairBuffer, irr::video::EHBT_COMPUTE);
	// Only this first pass resets the counter; a rebind alone keeps it, so the brute-force
	// passes below append onto the same list.
	m_driver->resetStructureCount(m_pairBuffer, 0);
	m_driver->dispatchComputeShaderBound(irr::core::vector3d<irr::u32>(leafGroups, 1, 1));
	m_driver->unbindComputeResources();

	// 3) large vs large, i < j - the tree never sees these, so nothing emits them twice.
	if (numLarge >= 2)
	{
		mat.MaterialType = (irr::video::E_MATERIAL_TYPE)m_largeLargePairMaterial;
		m_driver->setMaterial(mat);
		m_driver->bindComputeBuffer(0, m_pairParamBuffer, irr::video::EHBT_SHADER_RESOURCE);
		m_driver->bindComputeBuffer(8, m_largeAabbBuffer, irr::video::EHBT_SHADER_RESOURCE);
		m_driver->bindComputeBuffer(9, m_largeIndexBuffer, irr::video::EHBT_SHADER_RESOURCE);
		m_driver->bindComputeBuffer(0, m_pairBuffer, irr::video::EHBT_COMPUTE);
		m_driver->dispatchComputeShaderBound(irr::core::vector3d<irr::u32>(largeGroups, 1, 1));
		m_driver->unbindComputeResources();
	}

	// 4) small vs large, brute force
	if (numLarge >= 1)
	{
		mat.MaterialType = (irr::video::E_MATERIAL_TYPE)m_largePairMaterial;
		m_driver->setMaterial(mat);
		m_driver->bindComputeBuffer(0, m_pairParamBuffer, irr::video::EHBT_SHADER_RESOURCE);
		m_driver->bindComputeBuffer(3, m_aabbBuffer, irr::video::EHBT_SHADER_RESOURCE);
		m_driver->bindComputeBuffer(7, m_smallIndexBuffer, irr::video::EHBT_SHADER_RESOURCE);
		m_driver->bindComputeBuffer(8, m_largeAabbBuffer, irr::video::EHBT_SHADER_RESOURCE);
		m_driver->bindComputeBuffer(9, m_largeIndexBuffer, irr::video::EHBT_SHADER_RESOURCE);
		m_driver->bindComputeBuffer(0, m_pairBuffer, irr::video::EHBT_COMPUTE);
		m_driver->dispatchComputeShaderBound(irr::core::vector3d<irr::u32>(leafGroups, 1, 1));
		m_driver->unbindComputeResources();
	}

	// Hidden append counter -> a readable buffer. copyStructureCount resolves the UAV from the
	// buffer itself, so the slot binding no longer has to be live.
	m_driver->copyStructureCount(m_pairCountBuffer, 0, m_pairBuffer);

	m_pairCountBuffer->downloadFromGPU();
	unsigned int count = 0;
	memcpy(&count, m_pairCountBuffer->getBufferPointer(), sizeof(unsigned int));
	if (count > maxPairs)
	{
		// Append past capacity is discarded by the hardware, but the counter still advances,
		// so a count above capacity is exactly how overflow becomes visible rather than silent.
		if (overflowed)
			*overflowed = true;
		count = maxPairs;
	}

	m_pairBuffer->downloadFromGPU();
	const IrrUint2* raw = (const IrrUint2*)m_pairBuffer->getBufferPointer();
	pairs.reserve(count);
	for (unsigned int i = 0; i < count; ++i)
		pairs.push_back(std::make_pair(raw[i].x, raw[i].y));

	// Append order varies run to run, so the pair SET is stable but its ORDER is not - anything
	// accumulating floats over these pairs needs a data-derived order to stay reproducible.
	if (sortForDeterminism && !pairs.empty())
		sortPairsForDeterminism(pairs);

	return true;
}

bool b3IrrlichtLbvh::isResidentPathAvailable() const
{
	return m_dispatch && m_dispatch->isAvailable() && m_pairsMaterial >= 0 && m_mortonMaterial >= 0;
}

irr::scene::IComputeBuffer* b3IrrlichtLbvh::ensureLeafIdentity(unsigned int count)
{
	// Separate from m_smallIndexBuffer on purpose: the vector path pre-gathers its AABBs on the
	// CPU, so its LeafToBody is identity even though its SmallToOriginal is a real map.
	if (m_leafIdentityCount != count || !m_leafIdentityBuffer ||
		m_leafIdentityBuffer->getStructureCount() < count)
	{
		ensureBuffer<unsigned int>(m_leafIdentityBuffer, count);
		unsigned int* dst = (unsigned int*)m_leafIdentityBuffer->getBufferPointer();
		for (unsigned int i = 0; i < count; ++i)
			dst[i] = i;
		m_leafIdentityBuffer->setDirty();
		m_leafIdentityCount = count;
	}
	return m_leafIdentityBuffer;
}

bool b3IrrlichtLbvh::ensureIdentityIndexMap(unsigned int numAabbs)
{
	if (m_identityIndexCount == numAabbs && m_smallIndexBuffer &&
		m_smallIndexBuffer->getStructureCount() >= numAabbs)
		return true;

	ensureBuffer<unsigned int>(m_smallIndexBuffer, numAabbs);
	unsigned int* dst = (unsigned int*)m_smallIndexBuffer->getBufferPointer();
	for (unsigned int i = 0; i < numAabbs; ++i)
		dst[i] = i;
	m_smallIndexBuffer->setDirty();

	m_identityIndexCount = numAabbs;
	return true;
}

bool b3IrrlichtLbvh::calculateOverlappingPairsResident(irr::scene::IComputeBuffer* aabbs,
													   unsigned int numAabbs, unsigned int maxPairs,
													   irr::scene::IComputeBuffer* subset,
													   unsigned int subsetCount, bool refit,
													   irr::scene::IComputeBuffer* sleepStates,
													   irr::scene::IComputeBuffer* refitSleepStates)
{
	// A subset drives the tree through the SAME SmallToOriginal indirection identity uses, so the
	// build/traverse kernels are untouched - only which bodies become leaves changes.
	const unsigned int leafCount = subset ? subsetCount : numAabbs;
	if (!isResidentPathAvailable() || !aabbs || leafCount < 2 || maxPairs == 0)
		return false;

	// Never assigned into m_smallIndexBuffer: that one is owned and dropped by the destructor, and
	// a caller's buffer aliased into it would be double-freed.
	irr::scene::IComputeBuffer* indexMap = subset;
	if (!indexMap)
	{
		if (!ensureIdentityIndexMap(numAabbs))
			return false;
		indexMap = m_smallIndexBuffer;
	}

	numAabbs = leafCount;

	ensureBuffer<b3IrrAabb>(m_mergedBuffer, 1);
	ensureBuffer<b3IrrSortData>(m_mortonBuffer, numAabbs);
	ensureBuffer<LbvhParams>(m_paramBuffer, 1);

	LbvhParams params;
	params.numAabbs = numAabbs;
	params.pad0 = params.pad1 = params.pad2 = 0;
	memcpy(m_paramBuffer->getBufferPointer(), &params, sizeof(params));
	m_paramBuffer->setDirty();

	// A refit is only valid while the leaf SET is unchanged - the cached sort maps leaf slot to
	// body, so a different body count means that mapping no longer describes this scene.
	const bool canRefit = refit && m_cachedSortedCodes && m_cachedLeafCount == numAabbs;

	irr::video::SMaterial mat;
	irr::scene::IComputeBuffer* sortedCodes = m_cachedSortedCodes;
	if (!canRefit)
	{
		mat.MaterialType = (irr::video::E_MATERIAL_TYPE)m_mergeMaterial;
		m_driver->setMaterial(mat);
		m_driver->bindComputeBuffer(0, m_paramBuffer, irr::video::EHBT_SHADER_RESOURCE);
		m_driver->bindComputeBuffer(1, aabbs, irr::video::EHBT_SHADER_RESOURCE);
		m_driver->bindComputeBuffer(4, indexMap, irr::video::EHBT_SHADER_RESOURCE);
		m_driver->bindComputeBuffer(0, m_mergedBuffer, irr::video::EHBT_COMPUTE);
		m_driver->dispatchComputeShaderBound(irr::core::vector3d<irr::u32>(1, 1, 1));
		m_driver->unbindComputeResources();

		// Written as a UAV above, read as an SRV below.
		m_driver->computeBarrier(m_mergedBuffer);

		mat.MaterialType = (irr::video::E_MATERIAL_TYPE)m_mortonMaterial;
		m_driver->setMaterial(mat);
		m_driver->bindComputeBuffer(0, m_paramBuffer, irr::video::EHBT_SHADER_RESOURCE);
		m_driver->bindComputeBuffer(1, aabbs, irr::video::EHBT_SHADER_RESOURCE);
		m_driver->bindComputeBuffer(2, m_mergedBuffer, irr::video::EHBT_SHADER_RESOURCE);
		m_driver->bindComputeBuffer(4, indexMap, irr::video::EHBT_SHADER_RESOURCE);
		m_driver->bindComputeBuffer(1, m_mortonBuffer, irr::video::EHBT_COMPUTE);
		m_driver->dispatchComputeShaderBound(irr::core::vector3d<irr::u32>((numAabbs + 127) / 128, 1, 1));
		m_driver->unbindComputeResources();
		m_driver->computeBarrier(m_mortonBuffer);

		if (!m_sort->executeResident(m_mortonBuffer, numAabbs, MORTON_BITS, &sortedCodes))
			return false;

		m_cachedSortedCodes = sortedCodes;
		m_cachedLeafCount = numAabbs;
	}


	int rootIndex = 0;
	if (!buildTreeCore(aabbs, sortedCodes, numAabbs, rootIndex, 0, canRefit, refitSleepStates, indexMap))
		return false;

	const irr::u32 numLeaf = numAabbs;
	const irr::u32 numInternal = numLeaf - 1;

	ensureBuffer<IrrInt2>(m_leafRangeBuffer, numInternal);
	ensureBuffer<IrrUint2>(m_pairBuffer, maxPairs, irr::video::EHBF_COMPUTE_APPEND);
	ensureBuffer<unsigned int>(m_pairCountBuffer, 4, irr::video::EHBF_DRAW_INDIRECT_ARGS);
	ensureBuffer<PairParams>(m_pairParamBuffer, 1);

	// No large side list: classifying by extent needs the AABBs on the CPU.
	PairParams pp;
	memset(&pp, 0, sizeof(pp));
	pp.numLeafNodes = numLeaf;
	pp.numInternalNodes = numInternal;
	pp.rootIndex = rootIndex;
	pp.numLargeAabbs = 0;
	pp.useSleepGate = sleepStates ? 1u : 0u;
	memcpy(m_pairParamBuffer->getBufferPointer(), &pp, sizeof(pp));
	m_pairParamBuffer->setDirty();

	// Two ways to skip a per-internal-node descent to the leftmost/rightmost leaf: the gated query
	// never reads the ranges at all, and otherwise they only change when the topology does.
	const bool rangesNeeded = !m_leafRangeCaching || sleepStates == 0;
	if (!rangesNeeded)
		m_cachedLeafRangeCount = 0;
	else if (!m_leafRangeCaching || !canRefit || m_cachedLeafRangeCount != numInternal)
	{
		mat.MaterialType = (irr::video::E_MATERIAL_TYPE)m_leafRangeMaterial;
		m_driver->setMaterial(mat);
		m_driver->bindComputeBuffer(0, m_pairParamBuffer, irr::video::EHBT_SHADER_RESOURCE);
		m_driver->bindComputeBuffer(1, m_childNodeBuffer, irr::video::EHBT_SHADER_RESOURCE);
		m_driver->bindComputeBuffer(1, m_leafRangeBuffer, irr::video::EHBT_COMPUTE);
		m_driver->dispatchComputeShaderBound(irr::core::vector3d<irr::u32>((numInternal + 127) / 128, 1, 1));
		m_driver->unbindComputeResources();
		m_cachedLeafRangeCount = numInternal;
	}

	mat.MaterialType = (irr::video::E_MATERIAL_TYPE)m_pairsMaterial;
	m_driver->setMaterial(mat);
	m_driver->bindComputeBuffer(0, m_pairParamBuffer, irr::video::EHBT_SHADER_RESOURCE);
	m_driver->bindComputeBuffer(1, m_childNodeBuffer, irr::video::EHBT_SHADER_RESOURCE);
	m_driver->bindComputeBuffer(2, sortedCodes, irr::video::EHBT_SHADER_RESOURCE);
	m_driver->bindComputeBuffer(3, aabbs, irr::video::EHBT_SHADER_RESOURCE);
	m_driver->bindComputeBuffer(4, m_internalAabbBuffer, irr::video::EHBT_SHADER_RESOURCE);
	m_driver->bindComputeBuffer(5, m_leafRangeBuffer, irr::video::EHBT_SHADER_RESOURCE);
	m_driver->bindComputeBuffer(7, indexMap, irr::video::EHBT_SHADER_RESOURCE);
	if (sleepStates)
		m_driver->bindComputeBuffer(10, sleepStates, irr::video::EHBT_SHADER_RESOURCE);
	m_driver->bindComputeBuffer(0, m_pairBuffer, irr::video::EHBT_COMPUTE);
	m_driver->resetStructureCount(m_pairBuffer, 0);
	m_driver->dispatchComputeShaderBound(irr::core::vector3d<irr::u32>((numLeaf + 127) / 128, 1, 1));
	m_driver->unbindComputeResources();

	m_driver->copyStructureCount(m_pairCountBuffer, 0, m_pairBuffer);
	m_driver->computeBarrier(m_pairBuffer);
	return true;
}

void b3IrrlichtLbvh::sortPairsForDeterminism(std::vector<std::pair<unsigned int, unsigned int> >& pairs)
{
	// Normalise each pair first: the traversal reports (query, other), so the same body pair can
	// arrive either way round depending on which leaf's query found it.
	for (size_t i = 0; i < pairs.size(); ++i)
		if (pairs[i].first > pairs[i].second)
			std::swap(pairs[i].first, pairs[i].second);

	// Two stable passes, least-significant key first - the same LSD discipline the GPU radix
	// sort uses, so this can move onto the GPU later without changing the resulting order.
	std::stable_sort(pairs.begin(), pairs.end(),
					 [](const std::pair<unsigned int, unsigned int>& a,
						const std::pair<unsigned int, unsigned int>& b) { return a.second < b.second; });
	std::stable_sort(pairs.begin(), pairs.end(),
					 [](const std::pair<unsigned int, unsigned int>& a,
						const std::pair<unsigned int, unsigned int>& b) { return a.first < b.first; });
}

bool b3IrrlichtLbvh::castRays(const std::vector<b3IrrAabb>& aabbs, const std::vector<b3IrrRay>& rays,
							  std::vector<std::pair<unsigned int, unsigned int> >& rayBodyPairs,
							  unsigned int maxPairs, bool* overflowed)
{
	if (overflowed)
		*overflowed = false;
	rayBodyPairs.clear();

	if (m_rayMaterial < 0 || aabbs.size() < 2 || rays.empty())
		return false;

	if (!prepareAabbSets(aabbs))
		return false;

	TreeResult tree;
	if (!buildTree(m_smallAabbs, tree))
		return false;

	const irr::u32 numLarge = (irr::u32)m_largeIndices.size();

	b3IrrGpu::uploadBuffer<b3IrrRay>(m_rayBuffer, &rays[0], (irr::u32)rays.size());
	ensureBuffer<IrrUint2>(m_rayPairBuffer, maxPairs, irr::video::EHBF_COMPUTE_APPEND);
	// Indirect-args destination: CopyStructureCount into a plain structured buffer silently
	// yields 0 (see the Phase 4c note) - this shape is also what indirect dispatch wants.
	ensureBuffer<unsigned int>(m_pairCountBuffer, 4, irr::video::EHBF_DRAW_INDIRECT_ARGS);

	ensureBuffer<PairParams>(m_pairParamBuffer, 1);

	// numLeafNodes doubles as the ray count for this kernel - the traversal never needs the
	// leaf count, since it walks from the root.
	PairParams pp;
	memset(&pp, 0, sizeof(pp));
	pp.numLeafNodes = (unsigned int)rays.size();
	pp.numInternalNodes = (unsigned int)m_smallAabbs.size() - 1;
	pp.rootIndex = tree.rootIndex;
	pp.numLargeAabbs = numLarge;
	memcpy(m_pairParamBuffer->getBufferPointer(), &pp, sizeof(pp));
	m_pairParamBuffer->setDirty();

	const irr::u32 rayGroups = ((irr::u32)rays.size() + 127) / 128;
	irr::video::SMaterial mat;
	mat.MaterialType = (irr::video::E_MATERIAL_TYPE)m_rayMaterial;
	m_driver->setMaterial(mat);
	m_driver->bindComputeBuffer(0, m_pairParamBuffer, irr::video::EHBT_SHADER_RESOURCE);
	m_driver->bindComputeBuffer(1, m_childNodeBuffer, irr::video::EHBT_SHADER_RESOURCE);
	m_driver->bindComputeBuffer(2, m_sortedMortonBuffer, irr::video::EHBT_SHADER_RESOURCE);
	m_driver->bindComputeBuffer(3, m_aabbBuffer, irr::video::EHBT_SHADER_RESOURCE);
	m_driver->bindComputeBuffer(4, m_internalAabbBuffer, irr::video::EHBT_SHADER_RESOURCE);
	m_driver->bindComputeBuffer(6, m_rayBuffer, irr::video::EHBT_SHADER_RESOURCE);
	m_driver->bindComputeBuffer(7, m_smallIndexBuffer, irr::video::EHBT_SHADER_RESOURCE);
	m_driver->bindComputeBuffer(2, m_rayPairBuffer, irr::video::EHBT_COMPUTE);
	m_driver->resetStructureCount(m_rayPairBuffer, 0);
	m_driver->dispatchComputeShaderBound(irr::core::vector3d<irr::u32>(rayGroups, 1, 1));
	m_driver->unbindComputeResources();

	if (numLarge >= 1)
	{
		mat.MaterialType = (irr::video::E_MATERIAL_TYPE)m_largeRayMaterial;
		m_driver->setMaterial(mat);
		m_driver->bindComputeBuffer(0, m_pairParamBuffer, irr::video::EHBT_SHADER_RESOURCE);
		m_driver->bindComputeBuffer(6, m_rayBuffer, irr::video::EHBT_SHADER_RESOURCE);
		m_driver->bindComputeBuffer(8, m_largeAabbBuffer, irr::video::EHBT_SHADER_RESOURCE);
		m_driver->bindComputeBuffer(9, m_largeIndexBuffer, irr::video::EHBT_SHADER_RESOURCE);
		m_driver->bindComputeBuffer(2, m_rayPairBuffer, irr::video::EHBT_COMPUTE);
		m_driver->dispatchComputeShaderBound(irr::core::vector3d<irr::u32>(rayGroups, 1, 1));
		m_driver->unbindComputeResources();
	}

	m_driver->copyStructureCount(m_pairCountBuffer, 0, m_rayPairBuffer);

	m_pairCountBuffer->downloadFromGPU();
	unsigned int count = 0;
	memcpy(&count, m_pairCountBuffer->getBufferPointer(), sizeof(unsigned int));
	if (count > maxPairs)
	{
		if (overflowed)
			*overflowed = true;
		count = maxPairs;
	}

	m_rayPairBuffer->downloadFromGPU();
	const IrrUint2* raw = (const IrrUint2*)m_rayPairBuffer->getBufferPointer();
	rayBodyPairs.reserve(count);
	for (unsigned int i = 0; i < count; ++i)
		rayBodyPairs.push_back(std::make_pair(raw[i].x, raw[i].y));

	return true;
}
