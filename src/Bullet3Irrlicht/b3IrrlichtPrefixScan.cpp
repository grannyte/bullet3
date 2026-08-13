#include "b3IrrlichtPrefixScan.h"
#include "b3IrrlichtGpuBuffers.h"

#include <irrlicht.h>
#include "ComputeBuffer.h"

#include <cstring>

namespace
{
// Mirrors ScanParams in B3PrefixScan.hlsl.
struct ScanParams
{
	unsigned int numElems;
	unsigned int numBlocks;
	unsigned int pad0;
	unsigned int pad1;
};

static_assert(sizeof(ScanParams) == 16, "ScanParams must match the HLSL struct stride");

using b3IrrGpu::ensureBuffer;
}  // namespace

b3IrrlichtPrefixScan::b3IrrlichtPrefixScan(irr::video::IVideoDriver* driver)
	: m_driver(driver), m_paramBuffer(0), m_srcBuffer(0), m_dstBuffer(0), m_blockSumBuffer(0),
	  m_localScanMaterial(-1), m_topLevelScanMaterial(-1), m_addOffsetMaterial(-1)
{
}

b3IrrlichtPrefixScan::~b3IrrlichtPrefixScan()
{
	if (m_paramBuffer) m_paramBuffer->drop();
	if (m_srcBuffer) m_srcBuffer->drop();
	if (m_dstBuffer) m_dstBuffer->drop();
	if (m_blockSumBuffer) m_blockSumBuffer->drop();
}

bool b3IrrlichtPrefixScan::init(irr::io::IFileSystem* fileSystem)
{
	if (!m_driver)
		return false;

	irr::video::IGPUProgrammingServices* gpu = m_driver->getGPUProgrammingServices();
	if (!gpu)
		return false;

	// addComputeShaderFromFile hands back a usable material id even when the file is missing,
	// which silently substitutes an unrelated material - check first.
	const irr::io::path path = "media/shaders/B3PrefixScan.hlsl";
	if (fileSystem && !fileSystem->existFile(path))
		return false;

	m_localScanMaterial = gpu->addComputeShaderFromFile(path, "CSLocalScan", irr::video::ECST_CS_5_0, 0);
	m_topLevelScanMaterial = gpu->addComputeShaderFromFile(path, "CSTopLevelScan", irr::video::ECST_CS_5_0, 0);
	m_addOffsetMaterial = gpu->addComputeShaderFromFile(path, "CSAddOffset", irr::video::ECST_CS_5_0, 0);

	return m_localScanMaterial >= 0 && m_topLevelScanMaterial >= 0 && m_addOffsetMaterial >= 0;
}

bool b3IrrlichtPrefixScan::scanBuffers(irr::scene::IComputeBuffer* src, irr::scene::IComputeBuffer* dst,
									   irr::scene::IComputeBuffer* blockSums, unsigned int numElems)
{
	if (m_localScanMaterial < 0 || !src || !dst || !blockSums || numElems == 0)
		return false;

	const irr::u32 numBlocks = (numElems + ELEMS_PER_BLOCK - 1) / ELEMS_PER_BLOCK;
	if (numBlocks > ELEMS_PER_BLOCK)
		return false;

	ensureBuffer<ScanParams>(m_paramBuffer, 1);

	ScanParams params;
	params.numElems = numElems;
	params.numBlocks = numBlocks;
	params.pad0 = 0;
	params.pad1 = 0;
	memcpy(m_paramBuffer->getBufferPointer(), &params, sizeof(params));
	m_paramBuffer->setDirty();

	irr::video::SMaterial mat;

	mat.MaterialType = (irr::video::E_MATERIAL_TYPE)m_localScanMaterial;
	m_driver->setMaterial(mat);
	m_driver->bindComputeBuffer(0, m_paramBuffer, irr::video::EHBT_SHADER_RESOURCE);
	m_driver->bindComputeBuffer(1, src, irr::video::EHBT_SHADER_RESOURCE);
	m_driver->bindComputeBuffer(0, dst, irr::video::EHBT_COMPUTE);
	m_driver->bindComputeBuffer(1, blockSums, irr::video::EHBT_COMPUTE);
	m_driver->dispatchComputeShaderBound(irr::core::vector3d<irr::u32>(numBlocks, 1, 1));
	m_driver->unbindComputeResources();

	mat.MaterialType = (irr::video::E_MATERIAL_TYPE)m_topLevelScanMaterial;
	m_driver->setMaterial(mat);
	m_driver->bindComputeBuffer(0, m_paramBuffer, irr::video::EHBT_SHADER_RESOURCE);
	m_driver->bindComputeBuffer(1, blockSums, irr::video::EHBT_COMPUTE);
	m_driver->dispatchComputeShaderBound(irr::core::vector3d<irr::u32>(1, 1, 1));
	m_driver->unbindComputeResources();

	mat.MaterialType = (irr::video::E_MATERIAL_TYPE)m_addOffsetMaterial;
	m_driver->setMaterial(mat);
	m_driver->bindComputeBuffer(0, m_paramBuffer, irr::video::EHBT_SHADER_RESOURCE);
	m_driver->bindComputeBuffer(0, dst, irr::video::EHBT_COMPUTE);
	m_driver->bindComputeBuffer(1, blockSums, irr::video::EHBT_COMPUTE);
	m_driver->dispatchComputeShaderBound(irr::core::vector3d<irr::u32>(numBlocks, 1, 1));
	m_driver->unbindComputeResources();

	return true;
}

bool b3IrrlichtPrefixScan::execute(const std::vector<unsigned int>& src, std::vector<unsigned int>& dst)
{
	if (m_localScanMaterial < 0 || src.empty())
		return false;

	const irr::u32 numElems = (irr::u32)src.size();
	const irr::u32 numBlocks = (numElems + ELEMS_PER_BLOCK - 1) / ELEMS_PER_BLOCK;

	// The top-level scan is a single group, so it can only scan ELEMS_PER_BLOCK block sums.
	// Beyond that this needs to recurse; not required by current call sites.
	if (numBlocks > ELEMS_PER_BLOCK)
		return false;

	ensureBuffer<ScanParams>(m_paramBuffer, 1);
	ensureBuffer<unsigned int>(m_srcBuffer, numElems);
	ensureBuffer<unsigned int>(m_dstBuffer, numElems);
	// Padded to a full block so the single-group top-level scan reads defined memory.
	ensureBuffer<unsigned int>(m_blockSumBuffer, ELEMS_PER_BLOCK);

	memcpy(m_srcBuffer->getBufferPointer(), &src[0], numElems * sizeof(unsigned int));
	m_srcBuffer->setDirty();

	ScanParams params;
	params.numElems = numElems;
	params.numBlocks = numBlocks;
	params.pad0 = 0;
	params.pad1 = 0;
	memcpy(m_paramBuffer->getBufferPointer(), &params, sizeof(params));
	m_paramBuffer->setDirty();

	irr::video::SMaterial mat;

	// Pass 1: per-block exclusive scan, collecting block totals.
	mat.MaterialType = (irr::video::E_MATERIAL_TYPE)m_localScanMaterial;
	m_driver->setMaterial(mat);
	m_driver->bindComputeBuffer(0, m_paramBuffer, irr::video::EHBT_SHADER_RESOURCE);
	m_driver->bindComputeBuffer(1, m_srcBuffer, irr::video::EHBT_SHADER_RESOURCE);
	m_driver->bindComputeBuffer(0, m_dstBuffer, irr::video::EHBT_COMPUTE);
	m_driver->bindComputeBuffer(1, m_blockSumBuffer, irr::video::EHBT_COMPUTE);
	m_driver->dispatchComputeShaderBound(irr::core::vector3d<irr::u32>(numBlocks, 1, 1));
	m_driver->unbindComputeResources();

	// Pass 2: scan the block totals in place.
	mat.MaterialType = (irr::video::E_MATERIAL_TYPE)m_topLevelScanMaterial;
	m_driver->setMaterial(mat);
	m_driver->bindComputeBuffer(0, m_paramBuffer, irr::video::EHBT_SHADER_RESOURCE);
	m_driver->bindComputeBuffer(1, m_blockSumBuffer, irr::video::EHBT_COMPUTE);
	m_driver->dispatchComputeShaderBound(irr::core::vector3d<irr::u32>(1, 1, 1));
	m_driver->unbindComputeResources();

	// Pass 3: fold each block's offset back in.
	mat.MaterialType = (irr::video::E_MATERIAL_TYPE)m_addOffsetMaterial;
	m_driver->setMaterial(mat);
	m_driver->bindComputeBuffer(0, m_paramBuffer, irr::video::EHBT_SHADER_RESOURCE);
	m_driver->bindComputeBuffer(0, m_dstBuffer, irr::video::EHBT_COMPUTE);
	m_driver->bindComputeBuffer(1, m_blockSumBuffer, irr::video::EHBT_COMPUTE);
	m_driver->dispatchComputeShaderBound(irr::core::vector3d<irr::u32>(numBlocks, 1, 1));
	m_driver->unbindComputeResources();

	m_dstBuffer->downloadFromGPU();
	dst.resize(numElems);
	memcpy(&dst[0], m_dstBuffer->getBufferPointer(), numElems * sizeof(unsigned int));

	return true;
}
