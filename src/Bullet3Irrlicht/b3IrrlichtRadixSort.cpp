#include "b3IrrlichtRadixSort.h"
#include "b3IrrlichtPrefixScan.h"
#include "b3IrrlichtGpuBuffers.h"

#include <irrlicht.h>
#include "ComputeBuffer.h"

#include <cstring>

namespace
{
// Mirrors RadixParams in B3RadixSort.hlsl.
struct RadixParams
{
	unsigned int numElems;
	unsigned int numBlocks;
	unsigned int bitShift;
	unsigned int useLiveCount;
};

static_assert(sizeof(RadixParams) == 16, "RadixParams must match the HLSL struct stride");
static_assert(sizeof(b3IrrSortData) == 8, "b3IrrSortData must match the HLSL uint2 stride");

using b3IrrGpu::ensureBuffer;
}  // namespace

b3IrrlichtRadixSort::b3IrrlichtRadixSort(irr::video::IVideoDriver* driver)
	: m_driver(driver), m_scan(0), m_paramBuffer(0), m_srcBuffer(0), m_dstBuffer(0),
	  m_histogram(0), m_scannedHistogram(0), m_histBlockSums(0),
	  m_streamCountMaterial(-1), m_sortScatterMaterial(-1)
{
}

b3IrrlichtRadixSort::~b3IrrlichtRadixSort()
{
	releaseBuffers();
	if (m_paramBuffer) m_paramBuffer->drop();
	delete m_scan;
}

void b3IrrlichtRadixSort::releaseBuffers()
{
	if (m_srcBuffer) { m_srcBuffer->drop(); m_srcBuffer = 0; }
	if (m_dstBuffer) { m_dstBuffer->drop(); m_dstBuffer = 0; }
	if (m_histogram) { m_histogram->drop(); m_histogram = 0; }
	if (m_scannedHistogram) { m_scannedHistogram->drop(); m_scannedHistogram = 0; }
	if (m_histBlockSums) { m_histBlockSums->drop(); m_histBlockSums = 0; }
}

bool b3IrrlichtRadixSort::init(irr::io::IFileSystem* fileSystem)
{
	if (!m_driver)
		return false;

	irr::video::IGPUProgrammingServices* gpu = m_driver->getGPUProgrammingServices();
	if (!gpu)
		return false;

	const irr::io::path path = "media/shaders/B3RadixSort.hlsl";
	if (fileSystem && !fileSystem->existFile(path))
		return false;

	m_streamCountMaterial = gpu->addComputeShaderFromFile(path, "CSStreamCount", irr::video::ECST_CS_5_0, 0);
	m_sortScatterMaterial = gpu->addComputeShaderFromFile(path, "CSSortAndScatter", irr::video::ECST_CS_5_0, 0);

	m_scan = new b3IrrlichtPrefixScan(m_driver);
	if (!m_scan->init(fileSystem))
		return false;

	return m_streamCountMaterial >= 0 && m_sortScatterMaterial >= 0;
}

bool b3IrrlichtRadixSort::ensureWorkBuffers(unsigned int numElems, unsigned int histSize)
{
	ensureBuffer<b3IrrSortData>(m_srcBuffer, numElems);
	ensureBuffer<b3IrrSortData>(m_dstBuffer, numElems);
	ensureBuffer<unsigned int>(m_histogram, histSize);
	ensureBuffer<unsigned int>(m_scannedHistogram, histSize);
	ensureBuffer<unsigned int>(m_histBlockSums, b3IrrlichtPrefixScan::ELEMS_PER_BLOCK);
	ensureBuffer<RadixParams>(m_paramBuffer, 1);
	return m_srcBuffer && m_dstBuffer && m_histogram && m_scannedHistogram && m_histBlockSums;
}

bool b3IrrlichtRadixSort::sortPasses(irr::scene::IComputeBuffer* input, unsigned int numElems,
									 unsigned int keyBits, irr::scene::IComputeBuffer** result,
									 irr::scene::IComputeBuffer* liveCount)
{
	const irr::u32 numBlocks = (numElems + ELEMS_PER_BLOCK - 1) / ELEMS_PER_BLOCK;

	irr::scene::IComputeBuffer* src = input;
	// The first pass always writes m_dstBuffer, so an externally owned input is never written.
	irr::scene::IComputeBuffer* dst = m_dstBuffer;

	const unsigned int numPasses = (keyBits + BITS_PER_PASS - 1) / BITS_PER_PASS;
	for (unsigned int pass = 0; pass < numPasses; ++pass)
	{
		RadixParams params;
		params.numElems = numElems;
		params.numBlocks = numBlocks;
		params.bitShift = pass * BITS_PER_PASS;
		params.useLiveCount = liveCount ? 1u : 0u;
		memcpy(m_paramBuffer->getBufferPointer(), &params, sizeof(params));
		m_paramBuffer->setDirty();

		irr::video::SMaterial mat;

		mat.MaterialType = (irr::video::E_MATERIAL_TYPE)m_streamCountMaterial;
		m_driver->setMaterial(mat);
		m_driver->bindComputeBuffer(0, m_paramBuffer, irr::video::EHBT_SHADER_RESOURCE);
		m_driver->bindComputeBuffer(1, src, irr::video::EHBT_SHADER_RESOURCE);
		m_driver->bindComputeBuffer(0, m_histogram, irr::video::EHBT_COMPUTE);
		if (liveCount)
			m_driver->bindComputeBuffer(2, liveCount, irr::video::EHBT_COMPUTE);
		m_driver->dispatchComputeShaderBound(irr::core::vector3d<irr::u32>(numBlocks, 1, 1));
		m_driver->unbindComputeResources();

		if (!m_scan->scanBuffers(m_histogram, m_scannedHistogram, m_histBlockSums,
								 NUM_BUCKETS * numBlocks))
			return false;

		mat.MaterialType = (irr::video::E_MATERIAL_TYPE)m_sortScatterMaterial;
		m_driver->setMaterial(mat);
		m_driver->bindComputeBuffer(0, m_paramBuffer, irr::video::EHBT_SHADER_RESOURCE);
		m_driver->bindComputeBuffer(1, src, irr::video::EHBT_SHADER_RESOURCE);
		m_driver->bindComputeBuffer(2, m_scannedHistogram, irr::video::EHBT_SHADER_RESOURCE);
		m_driver->bindComputeBuffer(1, dst, irr::video::EHBT_COMPUTE);
		if (liveCount)
			m_driver->bindComputeBuffer(2, liveCount, irr::video::EHBT_COMPUTE);
		m_driver->dispatchComputeShaderBound(irr::core::vector3d<irr::u32>(numBlocks, 1, 1));
		m_driver->unbindComputeResources();

		src = dst;
		dst = (src == m_dstBuffer) ? m_srcBuffer : m_dstBuffer;
	}

	// src holds the result after the final swap.
	*result = src;
	return true;
}

bool b3IrrlichtRadixSort::execute(std::vector<b3IrrSortData>& data, unsigned int keyBits)
{
	if (m_streamCountMaterial < 0 || data.empty() || keyBits == 0 || keyBits > 32)
		return false;

	const irr::u32 numElems = (irr::u32)data.size();
	const irr::u32 numBlocks = (numElems + ELEMS_PER_BLOCK - 1) / ELEMS_PER_BLOCK;
	const irr::u32 histSize = NUM_BUCKETS * numBlocks;

	// The histogram scan is the binding constraint, not the element count.
	if (histSize > b3IrrlichtPrefixScan::maxElements())
		return false;

	if (!ensureWorkBuffers(numElems, histSize))
		return false;

	memcpy(m_srcBuffer->getBufferPointer(), &data[0], numElems * sizeof(b3IrrSortData));
	m_srcBuffer->setDirty();

	irr::scene::IComputeBuffer* result = 0;
	if (!sortPasses(m_srcBuffer, numElems, keyBits, &result))
		return false;

	result->downloadFromGPU();
	memcpy(&data[0], result->getBufferPointer(), numElems * sizeof(b3IrrSortData));
	return true;
}

bool b3IrrlichtRadixSort::executeResident(irr::scene::IComputeBuffer* input, unsigned int numElems,
										  unsigned int keyBits, irr::scene::IComputeBuffer** result)
{
	if (m_streamCountMaterial < 0 || !input || !result || numElems == 0 || keyBits == 0 || keyBits > 32)
		return false;

	const irr::u32 numBlocks = (numElems + ELEMS_PER_BLOCK - 1) / ELEMS_PER_BLOCK;
	const irr::u32 histSize = NUM_BUCKETS * numBlocks;
	if (histSize > b3IrrlichtPrefixScan::maxElements())
		return false;

	if (!ensureWorkBuffers(numElems, histSize))
		return false;

	return sortPasses(input, numElems, keyBits, result);
}

unsigned int b3IrrlichtRadixSort::maxElements()
{
	return (b3IrrlichtPrefixScan::maxElements() / NUM_BUCKETS) * ELEMS_PER_BLOCK;
}

bool b3IrrlichtRadixSort::executeResidentCounted(irr::scene::IComputeBuffer* input,
												  unsigned int capacity,
												  irr::scene::IComputeBuffer* countBuffer,
												  unsigned int keyBits,
												  irr::scene::IComputeBuffer** result)
{
	if (m_streamCountMaterial < 0 || !input || !countBuffer || !result || capacity == 0 ||
		keyBits == 0 || keyBits > 32)
		return false;

	const irr::u32 numBlocks = (capacity + ELEMS_PER_BLOCK - 1) / ELEMS_PER_BLOCK;
	const irr::u32 histSize = NUM_BUCKETS * numBlocks;
	// Sized from capacity, never from the live count - refuse rather than sort a prefix.
	if (histSize > b3IrrlichtPrefixScan::maxElements())
		return false;

	if (!ensureWorkBuffers(capacity, histSize))
		return false;

	return sortPasses(input, capacity, keyBits, result, countBuffer);
}
