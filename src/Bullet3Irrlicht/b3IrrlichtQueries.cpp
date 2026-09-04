#include "b3IrrlichtQueries.h"
#include "b3IrrlichtLbvh.h"
#include "b3IrrlichtNarrowphase.h"
#include "b3IrrlichtGpuBuffers.h"

#include <irrlicht.h>
#include "ComputeBuffer.h"

#include <cstring>

namespace
{
struct QueryParams
{
	unsigned int numQueries;
	int rootIndex;
	unsigned int useOwnerFilter;
	unsigned int numBodies;
};

static_assert(sizeof(QueryParams) == 16, "QueryParams must match the HLSL struct stride");
static_assert(sizeof(b3IrrlichtQueries::b3IrrQuery) == 48, "b3IrrQuery must match the HLSL b3Query stride");
static_assert(sizeof(b3IrrlichtQueries::b3IrrQueryHit) == 48, "b3IrrQueryHit must match the HLSL b3QueryHit stride");

const unsigned int WG_SIZE = 64;
}  // namespace

b3IrrlichtQueries::b3IrrlichtQueries(irr::video::IVideoDriver* driver)
	: m_driver(driver), m_doubleSingle(false), m_asyncSupported(false), m_material(-1),
	  m_ownerRootsDirty(true), m_ownerRootsUploaded(0),
	  m_paramBuffer(0), m_queryBuffer(0), m_hitBuffer(0), m_ownerRootBuffer(0),
	  m_submitCount(0)
{
	for (int i = 0; i < READBACK_SLOTS; ++i)
	{
		m_pending[i].count = 0;
		m_pending[i].valid = false;
		m_pending[i].async = false;
	}
}

b3IrrlichtQueries::~b3IrrlichtQueries()
{
	b3IrrGpu::dropBuffer(m_paramBuffer);
	b3IrrGpu::dropBuffer(m_queryBuffer);
	b3IrrGpu::dropBuffer(m_hitBuffer);
	b3IrrGpu::dropBuffer(m_ownerRootBuffer);
}

bool b3IrrlichtQueries::init(irr::io::IFileSystem* fileSystem, bool doubleSingle)
{
	m_doubleSingle = doubleSingle;
	if (!m_driver)
		return false;

	irr::video::IGPUProgrammingServices* gpu = m_driver->getGPUProgrammingServices();
	if (!gpu)
		return false;

	const irr::io::path path = m_doubleSingle ? "media/shaders/B3QueriesDS.hlsl" : "media/shaders/B3Queries.hlsl";
	if (fileSystem && !fileSystem->existFile(path))
		return false;

	m_material = gpu->addComputeShaderFromFile(path, "CSQueryTraverse", irr::video::ECST_CS_5_0, 0);
	return m_material >= 0;
}

void b3IrrlichtQueries::setBodyOwnerRoots(const std::vector<int>& roots)
{
	m_ownerRoots = roots;
	m_ownerRootsDirty = true;
}

void b3IrrlichtQueries::beginBatch()
{
	m_batch.clear();
}

unsigned int b3IrrlichtQueries::addRay(const float from[3], const float to[3], int ownerRoot)
{
	b3IrrQuery q;
	memset(&q, 0, sizeof(q));
	for (int a = 0; a < 3; ++a)
	{
		q.from[a] = from[a];
		q.to[a] = to[a];
	}
	q.radius = 0.f;
	q.ownerRoot = ownerRoot;
	q.kind = QUERY_RAY;
	m_batch.push_back(q);
	return (unsigned int)m_batch.size() - 1;
}

unsigned int b3IrrlichtQueries::addSphereSweep(const float from[3], const float to[3], float radius,
											   int ownerRoot)
{
	const unsigned int index = addRay(from, to, ownerRoot);
	m_batch[index].radius = radius;
	m_batch[index].kind = QUERY_SPHERE;
	return index;
}

unsigned int b3IrrlichtQueries::getSubmittedCount() const
{
	if (m_submitCount == 0)
		return 0;
	return m_pending[(m_submitCount - 1) % READBACK_SLOTS].count;
}

bool b3IrrlichtQueries::submit(const b3IrrlichtLbvh& lbvh, const b3IrrlichtNarrowphase& narrowphase,
							   irr::scene::IComputeBuffer* bodies, unsigned int numBodies)
{
	if (m_material < 0 || !bodies || numBodies == 0)
		return false;

	irr::scene::IComputeBuffer* childNodes = lbvh.getResidentChildNodeBuffer();
	irr::scene::IComputeBuffer* sortedCodes = lbvh.getResidentSortedCodeBuffer();
	irr::scene::IComputeBuffer* internalAabbs = lbvh.getResidentInternalAabbBuffer();
	irr::scene::IComputeBuffer* leafToBody = lbvh.getResidentLeafToBodyBuffer();
	irr::scene::IComputeBuffer* worldAabbs = narrowphase.getWorldAabbBuffer();
	irr::scene::IComputeBuffer* collidables = narrowphase.getCollidableBuffer();
	if (!childNodes || !sortedCodes || !internalAabbs || !leafToBody || !worldAabbs || !collidables
		|| lbvh.getResidentLeafCount() < 2)
		return false;

	const unsigned int slot = m_submitCount % READBACK_SLOTS;
	const unsigned int count = (unsigned int)m_batch.size();

	m_pending[slot].count = count;
	m_pending[slot].valid = true;
	m_pending[slot].async = false;
	m_fallback[slot].clear();
	++m_submitCount;

	if (count == 0)
		return true;

	// Identity roots unless the caller supplied a map: a body then skips only itself.
	const bool customRoots = m_ownerRoots.size() == numBodies;
	if (m_ownerRootsDirty || m_ownerRootsUploaded != numBodies)
	{
		std::vector<int> roots(numBodies);
		for (unsigned int i = 0; i < numBodies; ++i)
			roots[i] = customRoots ? m_ownerRoots[i] : (int)i;
		b3IrrGpu::uploadBuffer<int>(m_ownerRootBuffer, &roots[0], numBodies);
		m_ownerRootsDirty = false;
		m_ownerRootsUploaded = numBodies;
	}

	b3IrrGpu::uploadBuffer<b3IrrQuery>(m_queryBuffer, &m_batch[0], count);

	// A growth drops the hardware buffer and any staging copy still queued on it: drain the other
	// slots' pending async readbacks into the CPU fallback first, or fetch() loses that batch.
	if (m_hitBuffer && m_hitBuffer->getStructureCount() < count)
	{
		for (unsigned int s = 0; s < (unsigned int)READBACK_SLOTS; ++s)
		{
			Pending& prev = m_pending[s];
			if (s != slot && prev.valid && prev.async && prev.count > 0)
			{
				m_fallback[s].resize(prev.count);
				const unsigned int bytes = prev.count * (unsigned int)sizeof(b3IrrQueryHit);
				if (m_driver->tryReadComputeBuffer(m_hitBuffer, s, &m_fallback[s][0], bytes, true))
					prev.async = false;
				else
				{
					m_fallback[s].clear();
					prev.valid = false;
				}
			}
		}
	}
	b3IrrGpu::ensureBuffer<b3IrrQueryHit>(m_hitBuffer, count);

	QueryParams params;
	params.numQueries = count;
	params.rootIndex = lbvh.getResidentRootIndex();
	params.useOwnerFilter = 1u;
	params.numBodies = numBodies;
	b3IrrGpu::uploadBuffer<QueryParams>(m_paramBuffer, &params, 1);

	irr::video::SMaterial mat;
	mat.MaterialType = (irr::video::E_MATERIAL_TYPE)m_material;
	m_driver->setMaterial(mat);
	m_driver->bindComputeBuffer(0, m_paramBuffer, irr::video::EHBT_SHADER_RESOURCE);
	m_driver->bindComputeBuffer(1, childNodes, irr::video::EHBT_SHADER_RESOURCE);
	m_driver->bindComputeBuffer(2, sortedCodes, irr::video::EHBT_SHADER_RESOURCE);
	m_driver->bindComputeBuffer(3, worldAabbs, irr::video::EHBT_SHADER_RESOURCE);
	m_driver->bindComputeBuffer(4, internalAabbs, irr::video::EHBT_SHADER_RESOURCE);
	m_driver->bindComputeBuffer(5, m_queryBuffer, irr::video::EHBT_SHADER_RESOURCE);
	m_driver->bindComputeBuffer(6, leafToBody, irr::video::EHBT_SHADER_RESOURCE);
	m_driver->bindComputeBuffer(7, bodies, irr::video::EHBT_SHADER_RESOURCE);
	m_driver->bindComputeBuffer(8, collidables, irr::video::EHBT_SHADER_RESOURCE);
	// Null while no hull is registered: a cleared slot reads zeros, and the kernel never reaches
	// hull data for a sphere-only registry.
	m_driver->bindComputeBuffer(9, narrowphase.getHullBuffer(), irr::video::EHBT_SHADER_RESOURCE);
	m_driver->bindComputeBuffer(10, narrowphase.getHullFaceBuffer(), irr::video::EHBT_SHADER_RESOURCE);
	m_driver->bindComputeBuffer(11, m_ownerRootBuffer, irr::video::EHBT_SHADER_RESOURCE);
	m_driver->bindComputeBuffer(0, m_hitBuffer, irr::video::EHBT_COMPUTE);
	m_driver->dispatchComputeShaderBound(irr::core::vector3d<irr::u32>((count + WG_SIZE - 1) / WG_SIZE, 1, 1));
	m_driver->unbindComputeResources();
	m_driver->computeBarrier(m_hitBuffer);

	// Queue this batch's staging copy now; fetch() reads it a tick later, fetchBlocking() waits.
	m_asyncSupported = m_driver->beginComputeReadback(m_hitBuffer, slot);
	m_pending[slot].async = m_asyncSupported;
	if (!m_asyncSupported)
	{
		m_hitBuffer->downloadFromGPU();
		m_fallback[slot].assign((const b3IrrQueryHit*)m_hitBuffer->getBufferPointer(),
								(const b3IrrQueryHit*)m_hitBuffer->getBufferPointer() + count);
	}
	return true;
}

bool b3IrrlichtQueries::readSlot(unsigned int slot, std::vector<b3IrrQueryHit>& out)
{
	out.clear();
	const Pending& p = m_pending[slot];
	if (!p.valid)
		return false;
	if (p.count == 0)
		return true;

	if (!p.async)
	{
		out = m_fallback[slot];
		return out.size() == p.count;
	}

	out.resize(p.count);
	const unsigned int bytes = p.count * (unsigned int)sizeof(b3IrrQueryHit);
	if (m_driver->tryReadComputeBuffer(m_hitBuffer, slot, &out[0], bytes, false))
		return true;
	if (m_driver->tryReadComputeBuffer(m_hitBuffer, slot, &out[0], bytes, true))
		return true;

	out.clear();
	return false;
}

bool b3IrrlichtQueries::fetch(std::vector<b3IrrQueryHit>& out)
{
	out.clear();
	if (m_submitCount < 2)
		return false;
	return readSlot((m_submitCount - 2) % READBACK_SLOTS, out);
}

bool b3IrrlichtQueries::fetchBlocking(std::vector<b3IrrQueryHit>& out)
{
	out.clear();
	if (m_submitCount < 1)
		return false;
	return readSlot((m_submitCount - 1) % READBACK_SLOTS, out);
}
