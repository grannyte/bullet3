#include "b3IrrlichtSolver.h"
#include "b3IrrlichtGpuBuffers.h"

#include <irrlicht.h>
#include "ComputeBuffer.h"

#include <cstdlib>
#include <cstring>

namespace
{
struct SolverParams
{
	unsigned int numContacts;
	unsigned int numBodies;
	float deltaTime;
	float erp;
};

struct IrrFloat4
{
	float x, y, z, w;
};

static_assert(sizeof(SolverParams) == 16, "SolverParams must match the HLSL struct stride");

using b3IrrGpu::dropBuffer;
using b3IrrGpu::ensureBuffer;
}  // namespace

b3IrrlichtSolver::b3IrrlichtSolver(irr::video::IVideoDriver* driver)
	: m_driver(driver), m_doubleSingle(false), m_solveMaterial(-1), m_applyMaterial(-1), m_clearImpulseMaterial(-1),
	  m_clearLoadMaterial(-1), m_accumLoadMaterial(-1), m_buildActiveMaterial(-1),
	  m_applyActiveMaterial(-1), m_clearLoadActiveMaterial(-1), m_dispatch(0), m_activeDispatch(0),
	  m_paramBuffer(0), m_bodyBuffer(0), m_contactBuffer(0), m_inertiaBuffer(0), m_deltaBuffer(0),
	  m_impulseBuffer(0), m_rollingAccumBuffer(0), m_loadBuffer(0), m_activeParamBuffer(0), m_activeBodyBuffer(0),
	  m_activeCountBuffer(0), m_claimBuffer(0), m_impulseFloats(0), m_activeGating(true),
	  m_activeStamp(0)
{
}

b3IrrlichtSolver::~b3IrrlichtSolver()
{
	releaseBuffers();
	dropBuffer(m_paramBuffer);
	dropBuffer(m_activeParamBuffer);
	delete m_dispatch;
	delete m_activeDispatch;
}

void b3IrrlichtSolver::releaseBuffers()
{
	dropBuffer(m_bodyBuffer);
	dropBuffer(m_contactBuffer);
	dropBuffer(m_inertiaBuffer);
	dropBuffer(m_deltaBuffer);
	dropBuffer(m_impulseBuffer);
	dropBuffer(m_rollingAccumBuffer);
	dropBuffer(m_loadBuffer);
	dropBuffer(m_activeBodyBuffer);
	dropBuffer(m_activeCountBuffer);
	dropBuffer(m_claimBuffer);
	m_impulseFloats = 0;
	m_activeStamp = 0;
}

bool b3IrrlichtSolver::init(irr::io::IFileSystem* fileSystem, bool doubleSingle)
{
	m_doubleSingle = doubleSingle;

	if (!m_driver)
		return false;

	irr::video::IGPUProgrammingServices* gpu = m_driver->getGPUProgrammingServices();
	if (!gpu)
		return false;

	const irr::io::path path = m_doubleSingle ? "media/shaders/B3SolveContactsDS.hlsl" : "media/shaders/B3SolveContacts.hlsl";
	if (fileSystem && !fileSystem->existFile(path))
		return false;

	m_solveMaterial = gpu->addComputeShaderFromFile(path, "CSSolveContacts", irr::video::ECST_CS_5_0, 0);
	m_applyMaterial = gpu->addComputeShaderFromFile(path, "CSApplyVelocityDeltas", irr::video::ECST_CS_5_0, 0);

	// Resident-only kernels: a caller staying on the std::vector path must not lose the solver
	// because these could not be built.
	m_clearImpulseMaterial = gpu->addComputeShaderFromFile(path, "CSClearAccumImpulse", irr::video::ECST_CS_5_0, 0);
	m_clearLoadMaterial = gpu->addComputeShaderFromFile(path, "CSClearBodyPointCounts", irr::video::ECST_CS_5_0, 0);
	m_accumLoadMaterial = gpu->addComputeShaderFromFile(path, "CSAccumBodyPointCounts", irr::video::ECST_CS_5_0, 0);
	m_buildActiveMaterial = gpu->addComputeShaderFromFile(path, "CSBuildActiveBodies", irr::video::ECST_CS_5_0, 0);
	m_applyActiveMaterial = gpu->addComputeShaderFromFile(path, "CSApplyVelocityDeltasActive", irr::video::ECST_CS_5_0, 0);
	m_clearLoadActiveMaterial = gpu->addComputeShaderFromFile(path, "CSClearBodyPointCountsActive", irr::video::ECST_CS_5_0, 0);

	if (!m_dispatch)
		m_dispatch = new b3IrrGpu::DispatchHelper(m_driver);
	m_dispatch->init(fileSystem);
	if (!m_activeDispatch)
		m_activeDispatch = new b3IrrGpu::DispatchHelper(m_driver);
	m_activeDispatch->init(fileSystem);

	return m_solveMaterial >= 0 && m_applyMaterial >= 0;
}

bool b3IrrlichtSolver::isActiveBodyGatingAvailable() const
{
	return m_buildActiveMaterial >= 0 && m_applyActiveMaterial >= 0 &&
		   m_clearLoadActiveMaterial >= 0 && m_activeDispatch && m_activeDispatch->isAvailable();
}

bool b3IrrlichtSolver::isResidentPathAvailable() const
{
	return m_dispatch && m_dispatch->isAvailable() && m_clearImpulseMaterial >= 0 &&
		   m_clearLoadMaterial >= 0 && m_accumLoadMaterial >= 0 && m_solveMaterial >= 0 &&
		   m_applyMaterial >= 0;
}

bool b3IrrlichtSolver::solveContacts(std::vector<b3RigidBodyData>& bodies,
									 const std::vector<float>& invInertiaDiag,
									 const std::vector<b3Contact4Data>& contacts,
									 int iterations, float deltaTime, float erp,
									 irr::scene::IComputeBuffer* rollingFriction)
{
	// The std::vector path moves 80-byte bodies; the df64 kernels index a 96-byte stride.
	if (m_doubleSingle)
		return false;

	if (m_solveMaterial < 0 || m_applyMaterial < 0 || bodies.empty())
		return false;

	if (contacts.empty())
		return true;

	const irr::u32 numBodies = (irr::u32)bodies.size();
	const irr::u32 numContacts = (irr::u32)contacts.size();

	b3IrrGpu::uploadBuffer<b3RigidBodyData>(m_bodyBuffer, &bodies[0], numBodies);
	b3IrrGpu::uploadBuffer<b3Contact4Data>(m_contactBuffer, &contacts[0], numContacts);

	ensureBuffer<IrrFloat4>(m_inertiaBuffer, numBodies);
	{
		IrrFloat4* dst = (IrrFloat4*)m_inertiaBuffer->getBufferPointer();
		for (irr::u32 i = 0; i < numBodies; ++i)
		{
			const bool haveData = (invInertiaDiag.size() >= (size_t)(i + 1) * 4);
			dst[i].x = haveData ? invInertiaDiag[i * 4 + 0] : 0.f;
			dst[i].y = haveData ? invInertiaDiag[i * 4 + 1] : 0.f;
			dst[i].z = haveData ? invInertiaDiag[i * 4 + 2] : 0.f;
			dst[i].w = 0.f;
		}
		m_inertiaBuffer->setDirty();
	}

	// 6 ints per body; zeroed here and re-zeroed by the apply kernel each iteration.
	ensureBuffer<int>(m_deltaBuffer, numBodies * 6);
	memset(m_deltaBuffer->getBufferPointer(), 0, numBodies * 6 * sizeof(int));
	m_deltaBuffer->setDirty();

	// Per-body contact-point load, driving the shader's Jacobi relaxation of the friction rows.
	// Static bodies stay at 0: they absorb no velocity, so they cannot be over-corrected.
	ensureBuffer<int>(m_loadBuffer, numBodies);
	{
		int* load = (int*)m_loadBuffer->getBufferPointer();
		memset(load, 0, numBodies * sizeof(int));
		for (irr::u32 i = 0; i < numContacts; ++i)
		{
			int points = (int)contacts[i].m_worldNormalOnB.w;
			points = points < 0 ? 0 : (points > 4 ? 4 : points);

			const irr::u32 bodyA = (irr::u32)abs(contacts[i].m_bodyAPtrAndSignBit);
			const irr::u32 bodyB = (irr::u32)abs(contacts[i].m_bodyBPtrAndSignBit);
			if (bodyA < numBodies && bodies[bodyA].m_invMass > 0.f)
				load[bodyA] += points;
			if (bodyB < numBodies && bodies[bodyB].m_invMass > 0.f)
				load[bodyB] += points;
		}
		m_loadBuffer->setDirty();
	}

	// Warm-start ledger: normal + 2 tangent accumulated impulses for each of 4 points.
	m_impulseFloats = numContacts * 12;
	ensureBuffer<float>(m_impulseBuffer, m_impulseFloats);
	memset(m_impulseBuffer->getBufferPointer(), 0, m_impulseFloats * sizeof(float));
	m_impulseBuffer->setDirty();

	ensureBuffer<float>(m_rollingAccumBuffer, numContacts);
	memset(m_rollingAccumBuffer->getBufferPointer(), 0, numContacts * sizeof(float));
	m_rollingAccumBuffer->setDirty();

	ensureBuffer<SolverParams>(m_paramBuffer, 1);

	SolverParams p;
	p.numContacts = numContacts;
	p.numBodies = numBodies;
	p.deltaTime = deltaTime;
	p.erp = erp;
	memcpy(m_paramBuffer->getBufferPointer(), &p, sizeof(p));
	m_paramBuffer->setDirty();

	const irr::u32 contactGroups = (numContacts + 63) / 64;
	const irr::u32 bodyGroups = (numBodies + 63) / 64;
	irr::video::SMaterial mat;

	for (int it = 0; it < (iterations < 1 ? 1 : iterations); ++it)
	{
		mat.MaterialType = (irr::video::E_MATERIAL_TYPE)m_solveMaterial;
		m_driver->setMaterial(mat);
		m_driver->bindComputeBuffer(0, m_paramBuffer, irr::video::EHBT_SHADER_RESOURCE);
		m_driver->bindComputeBuffer(1, m_contactBuffer, irr::video::EHBT_SHADER_RESOURCE);
		m_driver->bindComputeBuffer(2, m_inertiaBuffer, irr::video::EHBT_SHADER_RESOURCE);
		m_driver->bindComputeBuffer(3, m_loadBuffer, irr::video::EHBT_SHADER_RESOURCE);
		if (rollingFriction)
			m_driver->bindComputeBuffer(7, rollingFriction, irr::video::EHBT_SHADER_RESOURCE);
		m_driver->bindComputeBuffer(0, m_bodyBuffer, irr::video::EHBT_COMPUTE);
		m_driver->bindComputeBuffer(1, m_deltaBuffer, irr::video::EHBT_COMPUTE);
		m_driver->bindComputeBuffer(2, m_impulseBuffer, irr::video::EHBT_COMPUTE);
		m_driver->bindComputeBuffer(6, m_rollingAccumBuffer, irr::video::EHBT_COMPUTE);
		m_driver->dispatchComputeShaderBound(irr::core::vector3d<irr::u32>(contactGroups, 1, 1));
		m_driver->unbindComputeResources();

		mat.MaterialType = (irr::video::E_MATERIAL_TYPE)m_applyMaterial;
		m_driver->setMaterial(mat);
		m_driver->bindComputeBuffer(0, m_paramBuffer, irr::video::EHBT_SHADER_RESOURCE);
		m_driver->bindComputeBuffer(1, m_contactBuffer, irr::video::EHBT_SHADER_RESOURCE);
		m_driver->bindComputeBuffer(2, m_inertiaBuffer, irr::video::EHBT_SHADER_RESOURCE);
		m_driver->bindComputeBuffer(0, m_bodyBuffer, irr::video::EHBT_COMPUTE);
		m_driver->bindComputeBuffer(1, m_deltaBuffer, irr::video::EHBT_COMPUTE);
		m_driver->dispatchComputeShaderBound(irr::core::vector3d<irr::u32>(bodyGroups, 1, 1));
		m_driver->unbindComputeResources();
	}

	m_bodyBuffer->downloadFromGPU();
	memcpy(&bodies[0], m_bodyBuffer->getBufferPointer(), numBodies * sizeof(b3RigidBodyData));
	return true;
}

bool b3IrrlichtSolver::uploadInvInertia(const std::vector<float>& invInertiaDiag,
									   unsigned int numBodies)
{
	if (numBodies == 0)
		return false;

	ensureBuffer<IrrFloat4>(m_inertiaBuffer, numBodies);
	IrrFloat4* dst = (IrrFloat4*)m_inertiaBuffer->getBufferPointer();
	for (irr::u32 i = 0; i < numBodies; ++i)
	{
		const bool haveData = (invInertiaDiag.size() >= (size_t)(i + 1) * 4);
		dst[i].x = haveData ? invInertiaDiag[i * 4 + 0] : 0.f;
		dst[i].y = haveData ? invInertiaDiag[i * 4 + 1] : 0.f;
		dst[i].z = haveData ? invInertiaDiag[i * 4 + 2] : 0.f;
		dst[i].w = 0.f;
	}
	m_inertiaBuffer->setDirty();
	return true;
}

bool b3IrrlichtSolver::solveContactsResident(irr::scene::IComputeBuffer* bodies,
											 unsigned int numBodies,
											 irr::scene::IComputeBuffer* contacts,
											 irr::scene::IComputeBuffer* contactCount,
											 unsigned int maxContacts, int iterations,
											 float deltaTime, float erp,
											 irr::scene::IComputeBuffer* sleepState,
											 irr::scene::IComputeBuffer* rollingFriction)
{
	if (!isResidentPathAvailable() || !bodies || !contacts || !contactCount || numBodies == 0 ||
		maxContacts == 0)
		return false;

	// A body buffer of the other precision's stride would be solved misaligned, silently.
	if (!b3IrrGpu::strideMatches(bodies, b3IrrGpu::bodyStride(m_doubleSingle)))
		return false;

	if (!m_inertiaBuffer || m_inertiaBuffer->getStructureCount() < numBodies)
		return false;

	ensureBuffer<int>(m_deltaBuffer, numBodies * 6);
	ensureBuffer<int>(m_loadBuffer, numBodies);
	ensureBuffer<float>(m_impulseBuffer, maxContacts * 12);
	m_impulseFloats = maxContacts * 12;
	ensureBuffer<float>(m_rollingAccumBuffer, maxContacts);
	ensureBuffer<SolverParams>(m_paramBuffer, 1);

	// numContacts is filled in by the GPU below; the rest has to come from here.
	SolverParams p;
	p.numContacts = 0;
	p.numBodies = numBodies;
	p.deltaTime = deltaTime;
	p.erp = erp;
	memcpy(m_paramBuffer->getBufferPointer(), &p, sizeof(p));
	m_paramBuffer->setDirty();

	if (!m_dispatch->prepareIndirect(contactCount, m_paramBuffer, 64, maxContacts))
		return false;

	irr::scene::IComputeBuffer* args = m_dispatch->getArgsBuffer();
	const irr::u32 bodyGroups = (numBodies + 63) / 64;
	irr::video::SMaterial mat;

	mat.MaterialType = (irr::video::E_MATERIAL_TYPE)m_clearImpulseMaterial;
	m_driver->setMaterial(mat);
	m_driver->bindComputeBuffer(0, m_paramBuffer, irr::video::EHBT_SHADER_RESOURCE);
	m_driver->bindComputeBuffer(2, m_impulseBuffer, irr::video::EHBT_COMPUTE);
	m_driver->bindComputeBuffer(6, m_rollingAccumBuffer, irr::video::EHBT_COMPUTE);
	m_driver->dispatchComputeShaderIndirect(args, 0);
	m_driver->unbindComputeResources();

	const bool gated = m_activeGating && isActiveBodyGatingAvailable() &&
					   buildActiveBodyList(numBodies, contacts, args);
	irr::scene::IComputeBuffer* activeArgs = gated ? m_activeDispatch->getArgsBuffer() : 0;

	mat.MaterialType = (irr::video::E_MATERIAL_TYPE)(gated ? m_clearLoadActiveMaterial
														   : m_clearLoadMaterial);
	m_driver->setMaterial(mat);
	m_driver->bindComputeBuffer(0, m_paramBuffer, irr::video::EHBT_SHADER_RESOURCE);
	m_driver->bindComputeBuffer(3, m_loadBuffer, irr::video::EHBT_COMPUTE);
	if (gated)
	{
		m_driver->bindComputeBuffer(5, m_activeParamBuffer, irr::video::EHBT_SHADER_RESOURCE);
		m_driver->bindComputeBuffer(6, m_activeBodyBuffer, irr::video::EHBT_SHADER_RESOURCE);
		m_driver->dispatchComputeShaderIndirect(activeArgs, 0);
	}
	else
		m_driver->dispatchComputeShaderBound(irr::core::vector3d<irr::u32>(bodyGroups, 1, 1));
	m_driver->unbindComputeResources();

	mat.MaterialType = (irr::video::E_MATERIAL_TYPE)m_accumLoadMaterial;
	m_driver->setMaterial(mat);
	m_driver->bindComputeBuffer(0, m_paramBuffer, irr::video::EHBT_SHADER_RESOURCE);
	m_driver->bindComputeBuffer(1, contacts, irr::video::EHBT_SHADER_RESOURCE);
	m_driver->bindComputeBuffer(0, bodies, irr::video::EHBT_COMPUTE);
	m_driver->bindComputeBuffer(3, m_loadBuffer, irr::video::EHBT_COMPUTE);
	m_driver->dispatchComputeShaderIndirect(args, 0);
	m_driver->unbindComputeResources();

	for (int it = 0; it < (iterations < 1 ? 1 : iterations); ++it)
	{
		mat.MaterialType = (irr::video::E_MATERIAL_TYPE)m_solveMaterial;
		m_driver->setMaterial(mat);
		m_driver->bindComputeBuffer(0, m_paramBuffer, irr::video::EHBT_SHADER_RESOURCE);
		m_driver->bindComputeBuffer(1, contacts, irr::video::EHBT_SHADER_RESOURCE);
		m_driver->bindComputeBuffer(2, m_inertiaBuffer, irr::video::EHBT_SHADER_RESOURCE);
		m_driver->bindComputeBuffer(3, m_loadBuffer, irr::video::EHBT_SHADER_RESOURCE);
		if (sleepState)
			m_driver->bindComputeBuffer(4, sleepState, irr::video::EHBT_SHADER_RESOURCE);
		if (rollingFriction)
			m_driver->bindComputeBuffer(7, rollingFriction, irr::video::EHBT_SHADER_RESOURCE);
		m_driver->bindComputeBuffer(0, bodies, irr::video::EHBT_COMPUTE);
		m_driver->bindComputeBuffer(1, m_deltaBuffer, irr::video::EHBT_COMPUTE);
		m_driver->bindComputeBuffer(2, m_impulseBuffer, irr::video::EHBT_COMPUTE);
		m_driver->bindComputeBuffer(6, m_rollingAccumBuffer, irr::video::EHBT_COMPUTE);
		m_driver->dispatchComputeShaderIndirect(args, 0);
		m_driver->unbindComputeResources();

		mat.MaterialType = (irr::video::E_MATERIAL_TYPE)(gated ? m_applyActiveMaterial
															  : m_applyMaterial);
		m_driver->setMaterial(mat);
		m_driver->bindComputeBuffer(0, m_paramBuffer, irr::video::EHBT_SHADER_RESOURCE);
		m_driver->bindComputeBuffer(1, contacts, irr::video::EHBT_SHADER_RESOURCE);
		m_driver->bindComputeBuffer(2, m_inertiaBuffer, irr::video::EHBT_SHADER_RESOURCE);
		m_driver->bindComputeBuffer(0, bodies, irr::video::EHBT_COMPUTE);
		m_driver->bindComputeBuffer(1, m_deltaBuffer, irr::video::EHBT_COMPUTE);
		if (gated)
		{
			m_driver->bindComputeBuffer(5, m_activeParamBuffer, irr::video::EHBT_SHADER_RESOURCE);
			m_driver->bindComputeBuffer(6, m_activeBodyBuffer, irr::video::EHBT_SHADER_RESOURCE);
			m_driver->dispatchComputeShaderIndirect(activeArgs, 0);
		}
		else
			m_driver->dispatchComputeShaderBound(irr::core::vector3d<irr::u32>(bodyGroups, 1, 1));
		m_driver->unbindComputeResources();
	}

	return true;
}

bool b3IrrlichtSolver::buildActiveBodyList(unsigned int numBodies,
										  irr::scene::IComputeBuffer* contacts,
										  irr::scene::IComputeBuffer* contactArgs)
{
	// A fresh claim buffer reads 0 everywhere, so the stamp sequence has to restart with it.
	irr::scene::IComputeBuffer* previousClaim = m_claimBuffer;
	ensureBuffer<unsigned int>(m_claimBuffer, numBodies);
	if (m_claimBuffer != previousClaim)
		m_activeStamp = 0;
	if (++m_activeStamp == 0u)
		m_activeStamp = 1u;

	ensureBuffer<unsigned int>(m_activeBodyBuffer, numBodies, irr::video::EHBF_COMPUTE_APPEND);
	ensureBuffer<unsigned int>(m_activeCountBuffer, 4, irr::video::EHBF_DRAW_INDIRECT_ARGS);
	if (!m_claimBuffer || !m_activeBodyBuffer || !m_activeCountBuffer)
		return false;

	b3IrrGpu::CountParams ap;
	ap.count = 0;
	ap.cap = m_activeStamp;
	ap.reduceCount = 0;
	ap.numBodies = numBodies;
	b3IrrGpu::uploadBuffer<b3IrrGpu::CountParams>(m_activeParamBuffer, &ap, 1);
	if (!m_activeParamBuffer)
		return false;

	irr::video::SMaterial mat;
	mat.MaterialType = (irr::video::E_MATERIAL_TYPE)m_buildActiveMaterial;
	m_driver->setMaterial(mat);
	m_driver->bindComputeBuffer(0, m_paramBuffer, irr::video::EHBT_SHADER_RESOURCE);
	m_driver->bindComputeBuffer(1, contacts, irr::video::EHBT_SHADER_RESOURCE);
	m_driver->bindComputeBuffer(5, m_activeParamBuffer, irr::video::EHBT_SHADER_RESOURCE);
	m_driver->bindComputeBuffer(4, m_activeBodyBuffer, irr::video::EHBT_COMPUTE);
	m_driver->bindComputeBuffer(5, m_claimBuffer, irr::video::EHBT_COMPUTE);
	m_driver->resetStructureCount(m_activeBodyBuffer, 0);
	m_driver->dispatchComputeShaderIndirect(contactArgs, 0);
	m_driver->copyStructureCount(m_activeCountBuffer, 0, m_activeBodyBuffer);
	m_driver->unbindComputeResources();
	m_driver->computeBarrier(m_activeBodyBuffer);

	// Patches ActiveParams.x only, so the stamp uploaded above survives into the build next step.
	return m_activeDispatch->prepareIndirect(m_activeCountBuffer, m_activeParamBuffer, 64, numBodies);
}

bool b3IrrlichtSolver::downloadAccumulatedImpulses(std::vector<float>& out)
{
	if (!m_impulseBuffer || m_impulseFloats == 0)
		return false;

	m_impulseBuffer->downloadFromGPU();
	out.resize(m_impulseFloats);
	memcpy(&out[0], m_impulseBuffer->getBufferPointer(), m_impulseFloats * sizeof(float));
	return true;
}
