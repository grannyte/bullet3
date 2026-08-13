#include "b3IrrlichtJointSolver.h"
#include "b3IrrlichtGpuBuffers.h"

#include <irrlicht.h>
#include "ComputeBuffer.h"

#include <cstring>

namespace
{
struct JointSolverParams
{
	unsigned int numJoints;
	unsigned int numBodies;
	float deltaTime;
	float erp;
};

struct IrrFloat4
{
	float x, y, z, w;
};

static_assert(sizeof(JointSolverParams) == 16, "JointSolverParams must match the HLSL struct stride");
static_assert(sizeof(b3IrrJoint) == 80, "b3IrrJoint must match the HLSL struct stride");

using b3IrrGpu::dropBuffer;
using b3IrrGpu::ensureBuffer;
}  // namespace

b3IrrlichtJointSolver::b3IrrlichtJointSolver(irr::video::IVideoDriver* driver)
	: m_driver(driver), m_doubleSingle(false), m_solveMaterial(-1), m_applyMaterial(-1), m_debugMaterial(-1), m_paramBuffer(0),
	  m_bodyBuffer(0), m_jointBuffer(0), m_inertiaBuffer(0), m_scaleBuffer(0), m_deltaBuffer(0),
	  m_residentJoints(0)
{
}

b3IrrlichtJointSolver::~b3IrrlichtJointSolver()
{
	releaseBuffers();
	dropBuffer(m_paramBuffer);
}

void b3IrrlichtJointSolver::releaseBuffers()
{
	dropBuffer(m_bodyBuffer);
	dropBuffer(m_jointBuffer);
	dropBuffer(m_inertiaBuffer);
	dropBuffer(m_scaleBuffer);
	dropBuffer(m_deltaBuffer);
	m_residentJoints = 0;
}

b3IrrJoint b3IrrlichtJointSolver::makePoint2Point(int bodyA, int bodyB, const float* pivotA,
												 const float* pivotB)
{
	b3IrrJoint j;
	memset(&j, 0, sizeof(j));
	j.constraintType = B3_IRR_POINT2POINT_CONSTRAINT_TYPE;
	j.rbA = bodyA;
	j.rbB = bodyB;
	j.breakingImpulseThreshold = 0.f;
	for (int i = 0; i < 3; ++i)
	{
		j.pivotInA[i] = pivotA ? pivotA[i] : 0.f;
		j.pivotInB[i] = pivotB ? pivotB[i] : 0.f;
	}
	j.relTargetAB[3] = 1.f;
	j.flags = B3_IRR_CONSTRAINT_FLAG_ENABLED;
	return j;
}

bool b3IrrlichtJointSolver::init(irr::io::IFileSystem* fileSystem, bool doubleSingle)
{
	m_doubleSingle = doubleSingle;

	if (!m_driver)
		return false;

	irr::video::IGPUProgrammingServices* gpu = m_driver->getGPUProgrammingServices();
	if (!gpu)
		return false;

	const irr::io::path path = m_doubleSingle ? "media/shaders/B3SolveJointsDS.hlsl" : "media/shaders/B3SolveJoints.hlsl";
	if (fileSystem && !fileSystem->existFile(path))
		return false;

	m_solveMaterial = gpu->addComputeShaderFromFile(path, "CSSolveJoints", irr::video::ECST_CS_5_0, 0);
	m_applyMaterial = gpu->addComputeShaderFromFile(path, "CSApplyJointVelocityDeltas",
													irr::video::ECST_CS_5_0, 0);
	// Diagnostic only, so its absence must never fail init and disable the whole solver.
	m_debugMaterial = gpu->addComputeShaderFromFile(path, "CSDebugJointLayout",
												   irr::video::ECST_CS_5_0, 0);

	return m_solveMaterial >= 0 && m_applyMaterial >= 0;
}

bool b3IrrlichtJointSolver::debugReadJointLayout(const std::vector<b3IrrJoint>& joints,
												 std::vector<int>& out)
{
	if (!m_driver || m_debugMaterial < 0 || joints.empty())
		return false;

	const irr::u32 numJoints = (irr::u32)joints.size();

	b3IrrGpu::uploadBuffer<b3IrrJoint>(m_jointBuffer, &joints[0], numJoints);

	ensureBuffer<int>(m_deltaBuffer, numJoints * 8);
	memset(m_deltaBuffer->getBufferPointer(), 0, numJoints * 8 * sizeof(int));
	m_deltaBuffer->setDirty();

	ensureBuffer<JointSolverParams>(m_paramBuffer, 1);

	JointSolverParams p;
	p.numJoints = numJoints;
	p.numBodies = 0;
	p.deltaTime = 1.f;
	p.erp = 0.f;
	memcpy(m_paramBuffer->getBufferPointer(), &p, sizeof(p));
	m_paramBuffer->setDirty();

	irr::video::SMaterial mat;
	mat.MaterialType = (irr::video::E_MATERIAL_TYPE)m_debugMaterial;
	m_driver->setMaterial(mat);
	m_driver->bindComputeBuffer(0, m_paramBuffer, irr::video::EHBT_SHADER_RESOURCE);
	m_driver->bindComputeBuffer(1, m_jointBuffer, irr::video::EHBT_SHADER_RESOURCE);
	m_driver->bindComputeBuffer(1, m_deltaBuffer, irr::video::EHBT_COMPUTE);
	m_driver->dispatchComputeShaderBound(irr::core::vector3d<irr::u32>((numJoints + 63) / 64, 1, 1));
	m_driver->unbindComputeResources();

	m_deltaBuffer->downloadFromGPU();
	out.resize(numJoints * 8);
	memcpy(&out[0], m_deltaBuffer->getBufferPointer(), numJoints * 8 * sizeof(int));
	return true;
}

bool b3IrrlichtJointSolver::uploadJoints(const std::vector<b3IrrJoint>& joints,
										const std::vector<b3RigidBodyData>& bodies)
{
	if (joints.empty() || bodies.empty())
		return false;

	const irr::u32 numJoints = (irr::u32)joints.size();
	const irr::u32 numBodies = (irr::u32)bodies.size();

	b3IrrGpu::uploadBuffer<b3IrrJoint>(m_jointBuffer, &joints[0], numJoints);

	// A Jacobi sweep applies every joint against the same velocities, so a body shared by N joints
	// would be over-corrected N times; splitting by the busier endpoint's count keeps it stable.
	ensureBuffer<float>(m_scaleBuffer, numJoints);
	std::vector<unsigned int> jointsPerBody(numBodies, 0);
	for (irr::u32 j = 0; j < numJoints; ++j)
	{
		const int a = joints[j].rbA;
		const int b = joints[j].rbB;
		if (a >= 0 && (irr::u32)a < numBodies && bodies[a].m_invMass > 0.f)
			++jointsPerBody[a];
		if (b >= 0 && (irr::u32)b < numBodies && bodies[b].m_invMass > 0.f)
			++jointsPerBody[b];
	}

	float* dst = (float*)m_scaleBuffer->getBufferPointer();
	for (irr::u32 j = 0; j < numJoints; ++j)
	{
		const int a = joints[j].rbA;
		const int b = joints[j].rbB;
		unsigned int worst = 1;
		if (a >= 0 && (irr::u32)a < numBodies && jointsPerBody[a] > worst)
			worst = jointsPerBody[a];
		if (b >= 0 && (irr::u32)b < numBodies && jointsPerBody[b] > worst)
			worst = jointsPerBody[b];
		dst[j] = 1.f / (float)worst;
	}
	m_scaleBuffer->setDirty();

	m_residentJoints = numJoints;
	return true;
}

bool b3IrrlichtJointSolver::uploadInvInertia(const std::vector<float>& invInertiaDiag,
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

bool b3IrrlichtJointSolver::solveJointsResident(irr::scene::IComputeBuffer* bodies,
												unsigned int numBodies, int iterations,
												float deltaTime, float erp)
{
	if (m_solveMaterial < 0 || m_applyMaterial < 0 || !bodies || numBodies == 0)
		return false;

	if (m_residentJoints == 0 || !m_jointBuffer || !m_scaleBuffer || !m_inertiaBuffer)
		return false;

	ensureBuffer<int>(m_deltaBuffer, numBodies * 6);
	ensureBuffer<JointSolverParams>(m_paramBuffer, 1);

	JointSolverParams p;
	p.numJoints = m_residentJoints;
	p.numBodies = numBodies;
	p.deltaTime = deltaTime;
	p.erp = erp;
	memcpy(m_paramBuffer->getBufferPointer(), &p, sizeof(p));
	m_paramBuffer->setDirty();

	const irr::u32 jointGroups = (m_residentJoints + 63) / 64;
	const irr::u32 bodyGroups = (numBodies + 63) / 64;
	irr::video::SMaterial mat;

	for (int it = 0; it < (iterations < 1 ? 1 : iterations); ++it)
	{
		mat.MaterialType = (irr::video::E_MATERIAL_TYPE)m_solveMaterial;
		m_driver->setMaterial(mat);
		m_driver->bindComputeBuffer(0, m_paramBuffer, irr::video::EHBT_SHADER_RESOURCE);
		m_driver->bindComputeBuffer(1, m_jointBuffer, irr::video::EHBT_SHADER_RESOURCE);
		m_driver->bindComputeBuffer(2, m_inertiaBuffer, irr::video::EHBT_SHADER_RESOURCE);
		m_driver->bindComputeBuffer(3, m_scaleBuffer, irr::video::EHBT_SHADER_RESOURCE);
		m_driver->bindComputeBuffer(0, bodies, irr::video::EHBT_COMPUTE);
		m_driver->bindComputeBuffer(1, m_deltaBuffer, irr::video::EHBT_COMPUTE);
		m_driver->dispatchComputeShaderBound(irr::core::vector3d<irr::u32>(jointGroups, 1, 1));
		m_driver->unbindComputeResources();

		mat.MaterialType = (irr::video::E_MATERIAL_TYPE)m_applyMaterial;
		m_driver->setMaterial(mat);
		m_driver->bindComputeBuffer(0, m_paramBuffer, irr::video::EHBT_SHADER_RESOURCE);
		m_driver->bindComputeBuffer(1, m_jointBuffer, irr::video::EHBT_SHADER_RESOURCE);
		m_driver->bindComputeBuffer(2, m_inertiaBuffer, irr::video::EHBT_SHADER_RESOURCE);
		m_driver->bindComputeBuffer(3, m_scaleBuffer, irr::video::EHBT_SHADER_RESOURCE);
		m_driver->bindComputeBuffer(0, bodies, irr::video::EHBT_COMPUTE);
		m_driver->bindComputeBuffer(1, m_deltaBuffer, irr::video::EHBT_COMPUTE);
		m_driver->dispatchComputeShaderBound(irr::core::vector3d<irr::u32>(bodyGroups, 1, 1));
		m_driver->unbindComputeResources();
	}

	return true;
}

bool b3IrrlichtJointSolver::solveJoints(std::vector<b3RigidBodyData>& bodies,
										const std::vector<float>& invInertiaDiag,
										const std::vector<b3IrrJoint>& joints,
										int iterations, float deltaTime, float erp)
{
	// The std::vector path moves 80-byte bodies; the df64 kernels index a 96-byte stride.
	if (m_doubleSingle)
		return false;

	if (m_solveMaterial < 0 || m_applyMaterial < 0 || bodies.empty())
		return false;

	if (joints.empty())
		return true;

	const irr::u32 numBodies = (irr::u32)bodies.size();
	const irr::u32 numJoints = (irr::u32)joints.size();

	b3IrrGpu::uploadBuffer<b3RigidBodyData>(m_bodyBuffer, &bodies[0], numBodies);

	if (!uploadJoints(joints, bodies) || !uploadInvInertia(invInertiaDiag, numBodies))
		return false;

	// 6 ints per body; zeroed here and re-zeroed by the apply kernel each iteration.
	ensureBuffer<int>(m_deltaBuffer, numBodies * 6);
	memset(m_deltaBuffer->getBufferPointer(), 0, numBodies * 6 * sizeof(int));
	m_deltaBuffer->setDirty();

	ensureBuffer<JointSolverParams>(m_paramBuffer, 1);

	JointSolverParams p;
	p.numJoints = numJoints;
	p.numBodies = numBodies;
	p.deltaTime = deltaTime;
	p.erp = erp;
	memcpy(m_paramBuffer->getBufferPointer(), &p, sizeof(p));
	m_paramBuffer->setDirty();

	const irr::u32 jointGroups = (numJoints + 63) / 64;
	const irr::u32 bodyGroups = (numBodies + 63) / 64;
	irr::video::SMaterial mat;

	for (int it = 0; it < (iterations < 1 ? 1 : iterations); ++it)
	{
		mat.MaterialType = (irr::video::E_MATERIAL_TYPE)m_solveMaterial;
		m_driver->setMaterial(mat);
		m_driver->bindComputeBuffer(0, m_paramBuffer, irr::video::EHBT_SHADER_RESOURCE);
		m_driver->bindComputeBuffer(1, m_jointBuffer, irr::video::EHBT_SHADER_RESOURCE);
		m_driver->bindComputeBuffer(2, m_inertiaBuffer, irr::video::EHBT_SHADER_RESOURCE);
		m_driver->bindComputeBuffer(3, m_scaleBuffer, irr::video::EHBT_SHADER_RESOURCE);
		m_driver->bindComputeBuffer(0, m_bodyBuffer, irr::video::EHBT_COMPUTE);
		m_driver->bindComputeBuffer(1, m_deltaBuffer, irr::video::EHBT_COMPUTE);
		m_driver->dispatchComputeShaderBound(irr::core::vector3d<irr::u32>(jointGroups, 1, 1));
		m_driver->unbindComputeResources();

		mat.MaterialType = (irr::video::E_MATERIAL_TYPE)m_applyMaterial;
		m_driver->setMaterial(mat);
		m_driver->bindComputeBuffer(0, m_paramBuffer, irr::video::EHBT_SHADER_RESOURCE);
		m_driver->bindComputeBuffer(1, m_jointBuffer, irr::video::EHBT_SHADER_RESOURCE);
		m_driver->bindComputeBuffer(2, m_inertiaBuffer, irr::video::EHBT_SHADER_RESOURCE);
		m_driver->bindComputeBuffer(3, m_scaleBuffer, irr::video::EHBT_SHADER_RESOURCE);
		m_driver->bindComputeBuffer(0, m_bodyBuffer, irr::video::EHBT_COMPUTE);
		m_driver->bindComputeBuffer(1, m_deltaBuffer, irr::video::EHBT_COMPUTE);
		m_driver->dispatchComputeShaderBound(irr::core::vector3d<irr::u32>(bodyGroups, 1, 1));
		m_driver->unbindComputeResources();
	}

	m_bodyBuffer->downloadFromGPU();
	memcpy(&bodies[0], m_bodyBuffer->getBufferPointer(), numBodies * sizeof(b3RigidBodyData));
	return true;
}
