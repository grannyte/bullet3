#include "b3IrrlichtJointSolver.h"
#include "b3IrrlichtGpuBuffers.h"

#include <irrlicht.h>
#include "ComputeBuffer.h"

#include <cmath>
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

/// Mirrors b3IrrJointAccum in B3SolveJointsBody.hlsli.
struct JointAccum
{
	float lin[4];
	float ang[4];
	float linMotor[4];
	float angMotor[4];
};

static_assert(sizeof(JointSolverParams) == 16, "JointSolverParams must match the HLSL struct stride");
static_assert(sizeof(b3IrrJoint) == 240, "b3IrrJoint must match the HLSL struct stride");
static_assert(sizeof(JointAccum) == 64, "JointAccum must match the HLSL struct stride");

using b3IrrGpu::dropBuffer;
using b3IrrGpu::ensureBuffer;

const float kFree = 1.f;       // lower > upper reads as "no limit" on the GPU
const float kFreeUpper = -1.f;

void copy3(float* dst, const float* src)
{
	for (int i = 0; i < 3; ++i)
		dst[i] = src ? src[i] : 0.f;
	dst[3] = 0.f;
}

void setIdentity(float* q)
{
	q[0] = q[1] = q[2] = 0.f;
	q[3] = 1.f;
}

void setAll4(float* v, float x, float y, float z)
{
	v[0] = x;
	v[1] = y;
	v[2] = z;
	v[3] = 0.f;
}

void cross3(const float* a, const float* b, float* out)
{
	out[0] = a[1] * b[2] - a[2] * b[1];
	out[1] = a[2] * b[0] - a[0] * b[2];
	out[2] = a[0] * b[1] - a[1] * b[0];
}

bool normalize3(float* v)
{
	const float len = std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
	if (len < 1e-12f)
		return false;
	for (int i = 0; i < 3; ++i)
		v[i] /= len;
	return true;
}

/// q = a * b, xyzw, same convention as quatMul in the kernel.
void quatMul(const float* a, const float* b, float* out)
{
	float c[3];
	cross3(a, b, c);
	out[0] = a[3] * b[0] + b[3] * a[0] + c[0];
	out[1] = a[3] * b[1] + b[3] * a[1] + c[1];
	out[2] = a[3] * b[2] + b[3] * a[2] + c[2];
	out[3] = a[3] * b[3] - (a[0] * b[0] + a[1] * b[1] + a[2] * b[2]);
}

void quatConj(const float* q, float* out)
{
	out[0] = -q[0];
	out[1] = -q[1];
	out[2] = -q[2];
	out[3] = q[3];
}

/// Rotation whose columns are ex, ey, ez, as a quaternion (Shepperd's method).
void quatFromBasis(const float* ex, const float* ey, const float* ez, float* q)
{
	const float m00 = ex[0], m10 = ex[1], m20 = ex[2];
	const float m01 = ey[0], m11 = ey[1], m21 = ey[2];
	const float m02 = ez[0], m12 = ez[1], m22 = ez[2];
	const float trace = m00 + m11 + m22;
	if (trace > 0.f)
	{
		const float s = std::sqrt(trace + 1.f) * 2.f;
		q[3] = 0.25f * s;
		q[0] = (m21 - m12) / s;
		q[1] = (m02 - m20) / s;
		q[2] = (m10 - m01) / s;
	}
	else if (m00 > m11 && m00 > m22)
	{
		const float s = std::sqrt(1.f + m00 - m11 - m22) * 2.f;
		q[3] = (m21 - m12) / s;
		q[0] = 0.25f * s;
		q[1] = (m01 + m10) / s;
		q[2] = (m02 + m20) / s;
	}
	else if (m11 > m22)
	{
		const float s = std::sqrt(1.f + m11 - m00 - m22) * 2.f;
		q[3] = (m02 - m20) / s;
		q[0] = (m01 + m10) / s;
		q[1] = 0.25f * s;
		q[2] = (m12 + m21) / s;
	}
	else
	{
		const float s = std::sqrt(1.f + m22 - m00 - m11) * 2.f;
		q[3] = (m10 - m01) / s;
		q[0] = (m02 + m20) / s;
		q[1] = (m12 + m21) / s;
		q[2] = 0.25f * s;
	}
}

/// Joint frame with its x axis along `axis`; the perpendicular pick is the same for both bodies,
/// so equal axes on aligned bodies give coincident frames.
void frameFromAxis(const float* axis, float* q)
{
	float ex[3] = {axis ? axis[0] : 1.f, axis ? axis[1] : 0.f, axis ? axis[2] : 0.f};
	if (!normalize3(ex))
	{
		setIdentity(q);
		return;
	}
	const float helper[3] = {std::fabs(ex[0]) < 0.9f ? 1.f : 0.f, std::fabs(ex[0]) < 0.9f ? 0.f : 1.f, 0.f};
	float ez[3], ey[3];
	cross3(ex, helper, ez);
	normalize3(ez);
	cross3(ez, ex, ey);
	quatFromBasis(ex, ey, ez, q);
}

/// Common tail of every maker: identity frames, every axis free, no motors, global ERP.
b3IrrJoint blankJoint(int type, int bodyA, int bodyB, const float* pivotA, const float* pivotB)
{
	b3IrrJoint j;
	memset(&j, 0, sizeof(j));
	j.constraintType = type;
	j.rbA = bodyA;
	j.rbB = bodyB;
	copy3(j.pivotInA, pivotA);
	copy3(j.pivotInB, pivotB);
	setIdentity(j.relTargetAB);
	setIdentity(j.frameInA);
	setIdentity(j.frameInB);
	setAll4(j.linLower, kFree, kFree, kFree);
	setAll4(j.linUpper, kFreeUpper, kFreeUpper, kFreeUpper);
	setAll4(j.angLower, kFree, kFree, kFree);
	setAll4(j.angUpper, kFreeUpper, kFreeUpper, kFreeUpper);
	j.flags = B3_IRR_CONSTRAINT_FLAG_ENABLED;
	return j;
}
}  // namespace

b3IrrlichtJointSolver::b3IrrlichtJointSolver(irr::video::IVideoDriver* driver)
	: m_driver(driver), m_doubleSingle(false), m_solveMaterial(-1), m_applyMaterial(-1),
	  m_finishMaterial(-1), m_debugMaterial(-1), m_paramBuffer(0), m_bodyBuffer(0), m_jointBuffer(0),
	  m_inertiaBuffer(0), m_scaleBuffer(0), m_deltaBuffer(0), m_accumBuffer(0), m_statusBuffer(0),
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
	dropBuffer(m_accumBuffer);
	dropBuffer(m_statusBuffer);
	m_residentJoints = 0;
}

b3IrrJoint b3IrrlichtJointSolver::makePoint2Point(int bodyA, int bodyB, const float* pivotA,
												 const float* pivotB)
{
	return blankJoint(B3_IRR_POINT2POINT_CONSTRAINT_TYPE, bodyA, bodyB, pivotA, pivotB);
}

b3IrrJoint b3IrrlichtJointSolver::makeFixed(int bodyA, int bodyB, const float* pivotA,
											const float* pivotB, const float* relTargetAB)
{
	b3IrrJoint j = blankJoint(B3_IRR_FIXED_CONSTRAINT_TYPE, bodyA, bodyB, pivotA, pivotB);
	if (relTargetAB)
		for (int i = 0; i < 4; ++i)
			j.relTargetAB[i] = relTargetAB[i];
	return j;
}

b3IrrJoint b3IrrlichtJointSolver::makeHinge(int bodyA, int bodyB, const float* pivotA,
											const float* pivotB, const float* axisA,
											const float* axisB, float lower, float upper)
{
	b3IrrJoint j = blankJoint(B3_IRR_HINGE_CONSTRAINT_TYPE, bodyA, bodyB, pivotA, pivotB);
	frameFromAxis(axisA, j.frameInA);
	frameFromAxis(axisB, j.frameInB);
	setAll4(j.linLower, 0.f, 0.f, 0.f);
	setAll4(j.linUpper, 0.f, 0.f, 0.f);
	setAll4(j.angLower, lower, 0.f, 0.f);
	setAll4(j.angUpper, upper, 0.f, 0.f);
	return j;
}

b3IrrJoint b3IrrlichtJointSolver::makeSlider(int bodyA, int bodyB, const float* pivotA,
											 const float* pivotB, const float* axisA,
											 const float* axisB, float lower, float upper)
{
	b3IrrJoint j = blankJoint(B3_IRR_SLIDER_CONSTRAINT_TYPE, bodyA, bodyB, pivotA, pivotB);
	frameFromAxis(axisA, j.frameInA);
	frameFromAxis(axisB, j.frameInB);
	setAll4(j.linLower, lower, 0.f, 0.f);
	setAll4(j.linUpper, upper, 0.f, 0.f);
	setAll4(j.angLower, 0.f, 0.f, 0.f);
	setAll4(j.angUpper, 0.f, 0.f, 0.f);
	return j;
}

b3IrrJoint b3IrrlichtJointSolver::makeConeTwist(int bodyA, int bodyB, const float* pivotA,
												const float* pivotB, const float* axisA,
												const float* axisB, float swingSpan,
												float twistSpan)
{
	b3IrrJoint j = blankJoint(B3_IRR_CONETWIST_CONSTRAINT_TYPE, bodyA, bodyB, pivotA, pivotB);
	frameFromAxis(axisA, j.frameInA);
	frameFromAxis(axisB, j.frameInB);
	setAll4(j.linLower, 0.f, 0.f, 0.f);
	setAll4(j.linUpper, 0.f, 0.f, 0.f);
	// x: twist span; the kernel reads the cone half-angle from angUpper.y for this type.
	setAll4(j.angLower, -std::fabs(twistSpan), kFree, kFree);
	setAll4(j.angUpper, std::fabs(twistSpan), std::fabs(swingSpan), kFreeUpper);
	return j;
}

b3IrrJoint b3IrrlichtJointSolver::make6Dof(int bodyA, int bodyB, const float* pivotA,
										   const float* pivotB, const float* frameA,
										   const float* frameB)
{
	b3IrrJoint j = blankJoint(B3_IRR_D6_CONSTRAINT_TYPE, bodyA, bodyB, pivotA, pivotB);
	for (int i = 0; i < 4; ++i)
	{
		if (frameA)
			j.frameInA[i] = frameA[i];
		if (frameB)
			j.frameInB[i] = frameB[i];
	}
	return j;
}

void b3IrrlichtJointSolver::setLinearLimit(b3IrrJoint& joint, b3IrrJointAxis axis, float lower,
										   float upper)
{
	joint.linLower[axis] = lower;
	joint.linUpper[axis] = upper;
}

void b3IrrlichtJointSolver::setAngularLimit(b3IrrJoint& joint, b3IrrJointAxis axis, float lower,
											float upper)
{
	joint.angLower[axis] = lower;
	joint.angUpper[axis] = upper;
}

void b3IrrlichtJointSolver::setMotor(b3IrrJoint& joint, b3IrrJointAxis axis, bool angular,
									 float targetVelocity, float maxImpulse)
{
	float* vel = angular ? joint.angMotorVel : joint.linMotorVel;
	float* cap = angular ? joint.angMotorMaxImpulse : joint.linMotorMaxImpulse;
	vel[axis] = targetVelocity;
	cap[axis] = maxImpulse < 0.f ? 0.f : maxImpulse;
}

void b3IrrlichtJointSolver::setErpCfm(b3IrrJoint& joint, float erp, float cfm)
{
	joint.erp = erp;
	joint.cfm = cfm;
	joint.flags |= B3_IRR_CONSTRAINT_FLAG_OWN_ERP;
}

void b3IrrlichtJointSolver::alignFrameBToCurrentPose(b3IrrJoint& joint, const float* quatA,
													 const float* quatB)
{
	// qB * frameInB == qA * frameInA  =>  frameInB = conj(qB) * qA * frameInA
	float conjB[4], tmp[4];
	quatConj(quatB, conjB);
	quatMul(quatA, joint.frameInA, tmp);
	quatMul(conjB, tmp, joint.frameInB);
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
	m_finishMaterial = gpu->addComputeShaderFromFile(path, "CSFinishJoints", irr::video::ECST_CS_5_0, 0);
	// Diagnostic only, so its absence must never fail init and disable the whole solver.
	m_debugMaterial = gpu->addComputeShaderFromFile(path, "CSDebugJointLayout",
												   irr::video::ECST_CS_5_0, 0);

	return m_solveMaterial >= 0 && m_applyMaterial >= 0 && m_finishMaterial >= 0;
}

bool b3IrrlichtJointSolver::debugReadJointLayout(const std::vector<b3IrrJoint>& joints,
												 std::vector<int>& out)
{
	if (!m_driver || m_debugMaterial < 0 || joints.empty())
		return false;

	const irr::u32 numJoints = (irr::u32)joints.size();
	const irr::u32 ints = numJoints * DebugLayoutInts;

	b3IrrGpu::uploadBuffer<b3IrrJoint>(m_jointBuffer, &joints[0], numJoints);

	ensureBuffer<int>(m_deltaBuffer, ints);
	memset(m_deltaBuffer->getBufferPointer(), 0, ints * sizeof(int));
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
	out.resize(ints);
	memcpy(&out[0], m_deltaBuffer->getBufferPointer(), ints * sizeof(int));
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

	// Fresh joints start with nothing accumulated and nothing broken.
	ensureBuffer<JointAccum>(m_accumBuffer, numJoints);
	memset(m_accumBuffer->getBufferPointer(), 0, numJoints * sizeof(JointAccum));
	m_accumBuffer->setDirty();
	ensureBuffer<unsigned int>(m_statusBuffer, numJoints);
	memset(m_statusBuffer->getBufferPointer(), 0, numJoints * sizeof(unsigned int));
	m_statusBuffer->setDirty();

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

void b3IrrlichtJointSolver::runIterations(irr::scene::IComputeBuffer* bodies, unsigned int numBodies,
										  unsigned int numJoints, int iterations)
{
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
		m_driver->bindComputeBuffer(0, bodies, irr::video::EHBT_COMPUTE);
		m_driver->bindComputeBuffer(1, m_deltaBuffer, irr::video::EHBT_COMPUTE);
		m_driver->bindComputeBuffer(2, m_accumBuffer, irr::video::EHBT_COMPUTE);
		m_driver->dispatchComputeShaderBound(irr::core::vector3d<irr::u32>(jointGroups, 1, 1));
		m_driver->unbindComputeResources();

		mat.MaterialType = (irr::video::E_MATERIAL_TYPE)m_applyMaterial;
		m_driver->setMaterial(mat);
		m_driver->bindComputeBuffer(0, m_paramBuffer, irr::video::EHBT_SHADER_RESOURCE);
		m_driver->bindComputeBuffer(0, bodies, irr::video::EHBT_COMPUTE);
		m_driver->bindComputeBuffer(1, m_deltaBuffer, irr::video::EHBT_COMPUTE);
		m_driver->dispatchComputeShaderBound(irr::core::vector3d<irr::u32>(bodyGroups, 1, 1));
		m_driver->unbindComputeResources();
	}

	// Joints are written here (enable bit), so they are bound as u3 and their t1 view is NOT bound.
	mat.MaterialType = (irr::video::E_MATERIAL_TYPE)m_finishMaterial;
	m_driver->setMaterial(mat);
	m_driver->bindComputeBuffer(0, m_paramBuffer, irr::video::EHBT_SHADER_RESOURCE);
	m_driver->bindComputeBuffer(2, m_accumBuffer, irr::video::EHBT_COMPUTE);
	m_driver->bindComputeBuffer(3, m_jointBuffer, irr::video::EHBT_COMPUTE);
	m_driver->bindComputeBuffer(4, m_statusBuffer, irr::video::EHBT_COMPUTE);
	m_driver->dispatchComputeShaderBound(irr::core::vector3d<irr::u32>(jointGroups, 1, 1));
	m_driver->unbindComputeResources();
}

bool b3IrrlichtJointSolver::solveJointsResident(irr::scene::IComputeBuffer* bodies,
												unsigned int numBodies, int iterations,
												float deltaTime, float erp)
{
	if (m_solveMaterial < 0 || m_applyMaterial < 0 || m_finishMaterial < 0 || !bodies || numBodies == 0)
		return false;

	// A body buffer of the other precision's stride would be indexed silently wrong.
	if (!b3IrrGpu::strideMatches(bodies, b3IrrGpu::bodyStride(m_doubleSingle)))
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

	runIterations(bodies, numBodies, m_residentJoints, iterations);
	return true;
}

bool b3IrrlichtJointSolver::readJointStatus(std::vector<unsigned int>& broken)
{
	if (!m_statusBuffer || m_residentJoints == 0)
		return false;

	m_statusBuffer->downloadFromGPU();
	broken.resize(m_residentJoints);
	memcpy(&broken[0], m_statusBuffer->getBufferPointer(), m_residentJoints * sizeof(unsigned int));
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

	if (m_solveMaterial < 0 || m_applyMaterial < 0 || m_finishMaterial < 0 || bodies.empty())
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

	runIterations(m_bodyBuffer, numBodies, numJoints, iterations);

	m_bodyBuffer->downloadFromGPU();
	memcpy(&bodies[0], m_bodyBuffer->getBufferPointer(), numBodies * sizeof(b3RigidBodyData));
	return true;
}
