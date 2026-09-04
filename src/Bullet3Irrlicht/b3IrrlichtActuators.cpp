#include "b3IrrlichtActuators.h"
#include "b3IrrlichtGpuBuffers.h"
#include "b3IrrlichtQueries.h"   // layout contract only - no link dependency

#include <irrlicht.h>
#include "ComputeBuffer.h"

#include <cmath>
#include <cstddef>
#include <cstring>

// The query API's resident hit buffer is consumed as-is by setRayHitBuffer, and its query layout
// is what buildRaysResident emits, so the two must never drift apart.
static_assert(sizeof(b3IrrActuatorRayHit) == sizeof(b3IrrlichtQueries::b3IrrQueryHit),
			  "b3IrrActuatorRayHit must stay byte-identical to b3IrrQueryHit");
static_assert(offsetof(b3IrrActuatorRayHit, hitNormal) == offsetof(b3IrrlichtQueries::b3IrrQueryHit, normal)
				  && offsetof(b3IrrActuatorRayHit, body) == offsetof(b3IrrlichtQueries::b3IrrQueryHit, bodyIndex)
				  && offsetof(b3IrrActuatorRayHit, hit) == offsetof(b3IrrlichtQueries::b3IrrQueryHit, hit),
			  "b3IrrActuatorRayHit fields must line up with b3IrrQueryHit");
static_assert(sizeof(b3IrrActuatorRay) == sizeof(b3IrrlichtQueries::b3IrrQuery)
				  && offsetof(b3IrrActuatorRay, ownerRoot) == offsetof(b3IrrlichtQueries::b3IrrQuery, ownerRoot)
				  && offsetof(b3IrrActuatorRay, kind) == offsetof(b3IrrlichtQueries::b3IrrQuery, kind),
			  "b3IrrActuatorRay must stay layout-identical to b3IrrQuery");

namespace
{
// Mirrors ActuatorParams in B3ActuatorsBody.hlsli - 32 bytes.
struct ActuatorParams
{
	unsigned int numActuators;
	unsigned int numBodies;
	float deltaTime;
	float useGravityBuffer;
	float uniformGravity[4];
};

struct IrrFloat4
{
	float x, y, z, w;
};

static_assert(sizeof(ActuatorParams) == 32, "ActuatorParams must match the HLSL struct stride");
static_assert(sizeof(IrrFloat4) == 16, "State/InvInertia elements are float4 in HLSL");

const unsigned int kWorkGroup = 64;
const float kBulletMaxSuspensionForce = 6000.f;   // btVehicleTuning default, never set by the drive

using b3IrrGpu::dropBuffer;
using b3IrrGpu::ensureBuffer;

void zeroBuffer(irr::scene::IComputeBuffer* buffer, size_t bytes)
{
	memset(buffer->getBufferPointer(), 0, bytes);
	buffer->setDirty();
}

b3IrrActuator blankRow(int kind, int body, int ownerRoot)
{
	b3IrrActuator a;
	memset(&a, 0, sizeof(a));
	a.kind = kind;
	a.body = body;
	a.ownerRoot = ownerRoot;
	a.flags = B3_IRR_ACTUATOR_FLAG_ENABLED;
	return a;
}
}  // namespace

b3IrrlichtActuators::b3IrrlichtActuators(irr::video::IVideoDriver* driver)
	: m_driver(driver), m_doubleSingle(false), m_buildRaysMaterial(-1), m_bruteForceMaterial(-1),
	  m_applyMaterial(-1), m_foldMaterial(-1), m_paramBuffer(0), m_actuatorBuffer(0), m_stateBuffer(0),
	  m_inertiaBuffer(0), m_externalInertiaBuffer(0), m_gravityBuffer(0), m_ownerBuffer(0), m_rayBuffer(0),
	  m_hitBuffer(0), m_externalHitBuffer(0), m_deltaBuffer(0), m_bodyBuffer(0), m_residentActuators(0),
	  m_ownerCount(0), m_uniformGravity(b3MakeVector3(0.f, 0.f, 0.f))
{
}

b3IrrlichtActuators::~b3IrrlichtActuators()
{
	dropBuffer(m_paramBuffer);
	dropBuffer(m_actuatorBuffer);
	dropBuffer(m_stateBuffer);
	dropBuffer(m_inertiaBuffer);
	dropBuffer(m_ownerBuffer);
	dropBuffer(m_rayBuffer);
	dropBuffer(m_hitBuffer);
	dropBuffer(m_deltaBuffer);
	dropBuffer(m_bodyBuffer);
}

bool b3IrrlichtActuators::init(irr::io::IFileSystem* fileSystem, bool doubleSingle)
{
	m_doubleSingle = doubleSingle;

	if (!m_driver)
		return false;

	irr::video::IGPUProgrammingServices* gpu = m_driver->getGPUProgrammingServices();
	if (!gpu)
		return false;

	// addComputeShaderFromFile hands back a usable id for a missing file; check it ourselves.
	const irr::io::path path = m_doubleSingle ? "media/shaders/B3ActuatorsDS.hlsl" : "media/shaders/B3Actuators.hlsl";
	if (fileSystem && !fileSystem->existFile(path))
		return false;

	m_buildRaysMaterial = gpu->addComputeShaderFromFile(path, "CSBuildActuatorRays", irr::video::ECST_CS_5_0, 0);
	m_bruteForceMaterial = gpu->addComputeShaderFromFile(path, "CSRayHitsBruteForceAabb", irr::video::ECST_CS_5_0, 0);
	m_applyMaterial = gpu->addComputeShaderFromFile(path, "CSApplyActuators", irr::video::ECST_CS_5_0, 0);
	m_foldMaterial = gpu->addComputeShaderFromFile(path, "CSApplyActuatorDeltas", irr::video::ECST_CS_5_0, 0);

	return m_buildRaysMaterial >= 0 && m_bruteForceMaterial >= 0 && m_applyMaterial >= 0 && m_foldMaterial >= 0;
}

bool b3IrrlichtActuators::uploadActuators(const std::vector<b3IrrActuator>& actuators)
{
	if (actuators.empty())
	{
		m_residentActuators = 0;
		return false;
	}

	const irr::u32 count = (irr::u32)actuators.size();
	b3IrrGpu::uploadBuffer<b3IrrActuator>(m_actuatorBuffer, &actuators[0], count);

	// State carries the walker gait phase across steps; only a count change resets it.
	if (count != m_residentActuators || !m_stateBuffer)
	{
		ensureBuffer<IrrFloat4>(m_stateBuffer, count);
		zeroBuffer(m_stateBuffer, count * sizeof(IrrFloat4));
	}
	m_residentActuators = count;
	return true;
}

bool b3IrrlichtActuators::uploadInvInertia(const std::vector<float>& invInertiaDiag, unsigned int numBodies)
{
	if (numBodies == 0)
		return false;

	ensureBuffer<IrrFloat4>(m_inertiaBuffer, numBodies);
	IrrFloat4* dst = (IrrFloat4*)m_inertiaBuffer->getBufferPointer();
	for (irr::u32 i = 0; i < numBodies; ++i)
	{
		const bool haveData = invInertiaDiag.size() >= (size_t)(i + 1) * 4;
		dst[i].x = haveData ? invInertiaDiag[i * 4 + 0] : 0.f;
		dst[i].y = haveData ? invInertiaDiag[i * 4 + 1] : 0.f;
		dst[i].z = haveData ? invInertiaDiag[i * 4 + 2] : 0.f;
		dst[i].w = 0.f;
	}
	m_inertiaBuffer->setDirty();
	return true;
}

bool b3IrrlichtActuators::uploadBodyOwners(const std::vector<int>& owners, unsigned int numBodies)
{
	if (numBodies == 0)
		return false;

	ensureBuffer<int>(m_ownerBuffer, numBodies);
	int* dst = (int*)m_ownerBuffer->getBufferPointer();
	for (irr::u32 i = 0; i < numBodies; ++i)
		dst[i] = i < owners.size() ? owners[i] : (int)i;
	m_ownerBuffer->setDirty();
	m_ownerCount = numBodies;
	return true;
}

void b3IrrlichtActuators::prepare(unsigned int numBodies, float deltaTime)
{
	ensureBuffer<ActuatorParams>(m_paramBuffer, 1);
	ActuatorParams p;
	p.numActuators = m_residentActuators;
	p.numBodies = numBodies;
	p.deltaTime = deltaTime;
	p.useGravityBuffer = m_gravityBuffer ? 1.f : 0.f;
	p.uniformGravity[0] = m_uniformGravity.getX();
	p.uniformGravity[1] = m_uniformGravity.getY();
	p.uniformGravity[2] = m_uniformGravity.getZ();
	p.uniformGravity[3] = 0.f;
	memcpy(m_paramBuffer->getBufferPointer(), &p, sizeof(p));
	m_paramBuffer->setDirty();

	// 6 ints per body; zeroed on (re)creation, then re-zeroed by the fold kernel each step.
	const bool deltaGrew = !m_deltaBuffer || m_deltaBuffer->getStructureCount() < numBodies * 6;
	ensureBuffer<int>(m_deltaBuffer, numBodies * 6);
	if (deltaGrew)
		zeroBuffer(m_deltaBuffer, numBodies * 6 * sizeof(int));

	// Identity owners unless the caller uploaded a hull-tree map.
	if (!m_ownerBuffer || m_ownerCount < numBodies)
		uploadBodyOwners(std::vector<int>(), numBodies);
}

bool b3IrrlichtActuators::buildRaysResident(irr::scene::IComputeBuffer* bodies, unsigned int numBodies)
{
	if (m_buildRaysMaterial < 0 || !bodies || numBodies == 0 || m_residentActuators == 0)
		return false;

	prepare(numBodies, 0.f);
	const irr::u32 rayCount = m_residentActuators * B3_IRR_ACTUATOR_RAYS_PER_ROW;
	ensureBuffer<b3IrrActuatorRay>(m_rayBuffer, rayCount);

	irr::video::SMaterial mat;
	mat.MaterialType = (irr::video::E_MATERIAL_TYPE)m_buildRaysMaterial;
	m_driver->setMaterial(mat);
	m_driver->bindComputeBuffer(0, m_paramBuffer, irr::video::EHBT_SHADER_RESOURCE);
	m_driver->bindComputeBuffer(1, m_actuatorBuffer, irr::video::EHBT_SHADER_RESOURCE);
	if (m_gravityBuffer)
		m_driver->bindComputeBuffer(3, m_gravityBuffer, irr::video::EHBT_SHADER_RESOURCE);
	m_driver->bindComputeBuffer(0, bodies, irr::video::EHBT_COMPUTE);
	m_driver->bindComputeBuffer(4, m_rayBuffer, irr::video::EHBT_COMPUTE);
	m_driver->dispatchComputeShaderBound(
		irr::core::vector3d<irr::u32>((m_residentActuators + kWorkGroup - 1) / kWorkGroup, 1, 1));
	m_driver->unbindComputeResources();

	// Written as a UAV here, read as an SRV by the ray provider.
	m_driver->computeBarrier(m_rayBuffer);
	return true;
}

bool b3IrrlichtActuators::castRaysBruteForceResident(irr::scene::IComputeBuffer* worldAabbs, unsigned int numBodies)
{
	if (m_bruteForceMaterial < 0 || !worldAabbs || numBodies == 0 || !m_rayBuffer || m_residentActuators == 0)
		return false;

	prepare(numBodies, 0.f);
	const irr::u32 rayCount = m_residentActuators * B3_IRR_ACTUATOR_RAYS_PER_ROW;
	ensureBuffer<b3IrrActuatorRayHit>(m_hitBuffer, rayCount);

	irr::video::SMaterial mat;
	mat.MaterialType = (irr::video::E_MATERIAL_TYPE)m_bruteForceMaterial;
	m_driver->setMaterial(mat);
	m_driver->bindComputeBuffer(0, m_paramBuffer, irr::video::EHBT_SHADER_RESOURCE);
	m_driver->bindComputeBuffer(5, worldAabbs, irr::video::EHBT_SHADER_RESOURCE);
	m_driver->bindComputeBuffer(6, m_ownerBuffer, irr::video::EHBT_SHADER_RESOURCE);
	m_driver->bindComputeBuffer(7, m_rayBuffer, irr::video::EHBT_SHADER_RESOURCE);
	m_driver->bindComputeBuffer(5, m_hitBuffer, irr::video::EHBT_COMPUTE);
	m_driver->dispatchComputeShaderBound(irr::core::vector3d<irr::u32>((rayCount + kWorkGroup - 1) / kWorkGroup, 1, 1));
	m_driver->unbindComputeResources();

	// Written as a UAV here, read as an SRV by the apply kernel.
	m_driver->computeBarrier(m_hitBuffer);
	return true;
}

bool b3IrrlichtActuators::applyResident(irr::scene::IComputeBuffer* bodies, unsigned int numBodies,
										float deltaTime, irr::scene::IComputeBuffer* sleepState)
{
	if (m_applyMaterial < 0 || m_foldMaterial < 0 || !bodies || numBodies == 0)
		return false;
	if (m_residentActuators == 0 || !m_actuatorBuffer || !inertiaBuffer())
		return false;

	prepare(numBodies, deltaTime);

	irr::video::SMaterial mat;
	mat.MaterialType = (irr::video::E_MATERIAL_TYPE)m_applyMaterial;
	m_driver->setMaterial(mat);
	m_driver->bindComputeBuffer(0, m_paramBuffer, irr::video::EHBT_SHADER_RESOURCE);
	m_driver->bindComputeBuffer(1, m_actuatorBuffer, irr::video::EHBT_SHADER_RESOURCE);
	m_driver->bindComputeBuffer(2, inertiaBuffer(), irr::video::EHBT_SHADER_RESOURCE);
	if (m_gravityBuffer)
		m_driver->bindComputeBuffer(3, m_gravityBuffer, irr::video::EHBT_SHADER_RESOURCE);
	if (hitBuffer())
		m_driver->bindComputeBuffer(4, hitBuffer(), irr::video::EHBT_SHADER_RESOURCE);
	m_driver->bindComputeBuffer(0, bodies, irr::video::EHBT_COMPUTE);
	m_driver->bindComputeBuffer(1, m_deltaBuffer, irr::video::EHBT_COMPUTE);
	m_driver->bindComputeBuffer(2, m_stateBuffer, irr::video::EHBT_COMPUTE);
	if (sleepState)
		m_driver->bindComputeBuffer(3, sleepState, irr::video::EHBT_COMPUTE);
	m_driver->dispatchComputeShaderBound(
		irr::core::vector3d<irr::u32>((m_residentActuators + kWorkGroup - 1) / kWorkGroup, 1, 1));
	m_driver->unbindComputeResources();

	mat.MaterialType = (irr::video::E_MATERIAL_TYPE)m_foldMaterial;
	m_driver->setMaterial(mat);
	m_driver->bindComputeBuffer(0, m_paramBuffer, irr::video::EHBT_SHADER_RESOURCE);
	m_driver->bindComputeBuffer(0, bodies, irr::video::EHBT_COMPUTE);
	m_driver->bindComputeBuffer(1, m_deltaBuffer, irr::video::EHBT_COMPUTE);
	m_driver->dispatchComputeShaderBound(irr::core::vector3d<irr::u32>((numBodies + kWorkGroup - 1) / kWorkGroup, 1, 1));
	m_driver->unbindComputeResources();

	// Bodies are read as an SRV by whatever phase follows (narrowphase, solver).
	m_driver->computeBarrier(bodies);
	return true;
}

bool b3IrrlichtActuators::stepResident(irr::scene::IComputeBuffer* bodies, unsigned int numBodies,
									   float deltaTime, irr::scene::IComputeBuffer* worldAabbs,
									   irr::scene::IComputeBuffer* sleepState)
{
	if (!buildRaysResident(bodies, numBodies))
		return false;
	if (worldAabbs && !castRaysBruteForceResident(worldAabbs, numBodies))
		return false;
	return applyResident(bodies, numBodies, deltaTime, sleepState);
}

bool b3IrrlichtActuators::applyActuators(std::vector<b3RigidBodyData>& bodies,
										 const std::vector<float>& invInertiaDiag, float deltaTime)
{
	// The std::vector path moves 80-byte bodies; the df64 kernels index a 96-byte stride.
	if (m_doubleSingle || bodies.empty() || m_residentActuators == 0)
		return false;

	const irr::u32 numBodies = (irr::u32)bodies.size();
	b3IrrGpu::uploadBuffer<b3RigidBodyData>(m_bodyBuffer, &bodies[0], numBodies);
	if (!uploadInvInertia(invInertiaDiag, numBodies))
		return false;

	if (!applyResident(m_bodyBuffer, numBodies, deltaTime))
		return false;

	m_bodyBuffer->downloadFromGPU();
	memcpy(&bodies[0], m_bodyBuffer->getBufferPointer(), numBodies * sizeof(b3RigidBodyData));
	return true;
}

bool b3IrrlichtActuators::downloadState(std::vector<b3IrrActuatorState>& out)
{
	if (!m_stateBuffer || m_residentActuators == 0)
		return false;

	m_stateBuffer->downloadFromGPU();
	out.resize(m_residentActuators);
	memcpy(&out[0], m_stateBuffer->getBufferPointer(), m_residentActuators * sizeof(b3IrrActuatorState));
	return true;
}

bool b3IrrlichtActuators::downloadRayHits(std::vector<b3IrrActuatorRayHit>& out)
{
	irr::scene::IComputeBuffer* hits = hitBuffer();
	const unsigned int count = getRayCount();
	if (!hits || count == 0 || hits->getStructureCount() < count)
		return false;

	hits->downloadFromGPU();
	out.resize(count);
	memcpy(&out[0], hits->getBufferPointer(), count * sizeof(b3IrrActuatorRayHit));
	return true;
}

// ---------------------------------------------------------------------------------------------
// Row builders
// ---------------------------------------------------------------------------------------------

b3IrrActuator b3IrrlichtActuators::makeThruster(int body, const float* forceLocal, const float* pointLocal, float throttle)
{
	b3IrrActuator a = blankRow(B3_IRR_ACTUATOR_THRUSTER, body, body);
	for (int i = 0; i < 3; ++i)
	{
		a.thrustLocal[i] = forceLocal ? forceLocal[i] : 0.f;
		a.params0[i] = pointLocal ? pointLocal[i] : 0.f;
	}
	a.params0[3] = throttle;
	return a;
}

b3IrrActuator b3IrrlichtActuators::makeRotationWheel(int body, const float* torqueLocal, float throttle)
{
	b3IrrActuator a = blankRow(B3_IRR_ACTUATOR_ROTATION_WHEEL, body, body);
	for (int i = 0; i < 3; ++i)
		a.torqueLocal[i] = torqueLocal ? torqueLocal[i] : 0.f;
	a.params0[3] = throttle;
	return a;
}

b3IrrActuator b3IrrlichtActuators::makeHover(int body, int ownerRoot, const float* aabbMin, const float* aabbMax,
											 bool walker, float engineForce, float maxSpeed, float rideHeight,
											 float stiffness, float damping, float steerTorque, float gaitPeriodTicks)
{
	b3IrrActuator a = blankRow(walker ? B3_IRR_ACTUATOR_WALKER : B3_IRR_ACTUATOR_HOVER, body, ownerRoot);
	// Outrigger stance off the hull box, rays from the mid-height plane (see GroundVehicle.cpp).
	const float x = (aabbMax[0] - aabbMin[0]) * 0.75f;
	const float z = (aabbMax[2] - aabbMin[2]) * 0.75f;
	const float y = (aabbMin[1] + aabbMax[1]) * 0.5f;
	a.thrustLocal[2] = engineForce;
	a.thrustLocal[3] = maxSpeed;
	a.torqueLocal[0] = x;
	a.torqueLocal[1] = y;
	a.torqueLocal[2] = z;
	a.torqueLocal[3] = (y - aabbMin[1]) + rideHeight;
	a.params0[0] = stiffness;
	a.params0[1] = damping;
	a.params0[2] = steerTorque;
	a.params0[3] = gaitPeriodTicks < 2.f ? 2.f : gaitPeriodTicks;
	return a;
}

void b3IrrlichtActuators::appendWheels(std::vector<b3IrrActuator>& out, int body, int ownerRoot,
									   const float* aabbMin, const float* aabbMax, bool tracked,
									   float engineForce, float maxSpeed, float maxSteerAngle, float wheelRadius,
									   float suspensionRestLength, float suspensionStiffness, float wheelFriction)
{
	const float x = (aabbMax[0] - aabbMin[0]) * 0.75f;
	const float y = (aabbMin[1] + aabbMax[1]) * 0.5f;
	const float z = (aabbMax[2] - aabbMin[2]) * 0.75f;
	const float restFromCenter = suspensionRestLength + (y - aabbMin[1]);
	const float sqrtK = std::sqrt(suspensionStiffness);

	for (int w = 0; w < 4; ++w)
	{
		b3IrrActuator a = blankRow(B3_IRR_ACTUATOR_WHEEL, body, ownerRoot);
		const bool front = w < 2;
		a.flags |= (tracked ? B3_IRR_ACTUATOR_FLAG_TRACKED : 0)
				 | ((front && !tracked) ? B3_IRR_ACTUATOR_FLAG_FRONT_WHEEL : 0)
				 | (4 << B3_IRR_ACTUATOR_FLAG_WHEEL_COUNT_SHIFT);
		a.thrustLocal[2] = engineForce;
		a.thrustLocal[3] = maxSpeed;
		a.torqueLocal[0] = (w & 1) ? x : -x;
		a.torqueLocal[1] = y;
		a.torqueLocal[2] = front ? z : -z;
		a.torqueLocal[3] = restFromCenter;
		a.params0[0] = wheelRadius;
		a.params0[1] = suspensionStiffness;
		a.params0[2] = 1.4f * sqrtK;   // near-critical relaxation damping
		a.params0[3] = 1.2f * sqrtK;   // compression damping
		a.params1[0] = wheelFriction;
		a.params1[1] = maxSteerAngle;
		a.params1[2] = restFromCenter;  // m_maxSuspensionTravelCm = restFromCenter * 100
		a.params1[3] = kBulletMaxSuspensionForce;
		out.push_back(a);
	}
}

b3IrrActuator b3IrrlichtActuators::makeNaval(int body, float engineForce, float maxSpeed, float rudderTorque,
											 float keelDrag, float rightingStrength, float submersion)
{
	b3IrrActuator a = blankRow(B3_IRR_ACTUATOR_NAVAL, body, body);
	a.thrustLocal[2] = engineForce;
	a.thrustLocal[3] = maxSpeed;
	a.torqueLocal[0] = rudderTorque;
	a.torqueLocal[1] = keelDrag;
	a.torqueLocal[2] = rightingStrength;
	a.torqueLocal[3] = submersion;
	return a;
}

void b3IrrlichtActuators::setDriveCommand(b3IrrActuator& row, float throttle, float yaw)
{
	row.thrustLocal[0] = throttle < -1.f ? -1.f : (throttle > 1.f ? 1.f : throttle);
	row.thrustLocal[1] = yaw < -1.f ? -1.f : (yaw > 1.f ? 1.f : yaw);
}
