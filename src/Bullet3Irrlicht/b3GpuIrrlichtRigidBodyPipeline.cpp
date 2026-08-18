#include "b3GpuIrrlichtRigidBodyPipeline.h"
#include "b3IrrlichtGpuBuffers.h"

#include <irrlicht.h>
#include "ComputeBuffer.h"

#include <cstring>

// The HLSL side declares the same fields in the same order; a mismatch here silently shifts
// every field rather than failing to compile.
static_assert(sizeof(b3RigidBodyData) == 80, "b3RigidBodyData must stay 80 bytes to match B3IntegrateTransforms.hlsl");
static_assert(sizeof(b3IrrGpu::b3IrrBodyTransformDS) == 40, "b3IrrBodyTransformDS must match the OS_DS HLSL transform stride");

namespace
{
// Mirrors IntegrateParams in B3IntegrateTransforms.hlsl - 32 bytes, float4 first.
struct IntegrateParams
{
	float gravityAcceleration[4];
	float timeStep;
	float angularDamping;
	float numNodes;
	float pad0;
};

static_assert(sizeof(IntegrateParams) == 32, "IntegrateParams must match the HLSL struct stride");
// 28 bytes: the demo's DemoTransform and B3IntegrateTransforms.hlsl's b3IrrBodyTransform.
static_assert(sizeof(b3GpuIrrlichtRigidBodyPipeline::b3IrrBodyTransform) == 28,
			  "The packed transform must stay 28 bytes to match DemoTransform and the HLSL struct");
}  // namespace

b3GpuIrrlichtRigidBodyPipeline::b3GpuIrrlichtRigidBodyPipeline(irr::video::IVideoDriver* driver)
	: m_driver(driver), m_doubleSingle(false), m_bodyBuffer(0), m_paramBuffer(0), m_transformBuffer(0),
	  m_renderTransformBuffer(0), m_integrateMaterial(-1), m_integratePackMaterial(-1), m_gravity(b3MakeVector3(0.f, -9.8f, 0.f)), m_angularDamping(0.99f)
{
}

b3GpuIrrlichtRigidBodyPipeline::~b3GpuIrrlichtRigidBodyPipeline()
{
	if (m_bodyBuffer)
		m_bodyBuffer->drop();
	if (m_paramBuffer)
		m_paramBuffer->drop();
	b3IrrGpu::dropBuffer(m_transformBuffer);
	b3IrrGpu::dropBuffer(m_renderTransformBuffer);
}

bool b3GpuIrrlichtRigidBodyPipeline::init(irr::io::IFileSystem* fileSystem, bool doubleSingle)
{
	m_doubleSingle = doubleSingle;

	if (!m_driver)
		return false;

	irr::video::IGPUProgrammingServices* gpu = m_driver->getGPUProgrammingServices();
	if (!gpu)
		return false;

	// addComputeShaderFromFile still hands back a usable material id when the file is missing,
	// which silently substitutes an unrelated material - check the file ourselves first.
	const irr::io::path shaderPath = m_doubleSingle ? "media/shaders/B3IntegrateTransformsDS.hlsl" : "media/shaders/B3IntegrateTransforms.hlsl";
	if (fileSystem && !fileSystem->existFile(shaderPath))
		return false;

	m_integrateMaterial = gpu->addComputeShaderFromFile(shaderPath, "CSMain", irr::video::ECST_CS_5_0, 0);
	m_integratePackMaterial = gpu->addComputeShaderFromFile(shaderPath, "CSIntegrateAndPack",
															irr::video::ECST_CS_5_0, 0);

	return m_integrateMaterial >= 0;
}

int b3GpuIrrlichtRigidBodyPipeline::registerRigidBody(const b3Vector3& position, const b3Quaternion& orientation, float mass)
{
	b3RigidBodyData body;
	memset(&body, 0, sizeof(body));

	body.m_pos = b3MakeVector3(position.getX(), position.getY(), position.getZ());
	body.m_quat = orientation;
	body.m_linVel = b3MakeVector3(0.f, 0.f, 0.f);
	body.m_angVel = b3MakeVector3(0.f, 0.f, 0.f);
	body.m_collidableIdx = -1;
	body.m_invMass = (mass > 0.f) ? 1.f / mass : 0.f;
	body.m_restituitionCoeff = 0.f;
	body.m_frictionCoeff = 0.3f;

	m_cpuBodies.push_back(body);
	return (int)m_cpuBodies.size() - 1;
}

void b3GpuIrrlichtRigidBodyPipeline::setLinearVelocity(int bodyIndex, const b3Vector3& velocity)
{
	m_cpuBodies[bodyIndex].m_linVel = b3MakeVector3(velocity.getX(), velocity.getY(), velocity.getZ());
}

void b3GpuIrrlichtRigidBodyPipeline::setAngularVelocity(int bodyIndex, const b3Vector3& velocity)
{
	m_cpuBodies[bodyIndex].m_angVel = b3MakeVector3(velocity.getX(), velocity.getY(), velocity.getZ());
}

void b3GpuIrrlichtRigidBodyPipeline::writeBodiesToGpu()
{
	// 80-byte bodies; a double-single instance indexes a 96-byte stride and owns no bodies here.
	if (m_doubleSingle)
		return;

	if (m_cpuBodies.empty())
		return;

	if (!m_bodyBuffer)
	{
		irr::scene::ComputeBuffer<b3RigidBodyData>* buffer = new irr::scene::ComputeBuffer<b3RigidBodyData>();
		buffer->setHardwareMappingHint(irr::scene::EHM_DYNAMIC);
		m_bodyBuffer = buffer;
	}

	m_bodyBuffer->set_used((irr::u32)m_cpuBodies.size());
	memcpy(m_bodyBuffer->getBufferPointer(), &m_cpuBodies[0], m_cpuBodies.size() * sizeof(b3RigidBodyData));
	m_bodyBuffer->setDirty();
}

void b3GpuIrrlichtRigidBodyPipeline::stepSimulation(float deltaTime)
{
	if (m_integrateMaterial < 0 || !m_bodyBuffer || m_cpuBodies.empty())
		return;

	if (!m_paramBuffer)
	{
		irr::scene::ComputeBuffer<IntegrateParams>* buffer = new irr::scene::ComputeBuffer<IntegrateParams>();
		buffer->setHardwareMappingHint(irr::scene::EHM_DYNAMIC);
		buffer->set_used(1);
		m_paramBuffer = buffer;
	}

	IntegrateParams params;
	params.gravityAcceleration[0] = m_gravity.getX();
	params.gravityAcceleration[1] = m_gravity.getY();
	params.gravityAcceleration[2] = m_gravity.getZ();
	params.gravityAcceleration[3] = 0.f;
	params.timeStep = deltaTime;
	params.angularDamping = m_angularDamping;
	params.numNodes = (float)m_cpuBodies.size();
	params.pad0 = 0.f;

	memcpy(m_paramBuffer->getBufferPointer(), &params, sizeof(params));
	m_paramBuffer->setDirty();

	irr::video::SMaterial mat;
	mat.MaterialType = (irr::video::E_MATERIAL_TYPE)m_integrateMaterial;
	m_driver->setMaterial(mat);

	m_driver->bindComputeBuffer(0, m_paramBuffer, irr::video::EHBT_SHADER_RESOURCE);
	m_driver->bindComputeBuffer(0, m_bodyBuffer, irr::video::EHBT_COMPUTE);

	// numthreads(64,1,1) in B3IntegrateTransforms.hlsl - the shader guards the tail.
	const irr::u32 groups = ((irr::u32)m_cpuBodies.size() + 63) / 64;
	m_driver->dispatchComputeShaderBound(irr::core::vector3d<irr::u32>(groups, 1, 1));

	m_driver->unbindComputeResources();
}

void b3GpuIrrlichtRigidBodyPipeline::readBodiesFromGpu()
{
	if (!m_bodyBuffer || m_cpuBodies.empty())
		return;

	m_bodyBuffer->downloadFromGPU();
	memcpy(&m_cpuBodies[0], m_bodyBuffer->getBufferPointer(), m_cpuBodies.size() * sizeof(b3RigidBodyData));
}

bool b3GpuIrrlichtRigidBodyPipeline::integrateResident(irr::scene::IComputeBuffer* bodies,
													   unsigned int numBodies, float deltaTime,
													   irr::scene::IComputeBuffer* sleepState)
{
	return dispatchIntegrate(m_integrateMaterial, bodies, numBodies, deltaTime, false, sleepState);
}

bool b3GpuIrrlichtRigidBodyPipeline::integrateAndPackResident(irr::scene::IComputeBuffer* bodies,
															  unsigned int numBodies, float deltaTime,
															  irr::scene::IComputeBuffer* sleepState)
{
	return dispatchIntegrate(m_integratePackMaterial, bodies, numBodies, deltaTime, true, sleepState);
}

bool b3GpuIrrlichtRigidBodyPipeline::dispatchIntegrate(int material,
													   irr::scene::IComputeBuffer* bodies,
													   unsigned int numBodies, float deltaTime,
													   bool packTransforms,
													   irr::scene::IComputeBuffer* sleepState)
{
	if (material < 0 || !bodies || numBodies == 0)
		return false;

	b3IrrGpu::ensureBuffer<IntegrateParams>(m_paramBuffer, 1);
	if (packTransforms)
	{
		if (m_doubleSingle)
		{
			b3IrrGpu::ensureBuffer<b3IrrGpu::b3IrrBodyTransformDS>(m_transformBuffer, numBodies);
			b3IrrGpu::ensureBuffer<b3IrrBodyTransform>(m_renderTransformBuffer, numBodies);
		}
		else
			b3IrrGpu::ensureBuffer<b3IrrBodyTransform>(m_transformBuffer, numBodies);
	}

	IntegrateParams params;
	params.gravityAcceleration[0] = m_gravity.getX();
	params.gravityAcceleration[1] = m_gravity.getY();
	params.gravityAcceleration[2] = m_gravity.getZ();
	params.gravityAcceleration[3] = 0.f;
	params.timeStep = deltaTime;
	params.angularDamping = m_angularDamping;
	params.numNodes = (float)numBodies;
	params.pad0 = 0.f;
	memcpy(m_paramBuffer->getBufferPointer(), &params, sizeof(params));
	m_paramBuffer->setDirty();

	irr::video::SMaterial mat;
	mat.MaterialType = (irr::video::E_MATERIAL_TYPE)material;
	m_driver->setMaterial(mat);
	m_driver->bindComputeBuffer(0, m_paramBuffer, irr::video::EHBT_SHADER_RESOURCE);
	if (sleepState)
		m_driver->bindComputeBuffer(1, sleepState, irr::video::EHBT_SHADER_RESOURCE);
	m_driver->bindComputeBuffer(0, bodies, irr::video::EHBT_COMPUTE);
	if (packTransforms)
	{
		m_driver->bindComputeBuffer(1, m_transformBuffer, irr::video::EHBT_COMPUTE);
		if (m_doubleSingle)
			m_driver->bindComputeBuffer(2, m_renderTransformBuffer, irr::video::EHBT_COMPUTE);
	}
	m_driver->dispatchComputeShaderBound(irr::core::vector3d<irr::u32>((numBodies + 63) / 64, 1, 1));
	m_driver->unbindComputeResources();

	// Written as a UAV here, read as an SRV by the next step's first phase.
	m_driver->computeBarrier(bodies);
	return true;
}

bool b3GpuIrrlichtRigidBodyPipeline::downloadTransformsDS(
	std::vector<b3IrrGpu::b3IrrBodyTransformDS>& out, unsigned int numBodies)
{
	if (!m_doubleSingle || !m_transformBuffer || numBodies == 0 ||
		m_transformBuffer->getStructureCount() < numBodies)
		return false;

	m_transformBuffer->downloadFromGPU();
	out.resize(numBodies);
	memcpy(&out[0], m_transformBuffer->getBufferPointer(),
		   numBodies * sizeof(b3IrrGpu::b3IrrBodyTransformDS));
	return true;
}

bool b3GpuIrrlichtRigidBodyPipeline::downloadTransforms(std::vector<b3IrrBodyTransform>& out,
														unsigned int numBodies)
{
	if (m_doubleSingle || !m_transformBuffer || numBodies == 0 ||
		m_transformBuffer->getStructureCount() < numBodies)
		return false;

	m_transformBuffer->downloadFromGPU();
	out.resize(numBodies);
	memcpy(&out[0], m_transformBuffer->getBufferPointer(), numBodies * sizeof(b3IrrBodyTransform));
	return true;
}
