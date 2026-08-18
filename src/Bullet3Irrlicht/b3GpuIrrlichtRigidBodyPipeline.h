#ifndef B3_GPU_IRRLICHT_RIGIDBODY_PIPELINE_H
#define B3_GPU_IRRLICHT_RIGIDBODY_PIPELINE_H

#include "Bullet3Collision/NarrowPhaseCollision/shared/b3RigidBodyData.h"
#include "Bullet3Common/b3Vector3.h"
#include "Bullet3Common/b3Quaternion.h"
#include "b3IrrlichtGpuBuffers.h"

#include <vector>

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

/// Irrlicht-compute analog of b3GpuRigidBodyPipeline. Everything GPU-side goes through
/// IVideoDriver, never a raw D3D11/D3D12 call - see the migration plan's cross-cutting rules.
class b3GpuIrrlichtRigidBodyPipeline
{
public:
	b3GpuIrrlichtRigidBodyPipeline(irr::video::IVideoDriver* driver);
	virtual ~b3GpuIrrlichtRigidBodyPipeline();

	/// Compiles the kernels. False means no compute pipeline is available, or the shader
	/// file is missing. Pass the device's filesystem so a missing file is caught, not silently
	/// substituted by an unrelated material.
	bool init(irr::io::IFileSystem* fileSystem = 0, bool doubleSingle = false);

	/// Whether this instance runs the emulated-double integrate kernel. Its own CPU-side body list
	/// stays f32-only, so the df64 path is the resident one plus downloadTransformsDS.
	bool isDoubleSingle() const { return m_doubleSingle; }

	/// mass 0 makes the body static (invMass 0), matching Bullet's convention.
	int registerRigidBody(const b3Vector3& position, const b3Quaternion& orientation, float mass);

	void setGravity(const b3Vector3& gravity) { m_gravity = gravity; }
	void setAngularDamping(float damping) { m_angularDamping = damping; }

	void setLinearVelocity(int bodyIndex, const b3Vector3& velocity);
	void setAngularVelocity(int bodyIndex, const b3Vector3& velocity);

	/// Uploads whatever registerRigidBody/setVelocity built. Call once before stepping.
	void writeBodiesToGpu();

	/// One integration step, entirely GPU-side - no readback.
	void stepSimulation(float deltaTime);

	/// Stalls until the GPU is done; only for validation and debug drawing.
	void readBodiesFromGpu();

	/// Renderer-facing transform: 28 bytes, matching the demo's DemoTransform element for element.
	struct b3IrrBodyTransform
	{
		float position[3];
		float orientation[4];   // xyzw
	};

	/**
	 * @brief One dispatch: gravity, integration, and the renderer's compact copy.
	 *
	 * Gravity is folded into the integrate kernel and the packed transforms come from the same
	 * registers, so a step ends with one buffer the renderer takes in a single memcpy.
	 *
	 * @param bodies Device-resident bodies, updated in place.
	 * @param numBodies Bodies in that buffer.
	 * @param deltaTime Timestep.
	 * @param sleepState Per-body sleep state from b3IrrlichtSleep; a sleeping body skips
	 *        integration but still publishes its transform. 0 reads as all-awake.
	 * @return False if the fused kernel is unavailable.
	 */
	bool integrateAndPackResident(irr::scene::IComputeBuffer* bodies, unsigned int numBodies,
								  float deltaTime, irr::scene::IComputeBuffer* sleepState = 0);

	/**
	 * @brief Integrates a caller-owned body buffer without producing renderer output.
	 * @param bodies Device-resident bodies, updated in place.
	 * @param numBodies Bodies in that buffer.
	 * @param deltaTime Timestep.
	 * @param sleepState Per-body sleep state; sleeping bodies are skipped. 0 reads as all-awake.
	 * @return False if the kernel is unavailable.
	 */
	bool integrateResident(irr::scene::IComputeBuffer* bodies, unsigned int numBodies,
						   float deltaTime, irr::scene::IComputeBuffer* sleepState = 0);

	/**
	 * @brief The one readback a step still owes the renderer: 28 bytes per body, not 80.
	 * @param out Receives position and orientation per body, in body order; copyable straight
	 *        into a DemoTransform array.
	 * @param numBodies Bodies to read.
	 * @return False if integrateAndPackResident has not run.
	 */
	bool downloadTransforms(std::vector<b3IrrBodyTransform>& out, unsigned int numBodies);

	/**
	 * @brief The df64 form of downloadTransforms: hi and lo halves per axis.
	 * @param out Receives position hi/lo and orientation per body, in body order.
	 * @param numBodies Bodies to read.
	 * @return False if this instance is not double-single, or the pack kernel has not run.
	 */
	bool downloadTransformsDS(std::vector<b3IrrGpu::b3IrrBodyTransformDS>& out, unsigned int numBodies);

	/// Packed transforms from the last integrateAndPackResident call.
	irr::scene::IComputeBuffer* getTransformBuffer() const { return m_transformBuffer; }

	/// The 28-byte transforms a renderer's cull pass reads; df64 emits them next to its hi/lo pair.
	irr::scene::IComputeBuffer* getRenderTransformBuffer() const
	{
		return m_doubleSingle ? m_renderTransformBuffer : m_transformBuffer;
	}

	/// The pipeline's own body buffer, so a resident chain can share it.
	irr::scene::IComputeBuffer* getBodyBuffer() const { return m_bodyBuffer; }

	int getNumBodies() const { return (int)m_cpuBodies.size(); }
	const b3RigidBodyData& getBody(int index) const { return m_cpuBodies[index]; }

private:
	b3GpuIrrlichtRigidBodyPipeline(const b3GpuIrrlichtRigidBodyPipeline&);
	b3GpuIrrlichtRigidBodyPipeline& operator=(const b3GpuIrrlichtRigidBodyPipeline&);

	/**
	 * @brief Shared body of the two resident integrate entry points.
	 * @param material Kernel to run.
	 * @param bodies Device-resident bodies.
	 * @param numBodies Bodies in that buffer.
	 * @param deltaTime Timestep.
	 * @param packTransforms Whether to also bind and fill the renderer's transform buffer.
	 * @param sleepState Per-body sleep state, or 0.
	 * @return False if the kernel is unavailable or an argument is missing.
	 */
	bool dispatchIntegrate(int material, irr::scene::IComputeBuffer* bodies, unsigned int numBodies,
						   float deltaTime, bool packTransforms,
						   irr::scene::IComputeBuffer* sleepState);

	irr::video::IVideoDriver* m_driver;
	bool m_doubleSingle;
	irr::scene::IComputeBuffer* m_bodyBuffer;
	irr::scene::IComputeBuffer* m_paramBuffer;
	irr::scene::IComputeBuffer* m_transformBuffer;
	irr::scene::IComputeBuffer* m_renderTransformBuffer;
	int m_integrateMaterial;
	int m_integratePackMaterial;

	std::vector<b3RigidBodyData> m_cpuBodies;
	b3Vector3 m_gravity;
	float m_angularDamping;
};

#endif  //B3_GPU_IRRLICHT_RIGIDBODY_PIPELINE_H
