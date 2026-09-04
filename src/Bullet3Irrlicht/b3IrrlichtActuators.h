#ifndef B3_IRRLICHT_ACTUATORS_H
#define B3_IRRLICHT_ACTUATORS_H

#include "Bullet3Collision/NarrowPhaseCollision/shared/b3RigidBodyData.h"
#include "Bullet3Common/b3Vector3.h"
#include "b3IrrActuator.h"

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

/**
 * @brief GPU actuators over a resident body buffer: thrusters, rotation wheels, hover/walker
 *        support rays, raycast-vehicle wheels, naval drives (B3Actuators.hlsl).
 *
 * Runs BEFORE the contact solve: build rays -> cast -> apply (fixed-point accumulate) -> fold.
 * Casting is an interim brute-force AABB provider; setRayHitBuffer() is the seam for the query API.
 */
class b3IrrlichtActuators
{
public:
	b3IrrlichtActuators(irr::video::IVideoDriver* driver);
	~b3IrrlichtActuators();

	/**
	 * @brief Compiles the kernels.
	 * @param fileSystem Device filesystem; without it a missing shader binds an unrelated material.
	 * @param doubleSingle Compile the emulated-double (df64) kernel set instead of the f32 one.
	 * @return False if a required shader is missing or failed to compile.
	 */
	bool init(irr::io::IFileSystem* fileSystem = 0, bool doubleSingle = false);

	/// Whether this instance runs the emulated-double kernels; the std::vector entry point refuses.
	bool isDoubleSingle() const { return m_doubleSingle; }

	/**
	 * @brief Uploads the actuator rows. Call whenever a setpoint changes (typically every step).
	 *
	 * Per-row state (gait phase, diagnostics) survives a re-upload of the same row count and is
	 * zeroed when the count changes.
	 *
	 * @param actuators Rows; body/ownerRoot are indices into the resident body buffer.
	 * @return False if there is nothing to upload.
	 */
	bool uploadActuators(const std::vector<b3IrrActuator>& actuators);

	/**
	 * @brief Uploads the per-body inverse inertia the apply kernel reads (own buffer).
	 * @param invInertiaDiag Diagonal inverse inertia per body, 4 floats each (w unused).
	 * @param numBodies Bodies the buffer must cover.
	 * @return False if the buffer could not be sized.
	 */
	bool uploadInvInertia(const std::vector<float>& invInertiaDiag, unsigned int numBodies);

	/// Shares a caller-owned float4-per-body inverse inertia buffer instead (e.g. the solver's).
	void setInvInertiaBuffer(irr::scene::IComputeBuffer* invInertia) { m_externalInertiaBuffer = invInertia; }

	/// Per-body gravity acceleration (float4 per body), the same buffer the integrator consumes.
	/// 0 falls back to setUniformGravity. Never computed here - one implementation rule.
	void setGravityBuffer(irr::scene::IComputeBuffer* gravityAccel) { m_gravityBuffer = gravityAccel; }
	void setUniformGravity(const b3Vector3& gravity) { m_uniformGravity = gravity; }

	/**
	 * @brief Uploads each body's hull root index; rays from an actuator skip every body sharing
	 *        its ownerRoot. Defaults to identity (every body its own root) when never called.
	 * @param owners Root body index per body.
	 * @param numBodies Bodies the buffer must cover; entries past owners.size() are their own root.
	 * @return False if the buffer could not be sized.
	 */
	bool uploadBodyOwners(const std::vector<int>& owners, unsigned int numBodies);

	/**
	 * @brief Writes this step's support/wheel rays from the current body poses.
	 * @param bodies Device-resident bodies.
	 * @param numBodies Bodies in that buffer.
	 * @return False if the kernel is unavailable or no actuators are uploaded.
	 */
	bool buildRaysResident(irr::scene::IComputeBuffer* bodies, unsigned int numBodies);

	/// Rays from the last buildRaysResident, B3_IRR_ACTUATOR_RAYS_PER_ROW per row (b3IrrActuatorRay).
	irr::scene::IComputeBuffer* getRayBuffer() const { return m_rayBuffer; }
	unsigned int getRayCount() const { return m_residentActuators * B3_IRR_ACTUATOR_RAYS_PER_ROW; }

	/**
	 * @brief INTERIM ray provider: closest world-AABB hit per ray, owner-filtered, brute force.
	 *        Exact for axis-aligned static boxes; O(rays x bodies). Replace with the query API.
	 * @param worldAabbs b3Aabb per body from b3IrrlichtNarrowphase::computeWorldAabbsResident.
	 * @param numBodies Bodies in that buffer.
	 * @return False if the kernel is unavailable or buildRaysResident never ran.
	 */
	bool castRaysBruteForceResident(irr::scene::IComputeBuffer* worldAabbs, unsigned int numBodies);

	/// Hit results from an external provider, b3IrrActuatorRayHit per ray slot; 0 restores the
	/// internal buffer. The slot order is getRayBuffer()'s.
	void setRayHitBuffer(irr::scene::IComputeBuffer* hits) { m_externalHitBuffer = hits; }

	/**
	 * @brief Applies every row and folds the accumulated deltas into the bodies. Run BEFORE the
	 *        contact solve so the impulses are resolved against this step's contacts.
	 * @param bodies Device-resident bodies; velocities are updated in place.
	 * @param numBodies Bodies in that buffer.
	 * @param deltaTime Timestep the impulses are scaled against.
	 * @param sleepState Per-body sleep bits; an actuated body is woken. 0 leaves it unbound.
	 * @return False if the kernels are unavailable or nothing is uploaded.
	 */
	bool applyResident(irr::scene::IComputeBuffer* bodies, unsigned int numBodies, float deltaTime,
					   irr::scene::IComputeBuffer* sleepState = 0);

	/**
	 * @brief buildRays -> brute-force cast (when worldAabbs is given) -> apply, in one call.
	 * @param bodies Device-resident bodies.
	 * @param numBodies Bodies in that buffer.
	 * @param deltaTime Timestep.
	 * @param worldAabbs World AABBs for the interim provider; 0 skips casting (external hits or none).
	 * @param sleepState Per-body sleep bits, or 0.
	 * @return False if any stage failed.
	 */
	bool stepResident(irr::scene::IComputeBuffer* bodies, unsigned int numBodies, float deltaTime,
					  irr::scene::IComputeBuffer* worldAabbs = 0, irr::scene::IComputeBuffer* sleepState = 0);

	/**
	 * @brief std::vector form for tests (f32 only): uploads the bodies, runs one apply, reads back.
	 * @param bodies Bodies; velocities are updated in place.
	 * @param invInertiaDiag Diagonal inverse inertia per body, 4 floats each.
	 * @param deltaTime Timestep.
	 * @return False if the kernels are unavailable or no actuators are uploaded.
	 */
	bool applyActuators(std::vector<b3RigidBodyData>& bodies, const std::vector<float>& invInertiaDiag,
						float deltaTime);

	/// Rows currently uploaded.
	unsigned int getResidentActuatorCount() const { return m_residentActuators; }

	/**
	 * @brief Reads back the per-row state/diagnostics (contacts, forward speed, gait phase).
	 * @param out Receives one entry per row.
	 * @return False if nothing has run yet.
	 */
	bool downloadState(std::vector<b3IrrActuatorState>& out);

	/**
	 * @brief Reads back the last ray hits, getRayCount() entries.
	 * @param out Receives one entry per ray slot.
	 * @return False if no cast has run.
	 */
	bool downloadRayHits(std::vector<b3IrrActuatorRayHit>& out);

	// --- row builders, mirroring GroundVehicle.cpp's constructors ---

	/**
	 * @brief Thruster row: local-frame force at a local point on its own body.
	 * @param body Body index.
	 * @param forceLocal Force (N) in the body frame, 3 floats.
	 * @param pointLocal Application point in the body frame, 3 floats; 0 = centre of mass.
	 * @param throttle [0,1] scale.
	 * @return An enabled row.
	 */
	static b3IrrActuator makeThruster(int body, const float* forceLocal, const float* pointLocal, float throttle);

	/**
	 * @brief Rotation-wheel row: local-frame torque on its own body.
	 * @param body Body index.
	 * @param torqueLocal Torque (N*m) in the body frame, 3 floats.
	 * @param throttle [-1,1] scale.
	 * @return An enabled row.
	 */
	static b3IrrActuator makeRotationWheel(int body, const float* torqueLocal, float throttle);

	/**
	 * @brief Hover or walker row from the hull's local AABB, as HoverWalkerDrive's constructor.
	 * @param body Hull body index.
	 * @param ownerRoot Hull root index the support rays skip.
	 * @param aabbMin Hull local AABB min, 3 floats.
	 * @param aabbMax Hull local AABB max, 3 floats.
	 * @param walker True for the gait-modulated walker.
	 * @param engineForce N.
	 * @param maxSpeed m/s.
	 * @param rideHeight m above the ground for the hull's bottom.
	 * @param stiffness Spring, per kg.
	 * @param damping Damper, per kg.
	 * @param steerTorque N*m.
	 * @param gaitPeriodTicks Steps per gait cycle (walker).
	 * @return An enabled row with throttle/yaw 0.
	 */
	static b3IrrActuator makeHover(int body, int ownerRoot, const float* aabbMin, const float* aabbMax,
								   bool walker, float engineForce, float maxSpeed, float rideHeight,
								   float stiffness, float damping, float steerTorque, float gaitPeriodTicks);

	/**
	 * @brief Appends the 4 wheel rows of a raycast vehicle, as RaycastVehicleDrive's constructor.
	 * @param out Receives 4 rows (front-left, front-right, rear-left, rear-right).
	 * @param body Chassis body index.
	 * @param ownerRoot Hull root index the wheel rays skip.
	 * @param aabbMin Chassis local AABB min, 3 floats.
	 * @param aabbMax Chassis local AABB max, 3 floats.
	 * @param tracked True for skid steer (all wheels drive, none steer).
	 * @param engineForce N, whole vehicle.
	 * @param maxSpeed m/s.
	 * @param maxSteerAngle rad.
	 * @param wheelRadius m.
	 * @param suspensionRestLength m, below the chassis bottom.
	 * @param suspensionStiffness Bullet tuning (per kg of chassis).
	 * @param wheelFriction Bullet frictionSlip.
	 */
	static void appendWheels(std::vector<b3IrrActuator>& out, int body, int ownerRoot,
							 const float* aabbMin, const float* aabbMax, bool tracked,
							 float engineForce, float maxSpeed, float maxSteerAngle, float wheelRadius,
							 float suspensionRestLength, float suspensionStiffness, float wheelFriction);

	/**
	 * @brief Naval row.
	 * @param body Hull body index.
	 * @param engineForce N.
	 * @param maxSpeed m/s.
	 * @param rudderTorque N*m.
	 * @param keelDrag Lateral drag per kg.
	 * @param rightingStrength Righting per kg.
	 * @param submersion Wet fraction [0,1]; every term is gated on it.
	 * @return An enabled row with throttle/yaw 0.
	 */
	static b3IrrActuator makeNaval(int body, float engineForce, float maxSpeed, float rudderTorque,
								   float keelDrag, float rightingStrength, float submersion);

	/**
	 * @brief Writes the drive setpoints shared by every vehicle kind into a row.
	 * @param row Row to edit (hover/walker/wheel/naval).
	 * @param throttle [-1,1].
	 * @param yaw [-1,1], positive = turn right (+Y yaw).
	 */
	static void setDriveCommand(b3IrrActuator& row, float throttle, float yaw);

private:
	b3IrrlichtActuators(const b3IrrlichtActuators&);
	b3IrrlichtActuators& operator=(const b3IrrlichtActuators&);

	/**
	 * @brief Fills the params block and sizes the per-body scratch buffers.
	 * @param numBodies Bodies in the resident buffer.
	 * @param deltaTime Timestep.
	 */
	void prepare(unsigned int numBodies, float deltaTime);

	irr::scene::IComputeBuffer* gravityBuffer() const { return m_gravityBuffer; }
	irr::scene::IComputeBuffer* inertiaBuffer() const
	{
		return m_externalInertiaBuffer ? m_externalInertiaBuffer : m_inertiaBuffer;
	}
	irr::scene::IComputeBuffer* hitBuffer() const
	{
		return m_externalHitBuffer ? m_externalHitBuffer : m_hitBuffer;
	}

	irr::video::IVideoDriver* m_driver;
	bool m_doubleSingle;

	int m_buildRaysMaterial;
	int m_bruteForceMaterial;
	int m_applyMaterial;
	int m_foldMaterial;

	irr::scene::IComputeBuffer* m_paramBuffer;
	irr::scene::IComputeBuffer* m_actuatorBuffer;
	irr::scene::IComputeBuffer* m_stateBuffer;
	irr::scene::IComputeBuffer* m_inertiaBuffer;
	irr::scene::IComputeBuffer* m_externalInertiaBuffer;
	irr::scene::IComputeBuffer* m_gravityBuffer;
	irr::scene::IComputeBuffer* m_ownerBuffer;
	irr::scene::IComputeBuffer* m_rayBuffer;
	irr::scene::IComputeBuffer* m_hitBuffer;
	irr::scene::IComputeBuffer* m_externalHitBuffer;
	irr::scene::IComputeBuffer* m_deltaBuffer;
	irr::scene::IComputeBuffer* m_bodyBuffer;   // std::vector path only

	unsigned int m_residentActuators;
	unsigned int m_ownerCount;
	b3Vector3 m_uniformGravity;
};

#endif  //B3_IRRLICHT_ACTUATORS_H
