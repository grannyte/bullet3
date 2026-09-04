#ifndef B3_IRR_ACTUATOR_H
#define B3_IRR_ACTUATOR_H

/// Actuator kinds consumed by B3ActuatorsBody.hlsli. One row = one actuator; several rows may push
/// the same body - accumulation is fixed-point atomic, so their order never matters.
enum b3IrrActuatorKind
{
	B3_IRR_ACTUATOR_THRUSTER = 0,       // local-frame force at a local point on its OWN body
	B3_IRR_ACTUATOR_ROTATION_WHEEL = 1, // local-frame torque on its own body
	B3_IRR_ACTUATOR_HOVER = 2,          // 4 corner support rays, spring-damper, righting, thrust
	B3_IRR_ACTUATOR_WALKER = 3,         // HOVER plus the alternating-diagonal gait
	B3_IRR_ACTUATOR_WHEEL = 4,          // ONE raycast-vehicle wheel; a chassis owns one row per wheel
	B3_IRR_ACTUATOR_NAVAL = 5           // propeller/rudder/keel/righting, gated on submersion
};

enum b3IrrActuatorFlags
{
	B3_IRR_ACTUATOR_FLAG_ENABLED = 1,        // a disabled row is a no-op (Hit_point <= 0 etc.)
	B3_IRR_ACTUATOR_FLAG_TRACKED = 2,        // WHEEL: skid steer (all wheels drive, no front steer)
	B3_IRR_ACTUATOR_FLAG_FRONT_WHEEL = 4,    // WHEEL: steers (wheeled only)
	B3_IRR_ACTUATOR_FLAG_WHEEL_COUNT_SHIFT = 8,  // WHEEL: bits 8..15 = wheels on this chassis
	B3_IRR_ACTUATOR_FLAG_WHEEL_COUNT_MASK = 0xFF00
};

/**
 * @brief One GPU actuator row, 80 bytes, matching B3ActuatorsBody.hlsli element for element.
 *
 * body/ownerRoot are BODY INDICES into the resident buffer (resolved at upload), not ObjectIDs;
 * ownerRoot is the hull root every support/wheel ray skips (standalone body = its own root).
 *
 * Per-kind packing, the layout contract ECS::ActuatorComponent packs to (unused fields must be 0):
 *
 * THRUSTER
 *   thrustLocal.xyz  force (N) in the body's local frame
 *   torqueLocal.xyz  extra local torque (N*m), usually 0
 *   params0.xyz      application point, local frame (m); 0 = centre of mass
 *   params0.w        throttle [0,1] scaling force and torque
 *
 * ROTATION_WHEEL
 *   torqueLocal.xyz  torque (N*m) in the body's local frame
 *   params0.w        throttle [-1,1]
 *
 * HOVER / WALKER   (HoverWalkerDrive::updateAction)
 *   thrustLocal      x throttle [-1,1], y yaw [-1,1], z engineForce (N), w maxSpeed (m/s)
 *   torqueLocal      x cornerX, y castPlaneY (mid-height cast plane), z cornerZ, w targetDistance
 *   params0          x stiffness, y damping, z steerTorque (N*m), w gaitPeriodTicks (>= 2, walker)
 *   params1          unused
 *   rays             4/row: corners (-x,y,z) (x,y,z) (-x,y,-z) (x,y,-z) along -up, len 2*targetDistance
 *
 * WHEEL   (btRaycastVehicle::updateVehicle + RaycastVehicleDrive governors; one row PER WHEEL)
 *   thrustLocal      x throttle [-1,1], y yaw [-1,1], z engineForce (N, whole vehicle), w maxSpeed (m/s)
 *   torqueLocal      xyz chassis connection point (local, mid-height plane), w suspensionRestLength (m, from that plane)
 *   params0          x wheelRadius, y suspensionStiffness, z dampingRelaxation, w dampingCompression
 *   params1          x frictionSlip, y maxSteerAngle (rad), z maxSuspensionTravel (m), w maxSuspensionForce (N)
 *   flags            TRACKED, FRONT_WHEEL, wheel count in bits 8..15
 *   rays             1 per row: from the connection point along local -Y, length rest + radius
 *
 * NAVAL   (NavalDrive::updateAction)
 *   thrustLocal      x throttle [-1,1], y yaw [-1,1], z engineForce (N), w maxSpeed (m/s)
 *   torqueLocal      x rudderTorque (N*m), y keelDrag, z rightingStrength, w submersion [0,1]
 *
 * Per-row state/diagnostics live in a separate float4 buffer (b3IrrActuatorState), so this row
 * stays pure input that the ECS column can overwrite every step.
 */
struct b3IrrActuator
{
	int kind;
	int body;
	int ownerRoot;
	int flags;
	float thrustLocal[4];
	float torqueLocal[4];
	float params0[4];
	float params1[4];
};

static_assert(sizeof(b3IrrActuator) == 80, "b3IrrActuator must match the HLSL struct stride (80 bytes)");

/// Kernel-owned per-row state and diagnostics, 16 bytes: x contacts/wet, y forward speed (m/s),
/// z gait phase (walker, advanced by the kernel), w unused.
struct b3IrrActuatorState
{
	float contacts;
	float forwardSpeed;
	float gaitPhase;
	float pad0;
};

static_assert(sizeof(b3IrrActuatorState) == 16, "b3IrrActuatorState must stay one float4");

/// Ray a support/wheel actuator wants cast, 48 bytes, laid out as b3IrrlichtQueries::b3IrrQuery
/// (radius 0, kind QUERY_RAY) so a resident submit consumes it as-is; inactive = degenerate ray.
struct b3IrrActuatorRay
{
	float from[4];
	float to[4];
	float radius;
	int ownerRoot;        // rejected by the ray provider (the hull's whole welded tree)
	unsigned int kind;    // 0 = ray
	unsigned int active;  // 0 = slot unused this step
};

static_assert(sizeof(b3IrrActuatorRay) == 48, "b3IrrActuatorRay must match the HLSL struct stride");

/// Closest hit per ray slot, 48 bytes, byte-identical to b3IrrlichtQueries::b3IrrQueryHit so
/// setRayHitBuffer(queries.getHitBuffer()) needs no conversion. hit == 0 means miss.
struct b3IrrActuatorRayHit
{
	float hitPoint[4];    // xyz world, w = fraction along from->to (1 on a miss)
	float hitNormal[4];   // xyz unit, w unused
	int body;             // -1 on a miss
	int hit;
	int pad0;
	int pad1;
};

static_assert(sizeof(b3IrrActuatorRayHit) == 48, "b3IrrActuatorRayHit must match the HLSL struct stride");

/// Ray slots reserved per actuator row; slot = row * B3_IRR_ACTUATOR_RAYS_PER_ROW + k.
#define B3_IRR_ACTUATOR_RAYS_PER_ROW 4

#endif  //B3_IRR_ACTUATOR_H
