// GPU actuators: thrusters, rotation wheels, hover/walker support rays, raycast-vehicle wheels,
// naval drives (ports GroundVehicle.cpp's updateAction and btRaycastVehicle::updateVehicle).
// Included by B3Actuators.hlsl (f32) and B3ActuatorsDS.hlsl (df64).
//
// One thread per row; deltas accumulate as fixed-point ints (InterlockedAdd) so rows on one body
// sum bit-exactly, while a row's own impulses update its local velocity copy in CPU order.

#include "B3Precision.hlsli"

#define WG_SIZE 64
#define FIXED_SCALE 65536.0
#define RAYS_PER_ROW 4

#define KIND_THRUSTER 0
#define KIND_ROTATION_WHEEL 1
#define KIND_HOVER 2
#define KIND_WALKER 3
#define KIND_WHEEL 4
#define KIND_NAVAL 5

#define FLAG_ENABLED 1
#define FLAG_TRACKED 2
#define FLAG_FRONT_WHEEL 4
#define FLAG_WHEEL_COUNT_SHIFT 8

#define B3_ASLEEP_BIT 0x80000000u

// Layout contract: bullet3MT/src/Bullet3Irrlicht/b3IrrActuator.h (80 bytes, static_asserted).
struct b3IrrActuator
{
	int kind;
	int body;
	int ownerRoot;
	int flags;
	float4 thrustLocal;
	float4 torqueLocal;
	float4 params0;
	float4 params1;
};

// Same layout as B3QueriesBody.hlsli's b3Query (radius 0, kind 0 = ray); `active` rides in its pad.
struct b3IrrActuatorRay
{
	float4 from;
	float4 to;
	float radius;
	int ownerRoot;
	uint kind;
	uint active;
};

struct b3IrrActuatorRayHit
{
	float4 hitPoint;    // w = fraction along from->to
	float4 hitNormal;
	int body;
	int hit;
	int pad0;
	int pad1;
};

struct ActuatorParams
{
	uint numActuators;
	uint numBodies;
	float deltaTime;
	float useGravityBuffer;   // 0: uniformGravity applies to every body
	float4 uniformGravity;
};

StructuredBuffer<ActuatorParams> Params : register(t0);
StructuredBuffer<b3IrrActuator> Actuators : register(t1);
StructuredBuffer<float4> InvInertia : register(t2);        // diagonal inverse inertia per body
StructuredBuffer<float4> GravityAccel : register(t3);      // per body, from GravityEffect - never computed here
StructuredBuffer<b3IrrActuatorRayHit> RayHits : register(t4);
// Interim ray provider inputs (CSRayHitsBruteForceAabb); replaced by b3IrrlichtQueries' traversal.
StructuredBuffer<b3Aabb> WorldAabbs : register(t5);
StructuredBuffer<int> BodyOwner : register(t6);             // per body: hull root index (self if standalone)
StructuredBuffer<b3IrrActuatorRay> RaysIn : register(t7);

RWStructuredBuffer<b3RigidBodyData> Bodies : register(u0);
// 6 ints per body: linear xyz then angular xyz, fixed-point.
RWStructuredBuffer<int> VelocityDelta : register(u1);
// Per row: x contacts, y forward speed, z gait phase (walker state), w unused.
RWStructuredBuffer<float4> State : register(u2);
// Sleep bits; a body an actuator pushed must wake. Unbound when sleeping is off (write discarded).
RWStructuredBuffer<uint> WakeState : register(u3);
RWStructuredBuffer<b3IrrActuatorRay> RaysOut : register(u4);
RWStructuredBuffer<b3IrrActuatorRayHit> RayHitsOut : register(u5);

float3 quatRotate(float4 q, float3 v)
{
	const float3 qv = q.xyz;
	return v + 2.f * cross(qv, cross(qv, v) + q.w * v);
}

float3 quatRotateInv(float4 q, float3 v)
{
	return quatRotate(float4(-q.xyz, q.w), v);
}

//! World-space inverse inertia for a diagonal local tensor: R * I^-1 * R^T applied to v.
float3 applyInvInertia(uint bodyIndex, float4 quat, float3 v)
{
	const float3 invI = InvInertia[bodyIndex].xyz;
	return quatRotate(quat, quatRotateInv(quat, v) * invI);
}

// Body position as float in both precisions; actuator geometry is body-relative so the df64 lo
// half only matters for the ray endpoints, which the brute-force provider tolerates.
float3 bodyPosF(b3RigidBodyData b)
{
#ifdef OS_DS
	return float3(dsToFloat(osPosX(b)), dsToFloat(osPosY(b)), dsToFloat(osPosZ(b)));
#else
	return b.pos.xyz;
#endif
}

// The 1/65536 truncation deadband is shared with the contact solver; see B3SolveContactsBody.
void atomicAddVel(uint bodyIndex, float3 dLin, float3 dAng)
{
	const uint base = bodyIndex * 6;
	int ignored;
	InterlockedAdd(VelocityDelta[base + 0], (int)(dLin.x * FIXED_SCALE), ignored);
	InterlockedAdd(VelocityDelta[base + 1], (int)(dLin.y * FIXED_SCALE), ignored);
	InterlockedAdd(VelocityDelta[base + 2], (int)(dLin.z * FIXED_SCALE), ignored);
	InterlockedAdd(VelocityDelta[base + 3], (int)(dAng.x * FIXED_SCALE), ignored);
	InterlockedAdd(VelocityDelta[base + 4], (int)(dAng.y * FIXED_SCALE), ignored);
	InterlockedAdd(VelocityDelta[base + 5], (int)(dAng.z * FIXED_SCALE), ignored);
}

float3 safeVec(float3 v)
{
	return (any(isnan(v)) || any(isinf(v))) ? float3(0.f, 0.f, 0.f) : v;
}

float3 gravityFor(uint bodyIndex, ActuatorParams p)
{
	return (p.useGravityBuffer != 0.f) ? GravityAccel[bodyIndex].xyz : p.uniformGravity.xyz;
}

// Fixed-order Taylor instead of sin/cos, as B3IntegrateTransformsBody does: transcendentals are
// not bit-exact across vendors. Steering angles stay well under 1 rad, where these hold ~1e-8.
float b3SinPoly(float x)
{
	precise float x2 = x * x;
	precise float r = x * (1.f + x2 * (-1.f / 6.f + x2 * (1.f / 120.f + x2 * (-1.f / 5040.f + x2 * (1.f / 362880.f)))));
	return r;
}

float b3CosPoly(float x)
{
	precise float x2 = x * x;
	precise float r = 1.f + x2 * (-0.5f + x2 * (1.f / 24.f + x2 * (-1.f / 720.f + x2 * (1.f / 40320.f + x2 * (-1.f / 3628800.f)))));
	return r;
}

//! Rodrigues rotation of v about unit axis by angle; used for the steering rotation of the axle.
float3 rotateAboutAxis(float3 v, float3 axis, float angle)
{
	const float c = b3CosPoly(angle);
	const float s = b3SinPoly(angle);
	return v * c + cross(axis, v) * s + axis * dot(axis, v) * (1.f - c);
}

float3 clampLength(float3 v, float maxLen)
{
	const float len = length(v);
	if (len > maxLen && maxLen > 0.f)
		v *= maxLen / len;
	return v;
}

// ---------------------------------------------------------------------------------------------
// Row-local impulse accumulation: mirrors btRigidBody::applyImpulse on a local velocity copy.
// ---------------------------------------------------------------------------------------------
struct RowVel
{
	float3 lin;
	float3 ang;
};

void applyCentralImpulse(inout RowVel v, float invMass, float3 P)
{
	v.lin += P * invMass;
}

void applyTorqueImpulse(inout RowVel v, uint bodyIndex, float4 q, float3 T)
{
	v.ang += applyInvInertia(bodyIndex, q, T);
}

void applyImpulseAt(inout RowVel v, uint bodyIndex, float4 q, float invMass, float3 P, float3 rWorld)
{
	v.lin += P * invMass;
	v.ang += applyInvInertia(bodyIndex, q, cross(rWorld, P));
}

//! Bullet's btJacobianEntry diagonal for a static partner: invMass + n . ((I^-1 (r x n)) x r).
float jacDiag(uint bodyIndex, float4 q, float invMass, float3 r, float3 n)
{
	return invMass + dot(n, cross(applyInvInertia(bodyIndex, q, cross(r, n)), r));
}

// ---------------------------------------------------------------------------------------------
// Kinds
// ---------------------------------------------------------------------------------------------

void actuateThruster(b3IrrActuator a, b3RigidBodyData body, uint bi, float dt, inout RowVel v)
{
	const float throttle = a.params0.w;
	const float3 F = quatRotate(body.quat, a.thrustLocal.xyz) * throttle;
	const float3 r = quatRotate(body.quat, a.params0.xyz);
	applyImpulseAt(v, bi, body.quat, body.invMass, F * dt, r);
	applyTorqueImpulse(v, bi, body.quat, quatRotate(body.quat, a.torqueLocal.xyz) * throttle * dt);
}

void actuateRotationWheel(b3IrrActuator a, b3RigidBodyData body, uint bi, float dt, inout RowVel v)
{
	applyTorqueImpulse(v, bi, body.quat, quatRotate(body.quat, a.torqueLocal.xyz) * a.params0.w * dt);
}

//! HoverWalkerDrive::updateAction.
void actuateHover(uint row, b3IrrActuator a, b3RigidBodyData body, uint bi, float dt, float3 gravity,
				  inout RowVel v, inout float4 state)
{
	const bool walker = (a.kind == KIND_WALKER);
	const float throttle = a.thrustLocal.x;
	const float yaw = a.thrustLocal.y;
	const float engineForce = a.thrustLocal.z;
	const float maxSpeed = a.thrustLocal.w;
	const float cornerX = a.torqueLocal.x;
	const float castY = a.torqueLocal.y;
	const float cornerZ = a.torqueLocal.z;
	const float targetDistance = a.torqueLocal.w;
	const float stiffness = a.params0.x;
	const float damping = a.params0.y;
	const float steerTorque = a.params0.z;
	const float gaitPeriod = max(a.params0.w, 2.f);

	const bool commanded = abs(throttle) > 0.05f || abs(yaw) > 0.05f;
	const float mass = 1.f / body.invMass;
	const float gLen = length(gravity);
	const float gMag = max(gLen, 1.f);
	const float3 localUpWorld = quatRotate(body.quat, float3(0.f, 1.f, 0.f));
	// Down is the LOCAL gravity direction, never world -Y: planet sub-worlds are radial.
	const float3 up = (gLen * gLen > 1e-12f) ? -gravity / gLen : localUpWorld;
	const float effectiveForce = min(engineForce, mass * gMag * 1.2f);

	float gaitPhase = state.z;
	if (walker && commanded)
		gaitPhase += 1.f / gaitPeriod;

	const float3 pos = bodyPosF(body);
	int contacts = 0;
	[unroll]
	for (int k = 0; k < 4; ++k)
	{
		const b3IrrActuatorRayHit hit = RayHits[row * RAYS_PER_ROW + (uint)k];
		if (hit.hit == 0)
			continue;
		++contacts;

		// Alternating-diagonal gait: swing legs keep a fraction of support; all stance when idle.
		float legWeight = 1.f;
		if (walker && commanded)
		{
			const bool diagonalA = (k == 0 || k == 3);
			const bool stance = (frac(gaitPhase) < 0.5f) == diagonalA;
			legWeight = stance ? 1.6f : 0.3f;
		}

		const float3 corner = float3((k & 1) ? cornerX : -cornerX, castY, (k & 2) ? -cornerZ : cornerZ);
		const float3 rWorld = quatRotate(body.quat, corner);
		const float3 from = pos + rWorld;
		const float distance = length(hit.hitPoint.xyz - from);
		const float error = targetDistance - distance;
		const float3 cornerVel = v.lin + cross(v.ang, rWorld);
		float force = (mass / 4.f) * (stiffness * error - damping * dot(cornerVel, up));
		force = clamp(force * legWeight, 0.f, mass * gMag * 1.5f);
		applyImpulseAt(v, bi, body.quat, body.invMass, up * (force * dt), rWorld);
	}
	state.x = (float)contacts;
	state.z = gaitPhase;
	if (contacts == 0)
		return;

	// Self-righting torque toward gravity-up, on top of the angular damping.
	applyTorqueImpulse(v, bi, body.quat, cross(localUpWorld, up) * (mass * 0.6f * gMag * dt));

	float3 forward = quatRotate(body.quat, float3(0.f, 0.f, 1.f));
	forward -= up * dot(forward, up);
	if (dot(forward, forward) < 1e-6f)
		return;
	forward = normalize(forward);

	const float3 velocity = v.lin;
	const float speed = abs(dot(velocity, forward));
	state.y = speed;
	const float speedScale = clamp((maxSpeed - speed) / max(maxSpeed * 0.2f, 1.f), 0.f, 1.f);
	applyCentralImpulse(v, body.invMass, forward * (effectiveForce * throttle * speedScale * dt));

	if (!commanded)
	{
		// Ground anchor: kill the gravity-tangent velocity, traction-clamped.
		const float3 tangential = velocity - up * dot(velocity, up);
		float3 kill = -tangential * mass;
		kill = clampLength(kill, (walker ? 3.f : 0.5f) * mass * gMag * dt);
		applyCentralImpulse(v, body.invMass, kill);
	}

	const float3 lateral = velocity - up * dot(velocity, up) - forward * dot(velocity, forward);
	applyCentralImpulse(v, body.invMass, -lateral * (mass * (walker ? 4.f : 1.5f) * dt));

	if (abs(yaw) > 0.01f)
	{
		const float3 localAngVel = quatRotateInv(body.quat, v.ang);
		const float MaxYawRate = 1.5f;
		const float governor = clamp(1.f - (localAngVel.y / MaxYawRate) * (yaw > 0.f ? 1.f : -1.f), 0.f, 1.f);
		applyTorqueImpulse(v, bi, body.quat,
						   localUpWorld * (yaw * governor * min(steerTorque, mass * 6.f * gMag) * dt));
	}
}

//! One wheel of RaycastVehicleDrive: btRaycastVehicle::updateVehicle for this wheel plus the
//! drive's governors, with the hull-level terms split 1/wheelCount so the rows sum to the CPU's.
void actuateWheel(uint row, b3IrrActuator a, b3RigidBodyData body, uint bi, float dt, float3 gravity,
				  inout RowVel v, inout float4 state)
{
	const float throttle = a.thrustLocal.x;
	const float yaw = a.thrustLocal.y;
	const float engineForce = a.thrustLocal.z;
	const float maxSpeed = a.thrustLocal.w;
	const float3 connCS = a.torqueLocal.xyz;
	const float restLength = a.torqueLocal.w;
	const float radius = a.params0.x;
	const float stiffness = a.params0.y;
	const float dampRelax = a.params0.z;
	const float dampComp = a.params0.w;
	const float baseFrictionSlip = a.params1.x;
	const float maxSteer = a.params1.y;
	const float maxTravel = a.params1.z;
	const float maxSuspForce = a.params1.w;
	const bool tracked = (a.flags & FLAG_TRACKED) != 0;
	const bool front = (a.flags & FLAG_FRONT_WHEEL) != 0;
	const float wheelCount = (float)max((a.flags >> FLAG_WHEEL_COUNT_SHIFT) & 0xFF, 1);
	const float rollInfluence = 0.05f;

	const float mass = 1.f / body.invMass;
	const float gMag = max(length(gravity), 1.f);
	const float4 q = body.quat;
	const float3 pos = bodyPosF(body);
	const float3 localUpWorld = quatRotate(q, float3(0.f, 1.f, 0.f));

	// Bullet's km/h speed is |v| signed by heading; the governors use its magnitude.
	const float speed = length(v.lin);
	const float speedScale = clamp((maxSpeed - speed) / max(maxSpeed * 0.2f, 1.f), 0.f, 1.f);
	const float effectiveForce = min(engineForce, mass * gMag * 1.2f);
	const float forwardForce = effectiveForce * throttle * speedScale;
	const bool idle = abs(throttle) < 0.05f && abs(yaw) < 0.05f;
	const bool pivoting = abs(yaw) > 0.05f && speed < 2.f;
	const float frictionSlip = baseFrictionSlip * (pivoting ? 0.3f : 1.f);

	float wheelEngineForce = 0.f;
	float steering = 0.f;
	if (tracked)
		wheelEngineForce = forwardForce * 0.25f;
	else if (front)
	{
		const float wheelbase = 2.f * abs(connCS.z);
		const float maxLateralAccel = 4.f;
		const float steerLimit = (speed > 1.f) ? min(maxSteer, maxLateralAccel * wheelbase / (speed * speed))
											   : maxSteer;
		steering = yaw * steerLimit;
	}
	else
		wheelEngineForce = forwardForce * 0.5f;
	const float brake = idle ? mass * gMag * radius * 0.5f : 0.f;

	// Governed hull yaw torque, applied before updateVehicle as on the CPU; this row's share.
	if (abs(yaw) > 0.01f)
	{
		const float authority = tracked ? 1.f : 0.6f * clamp(1.f - speed / 2.f, 0.f, 1.f);
		if (authority > 0.f)
		{
			const float3 localAngVel = quatRotateInv(q, v.ang);
			const float MaxYawRate = 1.5f;
			const float governor = clamp(1.f - (localAngVel.y / MaxYawRate) * (yaw > 0.f ? 1.f : -1.f), 0.f, 1.f);
			applyTorqueImpulse(v, bi, q, localUpWorld * (yaw * governor * authority * effectiveForce * 1.5f * dt / wheelCount));
		}
	}

	// --- btRaycastVehicle::rayCast ---
	const float3 wheelDirWS = quatRotate(q, float3(0.f, -1.f, 0.f));
	const float3 axleCS_WS = quatRotate(q, float3(-1.f, 0.f, 0.f));
	const float raylen = restLength + radius;
	const b3IrrActuatorRayHit hit = RayHits[row * RAYS_PER_ROW];
	const bool inContact = hit.hit != 0;

	float3 contactNormal = -wheelDirWS;
	float3 contactPoint = pos + quatRotate(q, connCS) + wheelDirWS * raylen;
	float suspLen = restLength;
	float suspRelVel = 0.f;
	float clippedInv = 1.f;
	if (inContact)
	{
		contactNormal = hit.hitNormal.xyz;
		contactPoint = hit.hitPoint.xyz;
		suspLen = clamp(hit.hitPoint.w * raylen - radius, restLength - maxTravel, restLength + maxTravel);
		const float denominator = dot(contactNormal, wheelDirWS);
		const float3 relpos = contactPoint - pos;
		const float projVel = dot(contactNormal, v.lin + cross(v.ang, relpos));
		if (denominator >= -0.1f)
		{
			suspRelVel = 0.f;
			clippedInv = 10.f;
		}
		else
		{
			const float inv = -1.f / denominator;
			suspRelVel = projVel * inv;
			clippedInv = inv;
		}
	}

	// --- updateSuspension + suspension impulse ---
	float suspForce = 0.f;
	if (inContact)
	{
		float force = stiffness * (restLength - suspLen) * clippedInv;
		force -= (suspRelVel < 0.f ? dampComp : dampRelax) * suspRelVel;
		suspForce = max(force * mass, 0.f);
	}
	const float3 relPos = contactPoint - pos;
	applyImpulseAt(v, bi, q, body.invMass, contactNormal * (min(suspForce, maxSuspForce) * dt), relPos);

	// --- updateFriction (ground is Bullet's fixed body, so the partner has no mass terms) ---
	if (inContact)
	{
		float3 axle = rotateAboutAxis(axleCS_WS, -wheelDirWS, steering);
		axle -= contactNormal * dot(axle, contactNormal);
		axle = normalize(axle);
		const float3 forwardWS = normalize(cross(contactNormal, axle));

		const float3 vel = v.lin + cross(v.ang, relPos);

		// resolveSingleBilateral: contactDamping 0.2, sideFrictionStiffness2 1.
		float sideImpulse = -0.2f * dot(axle, vel) / jacDiag(bi, q, body.invMass, relPos, axle);

		float rolling = 0.f;
		if (wheelEngineForce != 0.f)
			rolling = wheelEngineForce * dt;
		else
		{
			// calcRollingFriction; wheels on ground approximated by the row's wheel count.
			const float j1 = -dot(forwardWS, vel) / jacDiag(bi, q, body.invMass, relPos, forwardWS) / wheelCount;
			rolling = clamp(j1, -brake, brake);
		}

		const float maximp = suspForce * dt * frictionSlip;
		const float x = rolling * 0.5f;
		const float y = sideImpulse;
		const float impulseSquared = x * x + y * y;
		if (impulseSquared > maximp * maximp && sideImpulse != 0.f)
		{
			const float skid = maximp / sqrt(impulseSquared);
			rolling *= skid;
			sideImpulse *= skid;
		}

		if (rolling != 0.f)
			applyImpulseAt(v, bi, q, body.invMass, forwardWS * rolling, relPos);
		if (sideImpulse != 0.f)
		{
			// ROLLING_INFLUENCE_FIX: pull the side impulse's lever toward the CoM height.
			const float3 relPos2 = relPos - localUpWorld * (dot(localUpWorld, relPos) * (1.f - rollInfluence));
			applyImpulseAt(v, bi, q, body.invMass, axle * sideImpulse, relPos2);
		}
	}

	// --- RaycastVehicleDrive's anchor / track lateral kill, this row's share ---
	if (inContact)
	{
		const float3 velocity = v.lin;
		const float3 tangential = velocity - contactNormal * dot(velocity, contactNormal);
		float3 kill = float3(0.f, 0.f, 0.f);
		if (idle)
		{
			const float3 gravityTangent = gravity - contactNormal * dot(gravity, contactNormal);
			kill = -(tangential + gravityTangent * dt) * mass;
		}
		else if (tracked)
		{
			float3 lateralDir = quatRotate(q, float3(1.f, 0.f, 0.f));
			lateralDir -= contactNormal * dot(lateralDir, contactNormal);
			if (dot(lateralDir, lateralDir) > 1e-12f)
			{
				lateralDir = normalize(lateralDir);
				kill = -lateralDir * dot(tangential, lateralDir) * mass;
			}
		}
		const float maxImpulse = baseFrictionSlip * (tracked ? 2.f : 1.f) * mass * gMag * dt;
		kill = clampLength(kill, maxImpulse);
		applyCentralImpulse(v, body.invMass, kill / wheelCount);
	}

	state.x = inContact ? 1.f : 0.f;
	state.y = speed;
}

//! NavalDrive::updateAction. Up is -gravity when known, else radial from the sub-world origin.
void actuateNaval(b3IrrActuator a, b3RigidBodyData body, uint bi, float dt, float3 gravity,
				  inout RowVel v, inout float4 state)
{
	const float submersion = a.torqueLocal.w;
	state.x = submersion > 0.1f ? 1.f : 0.f;
	if (submersion <= 0.1f)
		return;

	const float throttle = a.thrustLocal.x;
	const float yaw = a.thrustLocal.y;
	const float engineForce = a.thrustLocal.z;
	const float maxSpeed = a.thrustLocal.w;
	const float rudderTorque = a.torqueLocal.x;
	const float keelDrag = a.torqueLocal.y;
	const float rightingStrength = a.torqueLocal.z;

	const float mass = 1.f / body.invMass;
	const float4 q = body.quat;
	const float3 pos = bodyPosF(body);
	const float gLen = length(gravity);
	float3 up;
	if (gLen * gLen > 1e-12f)
		up = -gravity / gLen;
	else
	{
		if (dot(pos, pos) < 1.f)
			return;
		up = normalize(pos);
	}
	const float3 localUpWorld = quatRotate(q, float3(0.f, 1.f, 0.f));

	applyTorqueImpulse(v, bi, q, cross(localUpWorld, up) * (mass * rightingStrength * submersion * dt));

	float3 forward = quatRotate(q, float3(0.f, 0.f, 1.f));
	forward -= up * dot(forward, up);
	if (dot(forward, forward) < 1e-6f)
		return;
	forward = normalize(forward);

	const float3 velocity = v.lin;
	const float speed = dot(velocity, forward);
	state.y = abs(speed);
	const float speedScale = clamp((maxSpeed - abs(speed)) / max(maxSpeed * 0.2f, 1.f), 0.f, 1.f);
	const float effectiveForce = min(engineForce, mass * 12.f);
	applyCentralImpulse(v, body.invMass, forward * (effectiveForce * throttle * speedScale * submersion * dt));

	const float3 lateral = velocity - up * dot(velocity, up) - forward * dot(velocity, forward);
	applyCentralImpulse(v, body.invMass, -lateral * (mass * keelDrag * submersion * dt));

	if (abs(yaw) > 0.01f)
	{
		const float flowScale = clamp(abs(speed) / max(maxSpeed * 0.25f, 1.f), 0.f, 1.f);
		const float3 localAngVel = quatRotateInv(q, v.ang);
		const float MaxYawRate = 0.6f;
		const float governor = clamp(1.f - (localAngVel.y / MaxYawRate) * (yaw > 0.f ? 1.f : -1.f), 0.f, 1.f);
		applyTorqueImpulse(v, bi, q, localUpWorld * (yaw * governor * flowScale * submersion
													 * min(rudderTorque, mass * 60.f) * dt));
	}
}

// ---------------------------------------------------------------------------------------------
// Kernels
// ---------------------------------------------------------------------------------------------

//! Writes this step's support/wheel rays for every row; other kinds mark their slots inactive.
[numthreads(WG_SIZE, 1, 1)]
void CSBuildActuatorRays(uint3 tid : SV_DispatchThreadID)
{
	const uint row = tid.x;
	if (row >= Params[0].numActuators)
		return;

	const b3IrrActuator a = Actuators[row];
	b3IrrActuatorRay ray;
	ray.from = float4(0.f, 0.f, 0.f, 0.f);
	ray.to = float4(0.f, 0.f, 0.f, 0.f);
	ray.radius = 0.f;
	ray.ownerRoot = a.ownerRoot;
	ray.kind = 0u;
	ray.active = 0u;

	const bool enabled = (a.flags & FLAG_ENABLED) != 0 && (uint)a.body < Params[0].numBodies;
	// Always a valid read (X4000 treats a conditionally assigned struct as an error).
	const b3RigidBodyData body = Bodies[enabled ? (uint)a.body : 0u];
	const float3 pos = bodyPosF(body);

	[unroll]
	for (uint k = 0; k < RAYS_PER_ROW; ++k)
	{
		b3IrrActuatorRay r = ray;
		if (enabled && (a.kind == KIND_HOVER || a.kind == KIND_WALKER))
		{
			const float3 gravity = gravityFor(a.body, Params[0]);
			const float gLen = length(gravity);
			const float3 up = (gLen * gLen > 1e-12f) ? -gravity / gLen : quatRotate(body.quat, float3(0.f, 1.f, 0.f));
			const float3 corner = float3((k & 1) ? a.torqueLocal.x : -a.torqueLocal.x, a.torqueLocal.y,
										 (k & 2) ? -a.torqueLocal.z : a.torqueLocal.z);
			const float3 from = pos + quatRotate(body.quat, corner);
			r.from = float4(from, 0.f);
			r.to = float4(from - up * (a.torqueLocal.w * 2.f), 0.f);
			r.active = 1u;
		}
		else if (enabled && a.kind == KIND_WHEEL && k == 0)
		{
			// Wheel rays follow the chassis' own down axis, as btRaycastVehicle does.
			const float3 from = pos + quatRotate(body.quat, a.torqueLocal.xyz);
			const float3 down = quatRotate(body.quat, float3(0.f, -1.f, 0.f));
			r.from = float4(from, 0.f);
			r.to = float4(from + down * (a.torqueLocal.w + a.params0.x), 0.f);
			r.active = 1u;
		}
		RaysOut[row * RAYS_PER_ROW + k] = r;
	}
}

//! INTERIM ray provider: closest world-AABB slab hit per ray, skipping the ray's own hull tree.
//! Exact for axis-aligned boxes (flat ground); to be replaced by the LBVH query API.
[numthreads(WG_SIZE, 1, 1)]
void CSRayHitsBruteForceAabb(uint3 tid : SV_DispatchThreadID)
{
	const uint rayIndex = tid.x;
	if (rayIndex >= Params[0].numActuators * RAYS_PER_ROW)
		return;

	const b3IrrActuatorRay ray = RaysIn[rayIndex];
	b3IrrActuatorRayHit best;
	best.hitPoint = float4(0.f, 0.f, 0.f, 1.f);
	best.hitNormal = float4(0.f, 0.f, 0.f, 0.f);
	best.body = -1;
	best.hit = 0;
	best.pad0 = 0;
	best.pad1 = 0;

	if (ray.active != 0u)
	{
		const float3 from = ray.from.xyz;
		const float3 d = ray.to.xyz - ray.from.xyz;
		// Division by a zero component yields +-inf; min/max return the non-NaN operand, which is
		// what makes an axis-parallel ray test its slab correctly (same care as CSRayTraverse).
		const float3 invD = 1.f / d;
		const uint numBodies = Params[0].numBodies;
		float bestT = 1.f;

		for (uint b = 0; b < numBodies; ++b)
		{
			if (BodyOwner[b] == ray.ownerRoot)
				continue;

			const b3Aabb box = WorldAabbs[b];
			const float3 t1 = (osAabbMinF(box) - from) * invD;
			const float3 t2 = (osAabbMaxF(box) - from) * invD;
			const float3 tmin = min(t1, t2);
			const float3 tmax = max(t1, t2);
			const float tNear = max(max(tmin.x, tmin.y), tmin.z);
			const float tFar = min(min(tmax.x, tmax.y), tmax.z);

			// A ray starting inside a box finds nothing, like btCollisionWorld::rayTest.
			if (tNear > tFar || tFar < 0.f || tNear < 0.f || tNear >= bestT)
				continue;
			if (isnan(tNear) || isinf(tNear))
				continue;

			bestT = tNear;
			float3 n = float3(0.f, 0.f, 0.f);
			if (tNear == tmin.x)
				n.x = (d.x > 0.f) ? -1.f : 1.f;
			else if (tNear == tmin.y)
				n.y = (d.y > 0.f) ? -1.f : 1.f;
			else
				n.z = (d.z > 0.f) ? -1.f : 1.f;
			best.hitPoint = float4(from + d * tNear, tNear);
			best.hitNormal = float4(n, 0.f);
			best.body = (int)b;
			best.hit = 1;
		}
	}

	RayHitsOut[rayIndex] = best;
}

//! One thread per actuator row: applies its kind and accumulates the body's velocity delta.
[numthreads(WG_SIZE, 1, 1)]
void CSApplyActuators(uint3 tid : SV_DispatchThreadID)
{
	const uint row = tid.x;
	const ActuatorParams p = Params[0];
	if (row >= p.numActuators)
		return;

	const b3IrrActuator a = Actuators[row];
	if ((a.flags & FLAG_ENABLED) == 0 || (uint)a.body >= p.numBodies)
		return;

	const uint bi = (uint)a.body;
	const b3RigidBodyData body = Bodies[bi];
	if (body.invMass == 0.f)
		return;

	const float dt = p.deltaTime;
	const float3 gravity = gravityFor(bi, p);

	RowVel v;
	v.lin = body.linVel.xyz;
	v.ang = body.angVel.xyz;
	float4 state = State[row];

	if (a.kind == KIND_THRUSTER)
		actuateThruster(a, body, bi, dt, v);
	else if (a.kind == KIND_ROTATION_WHEEL)
		actuateRotationWheel(a, body, bi, dt, v);
	else if (a.kind == KIND_HOVER || a.kind == KIND_WALKER)
		actuateHover(row, a, body, bi, dt, gravity, v, state);
	else if (a.kind == KIND_WHEEL)
		actuateWheel(row, a, body, bi, dt, gravity, v, state);
	else if (a.kind == KIND_NAVAL)
		actuateNaval(a, body, bi, dt, gravity, v, state);
	else
		return;

	State[row] = state;

	const float3 dLin = safeVec(v.lin - body.linVel.xyz);
	const float3 dAng = safeVec(v.ang - body.angVel.xyz);
	if (any(dLin != 0.f) || any(dAng != 0.f))
	{
		atomicAddVel(bi, dLin, dAng);
		// An actuated body must not stay asleep, matching the CPU's activate(true) on command.
		WakeState[bi] = 0u;
	}
}

//! Folds the fixed-point deltas into the bodies and clears them for the next step.
[numthreads(WG_SIZE, 1, 1)]
void CSApplyActuatorDeltas(uint3 tid : SV_DispatchThreadID)
{
	const uint bodyIndex = tid.x;
	if (bodyIndex >= Params[0].numBodies)
		return;

	const uint base = bodyIndex * 6;
	const float inv = 1.0 / FIXED_SCALE;

	b3RigidBodyData body = Bodies[bodyIndex];
	body.linVel.xyz += float3(VelocityDelta[base + 0], VelocityDelta[base + 1], VelocityDelta[base + 2]) * inv;
	body.angVel.xyz += float3(VelocityDelta[base + 3], VelocityDelta[base + 4], VelocityDelta[base + 5]) * inv;
	Bodies[bodyIndex] = body;

	[unroll]
	for (uint i = 0; i < 6; ++i)
		VelocityDelta[base + i] = 0;
}
