// Port of Bullet3OpenCL integrateKernel.cl / b3IntegrateTransforms.h.
// Body layout comes from B3Precision.hlsli - 80 bytes in f32, 96 in df64 (position gains a lo half).
// Included by B3IntegrateTransforms.hlsl (f32) and B3IntegrateTransformsDS.hlsl (df64).

#include "B3Precision.hlsli"

// Params ride in a structured buffer, not shader constants: this compute path reflects no
// variables at all, so setComputeShaderConstant can never resolve a name here.
struct IntegrateParams
{
	float4 gravityAcceleration;   // only read when useUniformGravity != 0
	float timeStep;
	float angularDamping;         // world default; overridden per body when usePerBodyDamping != 0
	float numNodes;
	float linearDamping;          // world default, multiplicative per step like angularDamping
	float useUniformGravity;      // 0: read GravityAccel[body] - the integrator never computes gravity
	float usePerBodyDamping;      // 0: BodyDamping unbound, world defaults apply
	float pad0;
	float pad1;
};

// Renderer-facing layout, 28 bytes (40 in df64). Scalars rather than float3/float4: a vector member
// risks FXC padding the element to 32 and no longer matching the buffer's StructureByteStride.
struct b3IrrBodyTransform
{
	float px, py, pz;
#ifdef OS_DS
	// The readback edge: hi+lo is recombined into a hardware double on the CPU side.
	float pxLo, pyLo, pzLo;
#endif
	float qx, qy, qz, qw;
};

#ifdef OS_DS
// The renderer's cull pass only ever reads a 28-byte transform, so df64 emits that alongside the
// hi/lo one; absolute world positions truncated to float, no camera-relative rebasing.
struct b3IrrBodyTransformF32
{
	float px, py, pz;
	float qx, qy, qz, qw;
};
#endif

// One externally driven pose. invMass stays 0, so no solver ever writes to such a body; the
// velocities exist only so the contact and joint rows see it actually moving.
struct b3KinematicTarget
{
	uint bodyIndex;
	uint pad0, pad1, pad2;
	float4 pos;
#ifdef OS_DS
	float4 posLo;
#endif
	float4 quat;
	float4 linVel;
	float4 angVel;
};

struct KinematicParams
{
	uint numTargets;
	uint numBodies;
	uint pad0;
	uint pad1;
};

StructuredBuffer<IntegrateParams> Params : register(t0);
// Bit 31 = asleep, from B3SleepBody.hlsli. Left UNBOUND when sleeping is off, which reads 0
// (awake) everywhere and restores the pre-sleep behaviour exactly.
StructuredBuffer<uint> SleepState : register(t1);

#define B3_ASLEEP_BIT 0x80000000u

StructuredBuffer<KinematicParams> KinParams : register(t2);
StructuredBuffer<b3KinematicTarget> KinTargets : register(t3);

// Per-body gravity acceleration (xyz, m/s^2), index = body index. Filled by the GravityEffect
// system or its GPU twin - one implementation; this kernel only consumes it.
StructuredBuffer<float4> GravityAccel : register(t4);
// Per-body damping: x = linear, y = angular, both multiplicative per step. Index = body index.
StructuredBuffer<float4> BodyDamping : register(t5);

RWStructuredBuffer<b3RigidBodyData> bodies : register(u0);
RWStructuredBuffer<b3IrrBodyTransform> OutTransforms : register(u1);
#ifdef OS_DS
RWStructuredBuffer<b3IrrBodyTransformF32> OutRenderTransforms : register(u2);
#endif
// u3 in both precisions, so the slot does not move with OS_DS. Unbound when sleeping is off, and
// a discarded write is exactly the right behaviour there.
RWStructuredBuffer<uint> KinSleepState : register(u3);

float4 b3QuatMul(float4 a, float4 b)
{
	float4 ans;
	ans.xyz = cross(a.xyz, b.xyz) + a.w * b.xyz + b.w * a.xyz;
	ans.w = a.w * b.w - dot(a.xyz, b.xyz);
	return ans;
}

// Fixed-order Taylor instead of sin/cos: transcendentals are not bit-exact across vendors, and
// this path must stay reproducible. The clamp below bounds |x| to ~0.393 rad, where these are
// accurate to ~1e-9. precise blocks the compiler reassociating/contracting the Horner chain.
float b3SinPoly(float x)
{
	precise float x2 = x * x;
	precise float r = x * (1.f + x2 * (-1.f / 6.f + x2 * (1.f / 120.f + x2 * (-1.f / 5040.f))));
	return r;
}

float b3CosPoly(float x)
{
	precise float x2 = x * x;
	precise float r = 1.f + x2 * (-0.5f + x2 * (1.f / 24.f + x2 * (-1.f / 720.f + x2 * (1.f / 40320.f))));
	return r;
}

float4 b3QuatNormalized(float4 q)
{
	float len = sqrt(dot(q, q));
	if (len > 0.f)
		q *= 1.f / len;
	else
		q = float4(0.f, 0.f, 0.f, 1.f);
	return q;
}

//! Gravity is folded in here rather than dispatched separately: it is one FMA per body, and the
//! velocity is already in registers at this point.
b3RigidBodyData integrateOne(b3RigidBodyData body, IntegrateParams p, uint nodeID)
{
	const float timeStep = p.timeStep;
	const float BT_GPU_ANGULAR_MOTION_THRESHOLD = 0.25f * 3.14159254f;

	float linearDamping = p.linearDamping;
	float angularDamping = p.angularDamping;
	if (p.usePerBodyDamping != 0.f)
	{
		const float4 d = BodyDamping[nodeID];
		linearDamping = d.x;
		angularDamping = d.y;
	}
	body.linVel.xyz *= linearDamping;
	body.angVel.xyz *= angularDamping;

	float3 angvel = body.angVel.xyz;
	float fAngle = sqrt(dot(angvel, angvel));

	if (fAngle * timeStep > BT_GPU_ANGULAR_MOTION_THRESHOLD)
		fAngle = BT_GPU_ANGULAR_MOTION_THRESHOLD / timeStep;

	float3 axis;
	if (fAngle < 0.001f)
		axis = angvel * (0.5f * timeStep - (timeStep * timeStep * timeStep) * 0.020833333333f * fAngle * fAngle);
	else
		axis = angvel * (b3SinPoly(0.5f * fAngle * timeStep) / fAngle);

	float4 dorn = float4(axis, b3CosPoly(fAngle * timeStep * 0.5f));
	body.quat = b3QuatNormalized(b3QuatMul(dorn, body.quat));

	// A df64 position plus a float delta - exactly what dsAddF is for.
	osIntegratePos(body, body.linVel.xyz * timeStep);
	const float3 gravity = (p.useUniformGravity != 0.f) ? p.gravityAcceleration.xyz
														: GravityAccel[nodeID].xyz;
	body.linVel.xyz += gravity * timeStep;

	return body;
}

//! Drives the infinite-mass bodies the host owns. Must run BEFORE the world-AABB pass, or their
//! bounds describe the previous step and they stop colliding.
[numthreads(64, 1, 1)]
void CSApplyKinematic(uint3 tid : SV_DispatchThreadID)
{
	if (tid.x >= KinParams[0].numTargets)
		return;

	const b3KinematicTarget t = KinTargets[tid.x];
	const uint i = t.bodyIndex;
	if (i >= KinParams[0].numBodies)
		return;

	b3RigidBodyData body = bodies[i];
	body.pos.xyz = t.pos.xyz;
#ifdef OS_DS
	body.posLo.xyz = t.posLo.xyz;
#endif
	body.quat = t.quat;
	body.linVel.xyz = t.linVel.xyz;
	body.angVel.xyz = t.angVel.xyz;
	bodies[i] = body;

	// invMass 0 otherwise makes the sleep pass mark this body permanently asleep, which freezes its
	// world AABB and bars it from ever being the moving side of a wake.
	if (any(t.linVel.xyz != 0.f) || any(t.angVel.xyz != 0.f))
		KinSleepState[i] = 0u;
}

[numthreads(64, 1, 1)]
void CSMain(uint3 tid : SV_DispatchThreadID)
{
	const IntegrateParams p = Params[0];

	int nodeID = (int)tid.x;
	if (nodeID >= (int)p.numNodes)
		return;

	b3RigidBodyData body = bodies[nodeID];
	if (body.invMass == 0.f || (SleepState[nodeID] & B3_ASLEEP_BIT) != 0u)
		return;

	bodies[nodeID] = integrateOne(body, p, (uint)nodeID);
}

//! Integrate and emit the renderer's copy in one pass. Static bodies skip the integration but
//! still publish a transform - the renderer draws them too.
[numthreads(64, 1, 1)]
void CSIntegrateAndPack(uint3 tid : SV_DispatchThreadID)
{
	const IntegrateParams p = Params[0];

	int nodeID = (int)tid.x;
	if (nodeID >= (int)p.numNodes)
		return;

	// A sleeping body skips integration but still publishes its transform: the renderer's cull pass
	// reads this buffer every frame and would otherwise lose the body entirely.
	b3RigidBodyData body = bodies[nodeID];
	if (body.invMass != 0.f && (SleepState[nodeID] & B3_ASLEEP_BIT) == 0u)
	{
		body = integrateOne(body, p, (uint)nodeID);
		bodies[nodeID] = body;
	}

	b3IrrBodyTransform t;
	t.px = body.pos.x;
	t.py = body.pos.y;
	t.pz = body.pos.z;
#ifdef OS_DS
	t.pxLo = body.posLo.x;
	t.pyLo = body.posLo.y;
	t.pzLo = body.posLo.z;
#endif
	t.qx = body.quat.x;
	t.qy = body.quat.y;
	t.qz = body.quat.z;
	t.qw = body.quat.w;
	OutTransforms[nodeID] = t;

#ifdef OS_DS
	b3IrrBodyTransformF32 r;
	r.px = dsToFloat(osPosX(body));
	r.py = dsToFloat(osPosY(body));
	r.pz = dsToFloat(osPosZ(body));
	r.qx = body.quat.x;
	r.qy = body.quat.y;
	r.qz = body.quat.z;
	r.qw = body.quat.w;
	OutRenderTransforms[nodeID] = r;
#endif
}
