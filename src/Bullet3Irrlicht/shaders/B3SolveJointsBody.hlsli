// Port of Bullet3OpenCL jointSolver.cl, extended: point-to-point, fixed, hinge, slider, cone-twist
// and generic 6-DoF joints with per-axis limits/motors, per-joint ERP/CFM and impulse breaking.
//
// Same fixed-point InterlockedAdd accumulation as B3SolveContacts.hlsl: order-independent, so the
// original's batching/graph-colouring kernels are not needed (per-joint rows are single-writer).
//
// Included by B3SolveJoints.hlsl (f32) and B3SolveJointsDS.hlsl (df64); the PREFER_FLOW_CONTROL
// marker lives in those wrappers, since D3DCompile only scans the top-level source for it.

#include "B3Precision.hlsli"

#define WG_SIZE 64
// 2^20: a joint's per-iteration delta is often far below the old 2^16 quantum, and truncating it to
// zero froze small anchor errors permanently. Leaves +-2047 m/s of int32 headroom.
#define FIXED_SCALE 1048576.0
#define FIXED_MAX 2.0e9
#define B3_INF 3.0e38

#define B3_GPU_POINT2POINT_CONSTRAINT_TYPE 3
#define B3_GPU_FIXED_CONSTRAINT_TYPE 4
#define B3_GPU_HINGE_CONSTRAINT_TYPE 5
#define B3_GPU_SLIDER_CONSTRAINT_TYPE 6
#define B3_GPU_CONETWIST_CONSTRAINT_TYPE 7
#define B3_GPU_D6_CONSTRAINT_TYPE 8

#define B3_CONSTRAINT_FLAG_ENABLED 1
#define B3_CONSTRAINT_FLAG_BROKEN 2
#define B3_CONSTRAINT_FLAG_OWN_ERP 4

// Mirrors b3IrrJoint in b3IrrlichtJointSolver.h (240 bytes) and b3SleepJoint in B3SleepBody.hlsli.
// Limits: lower > upper free, == locked, else limited; a motor needs maxImpulse > 0, axis unlocked.
struct b3IrrJoint
{
	int constraintType;
	int rbA;
	int rbB;
	float breakingImpulseThreshold;
	float4 pivotInA;
	float4 pivotInB;
	float4 relTargetAB;        // fixed joints: target B orientation relative to A
	int flags;
	int uid;
	float erp;
	float cfm;
	float4 frameInA;           // joint frame orientation in A's local frame (xyzw)
	float4 frameInB;
	float4 linLower;
	float4 linUpper;
	float4 angLower;           // cone-twist: x = twist span; angUpper.y = swing span
	float4 angUpper;
	float4 linMotorVel;
	float4 angMotorVel;
	float4 linMotorMaxImpulse;
	float4 angMotorMaxImpulse;
};

// Per-joint, per-step accumulated row impulses. Rows stay clamped against these across the Jacobi
// iterations, and their sum is the breaking measure. Reset by CSFinishJoints every step.
struct b3IrrJointAccum
{
	float4 lin;
	float4 ang;
	float4 linMotor;
	float4 angMotor;
};

struct JointSolverParams
{
	uint numJoints;
	uint numBodies;
	float deltaTime;
	float erp;              // Baumgarte position-correction factor for joints without OWN_ERP
};

StructuredBuffer<JointSolverParams> Params : register(t0);
StructuredBuffer<b3IrrJoint> Joints : register(t1);
StructuredBuffer<float4> InvInertia : register(t2);   // diagonal inverse inertia per body
// Jacobi splitting weight per joint: 1/(joints touching the busier of its two bodies).
StructuredBuffer<float> JointScale : register(t3);

RWStructuredBuffer<b3RigidBodyData> Bodies : register(u0);
// 6 ints per body: linear xyz then angular xyz, fixed-point.
RWStructuredBuffer<int> VelocityDelta : register(u1);
RWStructuredBuffer<b3IrrJointAccum> JointAccum : register(u2);
// CSFinishJoints only: it clears the enable bit, so the joints are bound writable there and the
// t1 view is left unbound (SRV + UAV of one buffer in a dispatch reads garbage).
RWStructuredBuffer<b3IrrJoint> JointsRW : register(u3);
// 1 where the joint is broken, readable by the host; sticky until the joint is re-uploaded.
RWStructuredBuffer<uint> JointStatus : register(u4);

float3 quatRotate(float4 q, float3 v)
{
	const float3 qv = q.xyz;
	return v + 2.f * cross(qv, cross(qv, v) + q.w * v);
}

float4 quatMul(float4 a, float4 b)
{
	return float4(a.w * b.xyz + b.w * a.xyz + cross(a.xyz, b.xyz), a.w * b.w - dot(a.xyz, b.xyz));
}

float4 quatConj(float4 q)
{
	return float4(-q.xyz, q.w);
}

//! Rounds rather than truncates: truncation is a one-sided bias toward zero, and every Jacobi
//! iteration re-applies it, so a small residual can never converge.
int toFixed(float v)
{
	return (int)clamp(round(v * FIXED_SCALE), -FIXED_MAX, FIXED_MAX);
}

void atomicAddVel(uint bodyIndex, float3 dLin, float3 dAng)
{
	const uint base = bodyIndex * 6;
	int ignored;
	InterlockedAdd(VelocityDelta[base + 0], toFixed(dLin.x), ignored);
	InterlockedAdd(VelocityDelta[base + 1], toFixed(dLin.y), ignored);
	InterlockedAdd(VelocityDelta[base + 2], toFixed(dLin.z), ignored);
	InterlockedAdd(VelocityDelta[base + 3], toFixed(dAng.x), ignored);
	InterlockedAdd(VelocityDelta[base + 4], toFixed(dAng.y), ignored);
	InterlockedAdd(VelocityDelta[base + 5], toFixed(dAng.z), ignored);
}

//! World-space inverse inertia tensor R * diag(invI) * R^T, materialised because the joint's
//! effective mass is a full 3x3 that has to be inverted, not a scalar along one axis.
float3x3 worldInvInertia(uint bodyIndex, float4 quat)
{
	const float3 invI = InvInertia[bodyIndex].xyz;
	const float3 ex = quatRotate(quat, float3(1.f, 0.f, 0.f));
	const float3 ey = quatRotate(quat, float3(0.f, 1.f, 0.f));
	const float3 ez = quatRotate(quat, float3(0.f, 0.f, 1.f));

	const float3x3 rt = float3x3(ex, ey, ez);          // rows are the rotated basis, i.e. R^T
	const float3x3 r = transpose(rt);
	const float3x3 d = float3x3(invI.x, 0.f, 0.f,
								0.f, invI.y, 0.f,
								0.f, 0.f, invI.z);
	return mul(r, mul(d, rt));
}

float3x3 skew(float3 v)
{
	return float3x3(0.f, -v.z, v.y,
					v.z, 0.f, -v.x,
					-v.y, v.x, 0.f);
}

//! Returns false when m is singular, which a fully static or degenerate pair produces.
bool invert3x3(float3x3 m, out float3x3 result)
{
	const float3 c0 = float3(m._22 * m._33 - m._23 * m._32,
							 m._23 * m._31 - m._21 * m._33,
							 m._21 * m._32 - m._22 * m._31);
	const float det = m._11 * c0.x + m._12 * c0.y + m._13 * c0.z;

	result = float3x3(0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f);
	if (abs(det) < 1e-12f || isnan(det) || isinf(det))
		return false;

	const float invDet = 1.f / det;
	result = float3x3(c0.x, m._13 * m._32 - m._12 * m._33, m._12 * m._23 - m._13 * m._22,
					  c0.y, m._11 * m._33 - m._13 * m._31, m._13 * m._21 - m._11 * m._23,
					  c0.z, m._12 * m._31 - m._11 * m._32, m._11 * m._22 - m._12 * m._21) * invDet;
	return true;
}

bool finite1(float v)
{
	return !(isnan(v) || isinf(v));
}

bool finite3(float3 v)
{
	return !(isnan(v.x) || isnan(v.y) || isnan(v.z) || isinf(v.x) || isinf(v.y) || isinf(v.z));
}

//! One scalar row with accumulated clamping: returns the impulse to apply this iteration.
float solveRow(float k, float vrel, float bias, float lo, float hi, float scale, inout float acc)
{
	if (k <= 1e-12f || !finite1(k))
		return 0.f;
	const float lambda = -(vrel + bias) / k * scale;
	const float next = clamp(acc + lambda, lo, hi);
	const float applied = next - acc;
	if (!finite1(applied))
		return 0.f;
	acc = next;
	return applied;
}

//! Linear impulse p on B at rB and -p on A at rA, folded into the joint-local velocity copies so
//! later rows of the same joint see it (Gauss-Seidel inside the joint, Jacobi across joints).
void applyLinearImpulse(float3 p, float3 rA, float3 rB, float mA, float mB,
						float3x3 invIA, float3x3 invIB,
						inout float3 linA, inout float3 angA, inout float3 linB, inout float3 angB)
{
	linA -= p * mA;
	angA -= mul(invIA, cross(rA, p));
	linB += p * mB;
	angB += mul(invIB, cross(rB, p));
}

void applyAngularImpulse(float3 t, float3x3 invIA, float3x3 invIB,
						 inout float3 angA, inout float3 angB)
{
	angA -= mul(invIA, t);
	angB += mul(invIB, t);
}

//! Limit/lock row plus motor row along one linear axis n of the joint frame.
void solveLinearAxis(float3 n, float lower, float upper, float motorVel, float motorMax,
					 float erp, float cfm, float invDt, float scale, float3 d,
					 float3 rA, float3 rB, float mA, float mB, float3x3 invIA, float3x3 invIB,
					 inout float3 linA, inout float3 angA, inout float3 linB, inout float3 angB,
					 inout float accLimit, inout float accMotor)
{
	const float3 uA = cross(rA, n);
	const float3 uB = cross(rB, n);
	const float k = mA + mB + dot(uA, mul(invIA, uA)) + dot(uB, mul(invIB, uB)) + cfm;
	const float pos = dot(d, n);
	const bool locked = (lower == upper);

	if (lower <= upper)
	{
		float err = 0.f;
		float lo = -B3_INF;
		float hi = B3_INF;
		bool active = true;
		if (locked)
			err = pos - lower;
		else if (pos < lower)
		{
			err = pos - lower;
			lo = 0.f;
		}
		else if (pos > upper)
		{
			err = pos - upper;
			hi = 0.f;
		}
		else
			active = false;

		if (active)
		{
			const float vrel = dot(linB + cross(angB, rB) - linA - cross(angA, rA), n);
			const float lambda = solveRow(k, vrel, erp * invDt * err, lo, hi, scale, accLimit);
			applyLinearImpulse(lambda * n, rA, rB, mA, mB, invIA, invIB, linA, angA, linB, angB);
		}
	}

	if (motorMax > 0.f && !locked)
	{
		const float vrel = dot(linB + cross(angB, rB) - linA - cross(angA, rA), n);
		const float lambda = solveRow(k, vrel, -motorVel, -motorMax, motorMax, scale, accMotor);
		applyLinearImpulse(lambda * n, rA, rB, mA, mB, invIA, invIB, linA, angA, linB, angB);
	}
}

//! Limit/lock row plus motor row about one angular axis n; angle is the signed twist about n.
//! k is the row's effective mass, Schur-corrected by the caller when an anchor block is active.
void solveAngularAxis(float3 n, float angle, float lower, float upper, float motorVel, float motorMax,
					  float erp, float cfm, float invDt, float scale, float k,
					  float3x3 invIA, float3x3 invIB, inout float3 angA, inout float3 angB,
					  inout float accLimit, inout float accMotor)
{
	const bool locked = (lower == upper);

	if (lower <= upper)
	{
		float err = 0.f;
		float lo = -B3_INF;
		float hi = B3_INF;
		bool active = true;
		if (locked)
			err = angle - lower;
		else if (angle < lower)
		{
			err = angle - lower;
			lo = 0.f;
		}
		else if (angle > upper)
		{
			err = angle - upper;
			hi = 0.f;
		}
		else
			active = false;

		if (active)
		{
			const float vrel = dot(angB - angA, n);
			const float lambda = solveRow(k, vrel, erp * invDt * err, lo, hi, scale, accLimit);
			applyAngularImpulse(lambda * n, invIA, invIB, angA, angB);
		}
	}

	if (motorMax > 0.f && !locked)
	{
		const float vrel = dot(angB - angA, n);
		const float lambda = solveRow(k, vrel, -motorVel, -motorMax, motorMax, scale, accMotor);
		applyAngularImpulse(lambda * n, invIA, invIB, angA, angB);
	}
}

//! Signed twist of q about the axis whose quaternion component is passed; exact for a pure
//! single-axis rotation and a stable small-angle measure otherwise.
float twistAngle(float4 q, float component)
{
	return 2.f * atan2(component, q.w);
}

//! Angular row effective mass. With a locked anchor the bare inertia is wrong by the Schur term:
//! the true resistance about n is I + m r^2 (rotation about the PIVOT), not I about the COM.
float angularKEff(float3 n, float cfm, float3x3 invIA, float3x3 invIB,
				  bool anchorLocked, float3x3 kAnchorInv, float3 rA, float3 rB)
{
	float k = dot(n, mul(invIA, n)) + dot(n, mul(invIB, n)) + cfm;
	if (anchorLocked)
	{
		const float3 c = cross(mul(invIA, n), rA) + cross(mul(invIB, n), rB);
		k -= dot(c, mul(kAnchorInv, c));
	}
	return k;
}

[numthreads(WG_SIZE, 1, 1)]
void CSSolveJoints(uint3 tid : SV_DispatchThreadID)
{
	const uint jointIndex = tid.x;
	if (jointIndex >= Params[0].numJoints)
		return;

	const b3IrrJoint j = Joints[jointIndex];
	if ((j.flags & B3_CONSTRAINT_FLAG_ENABLED) == 0)
		return;
	if (j.constraintType < B3_GPU_POINT2POINT_CONSTRAINT_TYPE
		|| j.constraintType > B3_GPU_D6_CONSTRAINT_TYPE)
		return;

	const uint bodyA = (uint)j.rbA;
	const uint bodyB = (uint)j.rbB;
	if (bodyA >= Params[0].numBodies || bodyB >= Params[0].numBodies || bodyA == bodyB)
		return;

	const b3RigidBodyData a = Bodies[bodyA];
	const b3RigidBodyData b = Bodies[bodyB];
	if (a.invMass == 0.f && b.invMass == 0.f)
		return;

	const float3x3 invIA = worldInvInertia(bodyA, a.quat);
	const float3x3 invIB = worldInvInertia(bodyB, b.quat);

	const float invDt = 1.f / max(Params[0].deltaTime, 1e-6f);
	const float scale = JointScale[jointIndex];
	const bool ownErp = (j.flags & B3_CONSTRAINT_FLAG_OWN_ERP) != 0;
	const float erp = ownErp ? j.erp : Params[0].erp;
	const float cfm = ownErp ? max(j.cfm, 0.f) : 0.f;

	const float3 rA = quatRotate(a.quat, j.pivotInA.xyz);
	const float3 rB = quatRotate(b.quat, j.pivotInB.xyz);
	float3 pA, pB;
	osPairFrame(a, b, pA, pB);
	const float3 anchorA = pA + rA;
	const float3 anchorB = pB + rB;

	b3IrrJointAccum acc = JointAccum[jointIndex];

	if (j.constraintType == B3_GPU_POINT2POINT_CONSTRAINT_TYPE
		|| j.constraintType == B3_GPU_FIXED_CONSTRAINT_TYPE)
	{
		// Linear row set: hold the two anchors coincident. Shared by both types.
		{
			const float3 velA = a.linVel.xyz + cross(a.angVel.xyz, rA);
			const float3 velB = b.linVel.xyz + cross(b.angVel.xyz, rB);
			const float3 relVel = velB - velA;

			const float3x3 sA = skew(rA);
			const float3x3 sB = skew(rB);
			const float3x3 identity = float3x3(1.f, 0.f, 0.f, 0.f, 1.f, 0.f, 0.f, 0.f, 1.f);
			const float3x3 k = (a.invMass + b.invMass + cfm) * identity
							 - mul(sA, mul(invIA, sA)) - mul(sB, mul(invIB, sB));

			float3x3 kInv;
			if (invert3x3(k, kInv))
			{
				const float3 bias = erp * invDt * (anchorB - anchorA);
				const float3 impulse = mul(kInv, -(relVel + bias)) * scale;

				if (finite3(impulse))
				{
					acc.lin.xyz += impulse;
					if (a.invMass > 0.f)
						atomicAddVel(bodyA, -impulse * a.invMass, -mul(invIA, cross(rA, impulse)));
					if (b.invMass > 0.f)
						atomicAddVel(bodyB, impulse * b.invMass, mul(invIB, cross(rB, impulse)));
				}
			}
		}

		if (j.constraintType == B3_GPU_FIXED_CONSTRAINT_TYPE)
		{
			// Angular row set: drive B's orientation onto A's, offset by the authored relative target.
			const float4 target = quatMul(a.quat, j.relTargetAB);
			float4 delta = quatMul(target, quatConj(b.quat));
			if (delta.w < 0.f)
				delta = -delta;

			const float3 theta = 2.f * delta.xyz;   // small-angle axis*angle taking B onto its target
			const float3 relW = b.angVel.xyz - a.angVel.xyz;
			const float3x3 identity = float3x3(1.f, 0.f, 0.f, 0.f, 1.f, 0.f, 0.f, 0.f, 1.f);

			float3x3 kAngInv;
			if (invert3x3(invIA + invIB + cfm * identity, kAngInv))
			{
				const float3 torqueImpulse = mul(kAngInv, erp * invDt * theta - relW) * scale;
				if (finite3(torqueImpulse))
				{
					acc.ang.xyz += torqueImpulse;
					if (a.invMass > 0.f)
						atomicAddVel(bodyA, float3(0.f, 0.f, 0.f), -mul(invIA, torqueImpulse));
					if (b.invMass > 0.f)
						atomicAddVel(bodyB, float3(0.f, 0.f, 0.f), mul(invIB, torqueImpulse));
				}
			}
		}

		JointAccum[jointIndex] = acc;
		return;
	}

	// Generic path: hinge, slider, cone-twist and 6-DoF are all per-axis rows in the joint frame.
	const float4 qFA = quatMul(a.quat, j.frameInA);
	const float4 qFB = quatMul(b.quat, j.frameInB);
	const float3 ax = quatRotate(qFA, float3(1.f, 0.f, 0.f));
	const float3 ay = quatRotate(qFA, float3(0.f, 1.f, 0.f));
	const float3 az = quatRotate(qFA, float3(0.f, 0.f, 1.f));

	// B's frame expressed in A's frame; w >= 0 picks the short arc.
	float4 rel = quatMul(quatConj(qFA), qFB);
	if (rel.w < 0.f)
		rel = -rel;

	float3 linA = a.linVel.xyz;
	float3 angA = a.angVel.xyz;
	float3 linB = b.linVel.xyz;
	float3 angB = b.angVel.xyz;
	const float3 d = anchorB - anchorA;

	// All-locked anchors (hinge/cone-twist) get a coupled 3x3 block: per-axis rows converge at the
	// K condition number (~150 for a light bob), far past the iteration budget.
	const bool linLockedAll = (j.linLower.x == j.linUpper.x) && (j.linLower.y == j.linUpper.y)
							&& (j.linLower.z == j.linUpper.z);
	const float3x3 sA = skew(rA);
	const float3x3 sB = skew(rB);
	const float3x3 identity = float3x3(1.f, 0.f, 0.f, 0.f, 1.f, 0.f, 0.f, 0.f, 1.f);
	const float3x3 kAnchor = (a.invMass + b.invMass + cfm) * identity
						   - mul(sA, mul(invIA, sA)) - mul(sB, mul(invIB, sB));
	float3x3 kAnchorInv;
	const bool anchorBlock = linLockedAll && invert3x3(kAnchor, kAnchorInv);

	// Angular rows first, anchor last: Gauss-Seidel leaves the LAST row exact, and a limit row is
	// near-dependent with the anchor's tangential row, so whichever runs last wins - anchors must
	// never visibly separate, an angle may overshoot a little.
	solveAngularAxis(ax, twistAngle(rel, rel.x), j.angLower.x, j.angUpper.x, j.angMotorVel.x,
					 j.angMotorMaxImpulse.x, erp, cfm, invDt, scale,
					 angularKEff(ax, cfm, invIA, invIB, anchorBlock, kAnchorInv, rA, rB),
					 invIA, invIB, angA, angB, acc.ang.x, acc.angMotor.x);

	if (j.constraintType == B3_GPU_CONETWIST_CONSTRAINT_TYPE)
	{
		// Swing = rel with its x twist removed; one unilateral row about the swing axis keeps the
		// cone, instead of two per-axis limits that would square the cone.
		const float tl = max(length(float2(rel.x, rel.w)), 1e-12f);
		const float4 twist = float4(rel.x / tl, 0.f, 0.f, rel.w / tl);
		const float4 swing = quatMul(rel, quatConj(twist));
		const float sinHalf = length(swing.yz);
		const float swingAngle = 2.f * atan2(sinHalf, swing.w);
		const float span = j.angUpper.y;
		if (sinHalf > 1e-7f && swingAngle > span)
		{
			const float3 axis = quatRotate(qFA, float3(0.f, swing.y, swing.z) / sinHalf);
			solveAngularAxis(axis, swingAngle, -B3_INF, span, 0.f, 0.f, erp, cfm, invDt, scale,
							 angularKEff(axis, cfm, invIA, invIB, anchorBlock, kAnchorInv, rA, rB),
							 invIA, invIB, angA, angB, acc.ang.y, acc.angMotor.y);
		}
	}
	else
	{
		solveAngularAxis(ay, twistAngle(rel, rel.y), j.angLower.y, j.angUpper.y, j.angMotorVel.y,
						 j.angMotorMaxImpulse.y, erp, cfm, invDt, scale,
						 angularKEff(ay, cfm, invIA, invIB, anchorBlock, kAnchorInv, rA, rB),
						 invIA, invIB, angA, angB, acc.ang.y, acc.angMotor.y);
		solveAngularAxis(az, twistAngle(rel, rel.z), j.angLower.z, j.angUpper.z, j.angMotorVel.z,
						 j.angMotorMaxImpulse.z, erp, cfm, invDt, scale,
						 angularKEff(az, cfm, invIA, invIB, anchorBlock, kAnchorInv, rA, rB),
						 invIA, invIB, angA, angB, acc.ang.z, acc.angMotor.z);
	}

	if (anchorBlock)
	{
		const float3 target = anchorA + j.linLower.x * ax + j.linLower.y * ay + j.linLower.z * az;
		const float3 relVel = linB + cross(angB, rB) - linA - cross(angA, rA);
		const float3 bias = erp * invDt * (anchorB - target);
		const float3 impulse = mul(kAnchorInv, -(relVel + bias)) * scale;
		if (finite3(impulse))
		{
			acc.lin.xyz += impulse;
			applyLinearImpulse(impulse, rA, rB, a.invMass, b.invMass, invIA, invIB,
							   linA, angA, linB, angB);
		}
	}
	else
	{
		solveLinearAxis(ax, j.linLower.x, j.linUpper.x, j.linMotorVel.x, j.linMotorMaxImpulse.x,
						erp, cfm, invDt, scale, d, rA, rB, a.invMass, b.invMass, invIA, invIB,
						linA, angA, linB, angB, acc.lin.x, acc.linMotor.x);
		solveLinearAxis(ay, j.linLower.y, j.linUpper.y, j.linMotorVel.y, j.linMotorMaxImpulse.y,
						erp, cfm, invDt, scale, d, rA, rB, a.invMass, b.invMass, invIA, invIB,
						linA, angA, linB, angB, acc.lin.y, acc.linMotor.y);
		solveLinearAxis(az, j.linLower.z, j.linUpper.z, j.linMotorVel.z, j.linMotorMaxImpulse.z,
						erp, cfm, invDt, scale, d, rA, rB, a.invMass, b.invMass, invIA, invIB,
						linA, angA, linB, angB, acc.lin.z, acc.linMotor.z);
	}

	const float3 dLinA = linA - a.linVel.xyz;
	const float3 dAngA = angA - a.angVel.xyz;
	const float3 dLinB = linB - b.linVel.xyz;
	const float3 dAngB = angB - b.angVel.xyz;
	if (finite3(dLinA) && finite3(dAngA) && finite3(dLinB) && finite3(dAngB))
	{
		if (a.invMass > 0.f)
			atomicAddVel(bodyA, dLinA, dAngA);
		if (b.invMass > 0.f)
			atomicAddVel(bodyB, dLinB, dAngB);
		JointAccum[jointIndex] = acc;
	}
}

//! Layout probe: writes back what the GPU actually reads per field, so a C++/HLSL packing
//! divergence surfaces as data instead of as mysterious physics. 12 ints per joint.
[numthreads(WG_SIZE, 1, 1)]
void CSDebugJointLayout(uint3 tid : SV_DispatchThreadID)
{
	const uint jointIndex = tid.x;
	if (jointIndex >= Params[0].numJoints)
		return;

	const b3IrrJoint j = Joints[jointIndex];
	const uint base = jointIndex * 12;
	VelocityDelta[base + 0] = j.constraintType;
	VelocityDelta[base + 1] = j.rbA;
	VelocityDelta[base + 2] = j.rbB;
	VelocityDelta[base + 3] = asint(j.breakingImpulseThreshold);
	VelocityDelta[base + 4] = asint(j.pivotInA.y);
	VelocityDelta[base + 5] = asint(j.pivotInB.y);
	VelocityDelta[base + 6] = j.flags;
	VelocityDelta[base + 7] = j.uid;
	VelocityDelta[base + 8] = asint(j.erp);
	VelocityDelta[base + 9] = asint(j.cfm);
	VelocityDelta[base + 10] = asint(j.angUpper.x);
	VelocityDelta[base + 11] = asint(j.angMotorMaxImpulse.z);   // last field: proves the stride
}

//! Folds the accumulated fixed-point deltas into the bodies and clears them for the next
//! iteration, so the solve kernel never has to zero what it is concurrently adding to.
[numthreads(WG_SIZE, 1, 1)]
void CSApplyJointVelocityDeltas(uint3 tid : SV_DispatchThreadID)
{
	const uint bodyIndex = tid.x;
	if (bodyIndex >= Params[0].numBodies)
		return;

	const uint base = bodyIndex * 6;
	const float inv = 1.0 / FIXED_SCALE;

	b3RigidBodyData body = Bodies[bodyIndex];
	body.linVel.xyz += float3(VelocityDelta[base + 0], VelocityDelta[base + 1],
							  VelocityDelta[base + 2]) * inv;
	body.angVel.xyz += float3(VelocityDelta[base + 3], VelocityDelta[base + 4],
							  VelocityDelta[base + 5]) * inv;
	Bodies[bodyIndex] = body;

	for (uint i = 0; i < 6; ++i)
		VelocityDelta[base + i] = 0;
}

//! Once per step after the last iteration: breaks joints whose applied impulse passed their
//! threshold, publishes the broken set, and resets the per-joint accumulators.
[numthreads(WG_SIZE, 1, 1)]
void CSFinishJoints(uint3 tid : SV_DispatchThreadID)
{
	const uint jointIndex = tid.x;
	if (jointIndex >= Params[0].numJoints)
		return;

	b3IrrJoint j = JointsRW[jointIndex];
	const b3IrrJointAccum acc = JointAccum[jointIndex];

	float applied;
	if (j.constraintType == B3_GPU_POINT2POINT_CONSTRAINT_TYPE
		|| j.constraintType == B3_GPU_FIXED_CONSTRAINT_TYPE)
		applied = length(acc.lin.xyz) + length(acc.ang.xyz);
	else
	{
		const float3 one = float3(1.f, 1.f, 1.f);
		applied = dot(abs(acc.lin.xyz), one) + dot(abs(acc.ang.xyz), one)
				+ dot(abs(acc.linMotor.xyz), one) + dot(abs(acc.angMotor.xyz), one);
	}

	if ((j.flags & B3_CONSTRAINT_FLAG_ENABLED) != 0 && j.breakingImpulseThreshold > 0.f
		&& finite1(applied) && applied > j.breakingImpulseThreshold)
	{
		j.flags = (j.flags & ~B3_CONSTRAINT_FLAG_ENABLED) | B3_CONSTRAINT_FLAG_BROKEN;
		JointsRW[jointIndex] = j;
	}

	JointStatus[jointIndex] = ((j.flags & B3_CONSTRAINT_FLAG_BROKEN) != 0) ? 1u : 0u;

	b3IrrJointAccum zero;
	zero.lin = float4(0.f, 0.f, 0.f, 0.f);
	zero.ang = zero.lin;
	zero.linMotor = zero.lin;
	zero.angMotor = zero.lin;
	JointAccum[jointIndex] = zero;
}
