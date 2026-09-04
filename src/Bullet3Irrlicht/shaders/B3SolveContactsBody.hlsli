// Port of Bullet3OpenCL solveContact/solveFriction - sequential-impulse contact solver.
//
// Velocity deltas accumulate as FIXED-POINT INTEGERS via InterlockedAdd: SM5.0 has no float
// atomics, and integer accumulation is order-independent, so thread order cannot change the result.

// Included by B3SolveContacts.hlsl (f32) and B3SolveContactsDS.hlsl (df64).

// rsqrt/normalize are APPROXIMATE instructions with vendor-defined precision; sqrt and divide
// are correctly rounded, so these forms are bit-identical across GPUs. Needed for lockstep.
#include "B3Precision.hlsli"

#define WG_SIZE 64
#define FIXED_SCALE 65536.0

// relPosA maps onto b3Contact4Data::m_worldPosB; xyz is already rA, w the separation.
struct b3Contact4Data
{
	float4 relPosA[4];
	float4 worldNormalOnB;
	uint restitutionAndFriction;
	int batchIdx;
	int bodyAPtrAndSignBit;
	int bodyBPtrAndSignBit;
	int childIndexA;
	int childIndexB;
	int unused1;
	int unused2;
};

struct SolverParams
{
	uint numContacts;
	uint numBodies;
	float deltaTime;
	float erp;              // Baumgarte position-correction factor
};

StructuredBuffer<SolverParams> Params : register(t0);
StructuredBuffer<b3Contact4Data> Contacts : register(t1);
StructuredBuffer<float4> InvInertia : register(t2);   // diagonal inverse inertia per body
// Contact points referencing each DYNAMIC body this solve; 0 for static bodies.
StructuredBuffer<int> BodyPointCount : register(t3);
// Bit 31 = asleep, from B3SleepBody.hlsli. Left UNBOUND when sleeping is off, which reads 0
// (awake) everywhere and restores the pre-sleep behaviour exactly.
StructuredBuffer<uint> SleepState : register(t4);

#define B3_ASLEEP_BIT 0x80000000u

RWStructuredBuffer<b3RigidBodyData> Bodies : register(u0);
// 6 ints per body: linear xyz then angular xyz, fixed-point.
RWStructuredBuffer<int> VelocityDelta : register(u1);
// 12 floats per contact: (normal, tangent0, tangent1) accumulated impulse for each of 4 points.
// One thread owns a whole contact, so plain floats stay deterministic here - no atomics needed.
RWStructuredBuffer<float> AccumImpulse : register(u2);
// Same data as BodyPointCount, as a UAV: the resident path builds it on the GPU instead of
// uploading a CPU-side tally that would need the contacts read back first.
RWStructuredBuffer<int> OutBodyPointCount : register(u3);

// Bodies this step's contacts reference. Every other body's VelocityDelta is provably zero, so the
// per-body passes can run over this list instead of the world and stay bit-identical.
StructuredBuffer<uint4> ActiveParams : register(t5);   // .x = list length, .y = this step's stamp
StructuredBuffer<uint> ActiveBodies : register(t6);
AppendStructuredBuffer<uint> ActiveBodiesOut : register(u4);
// Monotonic per-step stamp rather than a flag: dedup then needs no per-body clear of its own.
RWStructuredBuffer<uint> ActiveClaim : register(u5);

// Rolling-friction coefficient per COLLIDABLE (not per body), from the narrowphase registry.
// Unbound reads 0 everywhere, which disables the row exactly.
StructuredBuffer<float> RollingFriction : register(t7);
// 1 float per contact: rolling impulse magnitude spent this step, against the rf * normal budget.
RWStructuredBuffer<float> RollingAccum : register(u6);

#define IMPULSES_PER_POINT 3
#define MAX_CONTACT_POINTS 4
// Bullet's m_restitutionVelocityThreshold: no bounce below this approach speed, so a resting
// body settles instead of micro-bouncing forever.
#define B3_RESTITUTION_VELOCITY_THRESHOLD 0.2f
// b3Contact4Data.unused1 from the narrowphase: w is a POSITIVE gap that may close within dt.
#define B3_CONTACT_SPECULATIVE 1

float3 quatRotate(float4 q, float3 v)
{
	const float3 qv = q.xyz;
	return v + 2.f * cross(qv, cross(qv, v) + q.w * v);
}

// The 1/65536 truncation deadband is load-bearing: removing it re-enables friction chatter and
// collapses the Wall to 3.5. Fix the chatter (batching) before making this finer.
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

//! World-space inverse inertia for a diagonal local tensor: R * I^-1 * R^T applied to v.
float3 applyInvInertia(uint bodyIndex, float4 quat, float3 v)
{
	const float3 invI = InvInertia[bodyIndex].xyz;
	// Rotate into local space, scale, rotate back - avoids materialising the 3x3.
	const float4 invQuat = float4(-quat.xyz, quat.w);
	const float3 local = quatRotate(invQuat, v);
	return quatRotate(quat, local * invI);
}

//! Two unit vectors spanning the plane of n, chosen off the smaller component so the
//! normalisation never divides by a near-zero length.
void planeSpace(float3 n, out float3 t0, out float3 t1)
{
	if (abs(n.z) > 0.70710678f)
	{
		const float a = n.y * n.y + n.z * n.z;
		const float k = 1.f / sqrt(max(a, 1e-12f));
		t0 = float3(0.f, -n.z * k, n.y * k);
		t1 = float3(a * k, -n.x * t0.z, n.x * t0.y);
	}
	else
	{
		const float a = n.x * n.x + n.y * n.y;
		const float k = 1.f / sqrt(max(a, 1e-12f));
		t0 = float3(-n.y * k, n.x * k, 0.f);
		t1 = float3(-n.z * t0.y, n.z * t0.x, a * k);
	}
}

/**
 * Solves one tangent direction of a contact point and returns the impulse actually applied
 * along it - the change in the ACCUMULATED value, which may be negative.
 */
float solveTangentImpulse(uint slot, float3 tangent, float3 rA, float3 rB,
						  uint bodyA, uint bodyB, b3RigidBodyData a, b3RigidBodyData b,
						  float3 relVelVec, float maxFriction, float share)
{
	const float3 angA = applyInvInertia(bodyA, a.quat, cross(rA, tangent));
	const float3 angB = applyInvInertia(bodyB, b.quat, cross(rB, tangent));
	const float denom = a.invMass + b.invMass
					  + dot(tangent, cross(angA, rA) + cross(angB, rB));
	if (denom <= 1e-9f)
		return 0.f;

	const float relVel = dot(tangent, relVelVec);
	const float previous = AccumImpulse[slot];
	float updated = clamp(previous - relVel / denom * share, -maxFriction, maxFriction);
	if (isnan(updated) || isinf(updated))
		updated = previous;

	AccumImpulse[slot] = updated;
	return updated - previous;
}

[numthreads(WG_SIZE, 1, 1)]
void CSSolveContacts(uint3 tid : SV_DispatchThreadID)
{
	const uint contactIndex = tid.x;
	if (contactIndex >= Params[0].numContacts)
		return;

	const b3Contact4Data c = Contacts[contactIndex];
	const uint bodyA = (uint)abs(c.bodyAPtrAndSignBit);
	const uint bodyB = (uint)abs(c.bodyBPtrAndSignBit);

	const b3RigidBodyData a = Bodies[bodyA];
	const b3RigidBodyData b = Bodies[bodyB];

	// Both static: nothing to solve, and dividing by a zero effective mass would produce inf.
	if (a.invMass == 0.f && b.invMass == 0.f)
		return;

	// Neither side can move, so this contact's impulse is zero by construction. This early-out IS
	// the settled-scene saving - ten Jacobi iterations over a sleeping stack collapse to a load.
	if (((SleepState[bodyA] & SleepState[bodyB]) & B3_ASLEEP_BIT) != 0u)
		return;

	const float3 normal = c.worldNormalOnB.xyz;
	const int numPoints = (int)c.worldNormalOnB.w;
	if (numPoints <= 0)
		return;

	const float restitution = (float)(c.restitutionAndFriction & 0xffff) / 65535.f;
	// Bullet's combination rule. Read off the bodies, not the packed contact field, which the
	// box and planet narrowphases leave at zero.
	const float mu = max(a.frictionCoeff * b.frictionCoeff, 0.f);

	// Jacobi over-corrects a body by its contact count; the normal row saturates but friction's
	// symmetric clamp chatters and pumps energy. 1/N is the stability bound.
	const int contactLoad = max(BodyPointCount[bodyA], BodyPointCount[bodyB]);
	const float frictionRelax = 1.f / (float)max(contactLoad, 1);

	float3 t0, t1;
	planeSpace(normal, t0, t1);

	// Small for a touching pair, so rB stays float-exact however far the pair is from the origin.
	float3 pA, pB;
	osPairFrame(a, b, pA, pB);
	const float3 aRelB = pA - pB;

	const bool speculative = (c.unused1 & B3_CONTACT_SPECULATIVE) != 0;
	const float invDt = 1.f / max(Params[0].deltaTime, 1e-6f);
	float normalTotal = 0.f;

	// Local velocity copies: later points and the second sweep see earlier impulses, so the
	// under-relaxed rows converge; the 1/numPoints share and positive-only increments are calibrated.
	float3 linA = a.linVel.xyz;
	float3 angVA = a.angVel.xyz;
	float3 linB = b.linVel.xyz;
	float3 angVB = b.angVel.xyz;

	// Two sweeps per dispatch: one share-relaxed pass left ~0.75 of an impact per iteration (a 120 m/s
	// shell kept 1.52 m/s through a wall). Friction runs only on the last sweep, keeping its 1/N gain.
	const float share = 1.f / (float)numPoints;
	[loop]
	for (int sweep = 0; sweep < 2; ++sweep)
	{
	// Constant bound with a runtime guard: an unbounded loop indexing relPosA fails FXC (X3511).
	[unroll]
	for (int p = 0; p < MAX_CONTACT_POINTS; ++p)
	{
		const float3 rA = c.relPosA[p].xyz;
		const float separation = c.relPosA[p].w;

		if (p < numPoints && (separation <= 0.f || speculative))
		{
			const uint slot = (contactIndex * MAX_CONTACT_POINTS + (uint)p) * IMPULSES_PER_POINT;

			const float3 rB = rA + aRelB;

			const float3 velA = linA + cross(angVA, rA);
			const float3 velB = linB + cross(angVB, rB);
			const float3 relVelVec = velB - velA;
			const float relVel = dot(normal, relVelVec);

			// Effective mass along the normal, including the angular term.
			const float3 angA = applyInvInertia(bodyA, a.quat, cross(rA, normal));
			const float3 angB = applyInvInertia(bodyB, b.quat, cross(rB, normal));
			const float denom = a.invMass + b.invMass
							  + dot(normal, cross(angA, rA) + cross(angB, rB));

			if (denom > 1e-9f)
			{
				// Restitution only on approach, and only above Bullet's velocity threshold.
				const float restitutionTerm = (relVel < -B3_RESTITUTION_VELOCITY_THRESHOLD)
											? -restitution * relVel : 0.f;
				float target;
				if (separation > 0.f)
				{
					// Speculative: close the gap this step, or bounce off the predicted impact - whichever
					// asks more. A zero restitution must NOT win the max: it stops a projectile a step short.
					const float approach = -separation * invDt;
					target = (restitutionTerm > 0.f) ? max(restitutionTerm, approach) : approach;
				}
				else
				{
					// Baumgarte or restitution, never their SUM: stacking them on a penetrating
					// bounce injects erp*pen/dt of extra exit speed. Identical when restitution is 0.
					target = max(-Params[0].erp * separation / max(Params[0].deltaTime, 1e-6f),
								 restitutionTerm);
				}
				const float increment = (target - relVel) / denom * share;

				// Per-increment, not accumulated: letting a later iteration take impulse back
				// stops support propagating up a deep stack in the few iterations Jacobi gets.
				float applied = max(increment, 0.f);
				if (isnan(applied) || isinf(applied))
					applied = 0.f;

				// The ledger's normal slot is therefore only the running total, kept for one
				// purpose: giving the friction rows below a Coulomb budget.
				const float accumulated = AccumImpulse[slot + 0] + applied;
				AccumImpulse[slot + 0] = accumulated;
				normalTotal += accumulated;

				float3 P = normal * applied;

				if (sweep == 1)
				{
					const float maxFriction = mu * accumulated;
					P += t0 * solveTangentImpulse(slot + 1, t0, rA, rB, bodyA, bodyB, a, b,
												  relVelVec, maxFriction, frictionRelax);
					P += t1 * solveTangentImpulse(slot + 2, t1, rA, rB, bodyA, bodyB, a, b,
												  relVelVec, maxFriction, frictionRelax);
				}

				if (a.invMass > 0.f)
				{
					linA -= P * a.invMass;
					angVA -= applyInvInertia(bodyA, a.quat, cross(rA, P));
				}
				if (b.invMass > 0.f)
				{
					linB += P * b.invMass;
					angVB += applyInvInertia(bodyB, b.quat, cross(rB, P));
				}
			}
		}
	}
	}

	if (a.invMass > 0.f)
		atomicAddVel(bodyA, linA - a.linVel.xyz, angVA - a.angVel.xyz);
	if (b.invMass > 0.f)
		atomicAddVel(bodyB, linB - b.linVel.xyz, angVB - b.angVel.xyz);

	// Rolling friction: one angular-only row per manifold opposing the relative spin, budgeted
	// like Coulomb friction by rf * (normal impulse so far). Bullet's combination rule for rf.
	const float rf = RollingFriction[a.collidableIdx] * b.frictionCoeff
				   + RollingFriction[b.collidableIdx] * a.frictionCoeff;
	if (rf > 0.f && normalTotal > 0.f)
	{
		const float3 relAng = angVB - angVA;
		const float spinSq = dot(relAng, relAng);
		if (spinSq > 1e-12f)
		{
			const float spin = sqrt(spinSq);
			const float3 axis = relAng / spin;
			const float denomR = dot(axis, applyInvInertia(bodyA, a.quat, axis)
										 + applyInvInertia(bodyB, b.quat, axis));
			if (denomR > 1e-9f)
			{
				// Magnitude ledger only: the axis follows the spin between iterations, and a
				// monotone spend can never reverse and chatter the way a signed clamp would.
				const float used = RollingAccum[contactIndex];
				const float budget = max(rf * normalTotal - used, 0.f);
				float mag = min(spin / denomR * frictionRelax, budget);
				if (isnan(mag) || isinf(mag))
					mag = 0.f;
				RollingAccum[contactIndex] = used + mag;

				const float3 L = axis * mag;
				if (a.invMass > 0.f)
					atomicAddVel(bodyA, float3(0.f, 0.f, 0.f), applyInvInertia(bodyA, a.quat, L));
				if (b.invMass > 0.f)
					atomicAddVel(bodyB, float3(0.f, 0.f, 0.f), -applyInvInertia(bodyB, b.quat, L));
			}
		}
	}
}

//! Zeroes the warm-start ledger for this step's contacts.
[numthreads(WG_SIZE, 1, 1)]
void CSClearAccumImpulse(uint3 tid : SV_DispatchThreadID)
{
	const uint contactIndex = tid.x;
	if (contactIndex >= Params[0].numContacts)
		return;

	const uint base = contactIndex * MAX_CONTACT_POINTS * IMPULSES_PER_POINT;
	[unroll]
	for (uint i = 0; i < MAX_CONTACT_POINTS * IMPULSES_PER_POINT; ++i)
		AccumImpulse[base + i] = 0.f;
	RollingAccum[contactIndex] = 0.f;
}

[numthreads(WG_SIZE, 1, 1)]
void CSClearBodyPointCounts(uint3 tid : SV_DispatchThreadID)
{
	if (tid.x >= Params[0].numBodies)
		return;
	OutBodyPointCount[tid.x] = 0;
}

//! Collects every body this step's contacts name, deduplicated against the step stamp.
[numthreads(WG_SIZE, 1, 1)]
void CSBuildActiveBodies(uint3 tid : SV_DispatchThreadID)
{
	if (tid.x >= Params[0].numContacts)
		return;

	const b3Contact4Data c = Contacts[tid.x];
	const uint numBodies = Params[0].numBodies;
	const uint stamp = ActiveParams[0].y;
	const uint2 pair = uint2((uint)abs(c.bodyAPtrAndSignBit), (uint)abs(c.bodyBPtrAndSignBit));

	[unroll]
	for (uint k = 0; k < 2; ++k)
	{
		const uint body = (k == 0) ? pair.x : pair.y;
		if (body < numBodies)
		{
			uint previous;
			InterlockedExchange(ActiveClaim[body], stamp, previous);
			if (previous != stamp)
				ActiveBodiesOut.Append(body);
		}
	}
}

//! CSClearBodyPointCounts over the active list. Bodies off the list are never read by the solve.
[numthreads(WG_SIZE, 1, 1)]
void CSClearBodyPointCountsActive(uint3 tid : SV_DispatchThreadID)
{
	if (tid.x >= ActiveParams[0].x)
		return;
	OutBodyPointCount[ActiveBodies[tid.x]] = 0;
}

//! Per-body contact-point tally, matching the CPU path: static bodies stay at 0.
[numthreads(WG_SIZE, 1, 1)]
void CSAccumBodyPointCounts(uint3 tid : SV_DispatchThreadID)
{
	const uint contactIndex = tid.x;
	if (contactIndex >= Params[0].numContacts)
		return;

	const b3Contact4Data c = Contacts[contactIndex];
	const int points = clamp((int)c.worldNormalOnB.w, 0, MAX_CONTACT_POINTS);
	const uint bodyA = (uint)abs(c.bodyAPtrAndSignBit);
	const uint bodyB = (uint)abs(c.bodyBPtrAndSignBit);
	const uint numBodies = Params[0].numBodies;

	int ignored;
	if (bodyA < numBodies && Bodies[bodyA].invMass > 0.f)
		InterlockedAdd(OutBodyPointCount[bodyA], points, ignored);
	if (bodyB < numBodies && Bodies[bodyB].invMass > 0.f)
		InterlockedAdd(OutBodyPointCount[bodyB], points, ignored);
}

//! Folds the accumulated fixed-point deltas into one body and clears them for the next
//! iteration, so the solve kernel never has to zero what it is concurrently adding to.
void applyVelocityDelta(uint bodyIndex)
{
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

[numthreads(WG_SIZE, 1, 1)]
void CSApplyVelocityDeltas(uint3 tid : SV_DispatchThreadID)
{
	if (tid.x >= Params[0].numBodies)
		return;
	applyVelocityDelta(tid.x);
}

//! The same fold over the active list; a body off it has a zero delta, so folding it is a no-op.
[numthreads(WG_SIZE, 1, 1)]
void CSApplyVelocityDeltasActive(uint3 tid : SV_DispatchThreadID)
{
	if (tid.x >= ActiveParams[0].x)
		return;
	applyVelocityDelta(ActiveBodies[tid.x]);
}
