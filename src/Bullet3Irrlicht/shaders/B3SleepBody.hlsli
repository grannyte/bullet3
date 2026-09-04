// PREFER_FLOW_CONTROL - the wake walk is data-dependent and must stay a real branch.
// Approximate body deactivation for the GPU pipeline. Bullet's CPU side sleeps whole islands;
// islands are a CPU device, so this wakes ONE contact hop per step instead of building a graph.
//
// Included by B3Sleep.hlsl (f32) and B3SleepDS.hlsl (df64).

#include "B3Precision.hlsli"

#define WG_SIZE 64

// Bit 31 is the asleep flag, bits 0..30 the below-threshold step counter. One uint, so an UNBOUND
// SleepState reads 0 everywhere - i.e. all-awake, which is exactly the sleeping-disabled behaviour.
#define B3_ASLEEP_BIT 0x80000000u
#define B3_TIMER_MASK 0x7fffffffu

struct SleepParams
{
	uint numBodies;
	float moveThresholdSq;
	float angThresholdSq;
	uint sleepSteps;
};

// Reference a body's stillness is measured against, NOT refreshed while under threshold so a slow
// drift accumulates. Velocity cannot be used: a settled body here still oscillates past 0.8 m/s.
struct b3SleepRef
{
	float4 pos;
#ifdef OS_DS
	float4 posLo;
#endif
};

// Matches B3SolveContactsBody.hlsli; only the body indices and point count are read here.
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

StructuredBuffer<SleepParams> Params : register(t0);
// x is the live contact count, patched from the GPU-side append counter by CSPatchCountParams.
StructuredBuffer<uint4> ContactParams : register(t1);
StructuredBuffer<b3Contact4Data> Contacts : register(t2);

RWStructuredBuffer<b3RigidBodyData> Bodies : register(u0);
RWStructuredBuffer<uint> SleepState : register(u1);
// Separate from SleepState so the wake pass cannot cascade: a body woken this step is still read
// as asleep by every other thread, which is what bounds propagation to exactly one hop.
RWStructuredBuffer<uint> WakeRequest : register(u2);
RWStructuredBuffer<uint> AwakeCount : register(u3);
RWStructuredBuffer<b3SleepRef> RefPos : register(u4);
// Contacts with at least one non-sleeping body. An early-out inside the solver still pays for the
// thread and its loads; dropping the contact entirely is what removes ten iterations of work.
AppendStructuredBuffer<b3Contact4Data> LiveContacts : register(u5);

// Broadphase pairs, compacted the same way one phase earlier - see CSCompactPairs.
StructuredBuffer<uint2> Pairs : register(t3);

// Mirrors b3IrrJoint in B3SolveJointsBody.hlsli (240 bytes); only the endpoints and the enable
// bit are read, but the stride must match or every joint past the first misindexes.
struct b3SleepJoint
{
	int constraintType;
	int rbA;
	int rbB;
	float breakingImpulseThreshold;
	float4 pivotInA;
	float4 pivotInB;
	float4 relTargetAB;
	int flags;
	int uid;
	float erp;
	float cfm;
	float4 frameInA;
	float4 frameInB;
	float4 linLower;
	float4 linUpper;
	float4 angLower;
	float4 angUpper;
	float4 linMotorVel;
	float4 angMotorVel;
	float4 linMotorMaxImpulse;
	float4 angMotorMaxImpulse;
};

StructuredBuffer<b3SleepJoint> Joints : register(t4);
#define B3_JOINT_FLAG_ENABLED 1
AppendStructuredBuffer<uint2> LivePairs : register(u6);

// u7 is the LAST slot CS 5.0 allows, so the awake and sleeping body lists share it and are filled
// by two dispatches of CSCompactBodySet rather than one kernel emitting both.
AppendStructuredBuffer<uint> BodySet : register(u7);

bool b3IsAsleep(uint state) { return (state & B3_ASLEEP_BIT) != 0u; }

// Timer 0 means the body actually MOVED. "Awake" is too weak a wake source: stack neighbours reach
// the threshold on different steps, so awake-but-still bodies trade places asleep forever.
bool b3IsMoving(uint state) { return state == 0u; }

#ifndef OS_DS
float3 osRefDelta(b3RigidBodyData body, b3SleepRef r) { return body.pos.xyz - r.pos.xyz; }

b3SleepRef osMakeRef(b3RigidBodyData body)
{
	b3SleepRef r;
	r.pos = float4(body.pos.xyz, 0.f);
	return r;
}
#else
float3 osRefDelta(b3RigidBodyData body, b3SleepRef r)
{
	return float3(dsToFloat(dsSub(osPosX(body), float2(r.pos.x, r.posLo.x))),
				  dsToFloat(dsSub(osPosY(body), float2(r.pos.y, r.posLo.y))),
				  dsToFloat(dsSub(osPosZ(body), float2(r.pos.z, r.posLo.z))));
}

b3SleepRef osMakeRef(b3RigidBodyData body)
{
	b3SleepRef r;
	r.pos = float4(body.pos.xyz, 0.f);
	r.posLo = float4(body.posLo.xyz, 0.f);
	return r;
}
#endif

[numthreads(WG_SIZE, 1, 1)]
void CSClearWake(uint3 tid : SV_DispatchThreadID)
{
	if (tid.x == 0)
		AwakeCount[0] = 0u;
	if (tid.x >= Params[0].numBodies)
		return;
	WakeRequest[tid.x] = 0u;
}

//! One hop: a MOVING body flags its sleeping contact partners. Statics are permanently asleep, so
//! they are never the moving side and never wake anything spuriously.
[numthreads(WG_SIZE, 1, 1)]
void CSPropagateWake(uint3 tid : SV_DispatchThreadID)
{
	if (tid.x >= ContactParams[0].x)
		return;

	const b3Contact4Data c = Contacts[tid.x];
	if ((int)c.worldNormalOnB.w <= 0)
		return;

	const uint numBodies = Params[0].numBodies;
	const uint bodyA = (uint)abs(c.bodyAPtrAndSignBit);
	const uint bodyB = (uint)abs(c.bodyBPtrAndSignBit);
	if (bodyA >= numBodies || bodyB >= numBodies)
		return;

	const uint stateA = SleepState[bodyA];
	const uint stateB = SleepState[bodyB];

	if (b3IsMoving(stateA) && b3IsAsleep(stateB))
		WakeRequest[bodyB] = 1u;
	if (b3IsMoving(stateB) && b3IsAsleep(stateA))
		WakeRequest[bodyA] = 1u;
}

//! Joint counterpart to CSPropagateWake; ContactParams.x carries the JOINT count here. Without it
//! a sleeping body jointed to a mover hangs there violating its constraint until a contact wakes it.
[numthreads(WG_SIZE, 1, 1)]
void CSPropagateJointWake(uint3 tid : SV_DispatchThreadID)
{
	if (tid.x >= ContactParams[0].x)
		return;

	const b3SleepJoint j = Joints[tid.x];
	if ((j.flags & B3_JOINT_FLAG_ENABLED) == 0 || j.rbA < 0 || j.rbB < 0)
		return;

	const uint numBodies = Params[0].numBodies;
	const uint bodyA = (uint)j.rbA;
	const uint bodyB = (uint)j.rbB;
	if (bodyA >= numBodies || bodyB >= numBodies)
		return;

	const uint stateA = SleepState[bodyA];
	const uint stateB = SleepState[bodyB];

	if (b3IsMoving(stateA) && b3IsAsleep(stateB))
		WakeRequest[bodyB] = 1u;
	if (b3IsMoving(stateB) && b3IsAsleep(stateA))
		WakeRequest[bodyA] = 1u;
}

//! Drops every contact whose two bodies are both asleep, so the solver's indirect dispatch shrinks
//! to the live set. Must run AFTER CSUpdateSleep, or a body woken this step loses its contacts.
[numthreads(WG_SIZE, 1, 1)]
void CSCompactContacts(uint3 tid : SV_DispatchThreadID)
{
	if (tid.x >= ContactParams[0].x)
		return;

	const b3Contact4Data c = Contacts[tid.x];
	const uint numBodies = Params[0].numBodies;
	const uint bodyA = (uint)abs(c.bodyAPtrAndSignBit);
	const uint bodyB = (uint)abs(c.bodyBPtrAndSignBit);
	if (bodyA >= numBodies || bodyB >= numBodies)
		return;

	if (b3IsAsleep(SleepState[bodyA]) && b3IsAsleep(SleepState[bodyB]))
		return;

	LiveContacts.Append(c);
}

//! Drops every pair whose two bodies are both asleep: neither moved, so no contact between them can
//! have changed. Runs BEFORE narrowphase on the previous step's state, which is what keeps a settled
//! world off the SAT/clipping path entirely rather than merely off the solver.
[numthreads(WG_SIZE, 1, 1)]
void CSCompactPairs(uint3 tid : SV_DispatchThreadID)
{
	if (tid.x >= ContactParams[0].x)
		return;

	const uint2 pair = Pairs[tid.x];
	const uint numBodies = Params[0].numBodies;
	if (pair.x >= numBodies || pair.y >= numBodies)
		return;

	if (b3IsAsleep(SleepState[pair.x]) && b3IsAsleep(SleepState[pair.y]))
		return;

	LivePairs.Append(pair);
}

//! Emits the body indices of ONE sleep set, for a tree built over just that set.
//! ContactParams.x = body count, .y != 0 selects the sleeping set instead of the awake one.
[numthreads(WG_SIZE, 1, 1)]
void CSCompactBodySet(uint3 tid : SV_DispatchThreadID)
{
	if (tid.x >= ContactParams[0].x)
		return;

	const bool asleep = b3IsAsleep(SleepState[tid.x]);
	if (asleep == (ContactParams[0].y != 0u))
		BodySet.Append(tid.x);
}

//! Advances one body's timer and publishes the awake tally. Velocities are zeroed on the step a
//! body falls asleep so it cannot creep while the integrator is skipping it.
void updateSleepBody(uint i)
{
	b3RigidBodyData body = Bodies[i];
	if (body.invMass == 0.f)
	{
		SleepState[i] = B3_ASLEEP_BIT;
		return;
	}

	const float3 moved = osRefDelta(body, RefPos[i]);
	const float angSq = dot(body.angVel.xyz, body.angVel.xyz);
	const bool still = (dot(moved, moved) < Params[0].moveThresholdSq)
					&& (angSq < Params[0].angThresholdSq);

	uint timer = SleepState[i] & B3_TIMER_MASK;
	if (WakeRequest[i] != 0u || !still)
	{
		// Only on a reset: leaving the reference in place is what makes a slow, steady drift
		// accumulate past the threshold instead of hiding under it forever.
		timer = 0u;
		RefPos[i] = osMakeRef(body);
	}
	else
		timer = min(timer + 1u, Params[0].sleepSteps);

	const bool asleep = (Params[0].sleepSteps > 0u) && (timer >= Params[0].sleepSteps);
	if (asleep)
	{
		body.linVel.xyz = float3(0.f, 0.f, 0.f);
		body.angVel.xyz = float3(0.f, 0.f, 0.f);
		Bodies[i] = body;
	}
	else
	{
		uint ignored;
		InterlockedAdd(AwakeCount[0], 1u, ignored);
	}

	SleepState[i] = timer | (asleep ? B3_ASLEEP_BIT : 0u);
}

[numthreads(WG_SIZE, 1, 1)]
void CSUpdateSleepFull(uint3 tid : SV_DispatchThreadID)
{
	if (tid.x < Params[0].numBodies)
		updateSleepBody(tid.x);
}

//! The integrator never moves a sleeping body, so its stillness cannot have changed and the whole
//! pass reduces to rezeroing a velocity some awake neighbour's contact wrote into it.
[numthreads(WG_SIZE, 1, 1)]
void CSUpdateSleep(uint3 tid : SV_DispatchThreadID)
{
	const uint i = tid.x;
	if (i >= Params[0].numBodies)
		return;

	if (b3IsAsleep(SleepState[i]) && WakeRequest[i] == 0u
		&& all(Bodies[i].linVel.xyz == 0.f) && all(Bodies[i].angVel.xyz == 0.f))
		return;

	updateSleepBody(i);
}
