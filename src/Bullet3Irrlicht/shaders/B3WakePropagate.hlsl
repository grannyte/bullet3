// PREFER_FLOW_CONTROL - the wake walk is data-dependent and must stay a real branch.
// Extra wake hops for B3SleepBody.hlsli: CSPropagateWake covers one contact hop per step; these
// kernels re-run the walk so an island of bounded depth wakes within the same step.
// No body data is touched, so one file serves both precisions.

#include "B3SleepBody.hlsli"

// .x = hop index, starting at 2: hop 1 is CSPropagateWake / CSPropagateJointWake themselves.
StructuredBuffer<uint4> HopParams : register(t5);

// A source is a moving body, or one flagged by a STRICTLY EARLIER hop. WakeRequest stores the hop
// that flagged it, so a flag written by a concurrent thread of this hop is ignored until the next
// hop - the walk is deterministic regardless of thread order.
bool b3IsWakeSource(uint state, uint request, uint hop)
{
	return b3IsMoving(state) || (request != 0u && request < hop);
}

void b3FlagWake(uint body, uint hop)
{
	uint previous;
	InterlockedCompareExchange(WakeRequest[body], 0u, hop, previous);
}

[numthreads(WG_SIZE, 1, 1)]
void CSPropagateWakeHop(uint3 tid : SV_DispatchThreadID)
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

	const uint hop = HopParams[0].x;
	const uint stateA = SleepState[bodyA];
	const uint stateB = SleepState[bodyB];
	const uint reqA = WakeRequest[bodyA];
	const uint reqB = WakeRequest[bodyB];

	if (b3IsWakeSource(stateA, reqA, hop) && b3IsAsleep(stateB) && reqB == 0u)
		b3FlagWake(bodyB, hop);
	if (b3IsWakeSource(stateB, reqB, hop) && b3IsAsleep(stateA) && reqA == 0u)
		b3FlagWake(bodyA, hop);
}

//! Joint counterpart; ContactParams.x carries the JOINT count here.
[numthreads(WG_SIZE, 1, 1)]
void CSPropagateJointWakeHop(uint3 tid : SV_DispatchThreadID)
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

	const uint hop = HopParams[0].x;
	const uint stateA = SleepState[bodyA];
	const uint stateB = SleepState[bodyB];
	const uint reqA = WakeRequest[bodyA];
	const uint reqB = WakeRequest[bodyB];

	if (b3IsWakeSource(stateA, reqA, hop) && b3IsAsleep(stateB) && reqB == 0u)
		b3FlagWake(bodyB, hop);
	if (b3IsWakeSource(stateB, reqB, hop) && b3IsAsleep(stateA) && reqA == 0u)
		b3FlagWake(bodyA, hop);
}
