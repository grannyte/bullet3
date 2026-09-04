// Port of Bullet3OpenCL b3ComputeAabb - world AABB per body from its collidable.
// Non-sphere shapes rotate a CPU-uploaded LOCAL aabb rather than walking vertices, so one kernel
// covers hulls, compounds and meshes at the cost of a looser bound broadphase already tolerates.
// Included by B3Narrowphase.hlsl (f32) and B3NarrowphaseDS.hlsl (df64) - see B3Precision.hlsli.

#include "B3Precision.hlsli"

#define WG_SIZE 64

#define SHAPE_HEIGHT_FIELD 1
#define SHAPE_CONVEX_HULL 3
#define SHAPE_PLANE 4
#define SHAPE_CONCAVE_TRIMESH 5
#define SHAPE_COMPOUND_OF_CONVEX_HULLS 6
#define SHAPE_SPHERE 7

struct b3Collidable
{
	int numChildShapes;   // aliases bvhIndex
	float radius;         // aliases compoundBvhIndex
	int shapeType;
	int shapeIndex;       // aliases height
};

struct AabbParams
{
	uint numBodies;
	float margin;
	uint useSleep;
	float sweepDt;   // > 0: bound also covers this step's motion (speculative contacts / CCD)
};

#define SLEEP_ASLEEP_BIT 0x80000000u

StructuredBuffer<AabbParams> Params : register(t0);
StructuredBuffer<b3RigidBodyData> Bodies : register(t1);
StructuredBuffer<b3Collidable> Collidables : register(t2);
StructuredBuffer<b3AabbF> LocalAabbs : register(t3);   // indexed by COLLIDABLE index
StructuredBuffer<uint> SleepStates : register(t4);     // previous step's; unbound reads 0 = awake

RWStructuredBuffer<b3Aabb> OutAabbs : register(u0);

float3 quatRotate(float4 q, float3 v)
{
	const float3 qv = q.xyz;
	return v + 2.f * cross(qv, cross(qv, v) + q.w * v);
}

//! Columns of q's rotation matrix, i.e. the rotated basis vectors.
void quatToBasis(float4 q, out float3 bx, out float3 by, out float3 bz)
{
	bx = quatRotate(q, float3(1.f, 0.f, 0.f));
	by = quatRotate(q, float3(0.f, 1.f, 0.f));
	bz = quatRotate(q, float3(0.f, 0.f, 1.f));
}

[numthreads(WG_SIZE, 1, 1)]
void CSComputeWorldAabbs(uint3 tid : SV_DispatchThreadID)
{
	const uint bodyIndex = tid.x;
	if (bodyIndex >= Params[0].numBodies)
		return;

	// A sleeping body did not integrate, so its bound in OutAabbs still stands. Safe only because
	// sleep state is written after narrowphase, i.e. this reads the state the last step ended in.
	if (Params[0].useSleep != 0 && (SleepStates[bodyIndex] & SLEEP_ASLEEP_BIT) != 0)
		return;

	const b3RigidBodyData body = Bodies[bodyIndex];
	const b3Collidable col = Collidables[body.collidableIdx];
	const float margin = Params[0].margin;

	float3 centerOffset;
	float3 extent;
	float extraMargin;

	if (col.shapeType == SHAPE_SPHERE)
	{
		// Rotation-invariant, so this is exact rather than a bound.
		centerOffset = float3(0.f, 0.f, 0.f);
		extent = (col.radius + margin).xxx;
		extraMargin = 0.f;
	}
	else
	{
		const b3AabbF local = LocalAabbs[body.collidableIdx];
		const float3 center = 0.5f * (local.maxVec.xyz + local.minVec.xyz);
		const float3 localExtent = 0.5f * (local.maxVec.xyz - local.minVec.xyz);

		float3 bx, by, bz;
		quatToBasis(body.quat, bx, by, bz);

		// abs-matrix bound of the rotated box: exact for the OBB, conservative for the shape.
		extent = float3(dot(abs(float3(bx.x, by.x, bz.x)), localExtent),
						dot(abs(float3(bx.y, by.y, bz.y)), localExtent),
						dot(abs(float3(bx.z, by.z, bz.z)), localExtent));

		centerOffset = quatRotate(body.quat, center);
		extraMargin = margin;
	}

	// Swept bound: the box the body occupies over [0, dt] for its current velocity, so the
	// broadphase still pairs a body that crosses a thin wall within one step. Gravity added by the
	// integrator is not known here; at 60 Hz it is ~3 mm, inside the usual margin.
	const float sweepDt = Params[0].sweepDt;
	if (sweepDt > 0.f)
	{
		const float3 sweep = body.linVel.xyz * sweepDt;
		centerOffset += 0.5f * sweep;
		extent += 0.5f * abs(sweep);
		if (col.shapeType != SHAPE_SPHERE)
		{
			// Rotation over the step moves a corner by at most |w|*dt*circumradius.
			const float spin = sqrt(dot(body.angVel.xyz, body.angVel.xyz)) * sweepDt;
			extent += (spin * sqrt(dot(extent, extent))).xxx;
		}
	}

	// w stays 0: this pipeline carries the rigid index in the Morton buffer, so Bullet's
	// index-in-w convention would only be a second source of truth to keep in sync.
	OutAabbs[bodyIndex] = osBodyAabb(body, centerOffset, extent, extraMargin);
}
