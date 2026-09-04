// Port of Bullet3OpenCL primitiveContacts.cl - sphere/sphere contact generation.
// Separate file from B3Narrowphase.hlsl only so the append buffer can own u0: an append buffer
// bound elsewhere produced a silent zero count.

#define WG_SIZE 64
#define SHAPE_SPHERE 7

struct b3RigidBodyData
{
	float4 pos;
	float4 quat;
	float4 linVel;
	float4 angVel;
	int collidableIdx;
	float invMass;
	float restituitionCoeff;
	float frictionCoeff;
};

struct b3Collidable
{
	int numChildShapes;
	float radius;
	int shapeType;
	int shapeIndex;
};

//! Layout-identical to b3Contact4Data (112 bytes). The two packed 16-bit coefficients share one
//! uint here; HLSL has no 16-bit type at SM5.0, and the CPU side unpacks.
struct b3Contact4Data
{
	// Maps onto b3Contact4Data::m_worldPosB: xyz is RELATIVE TO BODY A's origin, w the separation.
	float4 relPosA[4];
	float4 worldNormalOnB;   // w = point count
	uint restitutionAndFriction;
	int batchIdx;
	int bodyAPtrAndSignBit;
	int bodyBPtrAndSignBit;
	int childIndexA;
	int childIndexB;
	int unused1;
	int unused2;
};

struct ContactParams
{
	uint numPairs;
	float collisionMargin;
	float speculativeDt;        // > 0: emit a speculative contact for a pair that meets within dt
	float ccdMotionThreshold;   // closing motion per step below which no speculative contact is made
};

// Marks b3Contact4Data.unused1: the point's w is a POSITIVE gap the solver may close this step.
#define B3_CONTACT_SPECULATIVE 1

StructuredBuffer<ContactParams> Params : register(t0);
StructuredBuffer<b3RigidBodyData> Bodies : register(t1);
StructuredBuffer<b3Collidable> Collidables : register(t2);
StructuredBuffer<uint2> Pairs : register(t3);

AppendStructuredBuffer<b3Contact4Data> OutContacts : register(u0);

//! 16-bit fixed point, matching b3Contact4Data's packed coefficient fields.
uint packCoeff(float restitution, float friction)
{
	const uint r = (uint)(saturate(restitution) * 65535.f + 0.5f);
	const uint f = (uint)(saturate(friction) * 65535.f + 0.5f);
	return (f << 16) | (r & 0xffffu);
}

[numthreads(WG_SIZE, 1, 1)]
void CSSphereSphereContacts(uint3 tid : SV_DispatchThreadID)
{
	const uint pairIndex = tid.x;
	if (pairIndex >= Params[0].numPairs)
		return;

	const uint bodyA = Pairs[pairIndex].x;
	const uint bodyB = Pairs[pairIndex].y;

	const b3RigidBodyData a = Bodies[bodyA];
	const b3RigidBodyData b = Bodies[bodyB];
	const b3Collidable ca = Collidables[a.collidableIdx];
	const b3Collidable cb = Collidables[b.collidableIdx];

	// Other shape combinations are handled by their own kernels over the same pair list.
	if (ca.shapeType != SHAPE_SPHERE || cb.shapeType != SHAPE_SPHERE)
		return;

	const float3 delta = b.pos.xyz - a.pos.xyz;
	const float distSqr = dot(delta, delta);
	const float radiusSum = ca.radius + cb.radius + Params[0].collisionMargin;

	int flags = 0;
	if (distSqr > radiusSum * radiusSum)
	{
		// Separated: exact linear sweep of the centre line, |delta + v t| = rA + rB for t in [0, dt].
		const float specDt = Params[0].speculativeDt;
		if (specDt <= 0.f)
			return;
		const float3 v = b.linVel.xyz - a.linVel.xyz;
		const float qa = dot(v, v);
		const float qb = 2.f * dot(delta, v);
		const float sumR = ca.radius + cb.radius;
		const float qc = distSqr - sumR * sumR;
		const float disc = qb * qb - 4.f * qa * qc;
		if (qa < 1e-12f || qb >= 0.f || disc < 0.f)
			return;
		const float toi = (-qb - sqrt(disc)) / (2.f * qa);
		const float closing = -qb / (2.f * sqrt(max(distSqr, 1e-12f)));
		if (isnan(toi) || isinf(toi) || toi > specDt || closing * specDt < Params[0].ccdMotionThreshold)
			return;
		flags = B3_CONTACT_SPECULATIVE;
	}

	// Concentric spheres have no defined normal; pick one rather than emit a NaN the solver
	// would propagate into every body it touches.
	float3 normalOnB = float3(0.f, 1.f, 0.f);
	float dist = 0.f;
	if (distSqr > 1e-12f)
	{
		dist = sqrt(distSqr);
		normalOnB = delta / dist;
	}

	const float penetration = dist - (ca.radius + cb.radius);
	// Point on B's surface, expressed from A's origin - `delta` is already b.pos - a.pos.
	const float3 pointRelA = delta - normalOnB * cb.radius;

	b3Contact4Data contact;
	contact.relPosA[0] = float4(pointRelA, penetration);
	contact.relPosA[1] = float4(0.f, 0.f, 0.f, 0.f);
	contact.relPosA[2] = float4(0.f, 0.f, 0.f, 0.f);
	contact.relPosA[3] = float4(0.f, 0.f, 0.f, 0.f);
	contact.worldNormalOnB = float4(normalOnB, 1.f);
	contact.restitutionAndFriction = packCoeff(a.restituitionCoeff * b.restituitionCoeff,
											   a.frictionCoeff * b.frictionCoeff);
	contact.batchIdx = 0;
	contact.bodyAPtrAndSignBit = (a.invMass == 0.f) ? -(int)bodyA : (int)bodyA;
	contact.bodyBPtrAndSignBit = (b.invMass == 0.f) ? -(int)bodyB : (int)bodyB;
	contact.childIndexA = -1;
	contact.childIndexB = -1;
	contact.unused1 = flags;
	contact.unused2 = 0;

	OutContacts.Append(contact);
}
