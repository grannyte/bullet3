#ifndef B3_PRECISION_HLSLI
#define B3_PRECISION_HLSLI

// Scalar abstraction shared by the single-precision and emulated-double (df64) physics kernels.
// OS_DS selects which definition block compiles; every kernel body calls the same helper names.
//
// Only ABSOLUTE-magnitude quantities are emulated - body position and world AABBs. Velocities,
// quaternions, contact points (already stored relative to body A), impulses and Morton codes are
// bounded or relative, so df64 would cost registers for no precision.

#ifdef OS_DS
#include "DoubleSingle.hlsli"
#endif

struct b3RigidBodyData
{
	float4 pos;
#ifdef OS_DS
	float4 posLo;
#endif
	float4 quat;
	float4 linVel;
	float4 angVel;
	int collidableIdx;
	float invMass;
	float restituitionCoeff;
	float frictionCoeff;
};

struct b3Aabb
{
	float4 minVec;
	float4 maxVec;
#ifdef OS_DS
	float4 minLo;
	float4 maxLo;
#endif
};

//! Always single precision, in both modes: shape-local bounds and the whole-scene merged bound,
//! whose only consumer is the relative Morton quantise.
struct b3AabbF
{
	float4 minVec;
	float4 maxVec;
};

#ifndef OS_DS
// -------------------------------------------------------------------------------------------
// Single precision. Every helper is the arithmetic the kernels used before the split.
// -------------------------------------------------------------------------------------------

//! Positions of a body pair in one common frame; every pair kernel works in differences only.
void osPairFrame(b3RigidBodyData a, b3RigidBodyData b, out float3 pA, out float3 pB)
{
	pA = a.pos.xyz;
	pB = b.pos.xyz;
}

void osIntegratePos(inout b3RigidBodyData body, float3 delta)
{
	body.pos.xyz += delta;
}

b3Aabb osBodyAabb(b3RigidBodyData body, float3 centerOffset, float3 extent, float margin)
{
	const float3 c = body.pos.xyz + centerOffset;
	b3Aabb r;
	r.minVec = float4(c - extent - margin.xxx, 0.f);
	r.maxVec = float4(c + extent + margin.xxx, 0.f);
	return r;
}

//! Lossy in df64 mode, and deliberately so: the only consumers are the Morton grid and the
//! broadphase raycast, both of which quantise relative to the scene extent anyway.
float3 osAabbMinF(b3Aabb a) { return a.minVec.xyz; }
float3 osAabbMaxF(b3Aabb a) { return a.maxVec.xyz; }

bool osAabbOverlap(b3Aabb a, b3Aabb b)
{
	if (a.minVec.x > b.maxVec.x || a.maxVec.x < b.minVec.x) return false;
	if (a.minVec.y > b.maxVec.y || a.maxVec.y < b.minVec.y) return false;
	if (a.minVec.z > b.maxVec.z || a.maxVec.z < b.minVec.z) return false;
	return true;
}

b3Aabb osAabbUnion(b3Aabb a, b3Aabb b)
{
	b3Aabb r;
	r.minVec = float4(min(a.minVec.xyz, b.minVec.xyz), 0.f);
	r.maxVec = float4(max(a.maxVec.xyz, b.maxVec.xyz), 0.f);
	return r;
}

#else
// -------------------------------------------------------------------------------------------
// Emulated double (df64). hi lives in the same field the f32 layout uses; lo rides alongside.
// -------------------------------------------------------------------------------------------

float2 dsNeg(float2 a) { return -a; }
float2 dsSub(float2 a, float2 b) { return dsAdd(a, -b); }

//! Exact ordering: the hi terms decide unless they are equal.
bool dsLess(float2 a, float2 b) { return (a.x < b.x) || (a.x == b.x && a.y < b.y); }

float2 osPosX(b3RigidBodyData b) { return float2(b.pos.x, b.posLo.x); }
float2 osPosY(b3RigidBodyData b) { return float2(b.pos.y, b.posLo.y); }
float2 osPosZ(b3RigidBodyData b) { return float2(b.pos.z, b.posLo.z); }

//! A-relative: the pair difference is small, so everything downstream stays float-exact. This is
//! the whole point of the split - df64 never enters the solver or clipping inner loops.
void osPairFrame(b3RigidBodyData a, b3RigidBodyData b, out float3 pA, out float3 pB)
{
	pA = float3(0.f, 0.f, 0.f);
	pB = float3(dsToFloat(dsSub(osPosX(b), osPosX(a))),
				dsToFloat(dsSub(osPosY(b), osPosY(a))),
				dsToFloat(dsSub(osPosZ(b), osPosZ(a))));
}

void osIntegratePos(inout b3RigidBodyData body, float3 delta)
{
	const float2 x = dsAddF(osPosX(body), delta.x);
	const float2 y = dsAddF(osPosY(body), delta.y);
	const float2 z = dsAddF(osPosZ(body), delta.z);
	body.pos.xyz = float3(x.x, y.x, z.x);
	body.posLo.xyz = float3(x.y, y.y, z.y);
}

b3Aabb osBodyAabb(b3RigidBodyData body, float3 centerOffset, float3 extent, float margin)
{
	const float2 cx = dsAddF(osPosX(body), centerOffset.x);
	const float2 cy = dsAddF(osPosY(body), centerOffset.y);
	const float2 cz = dsAddF(osPosZ(body), centerOffset.z);

	const float2 lox = dsAddF(dsAddF(cx, -extent.x), -margin);
	const float2 loy = dsAddF(dsAddF(cy, -extent.y), -margin);
	const float2 loz = dsAddF(dsAddF(cz, -extent.z), -margin);
	const float2 hix = dsAddF(dsAddF(cx, extent.x), margin);
	const float2 hiy = dsAddF(dsAddF(cy, extent.y), margin);
	const float2 hiz = dsAddF(dsAddF(cz, extent.z), margin);

	b3Aabb r;
	r.minVec = float4(lox.x, loy.x, loz.x, 0.f);
	r.maxVec = float4(hix.x, hiy.x, hiz.x, 0.f);
	r.minLo = float4(lox.y, loy.y, loz.y, 0.f);
	r.maxLo = float4(hix.y, hiy.y, hiz.y, 0.f);
	return r;
}

float3 osAabbMinF(b3Aabb a) { return a.minVec.xyz; }
float3 osAabbMaxF(b3Aabb a) { return a.maxVec.xyz; }

bool osAabbOverlap(b3Aabb a, b3Aabb b)
{
	if (dsLess(float2(b.maxVec.x, b.maxLo.x), float2(a.minVec.x, a.minLo.x))
		|| dsLess(float2(a.maxVec.x, a.maxLo.x), float2(b.minVec.x, b.minLo.x))) return false;
	if (dsLess(float2(b.maxVec.y, b.maxLo.y), float2(a.minVec.y, a.minLo.y))
		|| dsLess(float2(a.maxVec.y, a.maxLo.y), float2(b.minVec.y, b.minLo.y))) return false;
	if (dsLess(float2(b.maxVec.z, b.maxLo.z), float2(a.minVec.z, a.minLo.z))
		|| dsLess(float2(a.maxVec.z, a.maxLo.z), float2(b.minVec.z, b.minLo.z))) return false;
	return true;
}

b3Aabb osAabbUnion(b3Aabb a, b3Aabb b)
{
	b3Aabb r;
	r.minVec = a.minVec;
	r.minLo = a.minLo;
	r.maxVec = a.maxVec;
	r.maxLo = a.maxLo;

	if (dsLess(float2(b.minVec.x, b.minLo.x), float2(a.minVec.x, a.minLo.x))) { r.minVec.x = b.minVec.x; r.minLo.x = b.minLo.x; }
	if (dsLess(float2(b.minVec.y, b.minLo.y), float2(a.minVec.y, a.minLo.y))) { r.minVec.y = b.minVec.y; r.minLo.y = b.minLo.y; }
	if (dsLess(float2(b.minVec.z, b.minLo.z), float2(a.minVec.z, a.minLo.z))) { r.minVec.z = b.minVec.z; r.minLo.z = b.minLo.z; }

	if (dsLess(float2(a.maxVec.x, a.maxLo.x), float2(b.maxVec.x, b.maxLo.x))) { r.maxVec.x = b.maxVec.x; r.maxLo.x = b.maxLo.x; }
	if (dsLess(float2(a.maxVec.y, a.maxLo.y), float2(b.maxVec.y, b.maxLo.y))) { r.maxVec.y = b.maxVec.y; r.maxLo.y = b.maxLo.y; }
	if (dsLess(float2(a.maxVec.z, a.maxLo.z), float2(b.maxVec.z, b.maxLo.z))) { r.maxVec.z = b.maxVec.z; r.maxLo.z = b.maxLo.z; }

	r.minVec.w = r.maxVec.w = r.minLo.w = r.maxLo.w = 0.f;
	return r;
}

#endif

#endif
