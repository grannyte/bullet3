#ifndef DOUBLE_SINGLE_HLSLI
#define DOUBLE_SINGLE_HLSLI

// GPU-emulated double precision (df64): float2(hi, lo), Dekker/Knuth error-free transforms, FMA-free.
// `precise` is load-bearing: strip it and FXC at /O3 folds this whole file down to plain float32.

// Knuth two-sum: exact, no ordering assumption on |a| vs |b|.
float2 dsTwoSum(float a, float b)
{
	precise float s = a + b;
	precise float bb = s - a;
	precise float err = (a - (s - bb)) + (b - bb);
	return float2(s, err);
}

// Quick two-sum: exact, but ONLY valid when |a| >= |b|.
float2 dsQuickTwoSum(float a, float b)
{
	precise float s = a + b;
	precise float err = b - (s - a);
	return float2(s, err);
}

// Veltkamp split into two ~12-bit halves whose products sum losslessly; 4097 = 2^12 + 1.
void dsSplit(float a, out precise float hi, out precise float lo)
{
	precise float c = 4097.0 * a;
	hi = c - (c - a);
	lo = a - hi;
}

// Exact product of two float32s as a DS pair (Dekker's twoProd via split, not FMA).
float2 dsTwoProd(float a, float b)
{
	precise float p = a * b;
	precise float aHi, aLo, bHi, bLo;
	dsSplit(a, aHi, aLo);
	dsSplit(b, bHi, bLo);
	precise float err = ((aHi * bHi - p) + aHi * bLo + aLo * bHi) + aLo * bLo;
	return float2(p, err);
}

float2 dsSet(float a)
{
	return float2(a, 0.0);
}

float dsToFloat(float2 a)
{
	return a.x + a.y;
}

float2 dsAdd(float2 a, float2 b)
{
	precise float2 s = dsTwoSum(a.x, b.x);
	s.y += (a.y + b.y);
	return dsQuickTwoSum(s.x, s.y);
}

float2 dsAddF(float2 a, float b)
{
	precise float2 s = dsTwoSum(a.x, b);
	s.y += a.y;
	return dsQuickTwoSum(s.x, s.y);
}

float2 dsMul(float2 a, float2 b)
{
	precise float2 p = dsTwoProd(a.x, b.x);
	p.y += a.x * b.y + a.y * b.x;
	return dsQuickTwoSum(p.x, p.y);
}

float2 dsMulF(float2 a, float b)
{
	precise float2 p = dsTwoProd(a.x, b);
	p.y += a.y * b;
	return dsQuickTwoSum(p.x, p.y);
}

// One ds pair per axis; unpacked so call sites name the component instead of indexing.
struct ds3
{
	float2 x, y, z;
};

ds3 ds3FromFloat3(float3 v)
{
	ds3 r;
	r.x = dsSet(v.x);
	r.y = dsSet(v.y);
	r.z = dsSet(v.z);
	return r;
}

float3 ds3ToFloat3(ds3 v)
{
	return float3(dsToFloat(v.x), dsToFloat(v.y), dsToFloat(v.z));
}

// Scale a unit float3 direction by a DS magnitude; the direction gains nothing from being DS,
// the radius must not round until every other term has been folded in.
ds3 ds3ScaleDirection(float3 direction, float2 scale)
{
	ds3 r;
	r.x = dsMulF(scale, direction.x);
	r.y = dsMulF(scale, direction.y);
	r.z = dsMulF(scale, direction.z);
	return r;
}

// DS point (implicit w = 1) times a float4x4, row-vector left to match mul(pos, World) here.
// The matrix stays float32; the ACCUMULATION is what must not re-round frame to frame.
ds3 ds3TransformPoint(ds3 p, float4x4 m)
{
	ds3 r;

	float2 c0 = dsMulF(p.x, m._11);
	c0 = dsAdd(c0, dsMulF(p.y, m._21));
	c0 = dsAdd(c0, dsMulF(p.z, m._31));
	c0 = dsAddF(c0, m._41);
	r.x = c0;

	float2 c1 = dsMulF(p.x, m._12);
	c1 = dsAdd(c1, dsMulF(p.y, m._22));
	c1 = dsAdd(c1, dsMulF(p.z, m._32));
	c1 = dsAddF(c1, m._42);
	r.y = c1;

	float2 c2 = dsMulF(p.x, m._13);
	c2 = dsAdd(c2, dsMulF(p.y, m._23));
	c2 = dsAdd(c2, dsMulF(p.z, m._33));
	c2 = dsAddF(c2, m._43);
	r.z = c2;

	return r;
}

// As above, plus mLo -- the residual of the exact double transform against m -- folded back in.
// Recovers what m's f32 entries never held: ~2 m at planet radius, re-rounded every frame it spins.
ds3 ds3TransformPointCompensated(ds3 p, float4x4 m, float4x4 mLo)
{
	ds3 r = ds3TransformPoint(p, m);

	// mLo's terms are ~1e-7 of m's, so f32 resolves them; only the sum back in has to stay DS.
	precise float3 c = mul(float4(ds3ToFloat3(p), 1.0), mLo).xyz;

	r.x = dsAddF(r.x, c.x);
	r.y = dsAddF(r.y, c.y);
	r.z = dsAddF(r.z, c.z);

	return r;
}

#endif
