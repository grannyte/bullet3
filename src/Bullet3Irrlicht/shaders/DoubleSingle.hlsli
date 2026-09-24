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

// Integer-bit seed, not rcp (D3D allows rcp 1 ULP of slack). Each product is named before it is
// subtracted: an unnamed one folds into a mad, which rounds once where the CPU rounds twice.
float dsRecipSeed(float b)
{
	precise float r0 = asfloat(0x7EF311C3u - asuint(b));
	precise float p0 = b * r0;
	precise float r1 = r0 * (2.0 - p0);
	precise float p1 = b * r1;
	precise float r2 = r1 * (2.0 - p1);
	precise float p2 = b * r2;
	precise float r3 = r2 * (2.0 - p2);
	return r3;
}

// Inverse-square-root seed, integer-only for the same reason as dsRecipSeed.
float dsRsqrtSeed(float a)
{
	precise float h = 0.5 * a;
	precise float r0 = asfloat(0x5F3759DFu - (asuint(a) >> 1));
	precise float p0 = h * (r0 * r0);
	precise float r1 = r0 * (1.5 - p0);
	precise float p1 = h * (r1 * r1);
	precise float r2 = r1 * (1.5 - p1);
	precise float p2 = h * (r2 * r2);
	precise float r3 = r2 * (1.5 - p2);
	return r3;
}

// Biased exponent field: 0 for zero or a denormal, 255 for inf/NaN.
int dsExponent(float a)
{
	return (int)((asuint(a) >> 23) & 0xFFu);
}

// Exact a * 2^k by exponent arithmetic; under floorExp, or from a zero exponent, a signed zero.
float dsScaleF(float a, int k, int floorExp)
{
	uint u = asuint(a);
	int e = (int)((u >> 23) & 0xFFu);
	uint sgn = u & 0x80000000u;
	int ne = e + k;
	uint scaled = (u & 0x807FFFFFu) | ((uint)clamp(ne, 0, 255) << 23);
	uint bits = (e == 0 || ne < floorExp) ? sgn : ((ne > 254) ? (sgn | 0x7F800000u) : scaled);
	return asfloat(bits);
}

float2 dsFlush(float2 a)
{
	return float2(dsScaleF(a.x, 0, 1), dsScaleF(a.y, 0, 1));
}

float2 dsPair(uint hi, uint lo)
{
	return float2(asfloat(hi), asfloat(lo));
}

// Biased exponent of 2^-48; see Df64.h's DsLoFloorExp.
static const int DS_LO_FLOOR_EXP = 79;

float2 dsTrimLo(float2 a)
{
	return float2(a.x, dsScaleF(a.y, 0, max(dsExponent(a.x) - 50, 1)));
}

bool dsBefore(float2 a, float2 b)
{
	return (a.x < b.x) || (a.x == b.x && a.y < b.y);
}

// Long division with two residual corrections on operands moved to [1, 2), so no step meets a
// denormal. A zero divisor answers +-FLT_MAX, and 0 for 0/0.
float2 dsDiv(float2 a, float2 b)
{
	int ea = dsExponent(a.x);
	int eb = dsExponent(b.x);
	int ka = 127 - ea;
	int kb = 127 - eb;

	precise float2 n = float2(dsScaleF(a.x, ka, 1), dsScaleF(a.y, ka, DS_LO_FLOOR_EXP));
	precise float2 d = float2(dsScaleF(b.x, kb, 1), dsScaleF(b.y, kb, DS_LO_FLOOR_EXP));
	d = (eb == 0) ? float2(1.0, 0.0) : d;
	precise float rb = dsRecipSeed(d.x);

	precise float q1 = n.x * rb;
	precise float2 r1 = dsAdd(n, -dsMulF(d, q1));
	precise float q2 = r1.x * rb;
	precise float2 r2 = dsAdd(r1, -dsMulF(d, q2));
	precise float q3 = r2.x * rb;

	precise float2 q = dsTwoSum(q1, q2);
	q.y += q3;
	precise float2 unit = dsQuickTwoSum(q.x, q.y);

	precise float2 quot = float2(dsScaleF(unit.x, kb - ka, 1), dsScaleF(unit.y, kb - ka, 1));
	quot = (ea == 0) ? float2(0.0, 0.0) : quot;

	precise float big = asfloat(0x7F7FFFFFu);
	precise float fallback = (ea == 0) ? 0.0 : (((asuint(a.x) & 0x80000000u) != 0u) ? -big : big);
	return (eb == 0) ? float2(fallback, 0.0) : quot;
}

float2 dsRecip(float2 a)
{
	return dsDiv(dsSet(1.0), a);
}

// Newton in df64 off the integer-derived seed, on the radicand moved to [1, 4) by an even power of
// two. A negative argument answers zero rather than NaN, matching the zero-divisor rule above.
float2 dsSqrt(float2 a)
{
	uint u = asuint(a.x);
	int e = (int)((u >> 23) & 0xFFu);
	bool zero = (e == 0);
	bool negative = !zero && (u & 0x80000000u) != 0u;
	int hk = (e - 127) >> 1;

	precise float2 s = float2(dsScaleF(a.x, -2 * hk, 1), dsScaleF(a.y, -2 * hk, DS_LO_FLOOR_EXP));
	s = (zero || negative) ? float2(1.0, 0.0) : s;
	precise float r = dsRsqrtSeed(s.x);
	precise float y = s.x * r;
	precise float h = 0.5 * r;

	precise float2 q0 = dsSet(y);
	precise float2 d0 = dsAdd(s, -dsMul(q0, q0));
	precise float2 q1 = dsAddF(q0, d0.x * h);
	precise float2 d1 = dsAdd(s, -dsMul(q1, q1));
	precise float2 q2 = dsAddF(q1, d1.x * h);

	precise float2 root = float2(dsScaleF(q2.x, hk, 1), dsScaleF(q2.y, hk, 1));
	return (zero || negative) ? dsSet(0.0) : root;
}

// Truncation toward zero by clearing fraction bits, so no rounding mode is involved.
float dsTruncF(float a)
{
	uint u = asuint(a);
	int e = (int)((u >> 23) & 0xFFu);
	uint mask = (1u << (uint)clamp(150 - e, 0, 31)) - 1u;
	uint bits = (e < 127) ? (u & 0x80000000u) : ((e >= 150) ? u : (u & ~mask));
	return asfloat(bits);
}

// A fractional hi decides alone; else lo is stepped toward hi's sign.
float2 dsTrunc(float2 x)
{
	precise float th = dsTruncF(x.x);
	precise float tl = dsTruncF(x.y);
	bool fractional = asuint(tl) != asuint(x.y);
	bool hiNeg = (asuint(x.x) & 0x80000000u) != 0u;
	bool loNeg = (asuint(x.y) & 0x80000000u) != 0u;
	precise float stepped = hiNeg ? tl + 1.0 : tl - 1.0;
	tl = (fractional && x.x != 0.0 && hiNeg != loNeg) ? stepped : tl;
	precise float2 whole = dsQuickTwoSum(x.x, tl);
	return (asuint(th) != asuint(x.x)) ? dsSet(th) : whole;
}

// a - b * trunc(a / b) on operands scaled so b is in [1, 2), then one correction toward a's sign;
// b == 0 or |a| < |b| answers a, |a / b| >= 2^100 answers 0.
float2 dsFmod(float2 a, float2 b)
{
	precise float2 x = dsFlush(a);
	int ea = dsExponent(x.x);
	int eb = dsExponent(b.x);
	bool passThrough = (eb == 0 || ea < eb);
	bool tooWide = (ea - eb > 100);
	int k = 127 - eb;

	precise float2 xs = dsTrimLo(float2(dsScaleF(x.x, k, 1), dsScaleF(x.y, k, 1)));
	precise float2 bs = float2(dsScaleF(b.x, k, 1), dsScaleF(b.y, k, DS_LO_FLOOR_EXP));
	bs = (eb == 0) ? float2(1.0, 0.0) : bs;
	precise float2 n = dsTrunc(dsDiv(xs, bs));
	precise float2 r = dsAdd(xs, -dsMul(bs, n));
	precise float2 m = ((asuint(bs.x) & 0x80000000u) != 0u) ? -bs : bs;

	bool nonNegative = (asuint(x.x) & 0x80000000u) == 0u;
	bool below = nonNegative ? dsBefore(r, float2(0.0, 0.0)) : !dsBefore(-m, r);
	bool above = nonNegative ? !dsBefore(r, m) : dsBefore(float2(0.0, 0.0), r);
	precise float2 raised = dsAdd(r, m);
	precise float2 lowered = dsAdd(r, -m);
	r = below ? raised : (above ? lowered : r);
	bool wrapped = nonNegative ? !dsBefore(r, m) : !dsBefore(-m, r);
	r = wrapped ? float2(0.0, 0.0) : r;
	precise float2 back = float2(dsScaleF(r.x, -k, 1), dsScaleF(r.y, -k, 1));
	return passThrough ? x : (tooWide ? float2(0.0, 0.0) : back);
}

// pi/2 reduction in three Cody-Waite parts, Taylor through x^9 in df64 and the rest of the tail in
// float; |x| >= 2^24 answers (0, 1).
void dsSinCos(float2 x, out float2 sinOut, out float2 cosOut)
{
	precise float2 xf = dsFlush(x);
	int ex = dsExponent(xf.x);

	precise float t = xf.x * asfloat(0x3F22F983u);
	precise float halfStep = ((asuint(t) & 0x80000000u) != 0u) ? -0.5 : 0.5;
	precise float rounded = dsTruncF(t + halfStep);
	precise float nf = (ex >= 107) ? rounded : 0.0;
	precise float2 p1 = dsTwoProd(asfloat(0x3FC90FDBu), nf);
	precise float2 p2 = dsTwoProd(asfloat(0xB33BBD2Eu), nf);
	precise float tail = asfloat(0xA6F72CEDu) * nf;
	precise float2 red = dsAdd(xf, -p1);
	red = dsAdd(red, -p2);
	red = dsAddF(red, -tail);
	precise float2 r = dsTrimLo(dsFlush((ex >= 107) ? red : xf));

	precise float quarter = dsTruncF(nf * 0.25);
	precise float fourQ = 4.0 * quarter;
	precise float qf = nf - fourQ;
	qf = (qf < 0.0) ? qf + 4.0 : qf;
	int quadrant = (int)qf;

	int er = dsExponent(r.x);

	precise float rh = r.x;
	precise float r2f = rh * rh;
	precise float r3f = r2f * rh;
	precise float sc = r3f * asfloat(0xBE2AAAABu);
	precise float cc = r2f * -0.5;
	precise float2 sMid = dsAddF(r, sc);
	precise float2 cMid = dsAddF(float2(1.0, 0.0), cc);

	precise float2 r2 = dsTrimLo(dsMul(r, r));
	precise float z = r2.x;

	precise float m = z * asfloat(0x274A963Cu);
	precise float ts = asfloat(0xAB573F9Fu) + m;
	m = z * ts;
	ts = asfloat(0x2F309231u) + m;
	m = z * ts;
	ts = asfloat(0xB2D7322Bu) + m;
	m = z * ts;
	precise float2 ps = dsTrimLo(dsAddF(dsPair(0x3638EF1Du, 0x292AD8E6u), m));
	ps = dsTrimLo(dsAdd(dsMul(ps, r2), dsPair(0xB9500D01u, 0x2C3FCBFDu)));
	ps = dsTrimLo(dsAdd(dsMul(ps, r2), dsPair(0x3C088889u, 0xAFEEEEEFu)));
	ps = dsTrimLo(dsAdd(dsMul(ps, r2), dsPair(0xBE2AAAABu, 0x31AAAAABu)));
	precise float2 sFull = dsAdd(r, dsMul(dsTrimLo(dsMul(r, r2)), ps));

	m = z * asfloat(0xA53413C3u);
	precise float tc = asfloat(0x29573F9Fu) + m;
	m = z * tc;
	tc = asfloat(0xAD49CBA5u) + m;
	m = z * tc;
	tc = asfloat(0x310F76C7u) + m;
	m = z * tc;
	tc = asfloat(0xB493F27Eu) + m;
	m = z * tc;
	precise float2 pc = dsTrimLo(dsAddF(dsPair(0x37D00D01u, 0xAABFCBFDu), m));
	pc = dsTrimLo(dsAdd(dsMul(pc, r2), dsPair(0xBAB60B61u, 0x2E13E93Fu)));
	pc = dsTrimLo(dsAdd(dsMul(pc, r2), dsPair(0x3D2AAAABu, 0xB0AAAAABu)));
	pc = dsTrimLo(dsAdd(dsMul(pc, r2), float2(-0.5, 0.0)));
	precise float2 cFull = dsAdd(float2(1.0, 0.0), dsMul(r2, pc));

	precise float2 s = (er < 97) ? r : ((er < 107) ? sMid : sFull);
	precise float2 c = (er < 97) ? float2(1.0, 0.0) : ((er < 107) ? cMid : cFull);

	precise float2 sq = (quadrant == 0) ? s : ((quadrant == 1) ? c : ((quadrant == 2) ? -s : -c));
	precise float2 cq = (quadrant == 0) ? c : ((quadrant == 1) ? -s : ((quadrant == 2) ? -c : s));
	sinOut = (ex >= 151) ? float2(0.0, 0.0) : dsFlush(sq);
	cosOut = (ex >= 151) ? float2(1.0, 0.0) : dsFlush(cq);
}

float2 dsSinFull(float2 x)
{
	float2 s, c;
	dsSinCos(x, s, c);
	return s;
}

float2 dsCosFull(float2 x)
{
	float2 s, c;
	dsSinCos(x, s, c);
	return c;
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
