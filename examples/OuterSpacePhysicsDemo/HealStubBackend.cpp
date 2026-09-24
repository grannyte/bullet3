/*******************************************************
* Copyright (C) 2026 David Delisle granyte@hotmail.com
*
* This file is part of Outerspace.
*
* Outerspace can not be copied and/or distributed without the express
* permission of David Delisle
*******************************************************/
//
// Deliberately not a physics model: it only has to be bit-deterministic, cheap, and perturbable, so
// the resync protocol can be measured without a GPU or a real solver in the way.

#include "HealStubBackend.h"

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>

namespace
{
const unsigned int kBucketPayloadMagic = 0x4c414548u;   // 'HEAL'
const double kGravity = -9.81;
const double kRestitution = 0.5;

/**
 * @brief Reads an integer environment knob.
 * @param name Variable name.
 * @param fallback Value when unset or unparsable.
 * @return Parsed value.
 */
int EnvInt(const char* name, int fallback)
{
	const char* env = getenv(name);
	return env && *env ? atoi(env) : fallback;
}

/**
 * @brief Reads a floating-point environment knob.
 * @param name Variable name.
 * @param fallback Value when unset or unparsable.
 * @return Parsed value.
 */
double EnvDouble(const char* name, double fallback)
{
	const char* env = getenv(name);
	return env && *env ? atof(env) : fallback;
}

/**
 * @brief FNV-1a over raw bytes.
 * @param h Accumulator, updated in place.
 * @param data First byte.
 * @param bytes Length.
 */
void HashRaw(unsigned long long& h, const void* data, size_t bytes)
{
	const unsigned char* p = (const unsigned char*)data;
	for (size_t i = 0; i < bytes; ++i)
	{
		h ^= (unsigned long long)p[i];
		h *= 1099511628211ULL;
	}
}
}  // namespace

int HealStubBackend::BucketSize()
{
	const int size = EnvInt("OSDEMO_STUB_BUCKET", 64);
	return size > 0 ? size : 64;
}

HealStubBackend::HealStubBackend()
	: m_root(0), m_hashValid(false), m_lastStepMs(0.0), m_tick(0), m_healing(0),
	  m_divergeTick((unsigned long long)EnvInt("OSDEMO_STUB_DIVERGE_TICK", 0)),
	  m_divergeAt(EnvInt("OSDEMO_STUB_DIVERGE_AT", 100)),
	  m_divergeCount(EnvInt("OSDEMO_STUB_DIVERGE_COUNT", 3)),
	  m_divergeEps(EnvDouble("OSDEMO_STUB_DIVERGE_EPS", 0.05)),
	  m_coupling(EnvDouble("OSDEMO_STUB_COUPLING", 0.0))
{
}

bool HealStubBackend::Reset(const SceneSpec& scene)
{
	const int bodies = scene.BoxCount();
	if (bodies <= 0)
		return false;
	m_pos.assign(scene.BoxPositions.begin(), scene.BoxPositions.end());
	m_vel.assign(m_pos.size(), 0.0);
	m_scratch.assign(m_pos.size(), 0.0);
	m_transforms.assign((size_t)bodies, DemoTransform());
	m_buckets.clear();
	m_hashValid = false;
	m_tick = 0;
	m_healing = 0;
	return true;
}

void HealStubBackend::Step(float deltaTime)
{
	const std::chrono::high_resolution_clock::time_point begin =
		std::chrono::high_resolution_clock::now();
	const double dt = (double)deltaTime;
	const size_t bodies = m_pos.size() / 3;

	// Snapshotted rather than read in place: an in-place neighbour read would make the result depend
	// on iteration order, which is exactly the determinism this backend exists to guarantee.
	if (m_coupling != 0.0)
		m_scratch = m_pos;

	for (size_t i = 0; i < bodies; ++i)
	{
		double* p = &m_pos[i * 3];
		double* v = &m_vel[i * 3];
		v[1] += kGravity * dt;
		if (m_coupling != 0.0)
		{
			const size_t previous = i == 0 ? bodies - 1 : i - 1;
			v[1] += m_coupling * (m_scratch[previous * 3 + 1] - m_scratch[i * 3 + 1]) * dt;
		}
		for (int a = 0; a < 3; ++a)
			p[a] += v[a] * dt;
		if (p[1] < (double)kBoxHalfExtent)
		{
			p[1] = (double)kBoxHalfExtent;
			v[1] = -v[1] * kRestitution;
			v[0] *= 0.98;
			v[2] *= 0.98;
		}
	}

	++m_tick;
	if (m_divergeTick != 0 && m_tick == m_divergeTick)
		for (int k = 0; k < m_divergeCount; ++k)
		{
			const size_t body = (size_t)(m_divergeAt + k);
			if (body < bodies)
				m_pos[body * 3 + 1] += m_divergeEps;
		}

	for (size_t i = 0; i < bodies; ++i)
	{
		m_transforms[i].position[0] = (float)m_pos[i * 3 + 0];
		m_transforms[i].position[1] = (float)m_pos[i * 3 + 1];
		m_transforms[i].position[2] = (float)m_pos[i * 3 + 2];
		m_transforms[i].orientation[0] = 0.f;
		m_transforms[i].orientation[1] = 0.f;
		m_transforms[i].orientation[2] = 0.f;
		m_transforms[i].orientation[3] = 1.f;
	}

	m_hashValid = false;
	m_lastStepMs = std::chrono::duration<double, std::milli>(
					   std::chrono::high_resolution_clock::now() - begin).count();
}

unsigned long long HealStubBackend::HashBucket(int bucket) const
{
	const size_t bodies = m_pos.size() / 3;
	const size_t first = (size_t)bucket * (size_t)BucketSize();
	size_t last = first + (size_t)BucketSize();
	if (last > bodies)
		last = bodies;
	unsigned long long h = 14695981039346656037ULL;
	for (size_t i = first; i < last; ++i)
	{
		HashRaw(h, &m_pos[i * 3], sizeof(double) * 3);
		HashRaw(h, &m_vel[i * 3], sizeof(double) * 3);
	}
	return h;
}

void HealStubBackend::Rehash() const
{
	const int count = HashBucketCount();
	m_buckets.resize(count > 0 ? (size_t)count : 0);
	unsigned long long root = 14695981039346656037ULL;
	for (int b = 0; b < count; ++b)
	{
		m_buckets[(size_t)b] = HashBucket(b);
		HashRaw(root, &m_buckets[(size_t)b], sizeof(unsigned long long));
	}
	m_root = root;
	m_hashValid = true;
}

int HealStubBackend::HashBucketCount() const
{
	const size_t bodies = m_pos.size() / 3;
	if (bodies == 0)
		return 0;
	const size_t size = (size_t)BucketSize();
	return (int)((bodies + size - 1) / size);
}

bool HealStubBackend::HashState(unsigned long long& out) const
{
	if (m_pos.empty())
		return false;
	if (!m_hashValid)
		Rehash();
	out = m_root;
	return true;
}

bool HealStubBackend::BeginHashState()
{
	if (m_pos.empty())
		return false;
	Rehash();
	return true;
}

bool HealStubBackend::FetchHashState(unsigned long long& out)
{
	return HashState(out);
}

bool HealStubBackend::FetchHashBuckets(std::vector<unsigned long long>& out)
{
	out.clear();
	if (m_pos.empty())
		return false;
	if (!m_hashValid)
		Rehash();
	out = m_buckets;
	return !out.empty();
}

bool HealStubBackend::CaptureBuckets(const std::vector<int>& buckets,
									 std::vector<unsigned char>& out) const
{
	out.clear();
	const int count = HashBucketCount();
	if (count <= 0)
		return false;

	const unsigned int header[4] = {kBucketPayloadMagic, (unsigned int)BucketSize(),
								   (unsigned int)(m_pos.size() / 3), (unsigned int)buckets.size()};
	out.insert(out.end(), (const unsigned char*)header,
			   (const unsigned char*)header + sizeof(header));
	for (size_t k = 0; k < buckets.size(); ++k)
	{
		const int bucket = buckets[k];
		if (bucket < 0 || bucket >= count)
			return false;
		const unsigned int index = (unsigned int)bucket;
		out.insert(out.end(), (const unsigned char*)&index,
				   (const unsigned char*)&index + sizeof(index));
	}

	const size_t bodies = m_pos.size() / 3;
	for (size_t k = 0; k < buckets.size(); ++k)
	{
		const size_t first = (size_t)buckets[k] * (size_t)BucketSize();
		size_t last = first + (size_t)BucketSize();
		if (last > bodies)
			last = bodies;
		for (size_t i = first; i < last; ++i)
		{
			out.insert(out.end(), (const unsigned char*)&m_pos[i * 3],
					   (const unsigned char*)&m_pos[i * 3] + sizeof(double) * 3);
			out.insert(out.end(), (const unsigned char*)&m_vel[i * 3],
					   (const unsigned char*)&m_vel[i * 3] + sizeof(double) * 3);
		}
	}
	return true;
}

bool HealStubBackend::ApplyBucketsConverging(const unsigned char* data, size_t bytes,
											 const HealSettings& settings, double dtSeconds)
{
	if (!data || bytes < sizeof(unsigned int) * 4)
		return false;
	unsigned int header[4];
	memcpy(header, data, sizeof(header));
	const size_t bodies = m_pos.size() / 3;
	if (header[0] != kBucketPayloadMagic || header[1] != (unsigned int)BucketSize()
		|| header[2] != (unsigned int)bodies)
		return false;

	const unsigned int bucketCount = header[3];
	size_t offset = sizeof(header);
	if (bytes < offset + bucketCount * sizeof(unsigned int))
		return false;
	std::vector<unsigned int> buckets(bucketCount);
	if (bucketCount)
		memcpy(&buckets[0], data + offset, bucketCount * sizeof(unsigned int));
	offset += bucketCount * sizeof(unsigned int);

	const double blend = settings.HorizonSeconds > 0.0
							 ? (dtSeconds / settings.HorizonSeconds > 1.0
									? 1.0
									: dtSeconds / settings.HorizonSeconds)
							 : 1.0;
	int healing = 0;
	for (unsigned int k = 0; k < bucketCount; ++k)
	{
		const size_t first = (size_t)buckets[k] * (size_t)BucketSize();
		size_t last = first + (size_t)BucketSize();
		if (last > bodies)
			last = bodies;
		for (size_t i = first; i < last; ++i)
		{
			if (bytes < offset + sizeof(double) * 6)
				return false;
			double target[6];
			memcpy(target, data + offset, sizeof(target));
			offset += sizeof(target);

			double* p = &m_pos[i * 3];
			double* v = &m_vel[i * 3];
			double error = 0.0;
			double speed = 0.0;
			for (int a = 0; a < 3; ++a)
			{
				const double d = target[a] - p[a];
				error += d * d;
				speed += target[3 + a] * target[3 + a];
			}
			error = sqrt(error);
			speed = sqrt(speed);

			const double travel = speed * dtSeconds;
			double lockAt = settings.LockTravelFraction * travel;
			if (lockAt < settings.LockFloorMeters)
				lockAt = settings.LockFloorMeters;
			double snapAt = settings.SnapBodyRadii * (double)kBoxHalfExtent;
			const double horizonTravel = speed * settings.HorizonSeconds * settings.SnapTravelFraction;
			if (horizonTravel > snapAt)
				snapAt = horizonTravel;
			if (snapAt < settings.SnapFloorMeters)
				snapAt = settings.SnapFloorMeters;

			// Bit-exact copy, not another blend step: a converging solve is asymptotic, so without a
			// lock band the hashes would approach each other and never actually match.
			if (error <= lockAt || error >= snapAt)
			{
				memcpy(p, target, sizeof(double) * 3);
				memcpy(v, target + 3, sizeof(double) * 3);
				continue;
			}
			for (int a = 0; a < 3; ++a)
			{
				p[a] += (target[a] - p[a]) * blend;
				v[a] += (target[3 + a] - v[a]) * blend;
			}
			++healing;
		}
	}

	m_healing = healing;
	m_hashValid = false;
	for (size_t i = 0; i < bodies; ++i)
	{
		m_transforms[i].position[0] = (float)m_pos[i * 3 + 0];
		m_transforms[i].position[1] = (float)m_pos[i * 3 + 1];
		m_transforms[i].position[2] = (float)m_pos[i * 3 + 2];
	}
	return true;
}
