/*******************************************************
* Copyright (C) 2026 David Delisle granyte@hotmail.com
*
* This file is part of Outerspace.
*
* Outerspace can not be copied and/or distributed without the express
* permission of David Delisle
*******************************************************/
#ifndef OUTERSPACE_PHYSICS_DEMO_HEAL_STUB_BACKEND_H
#define OUTERSPACE_PHYSICS_DEMO_HEAL_STUB_BACKEND_H

#include "PhysicsBackend.h"

/**
 * @brief A deterministic CPU backend that implements the whole heal interface, for testing the
 *        resync protocol before a real backend implements it.
 *
 * Exists because the protocol has to be provable on its own: this needs no GPU, so two processes can
 * verify recovery without contending for the device or waiting on the backend half.
 */
class HealStubBackend : public PhysicsBackend
{
public:
	HealStubBackend();

	const char* Name() const { return "Stub (deterministic, healable)"; }
	bool Reset(const SceneSpec& scene);
	void Step(float deltaTime);
	const std::vector<DemoTransform>& Transforms() const { return m_transforms; }
	int BodyCount() const { return (int)(m_pos.size() / 3); }
	double LastStepMs() const { return m_lastStepMs; }

	bool HashState(unsigned long long& out) const;
	bool BeginHashState();
	bool FetchHashState(unsigned long long& out);

	int HashBucketCount() const;
	bool FetchHashBuckets(std::vector<unsigned long long>& out);
	bool CaptureBuckets(const std::vector<int>& buckets, std::vector<unsigned char>& out) const;
	bool ApplyBucketsConverging(const unsigned char* data, size_t bytes,
								const HealSettings& settings, double dtSeconds);
	int HealingBodyCount() const { return m_healing; }

	/**
	 * @brief Bodies per hash bucket (OSDEMO_STUB_BUCKET, default 64).
	 * @return Bucket width in bodies.
	 */
	static int BucketSize();

private:
	/**
	 * @brief Hashes one bucket's bodies from their exact stored bits.
	 * @param bucket Bucket index.
	 * @return Bucket hash.
	 */
	unsigned long long HashBucket(int bucket) const;

	/// Recomputes every bucket hash and the root reduction over them.
	void Rehash() const;

	std::vector<double> m_pos;   ///< 3 per body
	std::vector<double> m_vel;
	std::vector<double> m_scratch;
	std::vector<DemoTransform> m_transforms;
	mutable std::vector<unsigned long long> m_buckets;
	mutable unsigned long long m_root;
	mutable bool m_hashValid;
	double m_lastStepMs;
	unsigned long long m_tick;
	int m_healing;

	/// Injected one-shot perturbation, so one peer can be made to diverge provably.
	unsigned long long m_divergeTick;
	int m_divergeAt;
	int m_divergeCount;
	double m_divergeEps;
	/// Couples each body to its predecessor, so a divergence SPREADS the way a contact solver's does.
	double m_coupling;
};

#endif  //OUTERSPACE_PHYSICS_DEMO_HEAL_STUB_BACKEND_H
