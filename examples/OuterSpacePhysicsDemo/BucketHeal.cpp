/*******************************************************
* Copyright (C) 2026 David Delisle granyte@hotmail.com
*
* This file is part of Outerspace.
*
* Outerspace can not be copied and/or distributed without the express
* permission of David Delisle
*******************************************************/
#include "BucketHeal.h"

#include "GpuPhysicsBackend.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace
{
const float kHealStep = 1.f / 60.f;

int EnvOr(const char* name, int fallback)
{
	const char* env = getenv(name);
	return env && *env ? atoi(env) : fallback;
}

double EnvOrDouble(const char* name, double fallback)
{
	const char* env = getenv(name);
	return env && *env ? atof(env) : fallback;
}

/**
 * @brief Dispatches a reduction and reads its per-bucket half.
 * @param backend Backend to hash.
 * @param out Receives one hash per bucket.
 * @return False if bucket hashing is unavailable.
 */
bool HashBuckets(GpuPhysicsBackend& backend, std::vector<unsigned long long>& out)
{
	if (!backend.BeginHashState())
		return false;
	return backend.FetchHashBuckets(out);
}

/**
 * @brief Dispatches a reduction and reads the root hash.
 * @param backend Backend to hash.
 * @param out Receives the root.
 * @return False if no hash could be produced.
 */
bool HashRoot(GpuPhysicsBackend& backend, unsigned long long& out)
{
	if (!backend.BeginHashState())
		return false;
	return backend.FetchHashState(out);
}

unsigned long long XorFold(const std::vector<unsigned long long>& buckets)
{
	unsigned long long folded = 0;
	for (size_t i = 0; i < buckets.size(); ++i)
		folded ^= buckets[i];
	return folded;
}

/// Largest position difference between two published transform sets, in metres.
double MaxSeparation(const std::vector<DemoTransform>& a, const std::vector<DemoTransform>& b)
{
	const size_t n = a.size() < b.size() ? a.size() : b.size();
	double worst = 0.0;
	for (size_t i = 0; i < n; ++i)
	{
		double d = 0.0;
		for (int x = 0; x < 3; ++x)
		{
			const double e = (double)a[i].position[x] - (double)b[i].position[x];
			d += e * e;
		}
		d = std::sqrt(d);
		if (d > worst)
			worst = d;
	}
	return worst;
}

void Check(const char* label, bool ok, const char* detail, int& failures)
{
	printf("  %-36s %s  %s\n", label, ok ? "PASS" : "FAIL", detail);
	if (!ok)
		++failures;
	fflush(stdout);
}

/**
 * @brief Restores a full snapshot and republishes its transforms.
 * @param backend Backend to restore into.
 * @param state Snapshot bytes.
 * @param out Receives the restored transform set.
 * @return False if the restore was refused.
 */
bool Restore(GpuPhysicsBackend& backend, const std::vector<unsigned char>& state,
			 std::vector<DemoTransform>& out)
{
	if (state.empty() || !backend.ApplyState(&state[0], state.size()))
		return false;
	out = backend.Transforms();
	return true;
}

struct RecoveryResult
{
	RecoveryResult()
		: Ticks(-1), HeldTicks(0), FirstGap(0.0), WorstGap(0.0), WorstBuckets(0), WorstPayload(0),
		  Broke(false) { }

	int Ticks;              ///< Heals applied before the hashes re-matched, or -1.
	int HeldTicks;          ///< Unhealed ticks the match then survived.
	double FirstGap;        ///< Worst body separation at injection, metres.
	double WorstGap;        ///< Worst separation seen at any point.
	int WorstBuckets;
	size_t WorstPayload;    ///< Largest heal payload sent, bytes.
	bool Broke;             ///< A backend call refused; the numbers mean nothing.
};

/**
 * @brief Diverges one peer, heals it every tick, and reports when the hashes re-matched.
 *
 * Both peers live in this one backend, swapped in and out by full snapshot, so the healing side
 * only ever sees states the authority actually reached.
 *
 * @param backend Backend both peers run in.
 * @param scene Scene both peers build.
 * @param settleSteps Steps run before the divergence is injected.
 * @param healTicks Ticks the heal is given before it is called a failure.
 * @param holdTicks Unhealed ticks a re-match must survive to count.
 * @param firstBody First displaced body index.
 * @param bodyCount Bodies displaced.
 * @param metres Displacement along x.
 * @param settings Heal tuning.
 * @param verbose Print a per-30-tick trace.
 * @param out Receives the measurements.
 */
void RunRecovery(GpuPhysicsBackend& backend, const SceneSpec& scene, int settleSteps, int healTicks,
				 int holdTicks, int firstBody, int bodyCount, double metres,
				 const PhysicsBackend::HealSettings& settings, bool verbose, RecoveryResult& out)
{
	out = RecoveryResult();
	if (!backend.Reset(scene))
	{
		out.Broke = true;
		return;
	}
	for (int i = 0; i < settleSteps; ++i)
		backend.Step(kHealStep);

	std::vector<unsigned char> authority;
	std::vector<unsigned char> local;
	const double delta[3] = {metres, 0.0, 0.0};
	if (!backend.CaptureState(authority) || authority.empty()
		|| !backend.DebugPerturbBodies(firstBody, bodyCount, delta)
		|| !backend.CaptureState(local))
	{
		out.Broke = true;
		return;
	}

	std::vector<DemoTransform> authorityPose;
	std::vector<DemoTransform> localPose;
	std::vector<unsigned long long> authorityBuckets;
	std::vector<unsigned long long> localBuckets;
	std::vector<int> diverged;
	std::vector<unsigned char> payload;

	int healsApplied = 0;
	for (int tick = 0; tick < healTicks; ++tick)
	{
		if (!Restore(backend, local, localPose) || !HashBuckets(backend, localBuckets)
			|| !Restore(backend, authority, authorityPose) || !HashBuckets(backend, authorityBuckets))
		{
			out.Broke = true;
			return;
		}

		diverged.clear();
		for (size_t b = 0; b < authorityBuckets.size() && b < localBuckets.size(); ++b)
			if (authorityBuckets[b] != localBuckets[b])
				diverged.push_back((int)b);
		if ((int)diverged.size() > out.WorstBuckets)
			out.WorstBuckets = (int)diverged.size();

		const double separation = MaxSeparation(authorityPose, localPose);
		if (tick == 0)
			out.FirstGap = separation;
		if (separation > out.WorstGap)
			out.WorstGap = separation;
		if (verbose && (tick % 30) == 0)
		{
			printf("    tick %4d: %d bucket(s) diverged, %d healing, worst gap %.6f m\n", tick,
				   (int)diverged.size(), backend.HealingBodyCount(), separation);
			fflush(stdout);
		}

		if (diverged.empty())
		{
			out.Ticks = healsApplied;
			break;
		}

		// Captured before the authority advances, so the healing side is corrected with the tick it
		// is actually on.
		const bool captured = backend.CaptureBuckets(diverged, payload);
		backend.Step(kHealStep);
		if (!captured || !backend.CaptureState(authority))
		{
			out.Broke = true;
			return;
		}
		if (payload.size() > out.WorstPayload)
			out.WorstPayload = payload.size();

		if (!Restore(backend, local, localPose)
			|| !backend.ApplyBucketsConverging(&payload[0], payload.size(), settings,
											   (double)kHealStep))
		{
			out.Broke = true;
			return;
		}
		backend.Step(kHealStep);
		if (!backend.CaptureState(local))
		{
			out.Broke = true;
			return;
		}
		++healsApplied;
	}

	if (out.Ticks < 0)
		return;

	// A matching hash is not proof of a matching SIMULATION: sleep state is not hashed. Run both
	// peers on with no further healing and require they stay identical.
	unsigned long long authorityHash = 0;
	unsigned long long localHash = 0;
	for (; out.HeldTicks < holdTicks; ++out.HeldTicks)
	{
		if (!Restore(backend, authority, authorityPose))
			break;
		backend.Step(kHealStep);
		if (!HashRoot(backend, authorityHash) || !backend.CaptureState(authority))
			break;

		if (!Restore(backend, local, localPose))
			break;
		backend.Step(kHealStep);
		if (!HashRoot(backend, localHash) || !backend.CaptureState(local))
			break;

		if (authorityHash != localHash)
			break;
	}
}
}  // namespace

int RunBucketHealTest(GpuPhysicsBackend& backend, const SceneSpec& handed, int settleSteps,
					  int healSteps)
{
	if (settleSteps < 1)
		settleSteps = 1;
	if (healSteps < 1)
		healSteps = 1;

	// Its own scene, not whatever Reset happened to be called with: the hook fires on the first
	// Reset of a run, which differs per entry point and made two runs incomparable.
	const int sceneKind = EnvOr("OSDEMO_HEAL_SCENE", -1);
	const SceneSpec built = sceneKind >= 0
		? BuildScene(sceneKind, EnvOr("OSDEMO_HEAL_SCALE", 4)) : handed;
	const SceneSpec& scene = built;

	printf("\nBucket heal: %s on %s (%d bodies), settle@%d heal<=%d\n", backend.Name(), scene.Name,
		   scene.BoxCount(), settleSteps, healSteps);
	fflush(stdout);

	int failures = 0;
	char detail[256];

	if (!backend.Reset(scene))
	{
		printf("  backend cannot reset - bucket heal not applicable\n");
		return 1;
	}
	const int bucketCount = backend.HashBucketCount();
	if (bucketCount <= 0)
	{
		printf("  bucket hashing unavailable (OSDEMO_BUCKET_HASH=0 or no reduction kernel)\n");
		return 1;
	}

	const int totalBodies = backend.BodyCount() + 1;   // body 0 is the static ground
	const int perBucket = (totalBodies + bucketCount - 1) / bucketCount;
	sprintf(detail, "%d buckets over %d bodies, %d per bucket", bucketCount, totalBodies, perBucket);
	Check("bucket split", true, detail, failures);

	for (int i = 0; i < settleSteps; ++i)
		backend.Step(kHealStep);

	// --- Root consistency: the root must be exactly the XOR of the buckets, unperturbed.
	unsigned long long root = 0;
	std::vector<unsigned long long> baseBuckets;
	const bool gotRoot = HashRoot(backend, root);
	const bool gotBuckets = HashBuckets(backend, baseBuckets);
	sprintf(detail, "%016llx vs %016llx over %d buckets", root, XorFold(baseBuckets),
			(int)baseBuckets.size());
	Check("root == XOR(buckets)",
		  gotRoot && gotBuckets && (int)baseBuckets.size() == bucketCount
			  && root == XorFold(baseBuckets), detail, failures);

	// --- Narrowing: one perturbed body must move exactly its own bucket's hash.
	const int probeBody = totalBodies / 2;
	const int probeBucket = probeBody / perBucket;
	const double probeDelta[3] = {0.25, 0.0, 0.0};
	std::vector<unsigned long long> probeBuckets;
	if (!backend.DebugPerturbBodies(probeBody, 1, probeDelta) || !HashBuckets(backend, probeBuckets))
	{
		Check("narrowing", false, "could not perturb or re-hash", failures);
	}
	else
	{
		int differing = 0;
		int firstDiffering = -1;
		for (size_t b = 0; b < probeBuckets.size() && b < baseBuckets.size(); ++b)
			if (probeBuckets[b] != baseBuckets[b])
			{
				++differing;
				if (firstDiffering < 0)
					firstDiffering = (int)b;
			}
		sprintf(detail, "body %d -> bucket %d; %d bucket(s) differ, first %d", probeBody,
				probeBucket, differing, firstDiffering);
		Check("only the perturbed bucket differs", differing == 1 && firstDiffering == probeBucket,
			  detail, failures);
	}

	// --- Recovery, at whatever the caller asked for.
	const int perturbBodies = EnvOr("OSDEMO_HEAL_BODIES", 3);
	// Unless the caller pinned it, measure true convergence: a forced snap would satisfy the
	// re-match check by construction rather than by healing.
	if (!getenv("OSDEMO_HEAL_MAX_ROUNDS"))
		backend.SetHealMaxRounds(0);
	const double perturbMetres = EnvOrDouble("OSDEMO_HEAL_PERTURB", 0.05);
	const int holdTicks = EnvOr("OSDEMO_HEAL_HOLD", 120);
	const int firstPerturbed = totalBodies / 2;
	// The struct default (5 s, the engine's) does NOT converge here: contact coupling spreads the
	// divergence faster than a 5 s rendezvous drains the healing set. See the sweep.
	PhysicsBackend::HealSettings settings;
	settings.HorizonSeconds = EnvOrDouble("OSDEMO_HEAL_HORIZON", 0.25);

	RecoveryResult primary;
	RunRecovery(backend, scene, settleSteps, healSteps, holdTicks, firstPerturbed, perturbBodies,
				perturbMetres, settings, true, primary);
	sprintf(detail, "%d ticks; %d bodies displaced %.4f m, horizon %.4f s, worst gap %.4f m, "
			"worst %d/%d buckets, payload <=%d bytes", primary.Ticks, perturbBodies, perturbMetres,
			settings.HorizonSeconds, primary.WorstGap, primary.WorstBuckets, bucketCount,
			(int)primary.WorstPayload);
	Check("hash re-matches exactly", !primary.Broke && primary.Ticks >= 0, detail, failures);
	if (primary.Ticks >= 0)
	{
		sprintf(detail, "%d/%d unhealed ticks stayed identical", primary.HeldTicks, holdTicks);
		Check("stays converged without healing", primary.HeldTicks == holdTicks, detail, failures);
	}

	// --- Sweep: where the boundary between "steers home" and "diverges" actually sits.
	if (EnvOr("OSDEMO_HEAL_SWEEP", 0) != 0)
	{
		const double horizons[] = {5.0, 1.0, 0.25, 0.05};
		const double sizes[] = {0.005, 0.05, 0.5};
		printf("    sweep (ticks to re-match, - = never within %d):\n", healSteps);
		printf("      %-12s", "displace\\h");
		for (size_t hh = 0; hh < sizeof(horizons) / sizeof(horizons[0]); ++hh)
			printf("%10.3fs", horizons[hh]);
		printf("\n");
		for (size_t s = 0; s < sizeof(sizes) / sizeof(sizes[0]); ++s)
		{
			printf("      %-12.4f", sizes[s]);
			for (size_t hh = 0; hh < sizeof(horizons) / sizeof(horizons[0]); ++hh)
			{
				PhysicsBackend::HealSettings sweep;
				sweep.HorizonSeconds = horizons[hh];
				RecoveryResult cell;
				RunRecovery(backend, scene, settleSteps, healSteps, 30, firstPerturbed,
							perturbBodies, sizes[s], sweep, false, cell);
				if (cell.Broke)
					printf("%11s", "err");
				else if (cell.Ticks < 0)
					printf("%11s", "-");
				else
					printf("%9d%s", cell.Ticks, cell.HeldTicks == 30 ? " " : "!");
			}
			printf("\n");
			fflush(stdout);
		}
	}

	backend.Reset(scene);
	printf("  %s\n", failures ? "BUCKET HEAL FAILED" : "Bucket heal: every check passed.");
	fflush(stdout);
	return failures;
}
