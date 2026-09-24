/*******************************************************
* Copyright (C) 2026 David Delisle granyte@hotmail.com
*
* This file is part of Outerspace.
*
* Outerspace can not be copied and/or distributed without the express
* permission of David Delisle
*******************************************************/
#ifndef OUTERSPACE_PHYSICS_DEMO_NETSYNC_H
#define OUTERSPACE_PHYSICS_DEMO_NETSYNC_H

#include "PhysicsBackend.h"

#include <memory>
#include <string>
#include <vector>

/// Which side of the link corrects the other, if either.
enum DemoNetMode
{
	DEMO_NET_LOCKSTEP = 0,   ///< Symmetric; hashes are compared, nothing is corrected.
	DEMO_NET_AUTHORITATIVE,  ///< Host snapshots, client applies.
	DEMO_NET_HEALING         ///< Lockstep, then a bucket-narrowed convergent resync on a mismatch.
};

/**
 * @brief What healing cost and whether it worked, for the HUD and the exit summary.
 *
 * Rounds staying 0 over a clean run is the assertion that lockstep pays nothing while in sync.
 */
struct DemoHealStats
{
	DemoHealStats()
		: Divergences(0), Recoveries(0), Rounds(0), BucketsCorrected(0), BucketsCompared(0),
		  BytesSent(0), LastTicksToRecovery(0), WorstTicksToRecovery(0), DivergedSinceTick(0),
		  Unsupported(false) { }

	int Divergences;                        ///< Root mismatches that opened a heal round.
	int Recoveries;                         ///< Divergences whose root later matched again.
	int Rounds;                             ///< Bucket exchanges performed.
	int BucketsCorrected;                   ///< Diverged buckets captured or applied, summed.
	int BucketsCompared;                    ///< Buckets diffed, summed; the narrowing denominator.
	unsigned long long BytesSent;           ///< Bucket hashes plus payloads: what a heal actually cost.
	unsigned long long LastTicksToRecovery;
	unsigned long long WorstTicksToRecovery;
	unsigned long long DivergedSinceTick;   ///< Barrier an outstanding divergence began at, else 0.
	bool Unsupported;                       ///< A peer cannot bucket its hash, so healing is off.
};

/**
 * @brief One networked run's settings, straight off the command line.
 */
struct DemoNetConfig
{
	DemoNetConfig()
		: Host(false), PeerHost("127.0.0.1"), Port(45654), Mode(DEMO_NET_LOCKSTEP), Interval(50),
		  CorrectEvery(2), ConnectTimeoutSeconds(60), StepSeconds(1.0 / 60.0)
	{
	}

	bool Host;                   ///< True listens for one peer, false dials it.
	std::string PeerHost;        ///< Address to dial when joining.
	int Port;
	DemoNetMode Mode;
	int Interval;                ///< Ticks between hash exchanges.
	int CorrectEvery;            ///< Hash exchanges between authoritative snapshots.
	int ConnectTimeoutSeconds;
	/// Fixed timestep, so a heal solve can turn a barrier into the seconds it is given to converge.
	double StepSeconds;
};

/**
 * @brief Runs one scene on both peers and reports whether their states agree.
 *
 * Both peers must be started with the same scene/steps flags; the handshake refuses the run rather
 * than reporting a divergence that is really a configuration mismatch.
 *
 * @param backends Every available backend, run one after another over the same link.
 * @param scene Scene both peers must have built identically.
 * @param steps Steps to run per backend.
 * @param fixedStep Timestep handed to every Step call.
 * @param cfg Transport and policy settings.
 * @return 0 when the peers ended in sync, 1 on a divergence still outstanding at the end, 2 on a
 *         link/handshake/capability failure.
 */
int RunNetSync(std::vector<std::unique_ptr<PhysicsBackend> >& backends, const SceneSpec& scene,
			   int steps, float fixedStep, const DemoNetConfig& cfg);

/**
 * @brief Lockstep link driven from a render loop instead of a batch loop.
 *
 * The barrier recv runs on a worker thread and MayStep only polls, so a peer that is behind stalls
 * the physics while the window keeps drawing and pumping messages rather than going unresponsive.
 */
class DemoNetInteractiveSession
{
public:
	/**
	 * @brief Starts listening or dialling on a worker thread and returns immediately.
	 *
	 * Accept/dial/handshake all block, so none of them may run on the render thread; poll
	 * Connecting/Connected/Failed instead.
	 *
	 * @param cfg Transport settings; only lockstep and healing are honoured, others become lockstep.
	 * @param scene Scene both peers must have built identically.
	 * @param backends Local backend list, compared by name during the handshake. Must outlive this.
	 * @return A session in the connecting state; never null.
	 */
	static DemoNetInteractiveSession* Begin(const DemoNetConfig& cfg, const SceneSpec& scene,
											const std::vector<std::unique_ptr<PhysicsBackend> >& backends);
	~DemoNetInteractiveSession();

	/**
	 * @brief Whether the local simulation may advance past the tick it has already reached.
	 * @param tick Ticks stepped so far.
	 * @return False while waiting on the peer's hash for the last barrier.
	 */
	bool MayStep(unsigned long long tick) const;

	/**
	 * @brief Hands this tick's hash to the peer. Call on barrier ticks only.
	 * @param tick Tick the hash describes.
	 * @param hash Local state hash.
	 */
	void PublishHash(unsigned long long tick, unsigned long long hash);

	/**
	 * @brief Advances the heal handshake by whatever step is ready, without blocking.
	 *
	 * Render thread only: every phase calls into the backend, and the GPU backend belongs to that
	 * thread. The worker only ever parks received frames for this to pick up.
	 *
	 * @param backend Backend being stepped this frame.
	 */
	void PumpHeal(PhysicsBackend* backend);

	DemoHealStats HealStats() const;
	/// A heal round is mid-handshake right now, so stepping is held at this barrier.
	bool HealingNow() const;
	/// Bodies the backend still has inside a heal horizon; 0 once converged and locked.
	int HealingBodies() const;
	/// A divergence is outstanding at this instant, unlike the latched Desynced().
	bool Diverged() const;

	/// Ticks between barriers, so callers need not carry the config around.
	int Interval() const;
	/// Still listening or dialling; nothing is gated yet.
	bool Connecting() const;
	bool Connected() const;
	/// The link never came up, or dropped. Carries a short reason for the HUD.
	bool Failed() const;
	const char* StatusText() const;
	bool Desynced() const;
	/// Barrier tick the peer and this side first disagreed on, or 0 while still in sync.
	unsigned long long FirstDivergentTick() const;
	int Matched() const;
	/// Last barrier the peer reported, so a stalled link is visible rather than looking like a hang.
	unsigned long long PeerTick() const;

private:
	DemoNetInteractiveSession();
	DemoNetInteractiveSession(const DemoNetInteractiveSession&);
	DemoNetInteractiveSession& operator=(const DemoNetInteractiveSession&);

	struct Impl;
	Impl* m_impl;
};

#endif  //OUTERSPACE_PHYSICS_DEMO_NETSYNC_H
