/*******************************************************
* Copyright (C) 2026 David Delisle granyte@hotmail.com
*
* This file is part of Outerspace.
*
* Outerspace can not be copied and/or distributed without the express
* permission of David Delisle
*******************************************************/
#ifndef OUTERSPACE_PHYSICS_DEMO_GPU_BACKEND_H
#define OUTERSPACE_PHYSICS_DEMO_GPU_BACKEND_H

#include "PhysicsBackend.h"

#include "Bullet3Irrlicht/b3IrrlichtLbvh.h"
#include "Bullet3Irrlicht/b3IrrlichtNarrowphase.h"
#include "Bullet3Irrlicht/b3IrrlichtSolver.h"
#include "Bullet3Irrlicht/b3GpuIrrlichtRigidBodyPipeline.h"
#include "Bullet3Irrlicht/b3IrrlichtJointSolver.h"
#include "Bullet3Irrlicht/b3IrrlichtSleep.h"
#include "Bullet3Collision/NarrowPhaseCollision/shared/b3RigidBodyData.h"

#include <map>
#include <memory>

namespace irr
{
namespace video
{
class IVideoDriver;
}
namespace io
{
class IFileSystem;
}
}  // namespace irr

/**
 * @brief The ported GPU pipeline: world AABBs, LBVH broadphase, SAT+clipping, contact and joint solve.
 *
 * On the vector path bodies live on the CPU between phases; LastTransferMs() measures exactly
 * what the resident path removes.
 */
class GpuPhysicsBackend : public PhysicsBackend
{
public:
	/**
	 * @brief Constructs one of the two GPU pipelines.
	 * @param driver Device driver every phase dispatches through.
	 * @param fileSystem Device filesystem, so a missing shader fails instead of binding a stray one.
	 * @param doubleSingle Emulated-double (df64) kernel set instead of the pure single-precision one.
	 */
	GpuPhysicsBackend(irr::video::IVideoDriver* driver, irr::io::IFileSystem* fileSystem,
					  bool doubleSingle = false);
	virtual ~GpuPhysicsBackend();

	/**
	 * @brief Compiles every kernel the pipeline needs.
	 * @return False if any shader is missing or fails to compile.
	 */
	bool Init();

	virtual const char* Name() const { return m_doubleSingle ? "GPU compute (df64)" : "GPU compute (f32)"; }
	virtual bool Reset(const SceneSpec& scene);
	/// Point-to-point joints are solved by b3IrrlichtJointSolver, which is what DemoConstraint means.
	virtual bool SupportsConstraints() const { return true; }
	/// Resident-only: the vector path integrates through std::vector uploads with no kinematic pass.
	virtual bool SupportsKinematics() const { return m_resident && m_kinematicEnabled; }
	virtual void Step(float deltaTime);
	virtual const std::vector<DemoTransform>& Transforms() const { return m_transforms; }
	virtual double LastStepMs() const { return m_lastStepMs; }
	virtual double LastTransferMs() const { return m_lastTransferMs; }
	virtual int LastPairCount() const { return m_lastPairCount; }

	/// The broadphase append exceeded its capacity, so pairs were dropped in an order neither
	/// machine controls. Any sync result after this point is meaningless.
	virtual bool LastPairOverflowed() const { return m_pairOverflow; }
	/**
	 * @brief Hashes the resident body buffer, not the packed transforms.
	 *
	 * The renderer-facing transform is float and drops df64's low word, so hashing it would call two
	 * states identical that the simulation itself can tell apart.
	 *
	 * @param out Receives the hash.
	 * @return False if no resident state exists yet.
	 */
	virtual bool HashState(unsigned long long& out) const;

	/**
	 * @brief Dispatches the reduction kernel; nothing is read back yet.
	 * @return False if the kernel is unavailable, leaving the caller on the synchronous path.
	 */
	virtual bool BeginHashState();

	/**
	 * @brief Maps the 8-byte reduction result.
	 * @param out Receives the hash.
	 * @return False if no reduction is outstanding.
	 */
	virtual bool FetchHashState(unsigned long long& out);

	/**
	 * @brief Snapshots the resident body buffer and the sleep state that gates it.
	 *
	 * Native layout, not Transforms(): the packed transform drops df64's low word, so a snapshot
	 * taken from it would restore a state the simulation can tell apart from the original.
	 *
	 * @param out Receives the payload; cleared first.
	 * @return False if no resident state exists yet.
	 */
	virtual bool CaptureState(std::vector<unsigned char>& out) const;

	/**
	 * @brief Overwrites the resident body and sleep buffers from a CaptureState payload.
	 *
	 * Every field is validated before anything is written, so a rejected payload leaves the
	 * simulation exactly as it was rather than half-applied.
	 *
	 * @param data First byte.
	 * @param bytes Length.
	 * @return False if the payload does not match this backend's precision, path, body count or size.
	 */
	virtual bool ApplyState(const unsigned char* data, size_t bytes);

	/**
	 * @brief Body-index buckets the reduction splits its hash into.
	 * @return 0 when OSDEMO_BUCKET_HASH=0, no reduction kernel, or no resident state.
	 */
	virtual int HashBucketCount() const;

	/**
	 * @brief Maps the per-bucket half of the reduction result.
	 * @param out Receives HashBucketCount() entries.
	 * @return False if no reduction is outstanding.
	 */
	virtual bool FetchHashBuckets(std::vector<unsigned long long>& out);

	/**
	 * @brief Snapshots only the bodies covered by the named buckets, sleep state included.
	 * @param buckets Bucket indices; order and duplicates do not matter.
	 * @param out Receives the payload; cleared first.
	 * @return False if bucket hashing is off or the buffers are unavailable.
	 */
	virtual bool CaptureBuckets(const std::vector<int>& buckets, std::vector<unsigned char>& out) const;

	/**
	 * @brief Steers the payload's bodies toward the authority, snapping once that lands exactly.
	 * @param data First byte of a CaptureBuckets payload.
	 * @param bytes Length.
	 * @param settings Heal tuning.
	 * @param dtSeconds Time since the previous heal application.
	 * @return False if the payload does not match this backend's precision, path, count or offset.
	 */
	virtual bool ApplyBucketsConverging(const unsigned char* data, size_t bytes,
										const HealSettings& settings, double dtSeconds);

	virtual int HealingBodyCount() const { return (int)m_healHorizon.size(); }

	/// Bodies snapped outright for exhausting OSDEMO_HEAL_MAX_ROUNDS correction attempts.
	int HealForcedSnapCount() const { return m_healForcedSnaps; }

	/// 0 disables the round cap. The heal test turns it off so it keeps measuring real convergence:
	/// a forced snap re-matches the hash by construction and would pass the check trivially.
	void SetHealMaxRounds(int rounds) { m_healMaxRounds = rounds; }

	/**
	 * @brief Test hook: displaces a contiguous run of bodies, producing a localised divergence.
	 * @param firstBody First body index, ground included.
	 * @param count Bodies to displace.
	 * @param delta Metres added to each body's position.
	 * @return False if there is no resident body buffer to perturb.
	 */
	bool DebugPerturbBodies(int firstBody, int count, const double delta[3]);

	virtual bool LastPhaseMs(double out[6]) const;
	virtual bool SetPhaseTiming(bool on);
	virtual int BodyCount() const { return (int)m_transforms.size(); }
	virtual int PairCapacity() const { return (int)m_maxPairs; }
	virtual int AwakeBodyCount() const { return m_awakeCount; }

	/**
	 * @brief Pulls the awake tally off the device now, outside the interval readback.
	 * @return False if sleeping is off or no step has run.
	 */
	virtual bool RefreshAwakeCount();

	/**
	 * @brief Rough device-buffer footprint for the current scene.
	 * @return Bytes across the pair, body, AABB and contact buffers.
	 */
	virtual long long GpuBufferBytes() const;

	/// Both precisions pack the 28-byte layout the renderer's cull pass expects; df64 emits it
	/// beside its hi/lo readback buffer, in absolute world coordinates truncated to float.
	virtual irr::scene::IComputeBuffer* GpuTransformBuffer() const;
	virtual bool ProducesGpuTransforms() const { return m_resident; }

private:
	/// Device-to-device chain: bodies stay on the GPU and only the packed transforms come back.
	void StepResident(float deltaTime);
	/// Original path: every phase uploads and reads back through std::vector.
	void StepVector(float deltaTime);

	/**
	 * @brief Seeds the resident body buffer in whichever precision this pipeline runs.
	 * @return False if the upload failed.
	 */
	bool UploadBodies();

	/**
	 * @brief Converts the packed GPU transforms into scene-local DemoTransforms.
	 * @param n Body count, ground included.
	 */
	void PublishTransforms(unsigned int n);

	/**
	 * @brief Runs the reduction kernel and reads its 8-byte result.
	 * @param out Receives the hash.
	 * @return False if the kernel is unavailable.
	 */
	bool ReduceHashGpu(unsigned long long& out) const;

	/**
	 * @brief Marks a compute buffer dirty and forces its upload to happen now, not at the next bind.
	 * @param buffer Buffer whose CPU-side contents were just overwritten.
	 */
	void PushToDevice(irr::scene::IComputeBuffer* buffer);

	/**
	 * @brief Republishes DemoTransforms straight off a restored body block.
	 *
	 * Without it Transforms() would keep showing the pre-restore pose until two steps later, since
	 * the deferred readback for the discarded step is dropped.
	 *
	 * @param bodySrc First body in the payload's native layout.
	 * @param n Body count, ground included.
	 */
	void PublishRestoredTransforms(const unsigned char* bodySrc, unsigned int n);

	/**
	 * @brief Bucket count the current body count actually uses.
	 * @return min(configured, body count), or 0 when bucket hashing is unavailable.
	 */
	int EffectiveBucketCount() const;

	/**
	 * @brief Folds the reduction buffer's bucket pairs into one root hash.
	 * @param out Receives the root, the XOR of every bucket.
	 * @return False if the buffer could not be mapped.
	 */
	bool FoldBucketRoot(unsigned long long& out) const;

	/**
	 * @brief Downloads the reduction result once, without consuming it.
	 * @return False if nothing has been reduced yet.
	 */
	bool EnsureHashDownloaded() const;

	/// Maps the 4-byte pair tally. Observability only, so it rides the deferred readback.
	void ReadPairCount();

	/**
	 * @brief Writes this step's kinematic poses into the resident body buffer.
	 *
	 * Runs before the world-AABB pass, so a moved body's broadphase bound tracks it.
	 *
	 * @param deltaTime Step length, advancing the scene's own clock.
	 * @return False if there is nothing to drive or the kernel is unavailable.
	 */
	bool ApplyKinematics(float deltaTime);

	/**
	 * @brief Resolves a DemoConstraint endpoint to a body-buffer index.
	 * @param endpoint DemoConstraint::bodyA or bodyB.
	 * @param boxCount Dynamic boxes in the scene.
	 * @param kinematicCount Kinematic bodies in the scene.
	 * @return Buffer index, or -1 when the endpoint names nothing that exists.
	 */
	int ResolveAnchor(int endpoint, int boxCount, int kinematicCount) const;

	/**
	 * @brief Diagnostic: blocks until every dispatch so far has retired, and returns ms since mark.
	 *
	 * Commands retire in order, so waiting on any small readback drains everything before it. This
	 * serialises phases that normally overlap - read the SHARES, never the total.
	 *
	 * @param mark Start point for this phase.
	 * @return Milliseconds that phase took once drained, or 0 if timing is off.
	 */
	double SyncPhase(const std::chrono::steady_clock::time_point& mark);

	GpuPhysicsBackend(const GpuPhysicsBackend&);
	GpuPhysicsBackend& operator=(const GpuPhysicsBackend&);

	irr::video::IVideoDriver* m_driver;
	irr::io::IFileSystem* m_fileSystem;

	std::unique_ptr<b3IrrlichtNarrowphase> m_narrowphase;
	std::unique_ptr<b3IrrlichtLbvh> m_lbvh;
	std::unique_ptr<b3IrrlichtSolver> m_solver;
	std::unique_ptr<b3IrrlichtJointSolver> m_jointSolver;
	std::unique_ptr<b3IrrlichtSleep> m_sleep;

	/// Scene-local positions in double, kept because the f32 mirror below cannot hold them at a
	/// planetary offset; these are what the df64 upload splits into hi/lo.
	std::vector<double> m_positions;   // 3 doubles per body, world (offset included)
	double m_worldOffset[3];
	/// f32 mirror. In df64 mode only its non-position fields (masses, collidables) are used.
	std::vector<b3RigidBodyData> m_bodies;
	std::vector<b3IrrJoint> m_joints;
	/// Authored motion, and the buffer index its first body landed on.
	std::vector<DemoKinematicBody> m_kinematics;
	unsigned int m_firstKinematicBody;
	/// Seconds simulated since Reset; the kinematic pose is a pure function of it, so a replay of
	/// the same step count reproduces the same path exactly.
	double m_simTime;
	std::vector<float> m_invInertia;      // 4 floats per body
	std::vector<b3IrrAabb> m_aabbs;
	std::vector<DemoTransform> m_transforms;

	int m_groundCollidable;
	int m_boxCollidable;

	double m_lastStepMs;
	double m_lastTransferMs;
	int m_lastPairCount;
	unsigned int m_maxPairs;
	bool m_overflowReported;
	bool m_pairOverflow;
	/// OSDEMO_DEFER_READBACK=0 disables. Trades one frame of pose latency for no Map stall.
	bool m_deferReadback;
	bool m_readbackPending;

	/// One step of full AABB recompute and full tree rebuild after ApplyState: a sleeping body's
	/// world bound and the BVH topology both carry over from the receiver's own history.
	bool m_forceFullRefresh;
	/// Guards the OSDEMO_STATE_ROUNDTRIP self-test against re-entering Reset.
	bool m_roundTripDone;
	/// Same guard for OSDEMO_BUCKET_HEAL.
	bool m_bucketHealDone;

	/// OSDEMO_PHASE_TIMING=1. Serialises the pipeline to attribute cost per phase; slow by design.
	bool m_phaseTiming;
	/// aabb, broadphase, pair-compact, narrowphase, sleep+solve, integrate.
	double m_phaseMs[6];
	/// Last completed window's per-step means; m_phaseMs is an accumulator and gets reset.
	double m_phasePublished[6];
	bool m_phaseHasResult;
	int m_phaseSteps;

	/// Steps between full tree REBUILDS (0 = every step); OSDEMO_REFIT_INTERVAL overrides.
	/// The rebuild (morton + radix sort + construct) is nearly all of a settled step's ~88% broadphase.
	int m_refitInterval;
	int m_stepsSinceTreeRebuild;
	/// Sleeping bodies stop querying the broadphase tree. OSDEMO_QUERY_GATE=0 disables.
	bool m_queryGate;

	/// GPU-side state fingerprint: reduces the body buffer so a sync check reads back 8 bytes.
	int m_hashClearMaterial;
	int m_hashMaterial;
	/// Mutable so the const HashState can use the same reduction the deferred path does - one
	/// algorithm everywhere, so --determinism files and live sync hashes stay comparable.
	mutable irr::scene::IComputeBuffer* m_hashParamBuffer;
	mutable irr::scene::IComputeBuffer* m_hashOutBuffer;
	mutable bool m_hashPending;
	/// Result is downloaded and still valid. Separate from m_hashPending because the root and the
	/// buckets are two reads of one reduction, and whichever runs first must not consume it.
	mutable bool m_hashReady;
	/// Buckets the reduction splits into. 1 disables narrowing (OSDEMO_BUCKET_HASH=0) and is
	/// bit-identical to the original whole-state hash.
	int m_hashBuckets;
	/// Bucket count and range width the outstanding reduction was dispatched with.
	mutable int m_activeBuckets;
	mutable int m_activePerBucket;
	/// Per-body seconds left to rendezvous, carried across ticks; empty means fully converged.
	std::map<int, double> m_healHorizon;
	/// Correction attempts per body since it last converged. A body re-diverged every round has its
	/// horizon reset each time, so the time-based terminal snap can never fire for it.
	std::map<int, int> m_healRounds;
	int m_healMaxRounds;
	int m_healForcedSnaps;
	/// Default ON; OSDEMO_HEAL_TERMINAL_SNAP=0 keeps steering past the deadline instead of landing.
	bool m_healTerminalSnap;
	/// A refit re-fits only the ancestors of leaves that moved. OSDEMO_SPARSE_REFIT=0 disables.
	bool m_sparseRefit;
	bool m_depthReported;

	std::unique_ptr<b3GpuIrrlichtRigidBodyPipeline> m_pipeline;
	std::vector<b3GpuIrrlichtRigidBodyPipeline::b3IrrBodyTransform> m_packed;
	std::vector<b3IrrGpu::b3IrrBodyTransformDS> m_packedDs;
	/// Selected at construction; the two pipelines never mix.
	bool m_doubleSingle;
	/// Default ON; OSDEMO_GPU_RESIDENT=0 forces the older per-phase upload path.
	bool m_resident;

	/// Default ON; OSDEMO_KINEMATIC=0 makes the backend decline kinematic scenes outright rather
	/// than simulate them with the bodies pinned.
	bool m_kinematicEnabled;
	/// Default ON; OSDEMO_JOINT_WAKE=0 restores contact-only wake propagation.
	bool m_jointWake;
	int m_kinematicMaterial;
	irr::scene::IComputeBuffer* m_kinParamBuffer;
	irr::scene::IComputeBuffer* m_kinTargetBuffer;

	/// Default ON; OSDEMO_NO_SLEEP=1 disables it, mirroring the CPU backend's own knob.
	bool m_sleepEnabled;
	/// -1 until the first readback; the tally is per DYNAMIC body, statics never count as awake.
	int m_awakeCount;
	/// Steps between awake-count readbacks - it is a 4-byte stall, so it stays off the hot path.
	int m_awakeInterval;
	int m_stepsSinceAwakeRead;
	/// 100, the CPU backends' count: at 10 a tower of ten collapses. OSDEMO_SOLVER_ITERATIONS overrides both.
	int m_contactIterations;
};

#endif  //OUTERSPACE_PHYSICS_DEMO_GPU_BACKEND_H
