/*******************************************************
* Copyright (C) 2026 David Delisle granyte@hotmail.com
*
* This file is part of Outerspace.
*
* Outerspace can not be copied and/or distributed without the express
* permission of David Delisle
*******************************************************/
#ifndef OUTERSPACE_PHYSICS_DEMO_BACKEND_H
#define OUTERSPACE_PHYSICS_DEMO_BACKEND_H

#include <cstring>
#include <string>
#include <vector>

namespace irr
{
namespace scene
{
class IComputeBuffer;
}
}  // namespace irr

/// One body's world transform, in the layout the instanced renderer consumes.
struct DemoTransform
{
	float position[3];
	float orientation[4];   // xyzw
};

/**
 * @brief The scene, described once and handed to every backend.
 *
 * Built in one place deliberately: each backend used to run its own copy of the stack-building
 * loop, so a drift between them would have silently invalidated every comparison.
 */
/// A point-to-point link, in body-index space. A negative endpoint names a non-dynamic body -
/// see kDemoAnchorGround / DemoAnchorKinematic.
struct DemoConstraint
{
	int bodyA;
	int bodyB;
	float pivotA[3];   // in bodyA's local frame
	float pivotB[3];
};

/// DemoConstraint endpoint naming the static ground slab.
const int kDemoAnchorGround = -1;

/**
 * @brief DemoConstraint endpoint naming a kinematic body.
 * @param index Index into SceneSpec::Kinematics.
 * @return The negative endpoint encoding.
 */
inline int DemoAnchorKinematic(int index) { return -2 - index; }

/**
 * @brief Whether an endpoint names a kinematic body.
 * @param endpoint DemoConstraint::bodyA or bodyB.
 * @return True for a DemoAnchorKinematic encoding.
 */
inline bool DemoAnchorIsKinematic(int endpoint) { return endpoint <= -2; }

/**
 * @brief Decodes a kinematic endpoint.
 * @param endpoint Value DemoAnchorIsKinematic accepted.
 * @return Index into SceneSpec::Kinematics.
 */
inline int DemoAnchorKinematicIndex(int endpoint) { return -2 - endpoint; }

/**
 * @brief A body with infinite mass whose pose is driven from outside the solver.
 *
 * The engine relies on these for celestial bodies (Asteroid/CPlanet/Star all set
 * CF_KINEMATIC_OBJECT), so the GPU pipeline has to represent one to be usable there.
 */
struct DemoKinematicBody
{
	DemoKinematicBody() : period(0.f), startDelay(0.f), spinRate(0.f)
	{
		position[0] = position[1] = position[2] = 0.f;
		travel[0] = travel[1] = travel[2] = 0.f;
	}

	float position[3];   ///< Rest pose, scene-local metres.
	float travel[3];     ///< One-way amplitude of the ping-pong; zero holds the rest pose.
	float period;        ///< Seconds for a full there-and-back cycle; <= 0 holds the rest pose.
	float startDelay;    ///< Seconds held at the rest pose before any motion starts.
	float spinRate;      ///< Radians per second about +Y, also held until startDelay.
};

/**
 * @brief Pose and velocity of one kinematic body at a given simulation time.
 *
 * Shared so every backend drives the identical path; a per-backend copy would invalidate the
 * comparison the same way a per-backend scene build would.
 *
 * @param body Authored motion.
 * @param time Seconds since the scene was reset.
 * @param outPos Scene-local position, 3 floats.
 * @param outQuat Orientation as xyzw, 4 floats.
 * @param outLinVel Linear velocity, 3 floats; this is what the solvers read as the body's motion.
 * @param outAngVel Angular velocity, 3 floats.
 */
void DemoKinematicPose(const DemoKinematicBody& body, double time, float outPos[3],
					   float outQuat[4], float outLinVel[3], float outAngVel[3]);

/**
 * @brief Whole-scene translation, from OSDEMO_WORLD_OFFSET (metres, default 0, applied to x/y/z).
 *
 * At 6371000 (Earth's radius, and this repo is 1u = 1 m) float32's ULP is ~0.38 m - coarser than
 * the 1 m boxes - so the f32 GPU backend visibly falls apart while df64 and the double CPU
 * backends do not. That contrast is the point of the knob.
 *
 * @return Offset in metres.
 */
double DemoWorldOffset();

/// Default ground slab; a scene wider than this grows its own (SceneSpec::GroundHalfExtent).
const float kDefaultGroundHalfExtent = 60.f;

struct SceneSpec
{
	SceneSpec() : Name(""), GroundHalfExtent(kDefaultGroundHalfExtent), IslandCount(0)
	{
		WorldOffset[0] = WorldOffset[1] = WorldOffset[2] = DemoWorldOffset();
	}

	const char* Name;
	std::vector<float> BoxPositions;   // 3 floats per dynamic box
	std::vector<DemoConstraint> Constraints;
	/// Externally driven bodies, laid out after every dynamic box so box indices never shift.
	std::vector<DemoKinematicBody> Kinematics;
	/// Grown by wide scenes: a 600k-body island grid overhangs the default slab and would free-fall.
	float GroundHalfExtent;
	/// Disjoint clusters the layout intends, for the scaling table. 0 when the scene is one blob.
	int IslandCount;
	/// Constant translation applied to bodies AND ground by every backend. BoxPositions stay local
	/// so the layout itself is never quantised; it is the simulation that has to hold the offset.
	double WorldOffset[3];

	int BoxCount() const { return (int)(BoxPositions.size() / 3); }
	int KinematicCount() const { return (int)Kinematics.size(); }
	bool NeedsConstraints() const { return !Constraints.empty(); }
	bool NeedsKinematics() const { return !Kinematics.empty(); }
};

enum DemoSceneKind
{
	SCENE_SINGLE_STACK = 0,
	SCENE_MULTIPLE_STACKS,
	SCENE_PYRAMID,
	SCENE_WALL,
	SCENE_CHAINS,        // constraints
	SCENE_ISLAND_STACKS, // appended: --scene indices above must not shift
	SCENE_CLUSTERED_ISLANDS,
	SCENE_KINEMATIC_ANCHORS,   // kinematic bodies, plus joints anchored to non-dynamic bodies
	SCENE_COUNT
};

/**
 * @brief Builds one of the benchmark scenes.
 * @param kind Which layout to build.
 * @param scale Size knob; roughly the edge length of the arrangement.
 * @return Positions of every dynamic box, all sharing kBoxHalfExtent.
 */
SceneSpec BuildScene(int kind, int scale);

/**
 * @brief Islands-of-stacks sized to a body count instead of an edge length.
 * @param targetBodies Wanted dynamic body count; rounded up to a whole island grid.
 * @return Scene with IslandCount and a ground wide enough to hold it.
 */
SceneSpec BuildIslandStackScene(int targetBodies);

/**
 * @brief Towers that settle and sleep, each with one box falling onto it from above.
 *
 * Built for the sleep test: the towers must be asleep before impact, so they start already
 * touching the ground rather than dropped into place like the benchmark scenes.
 *
 * @param towersPerSide Towers along each ground axis.
 * @param towerHeight Boxes per tower.
 * @param dropHeight Clear air between a tower's top face and its dropper, in metres.
 * @return Every tower box first, then one dropper per tower - split by index.
 */
SceneSpec BuildSleepDropScene(int towersPerSide, int towerHeight, float dropHeight);

/**
 * @brief Clusters of island grids, sized to a body count: two levels of separation.
 * @param targetBodies Wanted dynamic body count; rounded up to a whole cluster grid.
 * @return Scene with IslandCount and a ground wide enough to hold it.
 */
SceneSpec BuildClusteredIslandScene(int targetBodies);

/**
 * @brief Islands per side within one cluster (OSDEMO_CLUSTER_ISLANDS, default 8).
 * @return Towers per cluster edge.
 */
int ClusterIslandsPerSide();

/**
 * @brief Clear air between clusters, on top of each cluster's own span (OSDEMO_CLUSTER_GAP).
 * @return Gap in metres.
 */
float ClusterGap();

/**
 * @brief Tower height the island scene uses (OSDEMO_TOWER_HEIGHT, default 10).
 * @return Boxes per tower.
 */
int IslandTowerHeight();

/**
 * @brief Centre-to-centre island spacing (OSDEMO_ISLAND_SPACING, default 3.6 m).
 * @return Spacing in metres.
 */
float IslandSpacing();

/**
 * @brief A physics implementation the demo can swap at runtime.
 *
 * All three backends build the SAME scene and are timed the same way, so the only difference
 * between their numbers is the implementation - not the workload.
 */
class PhysicsBackend
{
public:
	virtual ~PhysicsBackend() {}

	/**
	 * @brief Name shown in the HUD.
	 * @return Short display label.
	 */
	virtual const char* Name() const = 0;

	/**
	 * @brief Builds the given scene from scratch.
	 * @param scene Shared description, so every backend simulates the identical layout.
	 * @return False if the backend cannot represent this scene - check SupportsConstraints first.
	 */
	virtual bool Reset(const SceneSpec& scene) = 0;

	/**
	 * @brief Whether this backend can honour SceneSpec::Constraints.
	 *
	 * A backend that cannot must DECLINE a constrained scene rather than silently simulate an
	 * unconstrained one, which would produce plausible timings for the wrong simulation.
	 *
	 * @return True if constraints are simulated.
	 */
	virtual bool SupportsConstraints() const { return false; }

	/**
	 * @brief Whether this backend can represent SceneSpec::Kinematics and non-dynamic anchors.
	 *
	 * Same contract as SupportsConstraints: a backend that cannot must DECLINE rather than silently
	 * drop the kinematic bodies and every joint anchored to them.
	 *
	 * @return True if kinematic bodies are simulated.
	 */
	virtual bool SupportsKinematics() const { return false; }

	/**
	 * @brief Advances the simulation by one fixed step.
	 * @param deltaTime Step length in seconds.
	 */
	virtual void Step(float deltaTime) = 0;

	/**
	 * @brief Current transforms, in body order, for rendering.
	 * @return One entry per dynamic body.
	 */
	virtual const std::vector<DemoTransform>& Transforms() const = 0;

	/**
	 * @brief Bit-exact fingerprint of the whole simulation state, for cross-machine determinism runs.
	 *
	 * Bodies are hashed in buffer-index order, which is already stable, so no canonical sort is
	 * needed the way the engine's ObjectID-ordered snapshot needs one.
	 *
	 * @param out Receives the hash.
	 * @return False if this backend cannot produce one.
	 */
	virtual bool HashState(unsigned long long& out) const
	{
		// Renderer-facing transforms only: float, and no velocities. A backend holding richer native
		// state should override, or a df64 low word would never reach the hash.
		const std::vector<DemoTransform>& t = Transforms();
		unsigned long long h = 14695981039346656037ULL;
		for (size_t i = 0; i < t.size(); ++i)
		{
			HashFloats(h, t[i].position, 3);
			HashFloats(h, t[i].orientation, 4);
		}
		out = h;
		return !t.empty();
	}

	/**
	 * @brief Heal tuning, mirroring the engine's HealSettings.
	 *
	 * Every threshold is relative to the body's own size and speed; a fixed metre value cannot work
	 * across a range spanning a parked crate and a ship crossing light-minutes per second.
	 */
	struct HealSettings
	{
		HealSettings()
			: HorizonSeconds(5.0), SnapBodyRadii(10.0), SnapTravelFraction(0.25),
			  SnapFloorMeters(50.0), LockTravelFraction(0.05), LockFloorMeters(0.01) { }

		double HorizonSeconds;      ///< Seconds a correction is given to rendezvous.
		double SnapBodyRadii;       ///< Snap once error exceeds this many of the body's own radii...
		double SnapTravelFraction;  ///< ...or this fraction of one horizon's travel.
		double SnapFloorMeters;     ///< Floor for a small, slow body.
		/// Snap exactly once error is under this fraction of one tick's travel. This is what lets
		/// hashes match bit-for-bit again - force convergence is asymptotic and never lands exactly.
		double LockTravelFraction;
		double LockFloorMeters;     ///< Floor for a stationary body, whose per-tick travel is zero.
	};

	/**
	 * @brief Buckets the state hash is split into, for narrowing a desync to part of the body array.
	 *
	 * Buckets are ranges of BODY INDEX, never BVH subtrees: after a divergence the peers' Morton
	 * ordering can differ, so subtree N would not describe the same bodies on both sides.
	 *
	 * @return 0 if this backend only produces a whole-state hash.
	 */
	virtual int HashBucketCount() const { return 0; }

	/**
	 * @brief Per-bucket hashes from the last BeginHashState.
	 * @param out Receives HashBucketCount() entries.
	 * @return False if unsupported or none outstanding.
	 */
	virtual bool FetchHashBuckets(std::vector<unsigned long long>& out) { (void)out; return false; }

	/**
	 * @brief Serialises only the bodies in the named buckets.
	 * @param buckets Ascending bucket indices.
	 * @param out Receives the payload; cleared first.
	 * @return False if unsupported.
	 */
	virtual bool CaptureBuckets(const std::vector<int>& buckets, std::vector<unsigned char>& out) const
	{
		(void)buckets; (void)out; return false;
	}

	/**
	 * @brief Steers the named bodies toward a CaptureBuckets payload instead of snapping to it.
	 *
	 * Solves to where the authority WILL be, not where it is; chasing its current pose trails a
	 * moving target forever.
	 *
	 * @param data First byte of a CaptureBuckets payload.
	 * @param bytes Length.
	 * @param settings Heal tuning.
	 * @param dtSeconds Time since the previous heal application.
	 * @return False if unsupported or the payload does not match this backend.
	 */
	virtual bool ApplyBucketsConverging(const unsigned char* data, size_t bytes,
										const HealSettings& settings, double dtSeconds)
	{
		(void)data; (void)bytes; (void)settings; (void)dtSeconds; return false;
	}

	/// Bodies still inside their heal horizon; 0 means fully converged and locked.
	virtual int HealingBodyCount() const { return 0; }

	/**
	 * @brief Starts computing a state hash without reading it back.
	 *
	 * Pairs with FetchHashState a frame later. HashState reads the step just issued, which drains
	 * the GPU; splitting it lets the readback land after the work has retired anyway.
	 *
	 * @return False if this backend has no deferred path, in which case Fetch is simply synchronous.
	 */
	virtual bool BeginHashState() { return false; }

	/**
	 * @brief Collects the hash BeginHashState started.
	 * @param out Receives the hash.
	 * @return False if none is available.
	 */
	virtual bool FetchHashState(unsigned long long& out) { return HashState(out); }

	/**
	 * @brief Serialises the full simulation state as opaque bytes.
	 *
	 * The layout is the backend's own business; only the same backend build ever reads it back, so
	 * the transport never interprets it.
	 *
	 * @param out Receives the state; cleared first.
	 * @return False if this backend cannot snapshot.
	 */
	virtual bool CaptureState(std::vector<unsigned char>& out) const { (void)out; return false; }

	/**
	 * @brief Overwrites the simulation state from CaptureState bytes.
	 * @param data First byte.
	 * @param bytes Length.
	 * @return False if this backend cannot apply, or the payload does not match its layout.
	 */
	virtual bool ApplyState(const unsigned char* data, size_t bytes) { (void)data; (void)bytes; return false; }

	/**
	 * @brief FNV-1a over the raw bits of a float run.
	 * @param h Hash accumulator, updated in place.
	 * @param values First float.
	 * @param count How many.
	 */
	static void HashFloats(unsigned long long& h, const float* values, size_t count)
	{
		for (size_t i = 0; i < count; ++i)
		{
			unsigned int bits;
			memcpy(&bits, &values[i], sizeof(bits));
			for (int b = 0; b < 4; ++b)
			{
				h ^= (unsigned long long)((bits >> (b * 8)) & 0xFFu);
				h *= 1099511628211ULL;
			}
		}
	}

	/**
	 * @brief Wall-clock cost of the last Step, in milliseconds.
	 * @return Total step time including any CPU/GPU transfer.
	 */
	virtual double LastStepMs() const = 0;

	/**
	 * @brief Portion of the last step spent moving data across the PCIe bus.
	 *
	 * Reported separately because this composition uploads and reads back per phase; a
	 * GPU-resident pipeline would pay almost none of it, so folding it into the total would
	 * understate the GPU path.
	 *
	 * @return Transfer milliseconds, or 0 for CPU backends.
	 */
	virtual double LastTransferMs() const { return 0.0; }

	/**
	 * @brief Number of contact pairs the last step resolved. Sanity signal that the backends are
	 *        actually doing comparable work rather than one quietly diverging.
	 * @return Pair count, or -1 if the backend cannot report it.
	 */
	virtual int LastPairCount() const { return -1; }

	/// Broadphase pair capacity was exceeded, so pairs were dropped nondeterministically.
	virtual bool LastPairOverflowed() const { return false; }

	/**
	 * @brief Per-phase cost attribution, when the backend supports it and it is enabled.
	 * @param out Receives ms/step for aabb, broad, paircompact, narrow, sleep+solve, integrate.
	 * @return False if unsupported or off, leaving out untouched.
	 */
	virtual bool LastPhaseMs(double out[6]) const { (void)out; return false; }

	/**
	 * @brief Turns per-phase attribution on/off at runtime; serialises the pipeline while on.
	 * @param on Whether to measure.
	 * @return False if this backend cannot measure phases.
	 */
	virtual bool SetPhaseTiming(bool on) { (void)on; return false; }

	/**
	 * @brief Dynamic body count in the current scene.
	 * @return Body count.
	 */
	virtual int BodyCount() const = 0;

	/**
	 * @brief Dynamic bodies not currently deactivated.
	 *
	 * Step time alone cannot distinguish sleeping from a scene that simply went quiet.
	 *
	 * @return Awake body count, or -1 when the backend has no deactivation.
	 */
	virtual int AwakeBodyCount() const { return -1; }

	/**
	 * @brief Pulls a fresh awake tally now, for a backend that only samples it periodically.
	 * @return False when the backend has no deactivation or nothing to refresh.
	 */
	virtual bool RefreshAwakeCount() { return false; }

	/**
	 * @brief Pair/contact buffer capacity this backend sized for the current scene.
	 * @return Capacity in entries, or -1 when the backend has no such cap.
	 */
	virtual int PairCapacity() const { return -1; }

	/**
	 * @brief Rough device-buffer footprint of the current scene, for diagnosing an OOM.
	 * @return Bytes, or -1 when the backend has no GPU buffers.
	 */
	virtual long long GpuBufferBytes() const { return -1; }

	/**
	 * @brief Device-resident transforms, 28-byte {float3 pos; float4 quat;} elements.
	 *
	 * Non-null is what lets the renderer cull on the GPU instead of walking Transforms(); a backend
	 * whose transforms only exist on the CPU returns 0 and the caller falls back.
	 *
	 * @return Transform buffer, or 0.
	 */
	virtual irr::scene::IComputeBuffer* GpuTransformBuffer() const { return 0; }

	/// Whether GpuTransformBuffer() will be non-null once a step has run; the buffer itself does
	/// not exist until then, so scene setup has to ask this instead.
	virtual bool ProducesGpuTransforms() const { return false; }
};

/// Half-extent of every dynamic box, and of the ground's thickness. Shared so the renderer's cube
/// mesh matches whatever the backends collide with.
const float kBoxHalfExtent = 0.5f;
const float kGroundHalfExtent = kDefaultGroundHalfExtent;
const float kGroundThickness = 1.f;

#endif  //OUTERSPACE_PHYSICS_DEMO_BACKEND_H
