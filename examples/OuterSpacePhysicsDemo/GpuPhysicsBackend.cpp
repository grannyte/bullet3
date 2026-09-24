/*******************************************************
* Copyright (C) 2026 David Delisle granyte@hotmail.com
*
* This file is part of Outerspace.
*
* Outerspace can not be copied and/or distributed without the express
* permission of David Delisle
*******************************************************/
#include "GpuPhysicsBackend.h"

#include "BucketHeal.h"
#include "StateRoundTrip.h"

#include <irrlicht.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace
{
const float kBoxMass = 1.f;
const unsigned int kMinPairs = 65536;

/// 1024 buckets is an 8 KB readback at any body count, against 55 MB for the whole body buffer at
/// 576k - and one bucket still localises a desync to ~0.1% of the array.
const int kDefaultHashBuckets = 1024;

/**
 * @brief Pair-buffer capacity for a body count.
 * @param bodies Body count including the ground.
 * @return Entries; OSDEMO_MAX_PAIRS pins it, OSDEMO_PAIR_RATIO scales it (default 4x bodies).
 */
unsigned int ChoosePairCapacity(size_t bodies)
{
	if (const char* pinned = getenv("OSDEMO_MAX_PAIRS"))
	{
		const long v = atol(pinned);
		if (v > 0)
			return (unsigned int)v;
	}

	const char* ratioEnv = getenv("OSDEMO_PAIR_RATIO");
	float ratio = ratioEnv ? (float)atof(ratioEnv) : 4.f;
	if (ratio < 1.f)
		ratio = 1.f;

	const double wanted = (double)bodies * ratio;
	if (wanted < (double)kMinPairs)
		return kMinPairs;
	return (unsigned int)wanted;
}

/**
 * @brief Reads an int from the environment.
 * @param name Variable name.
 * @param fallback Value when unset or unparseable.
 * @return Parsed value.
 */
int EnvInt(const char* name, int fallback)
{
	const char* env = getenv(name);
	return env && *env ? atoi(env) : fallback;
}

/**
 * @brief Reads a float from the environment.
 * @param name Variable name.
 * @param fallback Value when unset or unparseable.
 * @return Parsed value.
 */
float EnvFloat(const char* name, float fallback)
{
	const char* env = getenv(name);
	return env && *env ? (float)atof(env) : fallback;
}

double MillisSince(const std::chrono::steady_clock::time_point& from)
{
	return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - from).count();
}

/// Sentinel-stamped round trip of every b3IrrJoint field, so a C++/HLSL packing divergence is
/// proven absent rather than inferred from plausible-looking physics.
void ReportJointLayout(b3IrrlichtJointSolver& solver, const std::vector<b3IrrJoint>& joints)
{
	std::vector<b3IrrJoint> probe = joints;
	for (size_t i = 0; i < probe.size(); ++i)
	{
		probe[i].uid = 0x0C0FFEE0 + (int)i;
		probe[i].breakingImpulseThreshold = 12345.f + (float)i;
	}

	std::vector<int> got;
	if (!solver.debugReadJointLayout(probe, got))
	{
		printf("[joint-layout] probe kernel unavailable\n");
		fflush(stdout);
		return;
	}

	int mismatches = 0;
	for (size_t i = 0; i < probe.size(); ++i)
	{
		const int* g = &got[i * b3IrrlichtJointSolver::DebugLayoutInts];
		float thr, pa, pb;
		memcpy(&thr, &g[3], sizeof(float));
		memcpy(&pa, &g[4], sizeof(float));
		memcpy(&pb, &g[5], sizeof(float));

		const bool ok = g[0] == probe[i].constraintType && g[1] == probe[i].rbA
					 && g[2] == probe[i].rbB && thr == probe[i].breakingImpulseThreshold
					 && pa == probe[i].pivotInA[1] && pb == probe[i].pivotInB[1]
					 && g[6] == probe[i].flags && g[7] == probe[i].uid;
		if (!ok)
		{
			if (mismatches == 0)
				printf("[joint-layout] MISMATCH joint %d: gpu type=%d rbA=%d rbB=%d thr=%g "
					   "pivA.y=%g pivB.y=%g flags=%d uid=0x%08X\n",
					   (int)i, g[0], g[1], g[2], thr, pa, pb, g[6], g[7]);
			++mismatches;
		}
	}

	printf("[joint-layout] sizeof(b3IrrJoint)=%d  %d/%d joints round-tripped every field exactly\n",
		   (int)sizeof(b3IrrJoint), (int)probe.size() - mismatches, (int)probe.size());
	fflush(stdout);
}

/// Mirrors KinematicParams in B3IntegrateTransformsBody.hlsli.
struct KinematicParamsGpu
{
	unsigned int numTargets;
	unsigned int numBodies;
	unsigned int pad0;
	unsigned int pad1;
};

/// Mirrors b3KinematicTarget; the df64 form carries the position's low half alongside.
struct KinematicTargetGpu
{
	unsigned int bodyIndex;
	unsigned int pad[3];
	float pos[4];
	float quat[4];
	float linVel[4];
	float angVel[4];
};

struct KinematicTargetGpuDS
{
	unsigned int bodyIndex;
	unsigned int pad[3];
	float pos[4];
	float posLo[4];
	float quat[4];
	float linVel[4];
	float angVel[4];
};

static_assert(sizeof(KinematicParamsGpu) == 16, "KinematicParams must match the HLSL stride");
static_assert(sizeof(KinematicTargetGpu) == 80 && sizeof(KinematicTargetGpuDS) == 96,
			  "b3KinematicTarget must match the HLSL struct stride in both precisions");

const unsigned int kStateMagic = 0x53545331u;    // "1STS"
const unsigned int kStateVersion = 1u;
const unsigned int kFlagDoubleSingle = 1u << 0;
const unsigned int kFlagSleepBlock = 1u << 1;
const unsigned int kFlagResident = 1u << 2;

/// Payload header. Every field is a refusal check on the receiving side, which is the whole point:
/// only the same backend build reads these bytes, so a mismatch means a bug, not a negotiation.
struct GpuStateHeader
{
	unsigned int magic;
	unsigned int version;
	unsigned int bodyCount;
	unsigned int bodyStride;
	unsigned int sleepRefStride;   // 0 when no sleep block follows
	unsigned int flags;
	int awakeCount;
	unsigned int reserved;
	unsigned long long bodyHash;
	double worldOffset[3];
};
static_assert(sizeof(GpuStateHeader) == 64, "GpuStateHeader must stay a fixed 64 bytes");

/// FNV-1a, matching PhysicsBackend::HashFloats' constants so the two are read the same way.
unsigned long long HashBytes(const unsigned char* data, size_t bytes)
{
	unsigned long long h = 14695981039346656037ULL;
	for (size_t i = 0; i < bytes; ++i)
	{
		h ^= (unsigned long long)data[i];
		h *= 1099511628211ULL;
	}
	return h;
}

const unsigned int kBucketMagic = 0x4B425331u;   // "1SBK"
const unsigned int kBucketVersion = 1u;

/// Partial-state payload: the same body/sleep layout CaptureState writes, restricted to the bodies
/// the bucket hashes named, each prefixed by its index so the receiver can place it.
struct GpuBucketHeader
{
	unsigned int magic;
	unsigned int version;
	unsigned int bodyCount;        // whole-world count, so a mismatched scene is refused
	unsigned int bodyStride;
	unsigned int recordCount;      // bodies actually carried
	unsigned int sleepRefStride;   // 0 when no sleep block follows
	unsigned int bucketCount;      // the split both peers must agree on
	unsigned int flags;
	unsigned long long payloadHash;
	double worldOffset[3];
};
static_assert(sizeof(GpuBucketHeader) == 64, "GpuBucketHeader must stay a fixed 64 bytes");

/// Field offsets into whichever body layout this build runs, so nothing here hardcodes a stride.
struct BodyLayout
{
	explicit BodyLayout(bool doubleSingle)
	{
		if (doubleSingle)
		{
			Stride = (unsigned int)sizeof(b3IrrGpu::b3IrrRigidBodyDataDS);
			PosHi = (unsigned int)offsetof(b3IrrGpu::b3IrrRigidBodyDataDS, pos);
			PosLo = (unsigned int)offsetof(b3IrrGpu::b3IrrRigidBodyDataDS, posLo);
			LinVel = (unsigned int)offsetof(b3IrrGpu::b3IrrRigidBodyDataDS, linVel);
			InvMass = (unsigned int)offsetof(b3IrrGpu::b3IrrRigidBodyDataDS, invMass);
		}
		else
		{
			Stride = (unsigned int)sizeof(b3RigidBodyData);
			PosHi = (unsigned int)offsetof(b3RigidBodyData, m_pos);
			PosLo = PosHi;   // unused in f32
			LinVel = (unsigned int)offsetof(b3RigidBodyData, m_linVel);
			InvMass = (unsigned int)offsetof(b3RigidBodyData, m_invMass);
		}
	}

	unsigned int Stride;
	unsigned int PosHi;
	unsigned int PosLo;
	unsigned int LinVel;
	unsigned int InvMass;
};

/**
 * @brief Reads one body's world position out of a raw block.
 * @param body First byte of the body.
 * @param layout Field offsets for this build.
 * @param doubleSingle Combine the hi/lo pair instead of reading a single float.
 * @param out Receives x/y/z.
 */
void ReadBodyPosition(const unsigned char* body, const BodyLayout& layout, bool doubleSingle,
					  double out[3])
{
	const float* hi = (const float*)(body + layout.PosHi);
	if (!doubleSingle)
	{
		for (int a = 0; a < 3; ++a)
			out[a] = (double)hi[a];
		return;
	}
	const float* lo = (const float*)(body + layout.PosLo);
	for (int a = 0; a < 3; ++a)
		out[a] = b3IrrGpu::b3DsCombine(hi[a], lo[a]);
}

double Length3(const double v[3])
{
	return std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
}

void QuatNormalize(b3Quat& q)
{
	const float len = std::sqrt((float)(q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w));
	if (len > 1e-8f)
	{
		const float inv = 1.f / len;
		q.x *= inv;
		q.y *= inv;
		q.z *= inv;
		q.w *= inv;
	}
}
}  // namespace

GpuPhysicsBackend::GpuPhysicsBackend(irr::video::IVideoDriver* driver,
									 irr::io::IFileSystem* fileSystem, bool doubleSingle)
	: m_driver(driver), m_fileSystem(fileSystem), m_groundCollidable(-1), m_boxCollidable(-1),
	  m_lastStepMs(0.0), m_lastTransferMs(0.0), m_lastPairCount(0), m_maxPairs(kMinPairs),
	  m_overflowReported(false), m_pairOverflow(false), m_deferReadback(true), m_readbackPending(false),
	  m_forceFullRefresh(false), m_roundTripDone(false), m_bucketHealDone(false),
	  m_phaseTiming(false), m_phaseHasResult(false), m_phaseSteps(0),
	  m_refitInterval(8), m_stepsSinceTreeRebuild(0), m_queryGate(true), m_sparseRefit(true), m_depthReported(false),
	  m_hashClearMaterial(-1), m_hashMaterial(-1), m_hashParamBuffer(0), m_hashOutBuffer(0),
	  m_hashPending(false), m_hashReady(false), m_hashBuckets(kDefaultHashBuckets), m_activeBuckets(1),
	  m_activePerBucket(1), m_healTerminalSnap(true), m_healMaxRounds(5), m_healForcedSnaps(0),
	  m_resident(true), m_doubleSingle(doubleSingle),
	  m_firstKinematicBody(0), m_simTime(0.0), m_kinematicEnabled(true), m_jointWake(true),
	  m_kinematicMaterial(-1), m_kinParamBuffer(0), m_kinTargetBuffer(0),
	  m_sleepEnabled(true), m_awakeCount(-1), m_awakeInterval(30), m_stepsSinceAwakeRead(0),
	  m_contactIterations(100)
{
	if (const char* env = getenv("OSDEMO_SOLVER_ITERATIONS"))
	{
		const int parsed = atoi(env);
		if (parsed > 0)
			m_contactIterations = parsed;
	}
	if (const char* env = getenv("OSDEMO_KINEMATIC"))
		m_kinematicEnabled = atoi(env) != 0;
	if (const char* env = getenv("OSDEMO_JOINT_WAKE"))
		m_jointWake = atoi(env) != 0;
	if (const char* env = getenv("OSDEMO_DEFER_READBACK"))
		m_deferReadback = atoi(env) != 0;
	if (const char* env = getenv("OSDEMO_REFIT_INTERVAL"))
		m_refitInterval = atoi(env);
	if (const char* env = getenv("OSDEMO_QUERY_GATE"))
		m_queryGate = atoi(env) != 0;
	// Kill switch: 0 collapses the reduction back to one bucket, which is the pre-bucket hash exactly.
	if (const char* env = getenv("OSDEMO_BUCKET_HASH"))
	{
		const int parsed = atoi(env);
		m_hashBuckets = parsed > 1 ? parsed : 1;
	}
	if (const char* env = getenv("OSDEMO_HEAL_TERMINAL_SNAP"))
		m_healTerminalSnap = atoi(env) != 0;
	// Correction attempts a body may steer through before it is snapped outright; 0 disables.
	if (const char* env = getenv("OSDEMO_HEAL_MAX_ROUNDS"))
		m_healMaxRounds = atoi(env);
	if (const char* env = getenv("OSDEMO_SPARSE_REFIT"))
		m_sparseRefit = atoi(env) != 0;
	if (const char* env = getenv("OSDEMO_PHASE_TIMING"))
		m_phaseTiming = atoi(env) != 0;
	for (int i = 0; i < 6; ++i)
		m_phaseMs[i] = m_phasePublished[i] = 0.0;
	// Mirrors the CPU backend's OSDEMO_NO_SLEEP (which sets DISABLE_DEACTIVATION), so a benchmark
	// can hold both families in their awake steady state.
	if (const char* env = getenv("OSDEMO_NO_SLEEP"))
		m_sleepEnabled = atoi(env) == 0;
	if (const char* env = getenv("OSDEMO_AWAKE_INTERVAL"))
	{
		const int parsed = atoi(env);
		if (parsed >= 1)
			m_awakeInterval = parsed;
	}

	m_worldOffset[0] = m_worldOffset[1] = m_worldOffset[2] = 0.0;

	// OSDEMO_GPU_RESIDENT=0 forces the per-phase upload path. df64 stays resident-only: its kernels
	// index a wider body stride than that path's std::vector uploads.
	if (!m_doubleSingle)
		if (const char* env = getenv("OSDEMO_GPU_RESIDENT"))
			m_resident = atoi(env) != 0;
}

GpuPhysicsBackend::~GpuPhysicsBackend()
{
	b3IrrGpu::dropBuffer(m_kinParamBuffer);
	b3IrrGpu::dropBuffer(m_kinTargetBuffer);
}

bool GpuPhysicsBackend::Init()
{
	if (!m_driver)
		return false;

	m_narrowphase.reset(new b3IrrlichtNarrowphase(m_driver));
	if (!m_narrowphase->init(m_fileSystem, m_doubleSingle))
		return false;

	m_lbvh.reset(new b3IrrlichtLbvh(m_driver));
	if (!m_lbvh->init(m_fileSystem, m_doubleSingle))
		return false;

	// Runtime-compiled, so a broken refit shader would otherwise disable itself invisibly.
	if (m_sparseRefit && !m_lbvh->isSparseRefitAvailable())
	{
		printf("[gpu] sparse BVH refit unavailable - B3LbvhRefit%s.hlsl failed to compile.\n",
			   m_doubleSingle ? "DS" : "");
		m_sparseRefit = false;
	}
	m_lbvh->setLeafRangeCaching(EnvInt("OSDEMO_LEAF_RANGE_CACHE", 1) != 0);

	m_solver.reset(new b3IrrlichtSolver(m_driver));
	if (!m_solver->init(m_fileSystem, m_doubleSingle))
		return false;
	m_solver->setActiveBodyGating(EnvInt("OSDEMO_ACTIVE_BODIES", 1) != 0);
	if (!m_solver->isActiveBodyGatingAvailable())
		printf("[gpu] active-body solve gating unavailable - its kernels failed to compile.\n");

	m_jointSolver.reset(new b3IrrlichtJointSolver(m_driver));
	if (!m_jointSolver->init(m_fileSystem, m_doubleSingle))
		return false;

	// Optional: without it HashState falls back to reading the whole body buffer back.
	if (irr::video::IGPUProgrammingServices* gpu = m_driver->getGPUProgrammingServices())
	{
		const char* hashPath = m_doubleSingle ? "media/shaders/B3StateHashDS.hlsl"
											  : "media/shaders/B3StateHash.hlsl";
		if (!m_fileSystem || m_fileSystem->existFile(hashPath))
		{
			m_hashClearMaterial =
				gpu->addComputeShaderFromFile(hashPath, "CSClearHash", irr::video::ECST_CS_5_0, 0);
			m_hashMaterial =
				gpu->addComputeShaderFromFile(hashPath, "CSHashBodies", irr::video::ECST_CS_5_0, 0);
		}
		if (m_hashClearMaterial < 0 || m_hashMaterial < 0)
			printf("[gpu] state-hash reduction unavailable - %s failed to compile; sync checks will "
				   "read the whole body buffer back.\n", hashPath);
	}

	// Compiled here rather than by the pipeline so its buffers stay this backend's own; a failure
	// makes SupportsKinematics false, which DECLINES a kinematic scene instead of pinning its bodies.
	if (irr::video::IGPUProgrammingServices* gpu = m_driver->getGPUProgrammingServices())
	{
		const char* kinPath = m_doubleSingle ? "media/shaders/B3IntegrateTransformsDS.hlsl"
											 : "media/shaders/B3IntegrateTransforms.hlsl";
		if (!m_fileSystem || m_fileSystem->existFile(kinPath))
			m_kinematicMaterial = gpu->addComputeShaderFromFile(kinPath, "CSApplyKinematic",
																irr::video::ECST_CS_5_0, 0);
		if (m_kinematicMaterial < 0 && m_kinematicEnabled)
		{
			printf("[gpu] kinematic bodies unavailable - CSApplyKinematic in %s failed to compile.\n",
				   kinPath);
			m_kinematicEnabled = false;
		}
	}

	// Optional: a machine whose sleep kernels will not compile still gets a correct simulation,
	// just without deactivation.
	m_sleep.reset(new b3IrrlichtSleep(m_driver));
	if (!m_sleep->init(m_fileSystem, m_doubleSingle))
	{
		m_sleep.reset();
		if (m_sleepEnabled)
			printf("[gpu] sleeping unavailable - B3Sleep%s.hlsl failed to compile.\n",
				   m_doubleSingle ? "DS" : "");
	}
	else
	{
		m_sleep->setThresholds(EnvFloat("OSDEMO_SLEEP_MOVE", 0.05f),
							   EnvFloat("OSDEMO_SLEEP_ANGULAR", 1.0f),
							   (unsigned int)EnvInt("OSDEMO_SLEEP_STEPS",
													(int)b3IrrlichtSleep::kDefaultSleepSteps));
		m_sleep->setSkipIdleBodies(EnvInt("OSDEMO_SKIP_IDLE", 1) != 0);
	}

	// Only the fused integrate/pack kernel is taken from the pipeline; it owns no bodies here.
	m_pipeline.reset(new b3GpuIrrlichtRigidBodyPipeline(m_driver));
	if (!m_pipeline->init(m_fileSystem, m_doubleSingle))
	{
		m_pipeline.reset();
		if (m_resident)
			printf("[gpu] resident path unavailable - the integrate kernel failed to compile.\n");
	}

	// A df64 backend with no resident chain would silently be the f32 one under a df64 label, so
	// it declines outright and the caller drops it from the backend list.
	if (m_doubleSingle && !m_pipeline)
	{
		printf("[gpu] df64 backend unavailable - B3IntegrateTransformsDS.hlsl failed to compile.\n");
		return false;
	}

	return true;
}

bool GpuPhysicsBackend::Reset(const SceneSpec& scene)
{
	if (!m_narrowphase || !m_jointSolver)
		return false;

	// Declining beats simulating the bodies pinned: a kinematic scene with its drivers dropped
	// looks plausible and is a different simulation entirely.
	if (scene.NeedsKinematics() && !SupportsKinematics())
		return false;

	// A fresh registry per reset: collidable indices are positional, so reusing a partially
	// populated one across resets would silently misindex.
	m_narrowphase.reset(new b3IrrlichtNarrowphase(m_driver));
	if (!m_narrowphase->init(m_fileSystem, m_doubleSingle))
		return false;

	m_worldOffset[0] = scene.WorldOffset[0];
	m_worldOffset[1] = scene.WorldOffset[1];
	m_worldOffset[2] = scene.WorldOffset[2];

	const float groundHalf[3] = {scene.GroundHalfExtent, kGroundThickness, scene.GroundHalfExtent};
	const float boxHalf[3] = {kBoxHalfExtent, kBoxHalfExtent, kBoxHalfExtent};
	m_groundCollidable = m_narrowphase->registerBoxShape(groundHalf);
	m_boxCollidable = m_narrowphase->registerBoxShape(boxHalf);
	if (!m_narrowphase->writeShapesToGpu())
		return false;

	m_bodies.clear();
	m_positions.clear();
	m_transforms.clear();
	m_invInertia.clear();
	m_joints.clear();
	m_kinematics = scene.Kinematics;
	m_simTime = 0.0;
	// A pending map would target the previous scene's buffer.
	m_readbackPending = false;
	m_forceFullRefresh = false;
	m_healHorizon.clear();
	m_healRounds.clear();

	// Body 0 is the static ground, matching the CPU backend's ordering so the two scenes are
	// directly comparable body-for-body.
	{
		b3RigidBodyData ground;
		memset(&ground, 0, sizeof(ground));
		const double gp[3] = {m_worldOffset[0], m_worldOffset[1] - kGroundThickness, m_worldOffset[2]};
		ground.m_pos = b3MakeVector3((float)gp[0], (float)gp[1], (float)gp[2]);
		ground.m_quat = b3Quaternion(0.f, 0.f, 0.f, 1.f);
		m_positions.insert(m_positions.end(), gp, gp + 3);
		ground.m_collidableIdx = m_groundCollidable;
		ground.m_invMass = 0.f;
		ground.m_restituitionCoeff = 0.f;
		ground.m_frictionCoeff = 0.5f;
		m_bodies.push_back(ground);
		m_invInertia.insert(m_invInertia.end(), 4, 0.f);
	}

	const float ix = (kBoxMass / 12.f) * (4.f * kBoxHalfExtent * kBoxHalfExtent * 2.f);
	const float invI = 1.f / ix;

	for (int i = 0; i < scene.BoxCount(); ++i)
	{
		const double bp[3] = {m_worldOffset[0] + scene.BoxPositions[i * 3 + 0],
							  m_worldOffset[1] + scene.BoxPositions[i * 3 + 1],
							  m_worldOffset[2] + scene.BoxPositions[i * 3 + 2]};

		b3RigidBodyData body;
		memset(&body, 0, sizeof(body));
		// f32 by construction: this IS the precision the single-precision pipeline works in.
		body.m_pos = b3MakeVector3((float)bp[0], (float)bp[1], (float)bp[2]);
		body.m_quat = b3Quaternion(0.f, 0.f, 0.f, 1.f);
		m_positions.insert(m_positions.end(), bp, bp + 3);
		body.m_collidableIdx = m_boxCollidable;
		body.m_invMass = 1.f / kBoxMass;
		body.m_restituitionCoeff = 0.f;
		body.m_frictionCoeff = 0.5f;
		m_bodies.push_back(body);

		m_invInertia.push_back(invI);
		m_invInertia.push_back(invI);
		m_invInertia.push_back(invI);
		m_invInertia.push_back(0.f);

		DemoTransform dt;
		dt.position[0] = (float)(bp[0] - m_worldOffset[0]);
		dt.position[1] = (float)(bp[1] - m_worldOffset[1]);
		dt.position[2] = (float)(bp[2] - m_worldOffset[2]);
		dt.orientation[0] = 0.f;
		dt.orientation[1] = 0.f;
		dt.orientation[2] = 0.f;
		dt.orientation[3] = 1.f;
		m_transforms.push_back(dt);
	}

	// Kinematic bodies come after every dynamic box, so a box index never shifts when a scene
	// gains one. invMass 0 is what bars the solvers from ever writing to them.
	m_firstKinematicBody = (unsigned int)m_bodies.size();
	for (int k = 0; k < scene.KinematicCount(); ++k)
	{
		float pos[3], quat[4], linVel[3], angVel[3];
		DemoKinematicPose(scene.Kinematics[k], 0.0, pos, quat, linVel, angVel);

		const double kp[3] = {m_worldOffset[0] + pos[0], m_worldOffset[1] + pos[1],
							  m_worldOffset[2] + pos[2]};

		b3RigidBodyData body;
		memset(&body, 0, sizeof(body));
		body.m_pos = b3MakeVector3((float)kp[0], (float)kp[1], (float)kp[2]);
		body.m_quat = b3Quaternion(quat[0], quat[1], quat[2], quat[3]);
		m_positions.insert(m_positions.end(), kp, kp + 3);
		body.m_collidableIdx = m_boxCollidable;
		body.m_invMass = 0.f;
		body.m_restituitionCoeff = 0.f;
		body.m_frictionCoeff = 0.5f;
		m_bodies.push_back(body);
		m_invInertia.insert(m_invInertia.end(), 4, 0.f);

		DemoTransform dt;
		for (int a = 0; a < 3; ++a)
			dt.position[a] = pos[a];
		memcpy(dt.orientation, quat, sizeof(dt.orientation));
		m_transforms.push_back(dt);
	}

	for (size_t c = 0; c < scene.Constraints.size(); ++c)
	{
		const DemoConstraint& spec = scene.Constraints[c];
		const int a = ResolveAnchor(spec.bodyA, scene.BoxCount(), scene.KinematicCount());
		const int b = ResolveAnchor(spec.bodyB, scene.BoxCount(), scene.KinematicCount());
		if (a < 0 || b < 0)
			continue;

		m_joints.push_back(b3IrrlichtJointSolver::makePoint2Point(a, b, spec.pivotA, spec.pivotB));
	}

	if (!m_joints.empty() && getenv("OSDEMO_JOINT_LAYOUT"))
		ReportJointLayout(*m_jointSolver, m_joints);

	// Resident uploads happen once here; Step then never sends the body array again.
	if (m_resident && m_pipeline)
	{
		const unsigned int n = (unsigned int)m_bodies.size();
		m_pipeline->setGravity(b3MakeVector3(0.f, -10.f, 0.f));
		m_pipeline->setAngularDamping(0.99f);
		m_packed.assign(n, b3GpuIrrlichtRigidBodyPipeline::b3IrrBodyTransform());
		m_packedDs.assign(n, b3IrrGpu::b3IrrBodyTransformDS());

		if (!UploadBodies()
			|| !m_solver->uploadInvInertia(m_invInertia, n)
			|| !m_jointSolver->uploadInvInertia(m_invInertia, n)
			|| (!m_joints.empty() && !m_jointSolver->uploadJoints(m_joints, m_bodies)))
		{
			if (m_doubleSingle)
			{
				printf("[gpu] df64 resident upload failed.\n");
				return false;
			}
			printf("[gpu] resident upload failed - falling back to the vector path.\n");
			m_resident = false;
		}
	}

	if (m_sleep && !m_sleep->reset((unsigned int)m_bodies.size()))
		m_sleep.reset();

	// Contact-only wake leaves a sleeping body jointed to a mover hanging in violation of its
	// constraint; OSDEMO_JOINT_WAKE=0 restores that older behaviour.
	if (m_sleep)
	{
		const bool wantJointWake = m_jointWake && !m_joints.empty() && m_resident;
		m_sleep->setJoints(wantJointWake ? m_jointSolver->getJointBuffer() : 0,
						   m_jointSolver->getResidentJointCount());
		if (wantJointWake && !m_sleep->hasJointWake())
			printf("[gpu] joint-aware wake unavailable - CSPropagateJointWake failed to compile.\n");
	}

	// Sleeping is a resident-path feature: the vector path integrates on the CPU.
	m_awakeCount = (m_resident && m_sleepEnabled && m_sleep) ? scene.BoxCount() : -1;
	m_stepsSinceAwakeRead = 0;
	m_maxPairs = ChoosePairCapacity(m_bodies.size());
	m_overflowReported = false;
	m_pairOverflow = false;

	// Self-contained verification entry until the demo grows its own --state-roundtrip flag; the
	// guard is what stops the roundtrip's own Reset calls re-entering this.
	if (!m_roundTripDone && getenv("OSDEMO_STATE_ROUNDTRIP"))
	{
		m_roundTripDone = true;
		RunStateRoundTrip(*this, scene, EnvInt("OSDEMO_ROUNDTRIP_CAPTURE", 120),
						  EnvInt("OSDEMO_ROUNDTRIP_DIVERGE", 60),
						  EnvInt("OSDEMO_ROUNDTRIP_TRACK", 60));
		return Reset(scene);
	}
	if (!m_bucketHealDone && getenv("OSDEMO_BUCKET_HEAL"))
	{
		m_bucketHealDone = true;
		RunBucketHealTest(*this, scene, EnvInt("OSDEMO_HEAL_SETTLE", 120),
						  EnvInt("OSDEMO_HEAL_TICKS", 900));
		return Reset(scene);
	}
	return true;
}

bool GpuPhysicsBackend::SetPhaseTiming(bool on)
{
	m_phaseTiming = on;
	m_phaseSteps = 0;
	for (int i = 0; i < 6; ++i)
		m_phaseMs[i] = 0.0;
	if (!on)
		m_phaseHasResult = false;
	return true;
}

int GpuPhysicsBackend::EffectiveBucketCount() const
{
	if (!m_resident || m_hashMaterial < 0 || m_hashClearMaterial < 0 || m_bodies.empty())
		return 0;
	const int n = (int)m_bodies.size();
	return m_hashBuckets < n ? m_hashBuckets : n;
}

int GpuPhysicsBackend::HashBucketCount() const
{
	// One bucket is the whole-state hash, i.e. no narrowing on offer.
	const int buckets = EffectiveBucketCount();
	return buckets > 1 ? buckets : 0;
}

bool GpuPhysicsBackend::FoldBucketRoot(unsigned long long& out) const
{
	const unsigned int* got = (const unsigned int*)m_hashOutBuffer->getBufferPointer();
	if (!got)
		return false;

	// The root is the XOR of every bucket, so splitting the reduction costs nothing and cannot
	// change what a whole-state hash reports.
	unsigned int lo = 0;
	unsigned int hi = 0;
	for (int b = 0; b < m_activeBuckets; ++b)
	{
		lo ^= got[b * 2];
		hi ^= got[b * 2 + 1];
	}
	out = ((unsigned long long)hi << 32) | (unsigned long long)lo;
	return true;
}

bool GpuPhysicsBackend::ReduceHashGpu(unsigned long long& out) const
{
	if (!const_cast<GpuPhysicsBackend*>(this)->BeginHashState())
		return false;
	// Without this the fold reads a buffer that was never mapped back, and every whole-state hash
	// comes out 0 - which makes any comparison built on it pass vacuously.
	if (!EnsureHashDownloaded())
		return false;
	return FoldBucketRoot(out);
}

bool GpuPhysicsBackend::BeginHashState()
{
	if (!m_resident || m_hashMaterial < 0 || m_hashClearMaterial < 0 || m_bodies.empty())
		return false;
	irr::scene::IComputeBuffer* bodies = m_narrowphase ? m_narrowphase->getBodyBuffer() : 0;
	if (!bodies)
		return false;

	const unsigned int n = (unsigned int)m_bodies.size();
	const int buckets = EffectiveBucketCount() > 0 ? EffectiveBucketCount() : 1;
	const unsigned int perBucket = (n + (unsigned int)buckets - 1) / (unsigned int)buckets;
	b3IrrGpu::ensureBuffer<unsigned int>(m_hashOutBuffer, (irr::u32)buckets * 2);
	b3IrrGpu::ensureBuffer<unsigned int>(m_hashParamBuffer, 4);
	if (!m_hashOutBuffer || !m_hashParamBuffer)
		return false;
	m_activeBuckets = buckets;
	m_activePerBucket = (int)perBucket;

	unsigned int* params = (unsigned int*)m_hashParamBuffer->getBufferPointer();
	params[0] = n;
	params[1] = (unsigned int)buckets;
	params[2] = perBucket;
	params[3] = 0;
	PushToDevice(m_hashParamBuffer);

	irr::video::SMaterial mat;
	mat.MaterialType = (irr::video::E_MATERIAL_TYPE)m_hashClearMaterial;
	m_driver->setMaterial(mat);
	m_driver->bindComputeBuffer(0, m_hashParamBuffer, irr::video::EHBT_SHADER_RESOURCE);
	m_driver->bindComputeBuffer(0, m_hashOutBuffer, irr::video::EHBT_COMPUTE);
	m_driver->dispatchComputeShaderBound(
		irr::core::vector3d<irr::u32>(((irr::u32)buckets * 2 + 255) / 256, 1, 1));
	m_driver->unbindComputeResources();
	m_driver->computeBarrier(m_hashOutBuffer);

	mat.MaterialType = (irr::video::E_MATERIAL_TYPE)m_hashMaterial;
	m_driver->setMaterial(mat);
	m_driver->bindComputeBuffer(0, m_hashParamBuffer, irr::video::EHBT_SHADER_RESOURCE);
	m_driver->bindComputeBuffer(1, bodies, irr::video::EHBT_SHADER_RESOURCE);
	m_driver->bindComputeBuffer(0, m_hashOutBuffer, irr::video::EHBT_COMPUTE);
	m_driver->dispatchComputeShaderBound(irr::core::vector3d<irr::u32>((n + 255) / 256, 1, 1));
	m_driver->unbindComputeResources();
	m_driver->computeBarrier(m_hashOutBuffer);

	m_hashPending = true;
	m_hashReady = false;   // a new reduction invalidates whatever the last one left readable
	return true;
}

bool GpuPhysicsBackend::EnsureHashDownloaded() const
{
	// Downloading must not CONSUME the result: the root and the buckets are two reads of the same
	// reduction, and the root is always read first, which used to leave the buckets unreachable.
	if (!m_hashOutBuffer)
		return false;
	if (m_hashPending)
	{
		m_hashOutBuffer->downloadFromGPU();
		m_hashPending = false;
		m_hashReady = true;
	}
	return m_hashReady;
}

bool GpuPhysicsBackend::FetchHashState(unsigned long long& out)
{
	if (!EnsureHashDownloaded())
		return HashState(out);   // no reduction outstanding: fall back to the full readback
	return FoldBucketRoot(out);
}

bool GpuPhysicsBackend::FetchHashBuckets(std::vector<unsigned long long>& out)
{
	out.clear();
	if (!EnsureHashDownloaded() || m_activeBuckets < 2)
		return false;

	const unsigned int* got = (const unsigned int*)m_hashOutBuffer->getBufferPointer();
	if (!got)
		return false;

	out.resize((size_t)m_activeBuckets);
	for (int b = 0; b < m_activeBuckets; ++b)
		out[b] = ((unsigned long long)got[b * 2 + 1] << 32) | (unsigned long long)got[b * 2];
	return true;
}

bool GpuPhysicsBackend::HashState(unsigned long long& out) const
{
	if (!m_resident || !m_narrowphase || m_bodies.empty())
		return PhysicsBackend::HashState(out);

	// Same reduction the deferred path uses; only the readback timing differs.
	if (ReduceHashGpu(out))
		return true;

	irr::scene::IComputeBuffer* bodies = m_narrowphase->getBodyBuffer();
	if (!bodies)
		return false;
	bodies->downloadFromGPU();
	const void* raw = bodies->getBufferPointer();
	if (!raw)
		return false;

	const unsigned int n = (unsigned int)m_bodies.size();
	unsigned long long h = 14695981039346656037ULL;
	for (unsigned int i = 0; i < n; ++i)
	{
		// Only the fields the simulation integrates. The trailing collidable/mass/friction block is
		// scene constants, and pos[3]/quat's padding lane is never written, so both would add noise.
		if (m_doubleSingle)
		{
			const b3IrrGpu::b3IrrRigidBodyDataDS& b = ((const b3IrrGpu::b3IrrRigidBodyDataDS*)raw)[i];
			HashFloats(h, b.pos, 3);
			HashFloats(h, b.posLo, 3);
			HashFloats(h, b.quat, 4);
			HashFloats(h, b.linVel, 3);
			HashFloats(h, b.angVel, 3);
		}
		else
		{
			const b3RigidBodyData& b = ((const b3RigidBodyData*)raw)[i];
			HashFloats(h, (const float*)&b.m_pos, 3);
			HashFloats(h, (const float*)&b.m_quat, 4);
			HashFloats(h, (const float*)&b.m_linVel, 3);
			HashFloats(h, (const float*)&b.m_angVel, 3);
		}
	}
	out = h;
	return true;
}

void GpuPhysicsBackend::PushToDevice(irr::scene::IComputeBuffer* buffer)
{
	// setDirty only REQUESTS an upload that the next bind performs, and HashState's
	// downloadFromGPU would overwrite the pending write with stale device data first.
	buffer->setDirty();
	m_driver->bindComputeBuffer(0, buffer, irr::video::EHBT_SHADER_RESOURCE);
}

bool GpuPhysicsBackend::CaptureState(std::vector<unsigned char>& out) const
{
	out.clear();
	if (!m_narrowphase || m_bodies.empty())
		return false;

	const unsigned int n = (unsigned int)m_bodies.size();
	const unsigned int stride = m_doubleSingle ? (unsigned int)sizeof(b3IrrGpu::b3IrrRigidBodyDataDS)
											   : (unsigned int)sizeof(b3RigidBodyData);
	const unsigned char* bodySrc = 0;
	irr::scene::IComputeBuffer* sleepState = 0;
	irr::scene::IComputeBuffer* sleepRef = 0;

	if (m_resident)
	{
		irr::scene::IComputeBuffer* bodies = m_narrowphase->getBodyBuffer();
		if (!bodies || bodies->getStructureStride() != stride || bodies->getStructureCount() < n)
			return false;
		bodies->downloadFromGPU();
		bodySrc = (const unsigned char*)bodies->getBufferPointer();

		if (m_sleepEnabled && m_sleep)
		{
			sleepState = m_sleep->getSleepStateBuffer();
			sleepRef = m_sleep->getSleepRefBuffer();
			if (!sleepState || !sleepRef || sleepState->getStructureCount() < n
				|| sleepRef->getStructureCount() < n)
				return false;
			sleepState->downloadFromGPU();
			sleepRef->downloadFromGPU();
		}
	}
	else
	{
		// The vector path integrates on the CPU, so m_bodies IS the state rather than a mirror.
		bodySrc = (const unsigned char*)&m_bodies[0];
	}
	if (!bodySrc)
		return false;

	const size_t bodyBytes = (size_t)n * stride;
	const bool haveSleep = sleepState != 0;
	const unsigned int refStride = haveSleep ? sleepRef->getStructureStride() : 0u;
	const size_t sleepBytes = haveSleep ? (size_t)n * sizeof(unsigned int) + (size_t)n * refStride : 0;

	GpuStateHeader h;
	memset(&h, 0, sizeof(h));
	h.magic = kStateMagic;
	h.version = kStateVersion;
	h.bodyCount = n;
	h.bodyStride = stride;
	h.sleepRefStride = refStride;
	h.flags = (m_doubleSingle ? kFlagDoubleSingle : 0u) | (haveSleep ? kFlagSleepBlock : 0u)
			| (m_resident ? kFlagResident : 0u);
	h.awakeCount = m_awakeCount;
	h.bodyHash = HashBytes(bodySrc, bodyBytes);
	h.worldOffset[0] = m_worldOffset[0];
	h.worldOffset[1] = m_worldOffset[1];
	h.worldOffset[2] = m_worldOffset[2];

	out.resize(sizeof(h) + bodyBytes + sleepBytes);
	memcpy(&out[0], &h, sizeof(h));
	memcpy(&out[sizeof(h)], bodySrc, bodyBytes);
	if (haveSleep)
	{
		unsigned char* dst = &out[sizeof(h) + bodyBytes];
		memcpy(dst, sleepState->getBufferPointer(), (size_t)n * sizeof(unsigned int));
		memcpy(dst + (size_t)n * sizeof(unsigned int), sleepRef->getBufferPointer(),
			   (size_t)n * refStride);
	}
	return true;
}

bool GpuPhysicsBackend::ApplyState(const unsigned char* data, size_t bytes)
{
	if (!data || bytes < sizeof(GpuStateHeader) || !m_narrowphase || m_bodies.empty())
		return false;

	GpuStateHeader h;
	memcpy(&h, data, sizeof(h));
	if (h.magic != kStateMagic || h.version != kStateVersion)
		return false;
	if (((h.flags & kFlagDoubleSingle) != 0) != m_doubleSingle)
		return false;
	if (((h.flags & kFlagResident) != 0) != m_resident)
		return false;
	if (h.bodyCount != (unsigned int)m_bodies.size())
		return false;

	const unsigned int stride = m_doubleSingle ? (unsigned int)sizeof(b3IrrGpu::b3IrrRigidBodyDataDS)
											   : (unsigned int)sizeof(b3RigidBodyData);
	if (h.bodyStride != stride)
		return false;
	// A different offset means the positions mean something else; restoring them would place the
	// whole scene somewhere the ground is not.
	if (h.worldOffset[0] != m_worldOffset[0] || h.worldOffset[1] != m_worldOffset[1]
		|| h.worldOffset[2] != m_worldOffset[2])
		return false;

	const bool wantSleep = (h.flags & kFlagSleepBlock) != 0;
	const size_t bodyBytes = (size_t)h.bodyCount * stride;
	size_t need = sizeof(h) + bodyBytes;
	if (wantSleep)
		need += (size_t)h.bodyCount * sizeof(unsigned int) + (size_t)h.bodyCount * h.sleepRefStride;
	if (bytes != need)
		return false;
	if (HashBytes(data + sizeof(h), bodyBytes) != h.bodyHash)
		return false;

	// Resolve every destination BEFORE writing any of them: a refusal past the first memcpy would
	// leave a half-restored world, which is worse than not restoring at all.
	irr::scene::IComputeBuffer* bodies = 0;
	irr::scene::IComputeBuffer* sleepState = 0;
	irr::scene::IComputeBuffer* sleepRef = 0;
	if (m_resident)
	{
		bodies = m_narrowphase->getBodyBuffer();
		if (!bodies || bodies->getStructureStride() != stride
			|| bodies->getStructureCount() < h.bodyCount)
			return false;

		if (m_sleepEnabled && m_sleep)
		{
			sleepState = m_sleep->getSleepStateBuffer();
			sleepRef = m_sleep->getSleepRefBuffer();
			if (!sleepState || !sleepRef || sleepState->getStructureCount() < h.bodyCount
				|| sleepRef->getStructureCount() < h.bodyCount)
				return false;
		}
	}
	// Sleeping changes which bodies integrate at all, so a payload and a backend that disagree
	// about it would restore a state that cannot behave like the one captured.
	if (wantSleep != (sleepState != 0))
		return false;
	if (wantSleep && h.sleepRefStride != sleepRef->getStructureStride())
		return false;

	const unsigned char* bodySrc = data + sizeof(h);
	if (m_resident)
	{
		memcpy(bodies->getBufferPointer(), bodySrc, bodyBytes);
		PushToDevice(bodies);
		if (wantSleep)
		{
			const unsigned char* sleepSrc = bodySrc + bodyBytes;
			const size_t stateBytes = (size_t)h.bodyCount * sizeof(unsigned int);
			memcpy(sleepState->getBufferPointer(), sleepSrc, stateBytes);
			PushToDevice(sleepState);
			memcpy(sleepRef->getBufferPointer(), sleepSrc + stateBytes,
				   (size_t)h.bodyCount * h.sleepRefStride);
			PushToDevice(sleepRef);
		}
		m_driver->unbindComputeResources();
	}
	else
	{
		memcpy(&m_bodies[0], bodySrc, bodyBytes);
	}

	m_awakeCount = h.awakeCount;
	// That map would publish the pre-restore step's poses over the ones just restored.
	m_readbackPending = false;
	m_forceFullRefresh = true;
	PublishRestoredTransforms(bodySrc, h.bodyCount);
	return true;
}

bool GpuPhysicsBackend::CaptureBuckets(const std::vector<int>& buckets,
									   std::vector<unsigned char>& out) const
{
	out.clear();
	const int bucketCount = HashBucketCount();
	if (bucketCount <= 0 || !m_narrowphase || m_bodies.empty() || buckets.empty())
		return false;

	const unsigned int n = (unsigned int)m_bodies.size();
	const BodyLayout layout(m_doubleSingle);
	irr::scene::IComputeBuffer* bodies = m_narrowphase->getBodyBuffer();
	if (!bodies || bodies->getStructureStride() != layout.Stride || bodies->getStructureCount() < n)
		return false;

	// Same split BeginHashState dispatched with, so a bucket index means the same range both sides.
	const unsigned int perBucket = (n + (unsigned int)bucketCount - 1) / (unsigned int)bucketCount;
	std::vector<unsigned int> indices;
	for (size_t b = 0; b < buckets.size(); ++b)
	{
		if (buckets[b] < 0 || buckets[b] >= bucketCount)
			return false;
		const unsigned int first = (unsigned int)buckets[b] * perBucket;
		const unsigned int last = first + perBucket < n ? first + perBucket : n;
		for (unsigned int i = first; i < last; ++i)
			indices.push_back(i);
	}
	std::sort(indices.begin(), indices.end());
	indices.erase(std::unique(indices.begin(), indices.end()), indices.end());
	if (indices.empty())
		return false;

	irr::scene::IComputeBuffer* sleepState = 0;
	irr::scene::IComputeBuffer* sleepRef = 0;
	if (m_sleepEnabled && m_sleep)
	{
		sleepState = m_sleep->getSleepStateBuffer();
		sleepRef = m_sleep->getSleepRefBuffer();
		if (!sleepState || !sleepRef || sleepState->getStructureCount() < n
			|| sleepRef->getStructureCount() < n)
			return false;
		sleepState->downloadFromGPU();
		sleepRef->downloadFromGPU();
	}
	bodies->downloadFromGPU();
	const unsigned char* bodySrc = (const unsigned char*)bodies->getBufferPointer();
	if (!bodySrc)
		return false;

	const unsigned int records = (unsigned int)indices.size();
	const unsigned int refStride = sleepState ? sleepRef->getStructureStride() : 0u;
	const size_t indexBytes = (size_t)records * sizeof(unsigned int);
	const size_t bodyBytes = (size_t)records * layout.Stride;
	const size_t sleepBytes = sleepState ? indexBytes + (size_t)records * refStride : 0;

	GpuBucketHeader h;
	memset(&h, 0, sizeof(h));
	h.magic = kBucketMagic;
	h.version = kBucketVersion;
	h.bodyCount = n;
	h.bodyStride = layout.Stride;
	h.recordCount = records;
	h.sleepRefStride = refStride;
	h.bucketCount = (unsigned int)bucketCount;
	h.flags = (m_doubleSingle ? kFlagDoubleSingle : 0u) | (sleepState ? kFlagSleepBlock : 0u)
			| (m_resident ? kFlagResident : 0u);
	h.worldOffset[0] = m_worldOffset[0];
	h.worldOffset[1] = m_worldOffset[1];
	h.worldOffset[2] = m_worldOffset[2];

	out.resize(sizeof(h) + indexBytes + bodyBytes + sleepBytes);
	unsigned char* cursor = &out[sizeof(h)];
	memcpy(cursor, &indices[0], indexBytes);
	cursor += indexBytes;
	for (unsigned int r = 0; r < records; ++r)
		memcpy(cursor + (size_t)r * layout.Stride, bodySrc + (size_t)indices[r] * layout.Stride,
			   layout.Stride);
	cursor += bodyBytes;
	if (sleepState)
	{
		const unsigned int* stateSrc = (const unsigned int*)sleepState->getBufferPointer();
		const unsigned char* refSrc = (const unsigned char*)sleepRef->getBufferPointer();
		for (unsigned int r = 0; r < records; ++r)
			memcpy(cursor + (size_t)r * sizeof(unsigned int), &stateSrc[indices[r]],
				   sizeof(unsigned int));
		cursor += indexBytes;
		for (unsigned int r = 0; r < records; ++r)
			memcpy(cursor + (size_t)r * refStride, refSrc + (size_t)indices[r] * refStride, refStride);
	}

	h.payloadHash = HashBytes(&out[sizeof(h)], out.size() - sizeof(h));
	memcpy(&out[0], &h, sizeof(h));
	return true;
}

bool GpuPhysicsBackend::ApplyBucketsConverging(const unsigned char* data, size_t bytes,
											   const HealSettings& settings, double dtSeconds)
{
	const int bucketCount = HashBucketCount();
	if (!data || bytes < sizeof(GpuBucketHeader) || bucketCount <= 0 || m_bodies.empty())
		return false;
	if (dtSeconds <= 0.0)
		return false;

	GpuBucketHeader h;
	memcpy(&h, data, sizeof(h));
	if (h.magic != kBucketMagic || h.version != kBucketVersion)
		return false;
	if (((h.flags & kFlagDoubleSingle) != 0) != m_doubleSingle)
		return false;
	if (((h.flags & kFlagResident) != 0) != m_resident)
		return false;
	if (h.bodyCount != (unsigned int)m_bodies.size() || h.bucketCount != (unsigned int)bucketCount)
		return false;

	const BodyLayout layout(m_doubleSingle);
	if (h.bodyStride != layout.Stride || h.recordCount == 0 || h.recordCount > h.bodyCount)
		return false;
	if (h.worldOffset[0] != m_worldOffset[0] || h.worldOffset[1] != m_worldOffset[1]
		|| h.worldOffset[2] != m_worldOffset[2])
		return false;

	const bool wantSleep = (h.flags & kFlagSleepBlock) != 0;
	const size_t indexBytes = (size_t)h.recordCount * sizeof(unsigned int);
	const size_t bodyBytes = (size_t)h.recordCount * layout.Stride;
	size_t need = sizeof(h) + indexBytes + bodyBytes;
	if (wantSleep)
		need += indexBytes + (size_t)h.recordCount * h.sleepRefStride;
	if (bytes != need)
		return false;
	if (HashBytes(data + sizeof(h), bytes - sizeof(h)) != h.payloadHash)
		return false;

	irr::scene::IComputeBuffer* bodies = m_narrowphase ? m_narrowphase->getBodyBuffer() : 0;
	if (!bodies || bodies->getStructureStride() != layout.Stride
		|| bodies->getStructureCount() < h.bodyCount)
		return false;

	irr::scene::IComputeBuffer* sleepState = 0;
	irr::scene::IComputeBuffer* sleepRef = 0;
	if (m_sleepEnabled && m_sleep)
	{
		sleepState = m_sleep->getSleepStateBuffer();
		sleepRef = m_sleep->getSleepRefBuffer();
		if (!sleepState || !sleepRef || sleepState->getStructureCount() < h.bodyCount
			|| sleepRef->getStructureCount() < h.bodyCount)
			return false;
	}
	if (wantSleep != (sleepState != 0))
		return false;
	if (wantSleep && (h.sleepRefStride != sleepRef->getStructureStride()
					  || h.sleepRefStride < 4 * sizeof(float)))
		return false;

	const unsigned int* indices = (const unsigned int*)(data + sizeof(h));
	for (unsigned int r = 0; r < h.recordCount; ++r)
		if (indices[r] >= h.bodyCount)
			return false;

	// Download BEFORE any write: PushToDevice only requests an upload, and a download after it would
	// discard the pending write and hand back stale device data.
	bodies->downloadFromGPU();
	unsigned char* bodyDst = (unsigned char*)bodies->getBufferPointer();
	if (!bodyDst)
		return false;
	unsigned int* stateDst = 0;
	unsigned char* refDst = 0;
	if (wantSleep)
	{
		sleepState->downloadFromGPU();
		sleepRef->downloadFromGPU();
		stateDst = (unsigned int*)sleepState->getBufferPointer();
		refDst = (unsigned char*)sleepRef->getBufferPointer();
		if (!stateDst || !refDst)
			return false;
	}

	const unsigned char* bodySrc = data + sizeof(h) + indexBytes;
	const unsigned int* stateSrc = wantSleep
		? (const unsigned int*)(bodySrc + bodyBytes) : 0;
	const unsigned char* refSrc = wantSleep
		? (const unsigned char*)(stateSrc) + indexBytes : 0;

	// Every dynamic body in this demo is the same unit box, so one bounding radius covers them all.
	const double boundsRadius = (double)kBoxHalfExtent * 1.7320508075688772;

	auto snapBody = [&](unsigned int idx, unsigned int r, unsigned char* dst,
						const unsigned char* src) {
		memcpy(dst, src, layout.Stride);
		if (wantSleep)
		{
			memcpy(&stateDst[idx], &stateSrc[r], sizeof(unsigned int));
			memcpy(refDst + (size_t)idx * h.sleepRefStride,
				   refSrc + (size_t)r * h.sleepRefStride, h.sleepRefStride);
		}
		m_healHorizon.erase((int)idx);
		m_healRounds.erase((int)idx);
	};

	for (unsigned int r = 0; r < h.recordCount; ++r)
	{
		const unsigned int idx = indices[r];
		const unsigned char* src = bodySrc + (size_t)r * layout.Stride;
		unsigned char* dst = bodyDst + (size_t)idx * layout.Stride;

		float invMass = 0.f;
		memcpy(&invMass, dst + layout.InvMass, sizeof(invMass));

		double current[3];
		double target[3];
		ReadBodyPosition(dst, layout, m_doubleSingle, current);
		ReadBodyPosition(src, layout, m_doubleSingle, target);
		const double error[3] = {target[0] - current[0], target[1] - current[1],
								 target[2] - current[2]};
		const double errorLength = Length3(error);

		float targetVel[3];
		memcpy(targetVel, src + layout.LinVel, sizeof(targetVel));
		const double speed = std::sqrt((double)targetVel[0] * targetVel[0]
									 + (double)targetVel[1] * targetVel[1]
									 + (double)targetVel[2] * targetVel[2]);

		const double snapThreshold = std::max(settings.SnapFloorMeters,
			std::max(settings.SnapBodyRadii * boundsRadius,
					 settings.SnapTravelFraction * speed * settings.HorizonSeconds));
		const double lockEpsilon = std::max(settings.LockFloorMeters,
											settings.LockTravelFraction * speed * dtSeconds);

		// Too far to close plausibly, already close enough that an exact landing is invisible, or a
		// static that never steers: take the authority's bytes outright. This is what re-matches a hash.
		if (invMass == 0.f || errorLength > snapThreshold || errorLength < lockEpsilon)
		{
			snapBody(idx, r, dst, src);
			continue;
		}

		if (m_healMaxRounds > 0)
		{
			int& rounds = m_healRounds[(int)idx];
			if (++rounds >= m_healMaxRounds)
			{
				++m_healForcedSnaps;
				snapBody(idx, r, dst, src);
				continue;
			}
		}

		std::map<int, double>::iterator horizon = m_healHorizon.find((int)idx);
		if (horizon == m_healHorizon.end())
			horizon = m_healHorizon.insert(std::make_pair((int)idx, settings.HorizonSeconds)).first;
		horizon->second = std::max(horizon->second - dtSeconds, dtSeconds);
		const double hz = horizon->second;

		// Past the deadline the Hermite gain is 6/dt^2 and oscillates, so the engine only ever
		// recovers by overshooting past SnapFloorMeters. OSDEMO_HEAL_TERMINAL_SNAP=0 restores that.
		if (m_healTerminalSnap && hz <= dtSeconds)
		{
			snapBody(idx, r, dst, src);
			continue;
		}

		float v0[3];
		memcpy(v0, dst + layout.LinVel, sizeof(v0));
		float steered[3];
		for (int a = 0; a < 3; ++a)
		{
			// Rendezvous with where the authority WILL be; aiming at its current pose trails a
			// moving target forever. Cubic Hermite, so arrival matches velocity too.
			const double fromHere = (target[a] + (double)targetVel[a] * hz) - current[a];
			const double accel = (fromHere * 6.0
								  - ((double)v0[a] * 4.0 + (double)targetVel[a] * 2.0) * hz) / (hz * hz);
			steered[a] = (float)((double)v0[a] + accel * dtSeconds);
		}
		memcpy(dst + layout.LinVel, steered, sizeof(steered));

		if (wantSleep)
		{
			// A sleeping body never integrates, so a steered velocity would do nothing at all.
			stateDst[idx] = 0u;
			unsigned char* ref = refDst + (size_t)idx * h.sleepRefStride;
			memset(ref, 0, h.sleepRefStride);
			memcpy(ref, dst + layout.PosHi, 3 * sizeof(float));
			if (m_doubleSingle)
				memcpy(ref + 4 * sizeof(float), dst + layout.PosLo, 3 * sizeof(float));
		}
	}

	PushToDevice(bodies);
	if (wantSleep)
	{
		PushToDevice(sleepState);
		PushToDevice(sleepRef);
	}
	m_driver->unbindComputeResources();

	// A snapped body teleports, so its world bound and the tree's topology both predate the heal.
	m_forceFullRefresh = true;
	m_readbackPending = false;
	PublishRestoredTransforms(bodyDst, h.bodyCount);
	return true;
}

bool GpuPhysicsBackend::DebugPerturbBodies(int firstBody, int count, const double delta[3])
{
	if (!m_resident || !m_narrowphase || m_bodies.empty() || count <= 0 || firstBody < 0)
		return false;

	const unsigned int n = (unsigned int)m_bodies.size();
	const BodyLayout layout(m_doubleSingle);
	irr::scene::IComputeBuffer* bodies = m_narrowphase->getBodyBuffer();
	if (!bodies || bodies->getStructureStride() != layout.Stride || bodies->getStructureCount() < n)
		return false;
	if ((unsigned int)firstBody + (unsigned int)count > n)
		return false;

	bodies->downloadFromGPU();
	unsigned char* dst = (unsigned char*)bodies->getBufferPointer();
	if (!dst)
		return false;

	for (int i = 0; i < count; ++i)
	{
		unsigned char* body = dst + (size_t)(firstBody + i) * layout.Stride;
		double pos[3];
		ReadBodyPosition(body, layout, m_doubleSingle, pos);
		for (int a = 0; a < 3; ++a)
		{
			pos[a] += delta[a];
			if (m_doubleSingle)
				b3IrrGpu::b3DsSplit(pos[a], ((float*)(body + layout.PosHi))[a],
									((float*)(body + layout.PosLo))[a]);
			else
				((float*)(body + layout.PosHi))[a] = (float)pos[a];
		}
	}

	PushToDevice(bodies);
	m_driver->unbindComputeResources();
	m_forceFullRefresh = true;
	m_readbackPending = false;
	PublishRestoredTransforms(dst, n);
	return true;
}

int GpuPhysicsBackend::ResolveAnchor(int endpoint, int boxCount, int kinematicCount) const
{
	if (endpoint == kDemoAnchorGround)
		return 0;
	if (DemoAnchorIsKinematic(endpoint))
	{
		const int k = DemoAnchorKinematicIndex(endpoint);
		return (k >= 0 && k < kinematicCount) ? (int)m_firstKinematicBody + k : -1;
	}
	return (endpoint >= 0 && endpoint < boxCount) ? endpoint + 1 : -1;
}

bool GpuPhysicsBackend::ApplyKinematics(float deltaTime)
{
	if (m_kinematics.empty() || m_kinematicMaterial < 0 || !m_narrowphase)
		return false;

	irr::scene::IComputeBuffer* bodies = m_narrowphase->getBodyBuffer();
	if (!bodies)
		return false;

	m_simTime += (double)deltaTime;

	const unsigned int count = (unsigned int)m_kinematics.size();
	const unsigned int n = (unsigned int)m_bodies.size();

	std::vector<KinematicTargetGpu> targets;
	std::vector<KinematicTargetGpuDS> targetsDs;
	if (m_doubleSingle)
		targetsDs.resize(count);
	else
		targets.resize(count);

	for (unsigned int k = 0; k < count; ++k)
	{
		float pos[3], quat[4], linVel[3], angVel[3];
		DemoKinematicPose(m_kinematics[k], m_simTime, pos, quat, linVel, angVel);

		const unsigned int index = m_firstKinematicBody + k;
		unsigned int* bodyIndex;
		float* posHi;
		float* posLo = 0;
		float* dstQuat;
		float* dstLin;
		float* dstAng;
		if (m_doubleSingle)
		{
			KinematicTargetGpuDS& t = targetsDs[k];
			memset(&t, 0, sizeof(t));
			bodyIndex = &t.bodyIndex; posHi = t.pos; posLo = t.posLo;
			dstQuat = t.quat; dstLin = t.linVel; dstAng = t.angVel;
		}
		else
		{
			KinematicTargetGpu& t = targets[k];
			memset(&t, 0, sizeof(t));
			bodyIndex = &t.bodyIndex; posHi = t.pos;
			dstQuat = t.quat; dstLin = t.linVel; dstAng = t.angVel;
		}

		*bodyIndex = index;
		for (int a = 0; a < 3; ++a)
		{
			const double world = m_worldOffset[a] + (double)pos[a];
			if (posLo)
				b3IrrGpu::b3DsSplit(world, posHi[a], posLo[a]);
			else
				posHi[a] = (float)world;
			dstLin[a] = linVel[a];
			dstAng[a] = angVel[a];
		}
		memcpy(dstQuat, quat, 4 * sizeof(float));
	}

	if (m_doubleSingle)
		b3IrrGpu::uploadBuffer<KinematicTargetGpuDS>(m_kinTargetBuffer, &targetsDs[0], count);
	else
		b3IrrGpu::uploadBuffer<KinematicTargetGpu>(m_kinTargetBuffer, &targets[0], count);

	KinematicParamsGpu kp;
	kp.numTargets = count;
	kp.numBodies = n;
	kp.pad0 = kp.pad1 = 0;
	b3IrrGpu::uploadBuffer<KinematicParamsGpu>(m_kinParamBuffer, &kp, 1);
	if (!m_kinTargetBuffer || !m_kinParamBuffer)
		return false;

	irr::scene::IComputeBuffer* sleepState =
		(m_sleepEnabled && m_sleep) ? m_sleep->getSleepStateBuffer() : 0;

	irr::video::SMaterial mat;
	mat.MaterialType = (irr::video::E_MATERIAL_TYPE)m_kinematicMaterial;
	m_driver->setMaterial(mat);
	m_driver->bindComputeBuffer(2, m_kinParamBuffer, irr::video::EHBT_SHADER_RESOURCE);
	m_driver->bindComputeBuffer(3, m_kinTargetBuffer, irr::video::EHBT_SHADER_RESOURCE);
	m_driver->bindComputeBuffer(0, bodies, irr::video::EHBT_COMPUTE);
	if (sleepState)
		m_driver->bindComputeBuffer(3, sleepState, irr::video::EHBT_COMPUTE);
	m_driver->dispatchComputeShaderBound(irr::core::vector3d<irr::u32>((count + 63) / 64, 1, 1));
	m_driver->unbindComputeResources();

	m_driver->computeBarrier(bodies);
	if (sleepState)
		m_driver->computeBarrier(sleepState);
	return true;
}

void GpuPhysicsBackend::PublishRestoredTransforms(const unsigned char* bodySrc, unsigned int n)
{
	// Body 0 is the static ground, so the renderer's array starts at body index 1.
	const size_t count = m_transforms.size() < (size_t)(n - 1) ? m_transforms.size() : (size_t)(n - 1);
	for (size_t i = 0; i < count; ++i)
	{
		DemoTransform& dt = m_transforms[i];
		if (m_doubleSingle)
		{
			const b3IrrGpu::b3IrrRigidBodyDataDS& b =
				((const b3IrrGpu::b3IrrRigidBodyDataDS*)bodySrc)[i + 1];
			for (int a = 0; a < 3; ++a)
				dt.position[a] = (float)(b3IrrGpu::b3DsCombine(b.pos[a], b.posLo[a]) - m_worldOffset[a]);
			memcpy(dt.orientation, b.quat, sizeof(dt.orientation));
		}
		else
		{
			const b3RigidBodyData& b = ((const b3RigidBodyData*)bodySrc)[i + 1];
			const float* pos = (const float*)&b.m_pos;
			for (int a = 0; a < 3; ++a)
				dt.position[a] = (float)((double)pos[a] - m_worldOffset[a]);
			memcpy(dt.orientation, (const float*)&b.m_quat, sizeof(dt.orientation));
		}
	}
}

bool GpuPhysicsBackend::LastPhaseMs(double out[6]) const
{
	if (!m_phaseTiming || !m_phaseHasResult)
		return false;

	for (int i = 0; i < 6; ++i)
		out[i] = m_phasePublished[i];
	return true;
}

double GpuPhysicsBackend::SyncPhase(const std::chrono::steady_clock::time_point& mark)
{
	if (!m_phaseTiming)
		return 0.0;

	if (irr::scene::IComputeBuffer* drain = m_lbvh ? m_lbvh->getPairCountBuffer() : 0)
		drain->downloadFromGPU();
	return MillisSince(mark);
}

void GpuPhysicsBackend::ReadPairCount()
{
	if (irr::scene::IComputeBuffer* pairCount = m_lbvh ? m_lbvh->getPairCountBuffer() : 0)
	{
		pairCount->downloadFromGPU();
		unsigned int count = 0;
		memcpy(&count, pairCount->getBufferPointer(), sizeof(unsigned int));

		// Past the cap the append drops pairs in scheduler-dependent order, so two machines lose
		// DIFFERENT ones - silent divergence, which is why this is fatal to a sync run.
		if (count > m_maxPairs)
		{
			m_pairOverflow = true;
			if (!m_overflowReported)
			{
				m_overflowReported = true;
				printf("[gpu] PAIR OVERFLOW: %u pairs against a %u cap - pairs were dropped "
					   "nondeterministically. Raise OSDEMO_PAIR_RATIO or OSDEMO_MAX_PAIRS.\n",
					   count, m_maxPairs);
				fflush(stdout);
			}
		}
		m_lastPairCount = (int)(count > m_maxPairs ? m_maxPairs : count);
	}
}

bool GpuPhysicsBackend::RefreshAwakeCount()
{
	if (!m_sleepEnabled || !m_sleep)
		return false;

	unsigned int awake = 0;
	if (!m_sleep->readAwakeCount(awake))
		return false;

	m_awakeCount = (int)awake;
	m_stepsSinceAwakeRead = 0;
	return true;
}

bool GpuPhysicsBackend::UploadBodies()
{
	if (!m_doubleSingle)
		return m_narrowphase->uploadBodiesResident(m_bodies);

	// The upload edge: double -> float2(hi, lo), once. Permanent, not transitional - the CPU keeps
	// hardware doubles and only the GPU side is emulated.
	std::vector<b3IrrGpu::b3IrrRigidBodyDataDS> ds(m_bodies.size());
	for (size_t i = 0; i < m_bodies.size(); ++i)
	{
		const b3RigidBodyData& src = m_bodies[i];
		b3IrrGpu::b3IrrRigidBodyDataDS& dst = ds[i];
		memset(&dst, 0, sizeof(dst));
		for (int a = 0; a < 3; ++a)
			b3IrrGpu::b3DsSplit(m_positions[i * 3 + a], dst.pos[a], dst.posLo[a]);
		dst.quat[0] = (float)src.m_quat.x;
		dst.quat[1] = (float)src.m_quat.y;
		dst.quat[2] = (float)src.m_quat.z;
		dst.quat[3] = (float)src.m_quat.w;
		dst.collidableIdx = src.m_collidableIdx;
		dst.invMass = src.m_invMass;
		dst.restituitionCoeff = src.m_restituitionCoeff;
		dst.frictionCoeff = src.m_frictionCoeff;
	}
	return m_narrowphase->uploadBodiesResidentDS(ds);
}

void GpuPhysicsBackend::PublishTransforms(unsigned int n)
{
	// Body 0 is the static ground, so the renderer's array starts at packed index 1.
	const size_t count = m_transforms.size() < (size_t)(n - 1) ? m_transforms.size() : (size_t)(n - 1);

	if (!m_doubleSingle)
	{
		// DemoTransform and b3IrrBodyTransform are both 28-byte position+quaternion, so with no
		// world offset the renderer's copy stays one memcpy rather than a per-body loop.
		if (m_worldOffset[0] == 0.0 && m_worldOffset[1] == 0.0 && m_worldOffset[2] == 0.0)
		{
			memcpy(&m_transforms[0], &m_packed[1], count * sizeof(DemoTransform));
			return;
		}
		for (size_t i = 0; i < count; ++i)
		{
			const b3GpuIrrlichtRigidBodyPipeline::b3IrrBodyTransform& src = m_packed[i + 1];
			DemoTransform& dt = m_transforms[i];
			for (int a = 0; a < 3; ++a)
				dt.position[a] = (float)((double)src.position[a] - m_worldOffset[a]);
			memcpy(dt.orientation, src.orientation, sizeof(dt.orientation));
		}
		return;
	}

	// The readback edge: hi + lo -> double, then back to scene-local so every backend reports
	// directly comparable positions.
	for (size_t i = 0; i < count; ++i)
	{
		const b3IrrGpu::b3IrrBodyTransformDS& src = m_packedDs[i + 1];
		DemoTransform& dt = m_transforms[i];
		for (int a = 0; a < 3; ++a)
			dt.position[a] = (float)(b3IrrGpu::b3DsCombine(src.position[a], src.positionLo[a])
									 - m_worldOffset[a]);
		memcpy(dt.orientation, src.orientation, sizeof(dt.orientation));
	}
}

long long GpuPhysicsBackend::GpuBufferBytes() const
{
	// Pair buffer + body/inertia uploads + the contact buffer, which is sized to the live pair
	// count each step (a pair yields at most one contact, so that bound is exact).
	const long long pairs = (long long)m_maxPairs * 8;
	const long long bodyStride = m_doubleSingle ? (long long)sizeof(b3IrrGpu::b3IrrRigidBodyDataDS)
											   : (long long)sizeof(b3RigidBodyData);
	const long long aabbStride = m_doubleSingle ? (long long)sizeof(b3IrrAabbDS)
											   : (long long)sizeof(b3IrrAabb);
	const long long bodies = (long long)m_bodies.size() * bodyStride;
	const long long aabbs = (long long)m_bodies.size() * aabbStride;
	const long long contacts = (long long)m_lastPairCount * 112;
	return pairs + bodies + aabbs + contacts;
}

irr::scene::IComputeBuffer* GpuPhysicsBackend::GpuTransformBuffer() const
{
	if (!m_pipeline || !m_resident)
		return 0;
	return m_pipeline->getRenderTransformBuffer();
}

void GpuPhysicsBackend::Step(float deltaTime)
{
	if (!m_narrowphase || !m_lbvh || !m_solver || !m_jointSolver || m_bodies.empty())
		return;

	if (m_resident && m_pipeline)
		StepResident(deltaTime);
	else
		StepVector(deltaTime);
}

void GpuPhysicsBackend::StepResident(float deltaTime)
{
	const std::chrono::steady_clock::time_point stepBegin = std::chrono::steady_clock::now();
	const unsigned int n = (unsigned int)m_bodies.size();
	irr::scene::IComputeBuffer* bodies = m_narrowphase->getBodyBuffer();
	if (!bodies)
		return;

	// Deferred readback: map LAST step's transforms now, a whole frame after their dispatch, so the
	// GPU is long done and Map does not stall. Costs one frame of latency on the rendered pose.
	if (m_deferReadback && m_readbackPending)
	{
		const std::chrono::steady_clock::time_point deferMark = std::chrono::steady_clock::now();
		if (m_doubleSingle ? m_pipeline->downloadTransformsDS(m_packedDs, n)
						   : m_pipeline->downloadTransforms(m_packed, n))
		{
			if (n > 1 && !m_transforms.empty()
				&& (m_doubleSingle ? m_packedDs.size() : m_packed.size()) >= (size_t)n)
				PublishTransforms(n);
		}
		m_lastTransferMs = MillisSince(deferMark);
		// Same frame of latency, same reason: the pair buffer is not rewritten until the LBVH
		// dispatch below, so this still reads the step it belongs to.
		ReadPairCount();
		m_readbackPending = false;
	}

	// Last step's bits: a sleeping body keeps the bound it already has instead of recomputing it.
	irr::scene::IComputeBuffer* priorSleep =
		(m_sleepEnabled && m_sleep) ? m_sleep->getSleepStateBuffer() : 0;

	// First step after a restore: a sleeping body's bound is never recomputed and a sparse refit
	// only touches leaves that moved, so both would still describe the pre-restore world.
	const bool fullRefresh = m_forceFullRefresh;
	m_forceFullRefresh = false;

	// Before the AABB pass: an externally driven body whose bound is computed from last step's pose
	// stops colliding with whatever it has since swept into.
	ApplyKinematics(deltaTime);

	std::chrono::steady_clock::time_point phaseMark = std::chrono::steady_clock::now();

	if (!m_narrowphase->computeWorldAabbsResident(bodies, n, 0.02f, fullRefresh ? 0 : priorSleep))
		return;
	m_phaseMs[0] += SyncPhase(phaseMark);
	phaseMark = std::chrono::steady_clock::now();

	// Refit between periodic full rebuilds: the topology is only re-sorted every N steps, which is
	// where the settled-world win comes from. Bounds stay conservative in between, never missing pairs.
	const bool refitThisStep = !fullRefresh && m_refitInterval > 0
							&& (m_stepsSinceTreeRebuild + 1) < m_refitInterval;
	// Sleeping leaves stop traversing: at 3% awake that is the bulk of the broadphase, and the
	// both-asleep pairs it forgoes are the ones compactPairs discards a phase later anyway.
	if (!m_lbvh->calculateOverlappingPairsResident(m_narrowphase->getWorldAabbBuffer(), n, m_maxPairs,
												   0, 0, refitThisStep,
												   m_queryGate ? priorSleep : 0,
												   (m_sparseRefit && !fullRefresh) ? priorSleep : 0))
		return;
	m_stepsSinceTreeRebuild = refitThisStep ? m_stepsSinceTreeRebuild + 1 : 0;
	// Tree depth sets how many dispatches the AABB fit costs, so it explains the broad phase.
	if (m_phaseTiming && !m_depthReported && !refitThisStep)
	{
		m_depthReported = true;
		printf("[gpu] lbvh leaves=%d maxDepth=%d\n", (int)n, m_lbvh->getLastMaxDistance());
		fflush(stdout);
	}
	m_phaseMs[1] += SyncPhase(phaseMark);
	phaseMark = std::chrono::steady_clock::now();

	// Both-asleep pairs cannot have changed since last step, so they never reach SAT/clipping.
	irr::scene::IComputeBuffer* narrowPairs = m_lbvh->getPairBuffer();
	irr::scene::IComputeBuffer* narrowPairCount = m_lbvh->getPairCountBuffer();
	if (priorSleep && m_sleep->compactPairs(narrowPairs, narrowPairCount, m_maxPairs))
	{
		narrowPairs = m_sleep->getLivePairBuffer();
		narrowPairCount = m_sleep->getLivePairCountBuffer();
	}
	m_phaseMs[2] += SyncPhase(phaseMark);
	phaseMark = std::chrono::steady_clock::now();

	if (!m_narrowphase->computeConvexContactsResident(bodies, n, narrowPairs, narrowPairCount,
													 m_maxPairs, m_maxPairs))
		return;
	m_phaseMs[3] += SyncPhase(phaseMark);
	phaseMark = std::chrono::steady_clock::now();

	// Before the solve, so a body woken by this step's contacts is solved this step rather than
	// next - which is what stops a drop landing on a frozen, non-reacting stack.
	irr::scene::IComputeBuffer* sleepState = 0;
	irr::scene::IComputeBuffer* solveContacts = m_narrowphase->getContactBuffer();
	irr::scene::IComputeBuffer* solveCount = m_narrowphase->getContactCountBuffer();
	if (m_sleepEnabled && m_sleep
		&& m_sleep->step(bodies, n, m_narrowphase->getContactBuffer(),
						 m_narrowphase->getContactCountBuffer(), m_maxPairs))
	{
		sleepState = m_sleep->getSleepStateBuffer();
		// The solver dispatches indirectly off this count, so a fully settled scene costs it
		// almost no threads at all rather than ten iterations of early-outs.
		solveContacts = m_sleep->getLiveContactBuffer();
		solveCount = m_sleep->getLiveContactCountBuffer();
	}

	m_solver->solveContactsResident(bodies, n, solveContacts, solveCount, m_maxPairs,
									m_contactIterations, deltaTime, 0.2f, sleepState);

	// Joints last so a link is never left visibly stretched by a contact solved after it.
	if (!m_joints.empty())
		m_jointSolver->solveJointsResident(bodies, n, 20, deltaTime, 0.2f, sleepState);

	m_phaseMs[4] += SyncPhase(phaseMark);
	phaseMark = std::chrono::steady_clock::now();

	// The one readback a step owes the renderer: 28 bytes per body instead of 80.
	const std::chrono::steady_clock::time_point transferMark = std::chrono::steady_clock::now();
	if (!m_pipeline->integrateAndPackResident(bodies, n, deltaTime, sleepState))
		return;

	if (m_deferReadback)
	{
		m_readbackPending = true;   // mapped at the top of the next step
	}
	else
	{
		if (m_doubleSingle ? !m_pipeline->downloadTransformsDS(m_packedDs, n)
						   : !m_pipeline->downloadTransforms(m_packed, n))
			return;
		m_lastTransferMs = MillisSince(transferMark);

		if (n > 1 && !m_transforms.empty()
			&& (m_doubleSingle ? m_packedDs.size() : m_packed.size()) >= (size_t)n)
			PublishTransforms(n);
	}

	m_phaseMs[5] += SyncPhase(phaseMark);

	if (m_phaseTiming && ++m_phaseSteps >= 30)
	{
		const double inv = 1.0 / (double)m_phaseSteps;
		double total = 0.0;
		for (int i = 0; i < 6; ++i)
		{
			m_phasePublished[i] = m_phaseMs[i] * inv;
			total += m_phasePublished[i];
			m_phaseMs[i] = 0.0;
		}
		m_phaseHasResult = true;
		m_phaseSteps = 0;

		printf("[phase ms/step, SERIALISED - read shares not totals] aabb %.2f  broad %.2f  "
			   "paircompact %.2f  narrow %.2f  sleep+solve %.2f  integrate %.2f  (sum %.2f)\n",
			   m_phasePublished[0], m_phasePublished[1], m_phasePublished[2], m_phasePublished[3],
			   m_phasePublished[4], m_phasePublished[5], total);
		// Redirected stdout is fully buffered, so without this the log stays empty until exit.
		fflush(stdout);
	}

	if (!m_deferReadback)
		ReadPairCount();

	// Gated to an interval: this is another 4-byte stall and the awake count is an observability
	// readout, not something the simulation itself consumes.
	if (sleepState && ++m_stepsSinceAwakeRead >= m_awakeInterval)
		RefreshAwakeCount();

	m_lastStepMs = MillisSince(stepBegin);
}

void GpuPhysicsBackend::StepVector(float deltaTime)
{
	const std::chrono::steady_clock::time_point stepBegin = std::chrono::steady_clock::now();
	double transferMs = 0.0;

	// Gravity is a trivial per-body add; it stays on the CPU rather than paying a dispatch for
	// what amounts to one FMA per body.
	for (size_t i = 0; i < m_bodies.size(); ++i)
	{
		if (m_bodies[i].m_invMass > 0.f)
			m_bodies[i].m_linVel.y -= 10.f * deltaTime;
	}

	std::chrono::steady_clock::time_point transferMark = std::chrono::steady_clock::now();
	if (!m_narrowphase->computeWorldAabbs(m_bodies, m_aabbs, 0.02f))
		return;
	transferMs += MillisSince(transferMark);

	std::vector<std::pair<unsigned int, unsigned int> > pairs;
	bool overflowed = false;
	transferMark = std::chrono::steady_clock::now();
	if (!m_lbvh->calculateOverlappingPairs(m_aabbs, pairs, m_maxPairs, &overflowed))
		return;
	transferMs += MillisSince(transferMark);
	m_lastPairCount = (int)pairs.size();

	std::vector<b3Contact4Data> contacts;
	if (!pairs.empty())
	{
		transferMark = std::chrono::steady_clock::now();
		// Exactly pairs.size(): the kernel appends at most one contact per pair, so a larger
		// allocation is pure waste - 268 MB of it at 600k bodies under the old flat cap.
		if (!m_narrowphase->computeConvexContacts(m_bodies, pairs, contacts,
												  (unsigned int)pairs.size(), &overflowed))
			return;
		transferMs += MillisSince(transferMark);
	}

	// Past the cap the GPU silently discards the excess, so an unreported overflow means
	// plausible timings for a simulation missing contacts.
	if (overflowed && !m_overflowReported)
	{
		m_overflowReported = true;
		printf("[gpu] WARNING: exceeded the %u pair/contact cap with %d bodies - contacts are being\n"
			   "      dropped and this run's physics is NOT trustworthy. Raise OSDEMO_PAIR_RATIO.\n",
			   m_maxPairs, (int)m_bodies.size());
		fflush(stdout);
	}

	if (!contacts.empty())
	{
		transferMark = std::chrono::steady_clock::now();
		m_solver->solveContacts(m_bodies, m_invInertia, contacts, m_contactIterations, deltaTime, 0.2f);
		transferMs += MillisSince(transferMark);
	}

	// Joints last so a link is never left visibly stretched by a contact solved after it.
	if (!m_joints.empty())
	{
		transferMark = std::chrono::steady_clock::now();
		m_jointSolver->solveJoints(m_bodies, m_invInertia, m_joints, 20, deltaTime, 0.2f);
		transferMs += MillisSince(transferMark);
	}

	// Integrate on the CPU for now: the ported integrator lives in b3GpuIrrlichtRigidBodyPipeline,
	// which owns its own body buffer, so wiring it here would mean a third copy of the same data.
	for (size_t i = 0; i < m_bodies.size(); ++i)
	{
		b3RigidBodyData& body = m_bodies[i];
		if (body.m_invMass <= 0.f)
			continue;

		body.m_pos.x += body.m_linVel.x * deltaTime;
		body.m_pos.y += body.m_linVel.y * deltaTime;
		body.m_pos.z += body.m_linVel.z * deltaTime;

		const b3Vector3 w = b3MakeVector3((float)body.m_angVel.x, (float)body.m_angVel.y,
										  (float)body.m_angVel.z);
		b3Quat q = body.m_quat;
		const b3Quat dq = b3Quaternion(w.x * deltaTime * 0.5f, w.y * deltaTime * 0.5f,
									   w.z * deltaTime * 0.5f, 0.f);
		q.x += dq.x * q.w + dq.y * q.z - dq.z * q.y;
		q.y += dq.y * q.w + dq.z * q.x - dq.x * q.z;
		q.z += dq.z * q.w + dq.x * q.y - dq.y * q.x;
		q.w -= dq.x * q.x + dq.y * q.y + dq.z * q.z;
		QuatNormalize(q);
		body.m_quat = q;

		// Light damping keeps the stack from jittering itself apart; the CPU backends get the
		// same effect from Bullet's own solver damping.
		body.m_linVel.x *= 0.999f;
		body.m_linVel.y *= 0.999f;
		body.m_linVel.z *= 0.999f;
		body.m_angVel.x *= 0.99f;
		body.m_angVel.y *= 0.99f;
		body.m_angVel.z *= 0.99f;
	}

	for (size_t i = 1; i < m_bodies.size(); ++i)
	{
		const size_t slot = i - 1;
		if (slot >= m_transforms.size())
			break;
		DemoTransform& dt = m_transforms[slot];
		dt.position[0] = (float)m_bodies[i].m_pos.x;
		dt.position[1] = (float)m_bodies[i].m_pos.y;
		dt.position[2] = (float)m_bodies[i].m_pos.z;
		dt.orientation[0] = (float)m_bodies[i].m_quat.x;
		dt.orientation[1] = (float)m_bodies[i].m_quat.y;
		dt.orientation[2] = (float)m_bodies[i].m_quat.z;
		dt.orientation[3] = (float)m_bodies[i].m_quat.w;
	}

	m_lastTransferMs = transferMs;
	m_lastStepMs = MillisSince(stepBegin);

	// OSDEMO_GPU_DEBUG=N traces the first N steps' pipeline counts. The phases each pass their
	// own unit tests, so a composition fault only shows as counts across a real sequence.
	static int debugStepsLeft = -1;
	if (debugStepsLeft < 0)
	{
		const char* env = getenv("OSDEMO_GPU_DEBUG");
		debugStepsLeft = env ? atoi(env) : 0;
	}
	if (debugStepsLeft > 0)
	{
		--debugStepsLeft;
		printf("[gpu] pairs=%d contacts=%d  body1 y=%.3f vy=%.3f\n", (int)pairs.size(),
			   (int)contacts.size(), (float)m_bodies[1].m_pos.y, (float)m_bodies[1].m_linVel.y);
		// Lateral state printed at full precision: "exactly zero" and "small but growing" are
		// different diagnoses and %.3f cannot tell them apart.
		static const size_t probe[3] = {1, 5, 10};
		for (int p = 0; p < 3; ++p)
		{
			if (probe[p] >= m_bodies.size())
				continue;
			const b3RigidBodyData& bd = m_bodies[probe[p]];
			printf("      b%-3d px=%+.9e pz=%+.9e vx=%+.9e vz=%+.9e wx=%+.9e wy=%+.9e wz=%+.9e\n",
				   (int)probe[p], (double)bd.m_pos.x, (double)bd.m_pos.z, (double)bd.m_linVel.x,
				   (double)bd.m_linVel.z, (double)bd.m_angVel.x, (double)bd.m_angVel.y,
				   (double)bd.m_angVel.z);
		}
		fflush(stdout);
	}
}
