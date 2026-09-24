/*******************************************************
* Copyright (C) 2026 David Delisle granyte@hotmail.com
*
* This file is part of Outerspace.
*
* Outerspace can not be copied and/or distributed without the express
* permission of David Delisle
*******************************************************/
#include "CpuBulletBackend.h"

#include "StateRoundTrip.h"

#include "btBulletDynamicsCommon.h"
#include "BulletCollision/CollisionDispatch/btCollisionDispatcherMt.h"
#include "BulletDynamics/Dynamics/btDiscreteDynamicsWorldMt.h"
#include "BulletDynamics/ConstraintSolver/btSequentialImpulseConstraintSolverMt.h"
#include "BulletDynamics/ConstraintSolver/btNNCGConstraintSolver.h"
#include "LinearMath/btQuickprof.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <set>

namespace
{
/// 100, not Bullet's default 10 (which CBulletThread uses): a 24-tall stack collapses at 10 purely
/// from iteration starvation, so the default measured a collapsed scene rather than the solver.
const int kSolverIterations = 100;
const btScalar kBoxMass = btScalar(1);

const unsigned int kCpuStateMagic = 0x43545331u;   // "1STC"
const unsigned int kCpuStateVersion = 1u;
const unsigned int kCpuFlagDoubleScalar = 1u << 0;

/// Shares its first 24 bytes with the GPU backend's header so both refuse the same way; the magic
/// is what keeps one backend's payload out of the other.
struct CpuStateHeader
{
	unsigned int magic;
	unsigned int version;
	unsigned int bodyCount;
	unsigned int bodyStride;
	unsigned int unusedStride;
	unsigned int flags;
	int awakeCount;
	unsigned int reserved;
	unsigned long long bodyHash;
	double worldOffset[3];
};
static_assert(sizeof(CpuStateHeader) == 64, "CpuStateHeader must stay a fixed 64 bytes");

/// Everything btDiscreteDynamicsWorld integrates for one body, plus the deactivation timer.
struct CpuBodyState
{
	btScalar origin[3];
	btScalar rotation[4];   // xyzw
	btScalar linVel[3];
	btScalar angVel[3];
	btScalar deactivationTime;
	int activationState;
	int pad;
};

/// FNV-1a, matching PhysicsBackend::HashFloats' constants.
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

/// @brief Reads an int env var, returning the fallback when unset or unparseable.
int EnvInt(const char* name, int fallback)
{
	if (const char* env = getenv(name))
	{
		const int parsed = atoi(env);
		if (parsed != 0 || env[0] == '0')
			return parsed;
	}
	return fallback;
}
}  // namespace

/// Timing/shape counters shared by the instrumented ST and MT solvers.
struct SolverStats
{
	SolverStats()
		: m_batching(0), m_phases(0), m_batches(0), m_contactConstraints(0), m_minBatch(0),
		  m_maxBatch(0), m_setupMs(0.0), m_iterMs(0.0), m_iterations(0), m_groups(0)
	{
	}

	int m_batching;
	int m_phases;
	int m_batches;
	int m_contactConstraints;
	int m_minBatch;
	int m_maxBatch;
	double m_setupMs;
	double m_iterMs;
	long long m_iterations;
	long long m_groups;
};

/**
 * @brief Stock single-threaded solver with the same setup/iteration timers as the MT one.
 */
class InstrumentedSolver : public btSequentialImpulseConstraintSolver
{
public:
	/**
	 * @brief Times solver setup.
	 * @return Whatever the base solver returns.
	 */
	virtual btScalar solveGroupCacheFriendlySetup(btCollisionObject** bodies, int numBodies,
												  btPersistentManifold** manifoldPtr, int numManifolds,
												  btTypedConstraint** constraints, int numConstraints,
												  const btContactSolverInfo& infoGlobal,
												  btIDebugDraw* debugDrawer) BT_OVERRIDE
	{
		const std::chrono::steady_clock::time_point b = std::chrono::steady_clock::now();
		const btScalar r = btSequentialImpulseConstraintSolver::solveGroupCacheFriendlySetup(
			bodies, numBodies, manifoldPtr, numManifolds, constraints, numConstraints, infoGlobal,
			debugDrawer);
		m_stats.m_setupMs += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - b).count();
		++m_stats.m_groups;
		m_stats.m_contactConstraints = m_tmpSolverContactConstraintPool.size();
		return r;
	}

	/**
	 * @brief Times one solver iteration.
	 * @return Whatever the base solver returns.
	 */
	virtual btScalar solveSingleIteration(int iteration, btCollisionObject** bodies, int numBodies,
										  btPersistentManifold** manifoldPtr, int numManifolds,
										  btTypedConstraint** constraints, int numConstraints,
										  const btContactSolverInfo& infoGlobal,
										  btIDebugDraw* debugDrawer) BT_OVERRIDE
	{
		const std::chrono::steady_clock::time_point b = std::chrono::steady_clock::now();
		const btScalar r = btSequentialImpulseConstraintSolver::solveSingleIteration(
			iteration, bodies, numBodies, manifoldPtr, numManifolds, constraints, numConstraints,
			infoGlobal, debugDrawer);
		m_stats.m_iterMs += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - b).count();
		++m_stats.m_iterations;
		return r;
	}

	SolverStats m_stats;
};

/**
 * @brief btSequentialImpulseConstraintSolverMt with batching/timing counters exposed.
 *
 * m_batchedContactConstraints is protected in Bullet, so a subclass is the only way to read the
 * phase/batch shape the MT solver actually built without patching the vendored library.
 */
class InstrumentedSolverMt : public btSequentialImpulseConstraintSolverMt
{
public:

	/**
	 * @brief Times solver setup and captures the batch/phase shape it produced.
	 * @return Whatever the base solver returns.
	 */
	virtual btScalar solveGroupCacheFriendlySetup(btCollisionObject** bodies, int numBodies,
												  btPersistentManifold** manifoldPtr, int numManifolds,
												  btTypedConstraint** constraints, int numConstraints,
												  const btContactSolverInfo& infoGlobal,
												  btIDebugDraw* debugDrawer) BT_OVERRIDE
	{
		const std::chrono::steady_clock::time_point b = std::chrono::steady_clock::now();
		const btScalar r = btSequentialImpulseConstraintSolverMt::solveGroupCacheFriendlySetup(
			bodies, numBodies, manifoldPtr, numManifolds, constraints, numConstraints, infoGlobal,
			debugDrawer);
		m_stats.m_setupMs += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - b).count();
		++m_stats.m_groups;

		m_stats.m_batching = m_useBatching ? 1 : 0;
		const btBatchedConstraints& bc = m_batchedContactConstraints;
		m_stats.m_phases = bc.m_phases.size();
		m_stats.m_batches = bc.m_batches.size();
		m_stats.m_contactConstraints = bc.m_constraintIndices.size();
		m_stats.m_minBatch = 0;
		m_stats.m_maxBatch = 0;
		for (int i = 0; i < bc.m_batches.size(); ++i)
		{
			const int sz = bc.m_batches[i].end - bc.m_batches[i].begin;
			if (i == 0 || sz < m_stats.m_minBatch)
				m_stats.m_minBatch = sz;
			if (sz > m_stats.m_maxBatch)
				m_stats.m_maxBatch = sz;
		}
		return r;
	}

	/**
	 * @brief Times one solver iteration so setup cost can be separated from per-iteration cost.
	 * @return Whatever the base solver returns.
	 */
	virtual btScalar solveSingleIteration(int iteration, btCollisionObject** bodies, int numBodies,
										  btPersistentManifold** manifoldPtr, int numManifolds,
										  btTypedConstraint** constraints, int numConstraints,
										  const btContactSolverInfo& infoGlobal,
										  btIDebugDraw* debugDrawer) BT_OVERRIDE
	{
		const std::chrono::steady_clock::time_point b = std::chrono::steady_clock::now();
		const btScalar r = btSequentialImpulseConstraintSolverMt::solveSingleIteration(
			iteration, bodies, numBodies, manifoldPtr, numManifolds, constraints, numConstraints,
			infoGlobal, debugDrawer);
		m_stats.m_iterMs += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - b).count();
		++m_stats.m_iterations;
		return r;
	}

	SolverStats m_stats;
};

CpuBulletBackend::CpuBulletBackend(bool multithreaded, int threadCount)
	: m_multithreaded(multithreaded), m_threadCount(threadCount),
	  m_name(multithreaded ? "Bullet CPU (multi-threaded)" : "Bullet CPU (single-threaded)"),
	  m_lastStepMs(0.0), m_lastPairCount(0), m_awakeCount(-1), m_stepCount(0),
	  m_roundTripDone(false), m_solverMt(0), m_solverSt(0)
{
	m_worldOffset[0] = m_worldOffset[1] = m_worldOffset[2] = 0.0;
}

CpuBulletBackend::~CpuBulletBackend()
{
	Teardown();
}

const char* CpuBulletBackend::Name() const
{
	return m_name.c_str();
}

void CpuBulletBackend::Teardown()
{
	if (m_world)
	{
		// Constraints first: they reference the bodies, so removing bodies underneath a live
		// constraint leaves the world holding dangling pointers.
		for (int i = m_world->getNumConstraints() - 1; i >= 0; --i)
		{
			btTypedConstraint* c = m_world->getConstraint(i);
			m_world->removeConstraint(c);
			delete c;
		}

		for (int i = m_world->getNumCollisionObjects() - 1; i >= 0; --i)
		{
			btCollisionObject* obj = m_world->getCollisionObjectArray()[i];
			btRigidBody* body = btRigidBody::upcast(obj);
			if (body && body->getMotionState())
				delete body->getMotionState();
			m_world->removeCollisionObject(obj);
			delete obj;
		}
	}
	m_world.reset();
	m_solver.reset();
	m_broadphase.reset();
	m_dispatcher.reset();
	m_config.reset();
	m_boxShape.reset();
	m_groundShape.reset();
	m_transforms.clear();
}

/**
 * @brief Prints island counts, MT batch shape and the solver's setup/iteration time split.
 */
void CpuBulletBackend::ReportStats()
{
	std::set<int> islands;
	int active = 0;
	const btCollisionObjectArray& objects = m_world->getCollisionObjectArray();
	for (int i = 0; i < objects.size(); ++i)
	{
		const btCollisionObject* o = objects[i];
		if (o->isStaticOrKinematicObject())
			continue;
		++active;
		islands.insert(o->getIslandTag());
	}

	const SolverStats* s = m_solverMt ? &m_solverMt->m_stats : (m_solverSt ? &m_solverSt->m_stats : 0);
	const int threads = m_multithreaded && btGetTaskScheduler() ? btGetTaskScheduler()->getNumThreads() : 1;

	printf("  [stats %s] step %d  bodies %d  islands %u  manifolds %d  threads %d",
		   m_multithreaded ? "MT" : "ST", m_stepCount, active, (unsigned)islands.size(),
		   m_lastPairCount, threads);
	if (s)
	{
		const double perIterUs = s->m_iterations ? (s->m_iterMs * 1000.0 / s->m_iterations) : 0.0;
		printf("  batching %d  phases %d  batches %d  cons %d  batchsz %d..%d"
			   "  setup %.3f  iters %.3f ms/step  %.2f us/iter (%lld iters, %lld groups)",
			   s->m_batching, s->m_phases, s->m_batches, s->m_contactConstraints, s->m_minBatch,
			   s->m_maxBatch, s->m_setupMs / m_stepCount, s->m_iterMs / m_stepCount, perIterUs,
			   s->m_iterations, s->m_groups);
	}
	printf("\n");
	fflush(stdout);
}

bool CpuBulletBackend::Reset(const SceneSpec& scene)
{
	Teardown();
	m_stepCount = 0;
	m_solverMt = 0;
	m_solverSt = 0;

	// Pools scale with the scene: a fixed 80k silently starves the dispatcher past ~40k bodies.
	const int pool = scene.BoxCount() * 2 > 80000 ? scene.BoxCount() * 2 : 80000;
	btDefaultCollisionConstructionInfo cci;
	cci.m_defaultMaxPersistentManifoldPoolSize = pool;
	cci.m_defaultMaxCollisionAlgorithmPoolSize = pool;
	btDefaultCollisionConfiguration* config = new btDefaultCollisionConfiguration(cci);
	m_config.reset(config);

	m_broadphase.reset(new btDbvtBroadphase());

	if (m_multithreaded)
	{
		btITaskScheduler* scheduler = btGetPPLTaskScheduler();
		if (scheduler)
		{
			// Leave a core for the render thread; the demo draws while physics steps.
			const int threads = EnvInt("OSDEMO_THREADS", m_threadCount > 1 ? m_threadCount : 2);
			scheduler->setNumThreads(threads < BT_MAX_THREAD_COUNT ? threads : BT_MAX_THREAD_COUNT - 1);
			btSetTaskScheduler(scheduler);
		}

		btCollisionDispatcherMt* dispatcher = new btCollisionDispatcherMt(config);
		dispatcher->setDispatcherFlags(btCollisionDispatcher::CD_DISABLE_CONTACTPOOL_DYNAMIC_ALLOCATION);
		m_dispatcher.reset(dispatcher);

		btConstraintSolver* solvers[BT_MAX_THREAD_COUNT];
		for (int i = 0; i < BT_MAX_THREAD_COUNT; ++i)
			solvers[i] = new btSequentialImpulseConstraintSolver();
		btConstraintSolverPoolMt* solverPool = new btConstraintSolverPoolMt(solvers, BT_MAX_THREAD_COUNT);

		InstrumentedSolverMt* solverMt = new InstrumentedSolverMt();
		solverMt->s_contactBatchingMethod = btBatchedConstraints::BATCHING_METHOD_SPATIAL_GRID_3D;
		solverMt->s_jointBatchingMethod = btBatchedConstraints::BATCHING_METHOD_SPATIAL_GRID_3D;
		// False deliberately - true reproduced a real hang even for a single world. See
		// CBulletThread::initMultiThreaded for the full reasoning.
		solverMt->s_allowNestedParallelForLoops = false;
		// Bullet's 250 sends any island above ~60 boxes through the batched intra-island solver,
		// which measured 1.7x SLOWER than serial there; it only starts winning past ~2500 manifolds.
		solverMt->s_minimumContactManifoldsForBatching = EnvInt("OSDEMO_MIN_MANIFOLDS_BATCH", 2500);
		solverMt->s_minBatchSize = EnvInt("OSDEMO_MIN_BATCH", solverMt->s_minBatchSize);
		solverMt->s_maxBatchSize = EnvInt("OSDEMO_MAX_BATCH", solverMt->s_maxBatchSize);
		m_solverMt = solverMt;
		m_solver.reset(solverMt);

		m_world.reset(new btDiscreteDynamicsWorldMt(m_dispatcher.get(), m_broadphase.get(),
													solverPool, solverMt, config));
	}
	else
	{
		m_dispatcher.reset(new btCollisionDispatcher(config));
		InstrumentedSolver* solverSt = new InstrumentedSolver();
		m_solverSt = solverSt;
		m_solver.reset(solverSt);
		m_world.reset(new btDiscreteDynamicsWorld(m_dispatcher.get(), m_broadphase.get(),
												  m_solver.get(), config));
	}

	m_world->setGravity(btVector3(0, -10, 0));
	m_world->getSolverInfo().m_solverMode = SOLVER_SIMD | SOLVER_USE_WARMSTARTING;
	// OSDEMO_SOLVER_ITERATIONS overrides the count: a tall stack that topples under the default 10
	// is starved of iterations, not misconfigured, and this is how to tell the two apart.
	int iterations = kSolverIterations;
	if (const char* env = getenv("OSDEMO_SOLVER_ITERATIONS"))
	{
		const int parsed = atoi(env);
		if (parsed > 0)
			iterations = parsed;
	}
	m_world->getSolverInfo().m_numIterations = iterations;
	// Split impulse runs its own numIterations-long parallel loop, so it doubles the barrier count.
	m_world->getSolverInfo().m_splitImpulse = EnvInt("OSDEMO_SPLIT_IMPULSE", 1) != 0;

	// Bullet here is BT_USE_DOUBLE_PRECISION, so the offset costs no accuracy; it is removed again
	// on readback so every backend reports scene-local positions and stays directly comparable.
	m_worldOffset[0] = scene.WorldOffset[0];
	m_worldOffset[1] = scene.WorldOffset[1];
	m_worldOffset[2] = scene.WorldOffset[2];

	const btScalar groundHalf = btScalar(scene.GroundHalfExtent);
	m_groundShape.reset(new btBoxShape(btVector3(groundHalf, kGroundThickness, groundHalf)));
	{
		btTransform t;
		t.setIdentity();
		t.setOrigin(btVector3(btScalar(m_worldOffset[0]),
							  btScalar(m_worldOffset[1] - kGroundThickness),
							  btScalar(m_worldOffset[2])));
		btDefaultMotionState* motion = new btDefaultMotionState(t);
		btRigidBody::btRigidBodyConstructionInfo info(0, motion, m_groundShape.get(), btVector3(0, 0, 0));
		m_world->addRigidBody(new btRigidBody(info));
	}

	m_boxShape.reset(new btBoxShape(btVector3(kBoxHalfExtent, kBoxHalfExtent, kBoxHalfExtent)));
	btVector3 inertia(0, 0, 0);
	m_boxShape->calculateLocalInertia(kBoxMass, inertia);

	m_transforms.clear();
	m_transforms.reserve((size_t)scene.BoxCount());

	std::vector<btRigidBody*> dynamicBodies;
	dynamicBodies.reserve((size_t)scene.BoxCount());

	for (int i = 0; i < scene.BoxCount(); ++i)
	{
		btTransform t;
		t.setIdentity();
		t.setOrigin(btVector3(btScalar(m_worldOffset[0] + scene.BoxPositions[i * 3 + 0]),
							  btScalar(m_worldOffset[1] + scene.BoxPositions[i * 3 + 1]),
							  btScalar(m_worldOffset[2] + scene.BoxPositions[i * 3 + 2])));

		btDefaultMotionState* motion = new btDefaultMotionState(t);
		btRigidBody::btRigidBodyConstructionInfo info(kBoxMass, motion, m_boxShape.get(), inertia);
		btRigidBody* body = new btRigidBody(info);
		// A settled scene sleeps and stops exercising the solver, which makes any ST/MT comparison
		// measure mostly idle steps; OSDEMO_NO_SLEEP holds the workload in its awake steady state.
		if (EnvInt("OSDEMO_NO_SLEEP", 0))
			body->setActivationState(DISABLE_DEACTIVATION);
		// Same knob the GPU pipeline reads, so a sleep comparison is not confounded by the two
		// families waiting different amounts of time before deactivating.
		gDeactivationTime = btScalar(EnvInt("OSDEMO_SLEEP_STEPS", 120)) / btScalar(60.);
		m_world->addRigidBody(body);
		dynamicBodies.push_back(body);

		DemoTransform dt;
		dt.position[0] = (float)((double)t.getOrigin().x() - m_worldOffset[0]);
		dt.position[1] = (float)((double)t.getOrigin().y() - m_worldOffset[1]);
		dt.position[2] = (float)((double)t.getOrigin().z() - m_worldOffset[2]);
		dt.orientation[0] = 0.f;
		dt.orientation[1] = 0.f;
		dt.orientation[2] = 0.f;
		dt.orientation[3] = 1.f;
		m_transforms.push_back(dt);
	}

	// Owned by the world; removeConstraint + delete happens in Teardown before the bodies go.
	for (size_t c = 0; c < scene.Constraints.size(); ++c)
	{
		const DemoConstraint& spec = scene.Constraints[c];
		if (spec.bodyA < 0 || spec.bodyB < 0
			|| spec.bodyA >= (int)dynamicBodies.size() || spec.bodyB >= (int)dynamicBodies.size())
			continue;

		btPoint2PointConstraint* p2p = new btPoint2PointConstraint(
			*dynamicBodies[spec.bodyA], *dynamicBodies[spec.bodyB],
			btVector3(spec.pivotA[0], spec.pivotA[1], spec.pivotA[2]),
			btVector3(spec.pivotB[0], spec.pivotB[1], spec.pivotB[2]));
		m_world->addConstraint(p2p, true);
	}

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
	return true;
}

void CpuBulletBackend::Step(float deltaTime)
{
	if (!m_world)
		return;

	const std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
	// maxSubSteps 1 with a matching fixed step: the demo compares per-step cost, so letting
	// Bullet insert extra substeps would silently change the workload between backends.
	m_world->stepSimulation(deltaTime, 1, deltaTime);
	const std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
	m_lastStepMs = std::chrono::duration<double, std::milli>(end - begin).count();

	m_lastPairCount = m_dispatcher ? m_dispatcher->getNumManifolds() : 0;

	++m_stepCount;
	if (EnvInt("OSDEMO_STATS", 0) && (m_stepCount % 300) == 0)
		ReportStats();

	// Body 0 is the static ground; dynamic bodies follow in creation order.
	const btCollisionObjectArray& objects = m_world->getCollisionObjectArray();
	const int numObjects = objects.size();
	m_awakeCount = 0;
	for (int i = 1; i < numObjects; ++i)
	{
		const size_t slot = (size_t)(i - 1);
		if (slot >= m_transforms.size())
			break;

		if (objects[i]->isActive())
			++m_awakeCount;

		const btTransform& t = objects[i]->getWorldTransform();
		const btQuaternion q = t.getRotation();
		DemoTransform& dt = m_transforms[slot];
		dt.position[0] = (float)((double)t.getOrigin().x() - m_worldOffset[0]);
		dt.position[1] = (float)((double)t.getOrigin().y() - m_worldOffset[1]);
		dt.position[2] = (float)((double)t.getOrigin().z() - m_worldOffset[2]);
		dt.orientation[0] = (float)q.x();
		dt.orientation[1] = (float)q.y();
		dt.orientation[2] = (float)q.z();
		dt.orientation[3] = (float)q.w();
	}
}

bool CpuBulletBackend::CaptureState(std::vector<unsigned char>& out) const
{
	out.clear();
	if (!m_world)
		return false;

	const btCollisionObjectArray& objects = m_world->getCollisionObjectArray();
	const int numObjects = objects.size();
	if (numObjects <= 0)
		return false;

	std::vector<CpuBodyState> state((size_t)numObjects);
	for (int i = 0; i < numObjects; ++i)
	{
		const btCollisionObject* obj = objects[i];
		const btTransform& t = obj->getWorldTransform();
		const btQuaternion q = t.getRotation();
		CpuBodyState& s = state[(size_t)i];
		memset(&s, 0, sizeof(s));
		for (int a = 0; a < 3; ++a)
			s.origin[a] = t.getOrigin()[a];
		s.rotation[0] = q.x();
		s.rotation[1] = q.y();
		s.rotation[2] = q.z();
		s.rotation[3] = q.w();
		if (const btRigidBody* body = btRigidBody::upcast(obj))
		{
			for (int a = 0; a < 3; ++a)
			{
				s.linVel[a] = body->getLinearVelocity()[a];
				s.angVel[a] = body->getAngularVelocity()[a];
			}
		}
		s.deactivationTime = obj->getDeactivationTime();
		s.activationState = obj->getActivationState();
	}

	const size_t bodyBytes = state.size() * sizeof(CpuBodyState);
	const unsigned char* bodySrc = (const unsigned char*)&state[0];

	CpuStateHeader h;
	memset(&h, 0, sizeof(h));
	h.magic = kCpuStateMagic;
	h.version = kCpuStateVersion;
	h.bodyCount = (unsigned int)numObjects;
	h.bodyStride = (unsigned int)sizeof(CpuBodyState);
	h.flags = sizeof(btScalar) == sizeof(double) ? kCpuFlagDoubleScalar : 0u;
	h.awakeCount = m_awakeCount;
	h.bodyHash = HashBytes(bodySrc, bodyBytes);
	h.worldOffset[0] = m_worldOffset[0];
	h.worldOffset[1] = m_worldOffset[1];
	h.worldOffset[2] = m_worldOffset[2];

	out.resize(sizeof(h) + bodyBytes);
	memcpy(&out[0], &h, sizeof(h));
	memcpy(&out[sizeof(h)], bodySrc, bodyBytes);
	return true;
}

bool CpuBulletBackend::ApplyState(const unsigned char* data, size_t bytes)
{
	if (!data || bytes < sizeof(CpuStateHeader) || !m_world)
		return false;

	CpuStateHeader h;
	memcpy(&h, data, sizeof(h));
	if (h.magic != kCpuStateMagic || h.version != kCpuStateVersion)
		return false;

	const unsigned int wantFlags = sizeof(btScalar) == sizeof(double) ? kCpuFlagDoubleScalar : 0u;
	if ((h.flags & kCpuFlagDoubleScalar) != wantFlags)
		return false;
	if (h.bodyStride != (unsigned int)sizeof(CpuBodyState))
		return false;
	if (bytes != sizeof(h) + (size_t)h.bodyCount * sizeof(CpuBodyState))
		return false;
	if (h.worldOffset[0] != m_worldOffset[0] || h.worldOffset[1] != m_worldOffset[1]
		|| h.worldOffset[2] != m_worldOffset[2])
		return false;

	const btCollisionObjectArray& objects = m_world->getCollisionObjectArray();
	if (h.bodyCount != (unsigned int)objects.size())
		return false;
	if (HashBytes(data + sizeof(h), (size_t)h.bodyCount * sizeof(CpuBodyState)) != h.bodyHash)
		return false;

	const CpuBodyState* state = (const CpuBodyState*)(data + sizeof(h));
	for (unsigned int i = 0; i < h.bodyCount; ++i)
	{
		const CpuBodyState& s = state[i];
		btCollisionObject* obj = objects[(int)i];

		btTransform t;
		t.setOrigin(btVector3(s.origin[0], s.origin[1], s.origin[2]));
		t.setRotation(btQuaternion(s.rotation[0], s.rotation[1], s.rotation[2], s.rotation[3]));
		obj->setWorldTransform(t);
		obj->setInterpolationWorldTransform(t);

		if (btRigidBody* body = btRigidBody::upcast(obj))
		{
			body->setLinearVelocity(btVector3(s.linVel[0], s.linVel[1], s.linVel[2]));
			body->setAngularVelocity(btVector3(s.angVel[0], s.angVel[1], s.angVel[2]));
			body->setInterpolationLinearVelocity(body->getLinearVelocity());
			body->setInterpolationAngularVelocity(body->getAngularVelocity());
			body->clearForces();
			if (body->getMotionState())
				body->getMotionState()->setWorldTransform(t);
		}
		// forceActivationState, not setActivationState: the latter refuses to overwrite a
		// DISABLE_DEACTIVATION body, which is exactly what OSDEMO_NO_SLEEP sets.
		obj->forceActivationState(s.activationState);
		obj->setDeactivationTime(s.deactivationTime);
		m_world->updateSingleAabb(obj);
	}

	// Warm-start impulses and cached contact points describe the world that was just overwritten.
	for (int i = 0; i < m_dispatcher->getNumManifolds(); ++i)
		m_dispatcher->getManifoldByIndexInternal(i)->clearManifold();

	m_awakeCount = h.awakeCount;
	for (int i = 1; i < objects.size(); ++i)
	{
		const size_t slot = (size_t)(i - 1);
		if (slot >= m_transforms.size())
			break;
		const btTransform& t = objects[i]->getWorldTransform();
		const btQuaternion q = t.getRotation();
		DemoTransform& dt = m_transforms[slot];
		for (int a = 0; a < 3; ++a)
			dt.position[a] = (float)((double)t.getOrigin()[a] - m_worldOffset[a]);
		dt.orientation[0] = (float)q.x();
		dt.orientation[1] = (float)q.y();
		dt.orientation[2] = (float)q.z();
		dt.orientation[3] = (float)q.w();
	}
	return true;
}
