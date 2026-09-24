/*******************************************************
* Copyright (C) 2026 David Delisle granyte@hotmail.com
*
* This file is part of Outerspace.
*
* Outerspace can not be copied and/or distributed without the express
* permission of David Delisle
*******************************************************/
#ifndef OUTERSPACE_PHYSICS_DEMO_CPU_BACKEND_H
#define OUTERSPACE_PHYSICS_DEMO_CPU_BACKEND_H

#include "PhysicsBackend.h"

#include <memory>

class btCollisionConfiguration;
class btCollisionDispatcher;
class btBroadphaseInterface;
class btConstraintSolver;
class btDiscreteDynamicsWorld;
class btCollisionShape;

/**
 * @brief Bullet on the CPU, single- or multi-threaded, selected at construction.
 *
 * Both modes share one class so the scene construction is provably identical between them - the
 * only divergence is which world/dispatcher/solver types get built.
 */
class CpuBulletBackend : public PhysicsBackend
{
public:
	/**
	 * @brief Constructs a CPU backend.
	 * @param multithreaded True for btDiscreteDynamicsWorldMt with a task scheduler.
	 * @param threadCount Worker threads for the MT case; ignored when single-threaded.
	 */
	CpuBulletBackend(bool multithreaded, int threadCount);
	virtual ~CpuBulletBackend();

	virtual const char* Name() const;
	virtual bool Reset(const SceneSpec& scene);
	virtual bool SupportsConstraints() const { return true; }
	virtual void Step(float deltaTime);
	virtual const std::vector<DemoTransform>& Transforms() const { return m_transforms; }
	virtual double LastStepMs() const { return m_lastStepMs; }
	virtual int LastPairCount() const { return m_lastPairCount; }
	virtual int BodyCount() const { return (int)m_transforms.size(); }
	virtual int AwakeBodyCount() const { return m_awakeCount; }

	/**
	 * @brief Snapshots every body's transform, velocities and activation state.
	 *
	 * The solver's persistent manifolds are NOT included - see ApplyState for what that costs.
	 *
	 * @param out Receives the payload; cleared first.
	 * @return False if no world has been built yet.
	 */
	virtual bool CaptureState(std::vector<unsigned char>& out) const;

	/**
	 * @brief Restores a CaptureState payload, then drops every cached manifold.
	 *
	 * Warm-start impulses describe the pre-restore contacts, so they are cleared rather than
	 * carried over; the first step after a restore therefore solves cold.
	 *
	 * @param data First byte.
	 * @param bytes Length.
	 * @return False if the payload does not match this world's body count, scalar width or size.
	 */
	virtual bool ApplyState(const unsigned char* data, size_t bytes);

private:
	CpuBulletBackend(const CpuBulletBackend&);
	CpuBulletBackend& operator=(const CpuBulletBackend&);

	void Teardown();
	void ReportStats();

	bool m_multithreaded;
	int m_threadCount;
	std::string m_name;

	std::unique_ptr<btCollisionConfiguration> m_config;
	std::unique_ptr<btCollisionDispatcher> m_dispatcher;
	std::unique_ptr<btBroadphaseInterface> m_broadphase;
	std::unique_ptr<btConstraintSolver> m_solver;
	std::unique_ptr<btDiscreteDynamicsWorld> m_world;
	std::unique_ptr<btCollisionShape> m_boxShape;
	std::unique_ptr<btCollisionShape> m_groundShape;

	std::vector<DemoTransform> m_transforms;
	/// SceneSpec::WorldOffset, added on creation and removed on readback.
	double m_worldOffset[3];
	double m_lastStepMs;
	int m_lastPairCount;
	/// Dynamic bodies Bullet still reports as active, for direct comparison with the GPU tally.
	int m_awakeCount;
	int m_stepCount;
	/// Guards the OSDEMO_STATE_ROUNDTRIP self-test against re-entering Reset.
	bool m_roundTripDone;
	/// Non-owning aliases of m_solver, typed so the instrumentation counters are reachable.
	class InstrumentedSolverMt* m_solverMt;
	class InstrumentedSolver* m_solverSt;
};

#endif  //OUTERSPACE_PHYSICS_DEMO_CPU_BACKEND_H
