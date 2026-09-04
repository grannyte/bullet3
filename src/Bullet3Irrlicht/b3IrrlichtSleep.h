#ifndef B3_IRRLICHT_SLEEP_H
#define B3_IRRLICHT_SLEEP_H

#include "b3IrrlichtGpuBuffers.h"
#include "Bullet3Collision/NarrowPhaseCollision/shared/b3Contact4Data.h"

#include <vector>

/**
 * @brief Approximate body deactivation for the Irrlicht-compute pipeline.
 *
 * Islands are a CPU device, so instead of building connected components this walks a bounded
 * number of contact/joint hops per step (setWakePropagationHops): an island deeper than that
 * takes extra frames to fully wake, in exchange for O(hops * contacts) passes.
 *
 * Header-only, matching b3IrrlichtGpuBuffers.h.
 */
class b3IrrlichtSleep
{
public:
	/// Bullet's own defaults: 0.8 m/s, 1.0 rad/s, 2 s at 60 Hz.
	static const unsigned int kDefaultSleepSteps = 120;

	/// Hard bound on wake hops per step: each hop is one more pass over contacts and joints.
	static const unsigned int kMaxWakePropagationHops = 8;
	/// An island up to this deep wakes within the step it is touched; deeper ones take more steps.
	static const unsigned int kDefaultWakePropagationHops = 4;

	b3IrrlichtSleep(irr::video::IVideoDriver* driver)
		: m_driver(driver), m_dispatch(driver), m_doubleSingle(false), m_clearMaterial(-1),
		  m_propagateMaterial(-1), m_jointWakeMaterial(-1), m_updateMaterial(-1),
		  m_updateFullMaterial(-1), m_compactMaterial(-1),
		  m_hopContactMaterial(-1), m_hopJointMaterial(-1), m_hopParamBuffer(0),
		  m_wakeHops(kDefaultWakePropagationHops),
		  m_skipIdle(true), m_jointBuffer(0), m_jointParamBuffer(0), m_numJoints(0),
		  m_compactPairMaterial(-1), m_bodySetMaterial(-1), m_paramBuffer(0),
		  m_contactParamBuffer(0), m_stateBuffer(0), m_wakeBuffer(0), m_awakeCountBuffer(0),
		  m_refBuffer(0), m_liveContactBuffer(0), m_liveCountBuffer(0), m_pairDispatch(driver),
		  m_pairParamBuffer(0), m_livePairBuffer(0), m_livePairCountBuffer(0),
		  m_bodySetParamBuffer(0), m_awakeListBuffer(0), m_sleepingListBuffer(0),
		  m_awakeListCountBuffer(0), m_sleepingListCountBuffer(0), m_numBodies(0),
		  m_moveThreshold(0.05f), m_angThreshold(1.0f), m_sleepSteps(kDefaultSleepSteps)
	{
	}

	~b3IrrlichtSleep()
	{
		b3IrrGpu::dropBuffer(m_paramBuffer);
		b3IrrGpu::dropBuffer(m_contactParamBuffer);
		b3IrrGpu::dropBuffer(m_stateBuffer);
		b3IrrGpu::dropBuffer(m_wakeBuffer);
		b3IrrGpu::dropBuffer(m_awakeCountBuffer);
		b3IrrGpu::dropBuffer(m_refBuffer);
		b3IrrGpu::dropBuffer(m_liveContactBuffer);
		b3IrrGpu::dropBuffer(m_liveCountBuffer);
		b3IrrGpu::dropBuffer(m_pairParamBuffer);
		b3IrrGpu::dropBuffer(m_livePairBuffer);
		b3IrrGpu::dropBuffer(m_livePairCountBuffer);
		b3IrrGpu::dropBuffer(m_jointParamBuffer);
		b3IrrGpu::dropBuffer(m_hopParamBuffer);
		b3IrrGpu::dropBuffer(m_bodySetParamBuffer);
		b3IrrGpu::dropBuffer(m_awakeListBuffer);
		b3IrrGpu::dropBuffer(m_sleepingListBuffer);
		b3IrrGpu::dropBuffer(m_awakeListCountBuffer);
		b3IrrGpu::dropBuffer(m_sleepingListCountBuffer);
	}

	/**
	 * @brief Compiles the sleep kernels.
	 * @param fileSystem Device filesystem; without it a missing shader binds an unrelated material.
	 * @param doubleSingle Compile the emulated-double (df64) kernel set instead of the f32 one.
	 * @return False if a required shader is missing or failed to compile.
	 */
	bool init(irr::io::IFileSystem* fileSystem, bool doubleSingle)
	{
		m_doubleSingle = doubleSingle;

		if (!m_driver)
			return false;

		irr::video::IGPUProgrammingServices* gpu = m_driver->getGPUProgrammingServices();
		if (!gpu)
			return false;

		const irr::io::path path = doubleSingle ? "media/shaders/B3SleepDS.hlsl"
												: "media/shaders/B3Sleep.hlsl";
		if (fileSystem && !fileSystem->existFile(path))
			return false;

		m_compactMaterial = gpu->addComputeShaderFromFile(path, "CSCompactContacts", irr::video::ECST_CS_5_0, 0);
		m_clearMaterial = gpu->addComputeShaderFromFile(path, "CSClearWake", irr::video::ECST_CS_5_0, 0);
		m_propagateMaterial = gpu->addComputeShaderFromFile(path, "CSPropagateWake", irr::video::ECST_CS_5_0, 0);
		m_jointWakeMaterial = gpu->addComputeShaderFromFile(path, "CSPropagateJointWake", irr::video::ECST_CS_5_0, 0);
		m_updateMaterial = gpu->addComputeShaderFromFile(path, "CSUpdateSleep", irr::video::ECST_CS_5_0, 0);
		m_updateFullMaterial = gpu->addComputeShaderFromFile(path, "CSUpdateSleepFull", irr::video::ECST_CS_5_0, 0);
		m_compactPairMaterial = gpu->addComputeShaderFromFile(path, "CSCompactPairs", irr::video::ECST_CS_5_0, 0);
		m_bodySetMaterial = gpu->addComputeShaderFromFile(path, "CSCompactBodySet", irr::video::ECST_CS_5_0, 0);

		// Optional: without it the wake walk stays at one hop per step.
		const irr::io::path hopPath = "media/shaders/B3WakePropagate.hlsl";
		if (!fileSystem || fileSystem->existFile(hopPath))
		{
			m_hopContactMaterial = gpu->addComputeShaderFromFile(hopPath, "CSPropagateWakeHop", irr::video::ECST_CS_5_0, 0);
			m_hopJointMaterial = gpu->addComputeShaderFromFile(hopPath, "CSPropagateJointWakeHop", irr::video::ECST_CS_5_0, 0);
		}

		m_dispatch.init(fileSystem);
		m_pairDispatch.init(fileSystem);
		return isAvailable();
	}

	/**
	 * @brief Sets how many contact/joint hops the wake walk takes per step.
	 * @param hops 1 restores the single hop; clamped to kMaxWakePropagationHops.
	 */
	void setWakePropagationHops(unsigned int hops)
	{
		m_wakeHops = hops < 1u ? 1u : (hops > kMaxWakePropagationHops ? kMaxWakePropagationHops : hops);
	}

	/// Whether the multi-hop wake kernels compiled; false means every step walks one hop only.
	bool hasMultiHopWake() const { return m_hopContactMaterial >= 0; }

	bool isAvailable() const
	{
		return m_clearMaterial >= 0 && m_propagateMaterial >= 0 && m_updateMaterial >= 0 &&
			   m_compactMaterial >= 0 && m_compactPairMaterial >= 0 && m_bodySetMaterial >= 0 && m_dispatch.isAvailable() &&
			   m_pairDispatch.isAvailable();
	}

	/**
	 * @brief Sets the deactivation thresholds.
	 * @param move Distance a body may drift from its reference and still count as still, metres.
	 * @param angular Angular speed below which a body counts as still, rad/s.
	 * @param sleepSteps Consecutive still steps before sleeping; 0 disables sleeping outright.
	 */
	void setThresholds(float move, float angular, unsigned int sleepSteps)
	{
		m_moveThreshold = move;
		m_angThreshold = angular;
		m_sleepSteps = sleepSteps;
	}

	/**
	 * @brief Lets the timer pass skip a sleeping body whose state provably cannot change.
	 * @param enabled False dispatches the unconditional per-body kernel instead.
	 */
	void setSkipIdleBodies(bool enabled) { m_skipIdle = enabled; }

	/**
	 * @brief Points the wake pass at the joint set, so a joint wakes its sleeping side.
	 * @param joints Device-resident b3IrrJoint array, or 0 to disable joint wake.
	 * @param numJoints Joints in that buffer.
	 */
	void setJoints(irr::scene::IComputeBuffer* joints, unsigned int numJoints)
	{
		m_jointBuffer = joints;
		m_numJoints = joints ? numJoints : 0;
	}

	/// Whether the joint wake kernel compiled; false means a joint scene falls back to contact-only.
	bool hasJointWake() const { return m_jointWakeMaterial >= 0; }

	/**
	 * @brief Sizes and zeroes the per-body state for a fresh scene. Zero means awake.
	 * @param numBodies Bodies in the scene, ground included.
	 * @return False if a buffer could not be sized.
	 */
	bool reset(unsigned int numBodies)
	{
		if (numBodies == 0)
			return false;

		m_numBodies = numBodies;
		b3IrrGpu::ensureBuffer<unsigned int>(m_stateBuffer, numBodies);
		b3IrrGpu::ensureBuffer<unsigned int>(m_wakeBuffer, numBodies);
		b3IrrGpu::ensureBuffer<unsigned int>(m_awakeCountBuffer, 1);
		if (m_doubleSingle)
			b3IrrGpu::ensureBuffer<SleepRefDS>(m_refBuffer, numBodies);
		else
			b3IrrGpu::ensureBuffer<SleepRef>(m_refBuffer, numBodies);
		if (!m_stateBuffer || !m_wakeBuffer || !m_awakeCountBuffer || !m_refBuffer)
			return false;

		// Zeroed, not seeded: a reference at the origin is simply "far", so the first step resets
		// every body's timer and writes the real reference. No validity flag needed.
		memset(m_refBuffer->getBufferPointer(), 0,
			   (size_t)numBodies * m_refBuffer->getStructureStride());
		m_refBuffer->setDirty();

		// ensureBuffer leaves contents undefined after a growth, and an undefined asleep bit here
		// would freeze bodies at step 0.
		memset(m_stateBuffer->getBufferPointer(), 0, numBodies * sizeof(unsigned int));
		m_stateBuffer->setDirty();
		memset(m_wakeBuffer->getBufferPointer(), 0, numBodies * sizeof(unsigned int));
		m_wakeBuffer->setDirty();
		*(unsigned int*)m_awakeCountBuffer->getBufferPointer() = numBodies;
		m_awakeCountBuffer->setDirty();
		return true;
	}

	/**
	 * @brief One step of wake propagation and timer advance, entirely device-side.
	 * @param bodies Device-resident bodies; a newly sleeping body has its velocities zeroed.
	 * @param numBodies Bodies in that buffer.
	 * @param contacts Device-resident manifolds from narrowphase.
	 * @param contactCount EHBF_DRAW_INDIRECT_ARGS buffer holding the appended contact count.
	 * @param maxContacts Capacity of the contact buffer; the count is clamped to it.
	 * @return False if the kernels are unavailable or an argument is missing.
	 */
	bool step(irr::scene::IComputeBuffer* bodies, unsigned int numBodies,
			  irr::scene::IComputeBuffer* contacts, irr::scene::IComputeBuffer* contactCount,
			  unsigned int maxContacts)
	{
		if (!isAvailable() || !bodies || !contacts || !contactCount || numBodies == 0)
			return false;
		if (m_numBodies < numBodies && !reset(numBodies))
			return false;

		SleepParams p;
		p.numBodies = numBodies;
		p.moveThresholdSq = m_moveThreshold * m_moveThreshold;
		p.angThresholdSq = m_angThreshold * m_angThreshold;
		p.sleepSteps = m_sleepSteps;
		b3IrrGpu::uploadBuffer<SleepParams>(m_paramBuffer, &p, 1);

		b3IrrGpu::CountParams cp;
		cp.count = 0;
		cp.cap = maxContacts;
		cp.reduceCount = 0;
		cp.numBodies = numBodies;
		b3IrrGpu::uploadBuffer<b3IrrGpu::CountParams>(m_contactParamBuffer, &cp, 1);
		if (!m_dispatch.prepareIndirect(contactCount, m_contactParamBuffer, 64, maxContacts))
			return false;

		const irr::u32 bodyGroups = (numBodies + 63) / 64;
		irr::video::SMaterial mat;

		mat.MaterialType = (irr::video::E_MATERIAL_TYPE)m_clearMaterial;
		m_driver->setMaterial(mat);
		m_driver->bindComputeBuffer(0, m_paramBuffer, irr::video::EHBT_SHADER_RESOURCE);
		m_driver->bindComputeBuffer(2, m_wakeBuffer, irr::video::EHBT_COMPUTE);
		m_driver->bindComputeBuffer(3, m_awakeCountBuffer, irr::video::EHBT_COMPUTE);
		m_driver->dispatchComputeShaderBound(irr::core::vector3d<irr::u32>(bodyGroups, 1, 1));
		m_driver->unbindComputeResources();

		mat.MaterialType = (irr::video::E_MATERIAL_TYPE)m_propagateMaterial;
		m_driver->setMaterial(mat);
		m_driver->bindComputeBuffer(0, m_paramBuffer, irr::video::EHBT_SHADER_RESOURCE);
		m_driver->bindComputeBuffer(1, m_contactParamBuffer, irr::video::EHBT_SHADER_RESOURCE);
		m_driver->bindComputeBuffer(2, contacts, irr::video::EHBT_SHADER_RESOURCE);
		m_driver->bindComputeBuffer(1, m_stateBuffer, irr::video::EHBT_COMPUTE);
		m_driver->bindComputeBuffer(2, m_wakeBuffer, irr::video::EHBT_COMPUTE);
		m_driver->dispatchComputeShaderIndirect(m_dispatch.getArgsBuffer(), 0);
		m_driver->unbindComputeResources();

		// Same one-hop model as the contact pass, on the joint graph. Ordered between propagate and
		// update so a body woken here is timed as awake by the very next kernel.
		if (m_jointBuffer && m_numJoints > 0 && m_jointWakeMaterial >= 0)
		{
			b3IrrGpu::CountParams jp;
			jp.count = m_numJoints;
			jp.cap = m_numJoints;
			jp.reduceCount = 0;
			jp.numBodies = numBodies;
			b3IrrGpu::uploadBuffer<b3IrrGpu::CountParams>(m_jointParamBuffer, &jp, 1);

			mat.MaterialType = (irr::video::E_MATERIAL_TYPE)m_jointWakeMaterial;
			m_driver->setMaterial(mat);
			m_driver->bindComputeBuffer(0, m_paramBuffer, irr::video::EHBT_SHADER_RESOURCE);
			m_driver->bindComputeBuffer(1, m_jointParamBuffer, irr::video::EHBT_SHADER_RESOURCE);
			m_driver->bindComputeBuffer(4, m_jointBuffer, irr::video::EHBT_SHADER_RESOURCE);
			m_driver->bindComputeBuffer(1, m_stateBuffer, irr::video::EHBT_COMPUTE);
			m_driver->bindComputeBuffer(2, m_wakeBuffer, irr::video::EHBT_COMPUTE);
			m_driver->dispatchComputeShaderBound(
				irr::core::vector3d<irr::u32>((m_numJoints + 63) / 64, 1, 1));
			m_driver->unbindComputeResources();
		}

		// Hops 2..N: a body flagged by an earlier hop is itself a wake source, so a touched island
		// up to m_wakeHops deep wakes this step rather than one layer per step.
		// m_jointParamBuffer is only filled by the hop-1 joint pass above.
		const bool jointHops = m_jointBuffer && m_numJoints > 0 && m_hopJointMaterial >= 0
							&& m_jointWakeMaterial >= 0 && m_jointParamBuffer;
		for (unsigned int hop = 2; hop <= m_wakeHops && m_hopContactMaterial >= 0; ++hop)
		{
			b3IrrGpu::CountParams hp;
			hp.count = hop;
			hp.cap = 0;
			hp.reduceCount = 0;
			hp.numBodies = numBodies;
			b3IrrGpu::uploadBuffer<b3IrrGpu::CountParams>(m_hopParamBuffer, &hp, 1);

			mat.MaterialType = (irr::video::E_MATERIAL_TYPE)m_hopContactMaterial;
			m_driver->setMaterial(mat);
			m_driver->bindComputeBuffer(0, m_paramBuffer, irr::video::EHBT_SHADER_RESOURCE);
			m_driver->bindComputeBuffer(1, m_contactParamBuffer, irr::video::EHBT_SHADER_RESOURCE);
			m_driver->bindComputeBuffer(2, contacts, irr::video::EHBT_SHADER_RESOURCE);
			m_driver->bindComputeBuffer(5, m_hopParamBuffer, irr::video::EHBT_SHADER_RESOURCE);
			m_driver->bindComputeBuffer(1, m_stateBuffer, irr::video::EHBT_COMPUTE);
			m_driver->bindComputeBuffer(2, m_wakeBuffer, irr::video::EHBT_COMPUTE);
			m_driver->dispatchComputeShaderIndirect(m_dispatch.getArgsBuffer(), 0);
			m_driver->unbindComputeResources();

			if (jointHops)
			{
				mat.MaterialType = (irr::video::E_MATERIAL_TYPE)m_hopJointMaterial;
				m_driver->setMaterial(mat);
				m_driver->bindComputeBuffer(0, m_paramBuffer, irr::video::EHBT_SHADER_RESOURCE);
				m_driver->bindComputeBuffer(1, m_jointParamBuffer, irr::video::EHBT_SHADER_RESOURCE);
				m_driver->bindComputeBuffer(4, m_jointBuffer, irr::video::EHBT_SHADER_RESOURCE);
				m_driver->bindComputeBuffer(5, m_hopParamBuffer, irr::video::EHBT_SHADER_RESOURCE);
				m_driver->bindComputeBuffer(1, m_stateBuffer, irr::video::EHBT_COMPUTE);
				m_driver->bindComputeBuffer(2, m_wakeBuffer, irr::video::EHBT_COMPUTE);
				m_driver->dispatchComputeShaderBound(
					irr::core::vector3d<irr::u32>((m_numJoints + 63) / 64, 1, 1));
				m_driver->unbindComputeResources();
			}
		}

		const bool useFull = !m_skipIdle && m_updateFullMaterial >= 0;
		mat.MaterialType = (irr::video::E_MATERIAL_TYPE)(useFull ? m_updateFullMaterial
																: m_updateMaterial);
		m_driver->setMaterial(mat);
		m_driver->bindComputeBuffer(0, m_paramBuffer, irr::video::EHBT_SHADER_RESOURCE);
		m_driver->bindComputeBuffer(0, bodies, irr::video::EHBT_COMPUTE);
		m_driver->bindComputeBuffer(1, m_stateBuffer, irr::video::EHBT_COMPUTE);
		m_driver->bindComputeBuffer(2, m_wakeBuffer, irr::video::EHBT_COMPUTE);
		m_driver->bindComputeBuffer(3, m_awakeCountBuffer, irr::video::EHBT_COMPUTE);
		m_driver->bindComputeBuffer(4, m_refBuffer, irr::video::EHBT_COMPUTE);
		m_driver->dispatchComputeShaderBound(irr::core::vector3d<irr::u32>(bodyGroups, 1, 1));
		m_driver->unbindComputeResources();

		b3IrrGpu::ensureBuffer<b3Contact4Data>(m_liveContactBuffer, maxContacts,
											   irr::video::EHBF_COMPUTE_APPEND);
		b3IrrGpu::ensureBuffer<unsigned int>(m_liveCountBuffer, 4,
											 irr::video::EHBF_DRAW_INDIRECT_ARGS);
		if (!m_liveContactBuffer || !m_liveCountBuffer)
			return false;

		mat.MaterialType = (irr::video::E_MATERIAL_TYPE)m_compactMaterial;
		m_driver->setMaterial(mat);
		m_driver->bindComputeBuffer(0, m_paramBuffer, irr::video::EHBT_SHADER_RESOURCE);
		m_driver->bindComputeBuffer(1, m_contactParamBuffer, irr::video::EHBT_SHADER_RESOURCE);
		m_driver->bindComputeBuffer(2, contacts, irr::video::EHBT_SHADER_RESOURCE);
		m_driver->bindComputeBuffer(1, m_stateBuffer, irr::video::EHBT_COMPUTE);
		m_driver->bindComputeBuffer(5, m_liveContactBuffer, irr::video::EHBT_COMPUTE);
		m_driver->resetStructureCount(m_liveContactBuffer, 0);
		m_driver->dispatchComputeShaderIndirect(m_dispatch.getArgsBuffer(), 0);
		m_driver->copyStructureCount(m_liveCountBuffer, 0, m_liveContactBuffer);
		m_driver->unbindComputeResources();

		m_driver->computeBarrier(bodies);
		m_driver->computeBarrier(m_liveContactBuffer);
		return true;
	}

	/**
	 * @brief Drops broadphase pairs whose two bodies were both asleep at the end of the last step.
	 *
	 * Called BEFORE narrowphase, so it reads the previous step's state - correct because neither
	 * body moved since it was written.
	 *
	 * @param pairs Device-resident (bodyA, bodyB) pairs from the broadphase.
	 * @param pairCount EHBF_DRAW_INDIRECT_ARGS buffer holding the appended pair count.
	 * @param maxPairs Capacity of the pair buffer; the count is clamped to it.
	 * @return False if the kernel is unavailable, or no step has run yet to produce state.
	 */
	bool compactPairs(irr::scene::IComputeBuffer* pairs, irr::scene::IComputeBuffer* pairCount,
					  unsigned int maxPairs)
	{
		if (!isAvailable() || m_compactPairMaterial < 0 || !pairs || !pairCount || !m_stateBuffer)
			return false;
		if (m_numBodies == 0)
			return false;

		SleepParams p;
		p.numBodies = m_numBodies;
		p.moveThresholdSq = m_moveThreshold * m_moveThreshold;
		p.angThresholdSq = m_angThreshold * m_angThreshold;
		p.sleepSteps = m_sleepSteps;
		b3IrrGpu::uploadBuffer<SleepParams>(m_paramBuffer, &p, 1);

		b3IrrGpu::CountParams cp;
		cp.count = 0;
		cp.cap = maxPairs;
		cp.reduceCount = 0;
		cp.numBodies = m_numBodies;
		b3IrrGpu::uploadBuffer<b3IrrGpu::CountParams>(m_pairParamBuffer, &cp, 1);
		if (!m_pairDispatch.prepareIndirect(pairCount, m_pairParamBuffer, 64, maxPairs))
			return false;

		b3IrrGpu::ensureBuffer<PairIndices>(m_livePairBuffer, maxPairs,
											irr::video::EHBF_COMPUTE_APPEND);
		b3IrrGpu::ensureBuffer<unsigned int>(m_livePairCountBuffer, 4,
											 irr::video::EHBF_DRAW_INDIRECT_ARGS);
		if (!m_livePairBuffer || !m_livePairCountBuffer)
			return false;

		irr::video::SMaterial mat;
		mat.MaterialType = (irr::video::E_MATERIAL_TYPE)m_compactPairMaterial;
		m_driver->setMaterial(mat);
		m_driver->bindComputeBuffer(0, m_paramBuffer, irr::video::EHBT_SHADER_RESOURCE);
		m_driver->bindComputeBuffer(1, m_pairParamBuffer, irr::video::EHBT_SHADER_RESOURCE);
		m_driver->bindComputeBuffer(3, pairs, irr::video::EHBT_SHADER_RESOURCE);
		m_driver->bindComputeBuffer(1, m_stateBuffer, irr::video::EHBT_COMPUTE);
		m_driver->bindComputeBuffer(6, m_livePairBuffer, irr::video::EHBT_COMPUTE);
		m_driver->resetStructureCount(m_livePairBuffer, 0);
		m_driver->dispatchComputeShaderIndirect(m_pairDispatch.getArgsBuffer(), 0);
		m_driver->copyStructureCount(m_livePairCountBuffer, 0, m_livePairBuffer);
		m_driver->unbindComputeResources();

		m_driver->computeBarrier(m_livePairBuffer);
		return true;
	}

	/**
	 * @brief Splits the bodies into an awake index list and a sleeping one, device-side.
	 *
	 * Feeds the two-tree broadphase: each list drives its own tree, so sleeping-vs-sleeping pairs
	 * are skipped by construction rather than by an unsound query-leaf early-out.
	 *
	 * @return False if the kernel is unavailable or no sleep state exists yet.
	 */
	bool compactBodySets()
	{
		if (!isAvailable() || m_bodySetMaterial < 0 || !m_stateBuffer || m_numBodies == 0)
			return false;

		b3IrrGpu::ensureBuffer<unsigned int>(m_awakeListBuffer, m_numBodies,
											 irr::video::EHBF_COMPUTE_APPEND);
		b3IrrGpu::ensureBuffer<unsigned int>(m_sleepingListBuffer, m_numBodies,
											 irr::video::EHBF_COMPUTE_APPEND);
		b3IrrGpu::ensureBuffer<unsigned int>(m_awakeListCountBuffer, 4,
											 irr::video::EHBF_DRAW_INDIRECT_ARGS);
		b3IrrGpu::ensureBuffer<unsigned int>(m_sleepingListCountBuffer, 4,
											 irr::video::EHBF_DRAW_INDIRECT_ARGS);
		if (!m_awakeListBuffer || !m_sleepingListBuffer || !m_awakeListCountBuffer ||
			!m_sleepingListCountBuffer)
			return false;

		for (int pass = 0; pass < 2; ++pass)
		{
			const bool wantAsleep = pass != 0;
			b3IrrGpu::CountParams cp;
			cp.count = m_numBodies;
			cp.cap = wantAsleep ? 1u : 0u;
			cp.reduceCount = 0;
			cp.numBodies = m_numBodies;
			b3IrrGpu::uploadBuffer<b3IrrGpu::CountParams>(m_bodySetParamBuffer, &cp, 1);

			irr::scene::IComputeBuffer* dst = wantAsleep ? m_sleepingListBuffer : m_awakeListBuffer;
			irr::scene::IComputeBuffer* cnt =
				wantAsleep ? m_sleepingListCountBuffer : m_awakeListCountBuffer;

			irr::video::SMaterial mat;
			mat.MaterialType = (irr::video::E_MATERIAL_TYPE)m_bodySetMaterial;
			m_driver->setMaterial(mat);
			m_driver->bindComputeBuffer(0, m_paramBuffer, irr::video::EHBT_SHADER_RESOURCE);
			m_driver->bindComputeBuffer(1, m_bodySetParamBuffer, irr::video::EHBT_SHADER_RESOURCE);
			m_driver->bindComputeBuffer(1, m_stateBuffer, irr::video::EHBT_COMPUTE);
			m_driver->bindComputeBuffer(7, dst, irr::video::EHBT_COMPUTE);
			m_driver->resetStructureCount(dst, 0);
			m_driver->dispatchComputeShaderBound(
				irr::core::vector3d<irr::u32>((m_numBodies + 63) / 64, 1, 1));
			m_driver->copyStructureCount(cnt, 0, dst);
			m_driver->unbindComputeResources();
			m_driver->computeBarrier(dst);
		}
		return true;
	}

	/// Body indices that are awake / asleep, from the last compactBodySets call.
	irr::scene::IComputeBuffer* getAwakeListBuffer() const { return m_awakeListBuffer; }
	irr::scene::IComputeBuffer* getSleepingListBuffer() const { return m_sleepingListBuffer; }
	/// Their append counts, in the indirect-args form a dispatch expects.
	irr::scene::IComputeBuffer* getAwakeListCountBuffer() const { return m_awakeListCountBuffer; }
	irr::scene::IComputeBuffer* getSleepingListCountBuffer() const { return m_sleepingListCountBuffer; }

	/// Pairs with at least one awake body; feed these to narrowphase in place of the raw list.
	irr::scene::IComputeBuffer* getLivePairBuffer() const { return m_livePairBuffer; }
	/// Append count for getLivePairBuffer, in the indirect-args form narrowphase expects.
	irr::scene::IComputeBuffer* getLivePairCountBuffer() const { return m_livePairCountBuffer; }

	/// Contacts with at least one awake body, from the last step; feed these to the solver.
	irr::scene::IComputeBuffer* getLiveContactBuffer() const { return m_liveContactBuffer; }
	/// Append count for getLiveContactBuffer, in the indirect-args form the solver expects.
	irr::scene::IComputeBuffer* getLiveContactCountBuffer() const { return m_liveCountBuffer; }

	/// Per-body sleep state the solver and integrator gate on; 0 when nothing has run yet.
	irr::scene::IComputeBuffer* getSleepStateBuffer() const { return m_stateBuffer; }

	/// Per-body stillness reference the timer measures against. Exposed so a state snapshot can
	/// carry it: restoring the timer without its reference re-times every body from a stale origin.
	irr::scene::IComputeBuffer* getSleepRefBuffer() const { return m_refBuffer; }

	/**
	 * @brief Reads the awake tally the last step published. Stalls, so call it on an interval.
	 * @param out Receives the number of DYNAMIC bodies still awake.
	 * @return False if no step has run yet.
	 */
	bool readAwakeCount(unsigned int& out)
	{
		if (!m_awakeCountBuffer)
			return false;

		m_awakeCountBuffer->downloadFromGPU();
		memcpy(&out, m_awakeCountBuffer->getBufferPointer(), sizeof(unsigned int));
		return true;
	}

	/**
	 * @brief Reads the whole per-body state, for tests and debugging. Stalls.
	 * @param out Receives one uint per body; bit 31 set means asleep.
	 * @param numBodies Bodies to read.
	 * @return False if the buffer is missing or too small.
	 */
	bool readSleepState(std::vector<unsigned int>& out, unsigned int numBodies)
	{
		if (!m_stateBuffer || numBodies == 0 || m_stateBuffer->getStructureCount() < numBodies)
			return false;

		m_stateBuffer->downloadFromGPU();
		out.resize(numBodies);
		memcpy(&out[0], m_stateBuffer->getBufferPointer(), numBodies * sizeof(unsigned int));
		return true;
	}

private:
	b3IrrlichtSleep(const b3IrrlichtSleep&);
	b3IrrlichtSleep& operator=(const b3IrrlichtSleep&);

	/// Mirrors SleepParams in B3SleepBody.hlsli.
	struct SleepParams
	{
		unsigned int numBodies;
		float moveThresholdSq;
		float angThresholdSq;
		unsigned int sleepSteps;
	};

	/// Mirrors b3SleepRef; the df64 form carries the position's low half alongside.
	struct SleepRef { float pos[4]; };
	struct SleepRefDS { float pos[4]; float posLo[4]; };

	/// Mirrors the broadphase's uint2 pair, so the append counter steps one pair at a time.
	struct PairIndices { unsigned int a; unsigned int b; };
	static_assert(sizeof(PairIndices) == 8, "PairIndices must match the HLSL uint2 stride");

	static_assert(sizeof(SleepParams) == 16, "SleepParams must match the HLSL struct stride");
	static_assert(sizeof(SleepRef) == 16 && sizeof(SleepRefDS) == 32,
				  "b3SleepRef must match the HLSL struct stride in both precisions");

	irr::video::IVideoDriver* m_driver;
	b3IrrGpu::DispatchHelper m_dispatch;
	bool m_doubleSingle;

	int m_clearMaterial;
	int m_propagateMaterial;
	int m_jointWakeMaterial;
	int m_updateMaterial;
	int m_updateFullMaterial;
	int m_hopContactMaterial;
	int m_hopJointMaterial;
	irr::scene::IComputeBuffer* m_hopParamBuffer;
	unsigned int m_wakeHops;
	bool m_skipIdle;
	int m_compactMaterial;
	int m_compactPairMaterial;
	int m_bodySetMaterial;

	/// Not owned: the joint solver owns the upload; this only reads it.
	irr::scene::IComputeBuffer* m_jointBuffer;
	irr::scene::IComputeBuffer* m_jointParamBuffer;
	unsigned int m_numJoints;

	irr::scene::IComputeBuffer* m_paramBuffer;
	irr::scene::IComputeBuffer* m_contactParamBuffer;
	irr::scene::IComputeBuffer* m_stateBuffer;
	irr::scene::IComputeBuffer* m_wakeBuffer;
	irr::scene::IComputeBuffer* m_awakeCountBuffer;
	irr::scene::IComputeBuffer* m_refBuffer;
	irr::scene::IComputeBuffer* m_liveContactBuffer;
	irr::scene::IComputeBuffer* m_liveCountBuffer;
	/// Its own dispatch helper: pair compaction runs a phase earlier than the contact one, so the
	/// two would otherwise overwrite each other's indirect args within a single step.
	b3IrrGpu::DispatchHelper m_pairDispatch;
	irr::scene::IComputeBuffer* m_pairParamBuffer;
	irr::scene::IComputeBuffer* m_livePairBuffer;
	irr::scene::IComputeBuffer* m_livePairCountBuffer;
	irr::scene::IComputeBuffer* m_bodySetParamBuffer;
	irr::scene::IComputeBuffer* m_awakeListBuffer;
	irr::scene::IComputeBuffer* m_sleepingListBuffer;
	irr::scene::IComputeBuffer* m_awakeListCountBuffer;
	irr::scene::IComputeBuffer* m_sleepingListCountBuffer;

	unsigned int m_numBodies;
	float m_moveThreshold;
	float m_angThreshold;
	unsigned int m_sleepSteps;
};

#endif  //B3_IRRLICHT_SLEEP_H
