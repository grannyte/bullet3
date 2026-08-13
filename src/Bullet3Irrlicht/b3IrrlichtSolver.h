#ifndef B3_IRRLICHT_SOLVER_H
#define B3_IRRLICHT_SOLVER_H

#include "Bullet3Collision/NarrowPhaseCollision/shared/b3Contact4Data.h"
#include "Bullet3Collision/NarrowPhaseCollision/shared/b3RigidBodyData.h"

#include <vector>

namespace irr
{
namespace video
{
class IVideoDriver;
}
namespace scene
{
class IComputeBuffer;
}
namespace io
{
class IFileSystem;
}
}  // namespace irr

namespace b3IrrGpu
{
class DispatchHelper;
}

/**
 * @brief Irrlicht-compute analog of b3Solver: sequential-impulse contact resolution on the GPU.
 *
 * Velocity deltas accumulate as fixed-point integers through InterlockedAdd, so the result is
 * bit-identical regardless of thread order - no graph colouring needed to stay deterministic.
 *
 * CONVENTION: m_worldPosB[i].xyz must be RELATIVE TO BODY A's origin; it is consumed directly as rA.
 */
class b3IrrlichtSolver
{
public:
	b3IrrlichtSolver(irr::video::IVideoDriver* driver);
	~b3IrrlichtSolver();

	/**
	 * @brief Compiles the solver kernels.
	 * @param fileSystem Device filesystem; without it a missing shader binds an unrelated material.
	 * @return False if no compute pipeline is available or a shader is missing.
	 */
	/**
	 * @brief Compiles the kernels.
	 * @param fileSystem Device filesystem; without it a missing shader binds an unrelated material.
	 * @param doubleSingle Compile the emulated-double (df64) kernel set instead of the f32 one.
	 * @return False if a required shader is missing or failed to compile.
	 */
	bool init(irr::io::IFileSystem* fileSystem = 0, bool doubleSingle = false);

	/// Whether this instance runs the emulated-double kernels; the std::vector entry point refuses.
	bool isDoubleSingle() const { return m_doubleSingle; }

	/**
	 * @brief Resolves contacts in place, iterating on the GPU without intermediate readback.
	 *
	 * Coulomb friction uses Bullet's combination rule on b3RigidBodyData::m_frictionCoeff
	 * (mu = muA * muB), relaxed per body by its contact-point count so the Jacobi iteration cannot
	 * chatter; friction is consequently weak for a body in a crowded stack. See the shader.
	 *
	 * @param bodies Bodies to solve; velocities are updated in place on return.
	 * @param invInertiaDiag Diagonal inverse inertia per body, 4 floats each (w unused).
	 * @param contacts Manifolds from narrowphase.
	 * @param iterations Sequential-impulse passes; more converges harder, never fewer than 1.
	 * @param deltaTime Timestep the impulses are scaled against.
	 * @param erp Baumgarte position-correction factor in [0,1].
	 * @return False if the kernels are unavailable.
	 */
	bool solveContacts(std::vector<b3RigidBodyData>& bodies,
					   const std::vector<float>& invInertiaDiag,
					   const std::vector<b3Contact4Data>& contacts,
					   int iterations = 4, float deltaTime = 1.f / 60.f, float erp = 0.2f);

	/**
	 * @brief Whether the device-buffer entry points below can run.
	 * @return True once B3GpuResident.hlsl compiled.
	 */
	bool isResidentPathAvailable() const;

	/**
	 * @brief Uploads the per-body inverse inertia the resident solve reads. Call on change only.
	 * @param invInertiaDiag Diagonal inverse inertia per body, 4 floats each (w unused).
	 * @param numBodies Bodies the buffer must cover.
	 * @return False if the buffer could not be sized.
	 */
	bool uploadInvInertia(const std::vector<float>& invInertiaDiag, unsigned int numBodies);

	/**
	 * @brief solveContacts against buffers that never leave the device.
	 *
	 * The contact count is read from the GPU counter, so the solve dispatch is indirect and
	 * nothing about the contact list has to come back first. Requires uploadInvInertia.
	 *
	 * @param bodies Device-resident bodies; velocities are updated in place.
	 * @param numBodies Bodies in that buffer.
	 * @param contacts Device-resident manifolds from narrowphase.
	 * @param contactCount EHBF_DRAW_INDIRECT_ARGS buffer holding the appended contact count.
	 * @param maxContacts Capacity of the contact buffer; the count is clamped to it.
	 * @param iterations Sequential-impulse passes; never fewer than 1.
	 * @param deltaTime Timestep the impulses are scaled against.
	 * @param erp Baumgarte position-correction factor in [0,1].
	 * @return False if the kernels are unavailable or an argument is missing.
	 */
	bool solveContactsResident(irr::scene::IComputeBuffer* bodies, unsigned int numBodies,
							   irr::scene::IComputeBuffer* contacts,
							   irr::scene::IComputeBuffer* contactCount,
							   unsigned int maxContacts, int iterations = 4,
							   float deltaTime = 1.f / 60.f, float erp = 0.2f);

	/**
	 * @brief Reads back the accumulated-impulse ledger left by the last solveContacts call.
	 * @param out Receives 12 floats per contact: normal, tangent0, tangent1 for each of 4 points.
	 *            The normal entry is the running total applied; the tangents are Coulomb-clamped.
	 * @return False if no solve has run yet.
	 */
	bool downloadAccumulatedImpulses(std::vector<float>& out);

private:
	b3IrrlichtSolver(const b3IrrlichtSolver&);
	b3IrrlichtSolver& operator=(const b3IrrlichtSolver&);

	void releaseBuffers();

	irr::video::IVideoDriver* m_driver;
	bool m_doubleSingle;

	int m_solveMaterial;
	int m_applyMaterial;
	int m_clearImpulseMaterial;
	int m_clearLoadMaterial;
	int m_accumLoadMaterial;

	b3IrrGpu::DispatchHelper* m_dispatch;

	irr::scene::IComputeBuffer* m_paramBuffer;
	irr::scene::IComputeBuffer* m_bodyBuffer;
	irr::scene::IComputeBuffer* m_contactBuffer;
	irr::scene::IComputeBuffer* m_inertiaBuffer;
	irr::scene::IComputeBuffer* m_deltaBuffer;
	irr::scene::IComputeBuffer* m_impulseBuffer;
	irr::scene::IComputeBuffer* m_loadBuffer;
	unsigned int m_impulseFloats;
};

#endif  //B3_IRRLICHT_SOLVER_H
