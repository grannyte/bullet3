#ifndef B3_IRRLICHT_JOINT_SOLVER_H
#define B3_IRRLICHT_JOINT_SOLVER_H

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

enum b3IrrJointType
{
	B3_IRR_POINT2POINT_CONSTRAINT_TYPE = 3,
	B3_IRR_FIXED_CONSTRAINT_TYPE = 4
};

enum b3IrrJointFlags
{
	B3_IRR_CONSTRAINT_FLAG_ENABLED = 1
};

/**
 * @brief One joint, laid out to match B3SolveJoints.hlsl's b3IrrJoint element for element.
 *
 * Field order mirrors b3GpuGenericConstraint so the original's semantics carry over unchanged.
 */
struct b3IrrJoint
{
	int constraintType;
	int rbA;
	int rbB;
	float breakingImpulseThreshold;   // recorded only; breaking is not ported

	float pivotInA[4];                // xyz in A's local frame, w unused
	float pivotInB[4];
	float relTargetAB[4];             // target relative rotation as xyzw, fixed joints only

	int flags;
	int uid;
	int padding[2];
};

/**
 * @brief Irrlicht-compute analog of b3GpuPgsConstraintSolver: point-to-point and fixed joints.
 *
 * Velocity deltas accumulate as fixed-point integers through InterlockedAdd, so the solve is a
 * deterministic Jacobi sweep and the original's batching kernels are not needed.
 */
class b3IrrlichtJointSolver
{
public:
	b3IrrlichtJointSolver(irr::video::IVideoDriver* driver);
	~b3IrrlichtJointSolver();

	/**
	 * @brief Compiles the joint kernels.
	 * @param fileSystem Device filesystem; without it a missing shader binds an unrelated material.
	 * @return False if no compute pipeline is available or the shader is missing.
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
	 * @brief Resolves joints in place, iterating on the GPU without intermediate readback.
	 * @param bodies Bodies to solve; velocities are updated in place on return.
	 * @param invInertiaDiag Diagonal inverse inertia per body, 4 floats each (w unused).
	 * @param joints Joints to satisfy; indices address bodies directly.
	 * @param iterations Jacobi passes; more converges harder, never fewer than 1.
	 * @param deltaTime Timestep the impulses are scaled against.
	 * @param erp Baumgarte position-correction factor in [0,1].
	 * @return False if the kernels are unavailable.
	 */
	bool solveJoints(std::vector<b3RigidBodyData>& bodies,
					 const std::vector<float>& invInertiaDiag,
					 const std::vector<b3IrrJoint>& joints,
					 int iterations = 20, float deltaTime = 1.f / 60.f, float erp = 0.2f);

	/**
	 * @brief Uploads the joint set and the per-joint Jacobi split it implies. Call on change only.
	 * @param joints Joints to satisfy; indices address the body buffer directly.
	 * @param bodies Bodies the joints reference; only m_invMass is read, to skip static endpoints.
	 * @return False if there is nothing to upload.
	 */
	bool uploadJoints(const std::vector<b3IrrJoint>& joints,
					  const std::vector<b3RigidBodyData>& bodies);

	/**
	 * @brief Uploads the per-body inverse inertia the resident solve reads.
	 * @param invInertiaDiag Diagonal inverse inertia per body, 4 floats each (w unused).
	 * @param numBodies Bodies the buffer must cover.
	 * @return False if the buffer could not be sized.
	 */
	bool uploadInvInertia(const std::vector<float>& invInertiaDiag, unsigned int numBodies);

	/**
	 * @brief solveJoints against a body buffer that never leaves the device.
	 * @param bodies Device-resident bodies; velocities are updated in place.
	 * @param numBodies Bodies in that buffer.
	 * @param iterations Jacobi passes; never fewer than 1.
	 * @param deltaTime Timestep the impulses are scaled against.
	 * @param erp Baumgarte position-correction factor in [0,1].
	 * @return False if the kernels are unavailable or uploadJoints/uploadInvInertia never ran.
	 */
	bool solveJointsResident(irr::scene::IComputeBuffer* bodies, unsigned int numBodies,
							 int iterations = 20, float deltaTime = 1.f / 60.f, float erp = 0.2f);

	/**
	 * @brief Builds a point-to-point joint between two bodies.
	 * @param bodyA Index of the first body.
	 * @param bodyB Index of the second body.
	 * @param pivotA Pivot in A's local frame, 3 floats.
	 * @param pivotB Pivot in B's local frame, 3 floats.
	 * @return An enabled joint ready to hand to solveJoints.
	 */
	static b3IrrJoint makePoint2Point(int bodyA, int bodyB, const float* pivotA, const float* pivotB);

	/**
	 * @brief Reads back the field values the GPU actually sees, proving the two layouts agree.
	 * @param joints Joints to upload and probe.
	 * @param out Receives 8 ints per joint: type, rbA, rbB, then the raw bits of
	 *            breakingImpulseThreshold, pivotInA.y and pivotInB.y, then flags and uid.
	 * @return False if the probe kernel is unavailable.
	 */
	bool debugReadJointLayout(const std::vector<b3IrrJoint>& joints, std::vector<int>& out);

private:
	b3IrrlichtJointSolver(const b3IrrlichtJointSolver&);
	b3IrrlichtJointSolver& operator=(const b3IrrlichtJointSolver&);

	void releaseBuffers();

	irr::video::IVideoDriver* m_driver;
	bool m_doubleSingle;

	int m_solveMaterial;
	int m_applyMaterial;
	int m_debugMaterial;

	irr::scene::IComputeBuffer* m_paramBuffer;
	irr::scene::IComputeBuffer* m_bodyBuffer;
	irr::scene::IComputeBuffer* m_jointBuffer;
	irr::scene::IComputeBuffer* m_inertiaBuffer;
	irr::scene::IComputeBuffer* m_scaleBuffer;
	irr::scene::IComputeBuffer* m_deltaBuffer;

	/// Joints currently uploaded, so the resident solve knows its dispatch size without a readback.
	unsigned int m_residentJoints;
};

#endif  //B3_IRRLICHT_JOINT_SOLVER_H
