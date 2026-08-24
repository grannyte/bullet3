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
	B3_IRR_FIXED_CONSTRAINT_TYPE = 4,
	B3_IRR_HINGE_CONSTRAINT_TYPE = 5,
	B3_IRR_SLIDER_CONSTRAINT_TYPE = 6,
	B3_IRR_CONETWIST_CONSTRAINT_TYPE = 7,
	B3_IRR_D6_CONSTRAINT_TYPE = 8
};

enum b3IrrJointFlags
{
	B3_IRR_CONSTRAINT_FLAG_ENABLED = 1,
	B3_IRR_CONSTRAINT_FLAG_BROKEN = 2,     // set by the GPU when breakingImpulseThreshold was passed
	B3_IRR_CONSTRAINT_FLAG_OWN_ERP = 4     // use the joint's erp/cfm instead of the global param
};

enum b3IrrJointAxis
{
	B3_IRR_JOINT_AXIS_X = 0,
	B3_IRR_JOINT_AXIS_Y = 1,
	B3_IRR_JOINT_AXIS_Z = 2
};

/**
 * @brief One joint, laid out to match B3SolveJointsBody.hlsl's b3IrrJoint element for element.
 *
 * The first 80 bytes keep b3GpuGenericConstraint's field order. Limits: lower > upper is free,
 * lower == upper is locked, otherwise limited. A motor runs when its max impulse is > 0.
 */
struct b3IrrJoint
{
	int constraintType;
	int rbA;
	int rbB;
	float breakingImpulseThreshold;   // 0 = unbreakable; compared to the step's applied impulse

	float pivotInA[4];                // xyz in A's local frame, w unused
	float pivotInB[4];
	float relTargetAB[4];             // target relative rotation as xyzw, fixed joints only

	int flags;
	int uid;
	float erp;                        // read only with B3_IRR_CONSTRAINT_FLAG_OWN_ERP
	float cfm;

	float frameInA[4];                // joint frame orientation in A's local frame, xyzw
	float frameInB[4];
	float linLower[4];                // per joint-frame axis; w unused
	float linUpper[4];
	float angLower[4];                // cone-twist: x = +-twist span; angUpper[1] = swing span
	float angUpper[4];
	float linMotorVel[4];
	float angMotorVel[4];
	float linMotorMaxImpulse[4];      // per step; 0 disables the motor
	float angMotorMaxImpulse[4];
};

/**
 * @brief Irrlicht-compute analog of b3GpuPgsConstraintSolver: point-to-point, fixed, hinge,
 *        slider, cone-twist and generic 6-DoF joints with limits, motors and breaking.
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
	 * @param joints Joints to satisfy; indices address bodies directly. Re-uploaded every call, so
	 *               a joint the GPU broke is live again next call unless the caller clears it.
	 * @param iterations Jacobi passes; more converges harder, never fewer than 1.
	 * @param deltaTime Timestep the impulses are scaled against.
	 * @param erp Baumgarte position-correction factor in [0,1] for joints without OWN_ERP.
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
	 * @param erp Baumgarte position-correction factor in [0,1] for joints without OWN_ERP.
	 * @return False if the kernels are unavailable or uploadJoints/uploadInvInertia never ran.
	 */
	bool solveJointsResident(irr::scene::IComputeBuffer* bodies, unsigned int numBodies,
							 int iterations = 20, float deltaTime = 1.f / 60.f, float erp = 0.2f);

	/**
	 * @brief Reads back which joints the GPU has broken since the last uploadJoints.
	 * @param broken Receives one entry per joint of the last solve, nonzero = broken.
	 * @return False before any solve has run.
	 */
	bool readJointStatus(std::vector<unsigned int>& broken);

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
	 * @brief Builds a fixed (weld) joint holding B's orientation at relTargetAB relative to A.
	 * @param bodyA Index of the first body.
	 * @param bodyB Index of the second body.
	 * @param pivotA Pivot in A's local frame, 3 floats.
	 * @param pivotB Pivot in B's local frame, 3 floats.
	 * @param relTargetAB Target B-relative-to-A rotation xyzw, or 0 for identity.
	 * @return An enabled joint.
	 */
	static b3IrrJoint makeFixed(int bodyA, int bodyB, const float* pivotA, const float* pivotB,
								const float* relTargetAB = 0);

	/**
	 * @brief Builds a hinge: anchors coincident, rotation only about the given axis.
	 * @param bodyA Index of the first body.
	 * @param bodyB Index of the second body.
	 * @param pivotA Pivot in A's local frame, 3 floats.
	 * @param pivotB Pivot in B's local frame, 3 floats.
	 * @param axisA Hinge axis in A's local frame, 3 floats.
	 * @param axisB Hinge axis in B's local frame, 3 floats.
	 * @param lower Lower angle limit (rad); pass lower > upper for an unlimited hinge.
	 * @param upper Upper angle limit (rad).
	 * @return An enabled joint whose zero angle is where both frames' perpendiculars coincide;
	 *         call alignFrameBToCurrentPose for bodies that do not start aligned.
	 */
	static b3IrrJoint makeHinge(int bodyA, int bodyB, const float* pivotA, const float* pivotB,
								const float* axisA, const float* axisB,
								float lower = 1.f, float upper = -1.f);

	/**
	 * @brief Builds a slider: translation only along the given axis, no rotation.
	 * @param bodyA Index of the first body.
	 * @param bodyB Index of the second body.
	 * @param pivotA Pivot in A's local frame, 3 floats.
	 * @param pivotB Pivot in B's local frame, 3 floats.
	 * @param axisA Slide axis in A's local frame, 3 floats.
	 * @param axisB Slide axis in B's local frame, 3 floats.
	 * @param lower Lower travel limit (m); lower > upper for unlimited travel.
	 * @param upper Upper travel limit (m).
	 * @return An enabled joint.
	 */
	static b3IrrJoint makeSlider(int bodyA, int bodyB, const float* pivotA, const float* pivotB,
								 const float* axisA, const float* axisB,
								 float lower = 1.f, float upper = -1.f);

	/**
	 * @brief Builds a cone-twist joint: anchors coincident, swing bounded by a cone, twist bounded.
	 * @param bodyA Index of the first body.
	 * @param bodyB Index of the second body.
	 * @param pivotA Pivot in A's local frame, 3 floats.
	 * @param pivotB Pivot in B's local frame, 3 floats.
	 * @param axisA Cone axis in A's local frame, 3 floats.
	 * @param axisB Cone axis in B's local frame, 3 floats.
	 * @param swingSpan Half-angle of the cone (rad).
	 * @param twistSpan Maximum |twist| about the axis (rad).
	 * @return An enabled joint.
	 */
	static b3IrrJoint makeConeTwist(int bodyA, int bodyB, const float* pivotA, const float* pivotB,
									const float* axisA, const float* axisB,
									float swingSpan, float twistSpan);

	/**
	 * @brief Builds a generic 6-DoF joint with every axis free; set limits/motors afterwards.
	 * @param bodyA Index of the first body.
	 * @param bodyB Index of the second body.
	 * @param pivotA Pivot in A's local frame, 3 floats.
	 * @param pivotB Pivot in B's local frame, 3 floats.
	 * @param frameA Joint frame orientation in A's local frame, xyzw, or 0 for identity.
	 * @param frameB Joint frame orientation in B's local frame, xyzw, or 0 for identity.
	 * @return An enabled joint.
	 */
	static b3IrrJoint make6Dof(int bodyA, int bodyB, const float* pivotA, const float* pivotB,
							   const float* frameA = 0, const float* frameB = 0);

	/**
	 * @brief Sets one linear axis' travel limit; lower > upper frees it, lower == upper locks it.
	 * @param joint Joint to edit.
	 * @param axis Joint-frame axis.
	 * @param lower Lower limit (m).
	 * @param upper Upper limit (m).
	 */
	static void setLinearLimit(b3IrrJoint& joint, b3IrrJointAxis axis, float lower, float upper);

	/**
	 * @brief Sets one angular axis' limit; lower > upper frees it, lower == upper locks it.
	 * @param joint Joint to edit.
	 * @param axis Joint-frame axis.
	 * @param lower Lower limit (rad).
	 * @param upper Upper limit (rad).
	 */
	static void setAngularLimit(b3IrrJoint& joint, b3IrrJointAxis axis, float lower, float upper);

	/**
	 * @brief Sets a velocity motor on one axis; a 0 max impulse turns it off.
	 * @param joint Joint to edit.
	 * @param axis Joint-frame axis; hinge, slider and cone-twist put their axis on X.
	 * @param angular True for the angular motor, false for the linear one.
	 * @param targetVelocity Target relative velocity (rad/s or m/s).
	 * @param maxImpulse Largest impulse the motor may apply per step.
	 */
	static void setMotor(b3IrrJoint& joint, b3IrrJointAxis axis, bool angular,
						 float targetVelocity, float maxImpulse);

	/**
	 * @brief Gives a joint its own ERP/CFM instead of the global solve parameter.
	 * @param joint Joint to edit.
	 * @param erp Baumgarte factor in [0,1].
	 * @param cfm Softness added to every row's effective-mass denominator.
	 */
	static void setErpCfm(b3IrrJoint& joint, float erp, float cfm);

	/**
	 * @brief Recomputes frameInB so the joint reads zero angle/offset at the bodies' current poses.
	 * @param joint Joint to edit; frameInA is kept.
	 * @param quatA Body A orientation, xyzw.
	 * @param quatB Body B orientation, xyzw.
	 */
	static void alignFrameBToCurrentPose(b3IrrJoint& joint, const float* quatA, const float* quatB);

	/// Device-resident joints from the last uploadJoints; 0 before one has run.
	irr::scene::IComputeBuffer* getJointBuffer() const { return m_jointBuffer; }
	/// Joints in that buffer.
	unsigned int getResidentJointCount() const { return m_residentJoints; }

	/**
	 * @brief Reads back the field values the GPU actually sees, proving the two layouts agree.
	 * @param joints Joints to upload and probe.
	 * @param out Receives 12 ints per joint: type, rbA, rbB, then the raw bits of
	 *            breakingImpulseThreshold, pivotInA.y and pivotInB.y, flags, uid, then the raw
	 *            bits of erp, cfm, angUpper.x and angMotorMaxImpulse.z.
	 * @return False if the probe kernel is unavailable.
	 */
	bool debugReadJointLayout(const std::vector<b3IrrJoint>& joints, std::vector<int>& out);

	/// Ints per joint written by debugReadJointLayout.
	static const unsigned int DebugLayoutInts = 12;

private:
	b3IrrlichtJointSolver(const b3IrrlichtJointSolver&);
	b3IrrlichtJointSolver& operator=(const b3IrrlichtJointSolver&);

	void releaseBuffers();
	void runIterations(irr::scene::IComputeBuffer* bodies, unsigned int numBodies,
					   unsigned int numJoints, int iterations);

	irr::video::IVideoDriver* m_driver;
	bool m_doubleSingle;

	int m_solveMaterial;
	int m_applyMaterial;
	int m_finishMaterial;
	int m_debugMaterial;

	irr::scene::IComputeBuffer* m_paramBuffer;
	irr::scene::IComputeBuffer* m_bodyBuffer;
	irr::scene::IComputeBuffer* m_jointBuffer;
	irr::scene::IComputeBuffer* m_inertiaBuffer;
	irr::scene::IComputeBuffer* m_scaleBuffer;
	irr::scene::IComputeBuffer* m_deltaBuffer;
	irr::scene::IComputeBuffer* m_accumBuffer;
	irr::scene::IComputeBuffer* m_statusBuffer;

	/// Joints currently uploaded, so the resident solve knows its dispatch size without a readback.
	unsigned int m_residentJoints;
};

#endif  //B3_IRRLICHT_JOINT_SOLVER_H
