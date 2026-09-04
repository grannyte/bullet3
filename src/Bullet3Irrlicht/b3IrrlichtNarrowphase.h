#ifndef B3_IRRLICHT_NARROWPHASE_H
#define B3_IRRLICHT_NARROWPHASE_H

#include "b3IrrlichtLbvh.h"
#include "b3IrrlichtGpuBuffers.h"
#include "Bullet3Collision/NarrowPhaseCollision/shared/b3Collidable.h"
#include "Bullet3Collision/NarrowPhaseCollision/shared/b3Contact4Data.h"
#include "Bullet3Collision/NarrowPhaseCollision/shared/b3ConvexPolyhedronData.h"
#include "Bullet3Collision/NarrowPhaseCollision/shared/b3RigidBodyData.h"

#include <vector>

namespace b3IrrGpu
{
class DispatchHelper;
}

/// One child of a compound: an already-registered collidable placed in the parent's frame.
struct b3IrrCompoundChild
{
	int collidableIndex;
	float position[3];
	float orientation[4];   // xyzw
};

/// Minimum-penetration axis for one pair, layout-identical to the HLSL SatResult (32 bytes).
struct b3IrrSatResult
{
	float axis[3];
	float depth;
	int hasSeparatingAxis;
	int pad0;
	int pad1;
	int pad2;
};

/**
 * @brief Irrlicht-compute analog of b3GpuNarrowPhase: collidable registry, world AABBs, contacts.
 *
 * Shapes live here rather than on the pipeline because every narrowphase kernel is keyed off a
 * collidable index - the registry is the shared vocabulary, not an implementation detail.
 *
 * CONVENTION: m_worldPosB[i].xyz is emitted RELATIVE TO BODY A's origin, not world-space as on the
 * CPU/OpenCL paths.
 */
class b3IrrlichtNarrowphase
{
public:
	b3IrrlichtNarrowphase(irr::video::IVideoDriver* driver);
	~b3IrrlichtNarrowphase();

	/**
	 * @brief Compiles the narrowphase kernels. The PlanetShape kernel is optional - see
	 *        isPlanetShapeAvailable().
	 * @param fileSystem Device filesystem; without it a missing shader silently binds an
	 *        unrelated material instead of failing.
	 * @param doubleSingle Compile the emulated-double (df64) kernel set instead of the f32 one.
	 * @return False if no compute pipeline is available or a REQUIRED shader is missing.
	 */
	bool init(irr::io::IFileSystem* fileSystem = 0, bool doubleSingle = false);

	/// Whether this instance runs the emulated-double kernels. The std::vector entry points are
	/// f32-only and refuse in that mode; the resident path is the df64 one.
	bool isDoubleSingle() const { return m_doubleSingle; }

	/**
	 * @brief Whether PlanetShape collision compiled. It #includes noise2.hlsl, which fails to
	 *        unroll under Release shader flags, so a box-only caller stays fully functional.
	 * @return True when computePlanetContacts can be used.
	 */
	bool isPlanetShapeAvailable() const { return m_planetMaterial >= 0; }

	/**
	 * @brief Enables speculative contacts (Bullet-style CCD) for every contact and AABB call.
	 *
	 * World AABBs grow to cover the step's motion, and a separated pair that the linear sweep says
	 * meets within deltaTime emits one contact with a POSITIVE gap the solver may close but not
	 * exceed. Translation only; rotation is covered by the AABB margin, not the contact.
	 *
	 * @param deltaTime Step the sweep covers; 0 disables (the default, bit-identical to before).
	 * @param motionThreshold Closing motion per step below which no speculative contact is made,
	 *        Bullet's ccdMotionThreshold analog; 0 applies it to every approaching pair.
	 */
	void setSpeculativeContacts(float deltaTime, float motionThreshold = 0.f)
	{
		m_speculativeDt = deltaTime > 0.f ? deltaTime : 0.f;
		m_ccdMotionThreshold = motionThreshold > 0.f ? motionThreshold : 0.f;
	}

	/// Step covered by speculative contacts; 0 when disabled.
	float getSpeculativeDeltaTime() const { return m_speculativeDt; }

	/**
	 * @brief Sets a collidable's rolling-friction coefficient, read by the contact solver.
	 *
	 * Combined per Bullet: rfA * muB + rfB * muA, then budgeted by the manifold's normal impulse.
	 * Takes effect on the next writeShapesToGpu.
	 *
	 * @param collidableIdx Index returned by a register* call.
	 * @param coefficient Rolling friction, typically 0.01 - 0.1; 0 disables (the default).
	 */
	void setRollingFriction(int collidableIdx, float coefficient);

	/// Per-collidable rolling friction, for b3IrrlichtSolver; 0 before writeShapesToGpu.
	irr::scene::IComputeBuffer* getRollingFrictionBuffer() const { return m_rollingFrictionBuffer; }

	/**
	 * @brief Registers a sphere collidable.
	 * @param radius Sphere radius in world units.
	 * @return Collidable index to store in b3RigidBodyData::m_collidableIdx.
	 */
	int registerSphereShape(float radius);

	/**
	 * @brief Registers any non-sphere shape by its local-space AABB.
	 * @param localAabb Shape bounds in its own frame; rotated conservatively per step.
	 * @param shapeType One of b3ShapeTypes.
	 * @param shapeIndex Index into whatever per-type shape data that type uses; -1 if none yet.
	 * @return Collidable index to store in b3RigidBodyData::m_collidableIdx.
	 */
	int registerShapeWithLocalAabb(const b3IrrAabb& localAabb, int shapeType, int shapeIndex = -1);

	/**
	 * @brief Registers a box as a 6-face convex hull, so one SAT kernel covers boxes and hulls.
	 * @param halfExtents Half-size along each local axis.
	 * @return Collidable index to store in b3RigidBodyData::m_collidableIdx.
	 */
	int registerBoxShape(const float halfExtents[3]);

	/**
	 * @brief Registers an arbitrary point cloud as a convex hull: faces, unique edge directions
	 *        and local AABB are derived on the host; the SAT/clip kernels need no special case.
	 *
	 * Every face is capped at MAX_POLY (8) vertices - a larger coplanar polygon is split into
	 * coplanar sub-faces, which only narrows the clip manifold, never the SAT. O(n^4) plane
	 * search: fine for shape registration up to a few hundred vertices, not for per-step use.
	 *
	 * @param xyz numVertices * 3 floats in the shape's local frame.
	 * @param numVertices At least 4 non-coplanar points.
	 * @return Collidable index, or -1 if the points do not span a volume.
	 */
	int registerConvexHullShape(const float* xyz, int numVertices);

	/**
	 * @brief Registers a Y-axis cylinder as an 8-sided prism hull.
	 *
	 * 8 sides because the flat caps are single faces and MAX_POLY caps a face at 8 vertices;
	 * a rounder wheel needs a dedicated sphere-swept kernel, not a denser hull.
	 *
	 * @param radius Cylinder radius.
	 * @param halfHeight Half-length along local Y.
	 * @return Collidable index.
	 */
	int registerCylinderShape(float radius, float halfHeight);

	/**
	 * @brief Registers a Y-axis capsule as a dense hull: 12 segments, two latitude rings and a
	 *        pole per end (74 vertices). No face exceeds 4 vertices, so rolling is round-ish.
	 * @param radius Capsule radius.
	 * @param halfHeight Half-length of the straight section along local Y.
	 * @return Collidable index.
	 */
	int registerCapsuleShape(float radius, float halfHeight);

	/**
	 * @brief Registers a compound of already-registered convex collidables.
	 *
	 * Broadphase uses the merged local AABB of the rotated children; narrowphase expands each
	 * body pair into child pairs on the GPU (see expandPairsResident). Children must be convex
	 * hulls; compounds do not nest and sphere children produce no contacts.
	 *
	 * @param children Child collidables with parent-relative transforms.
	 * @param numChildren Number of children; at least 1.
	 * @return Collidable index, or -1 on an invalid child.
	 */
	int registerCompoundShape(const b3IrrCompoundChild* children, int numChildren);

	/// True once a registered shape needs body pairs expanded into leaf pairs before clipping.
	/// The mesh path extends this condition when it lands.
	bool needsPairExpansion() const { return !m_cpuChildShapes.empty(); }

	/// Capacity of the leaf-pair append buffer; appends past it are dropped.
	void setMaxLeafPairs(unsigned int maxLeafPairs) { m_maxLeafPairs = maxLeafPairs; }

	/**
	 * @brief Uploads the collidable registry and any hull data. Call after the last register*.
	 * @return False if nothing has been registered.
	 */
	bool writeShapesToGpu();

	/**
	 * @brief Minimum-penetration axis per pair for convex shapes; sphere pairs report no axis.
	 * @param bodies Bodies indexed by the pair list.
	 * @param pairs Broadphase output; body-index pairs.
	 * @param results Receives one entry per pair, in pair order.
	 * @return False if the kernel is unavailable or the shapes were never uploaded.
	 */
	bool findSeparatingAxis(const std::vector<b3RigidBodyData>& bodies,
							const std::vector<std::pair<unsigned int, unsigned int> >& pairs,
							std::vector<b3IrrSatResult>& results);

	/**
	 * @brief Clips convex pairs into contact manifolds, running SAT internally first.
	 *
	 * The clipped point set is reduced on the GPU to the deepest point plus the extremes along two
	 * perpendicular in-plane axes, so a manifold never carries more than 4 points.
	 *
	 * @param bodies Bodies indexed by the pair list.
	 * @param pairs Broadphase output; body-index pairs.
	 * @param contacts Receives one b3Contact4Data per colliding pair, up to 4 points each.
	 * @param maxContacts Append capacity; overflow truncates rather than corrupting.
	 * @param overflowed Set when the GPU produced more contacts than maxContacts.
	 * @return False if the kernel is unavailable or the shapes were never uploaded.
	 */
	bool computeConvexContacts(const std::vector<b3RigidBodyData>& bodies,
							   const std::vector<std::pair<unsigned int, unsigned int> >& pairs,
							   std::vector<b3Contact4Data>& contacts,
							   unsigned int maxContacts = 65536,
							   bool* overflowed = 0);

	/**
	 * @brief World AABB per body, GPU-side, ready to feed straight into the broadphase.
	 * @param bodies Bodies whose m_collidableIdx indexes the registry.
	 * @param aabbs Receives one AABB per body, in body order.
	 * @param margin Collision margin added to every bound.
	 * @return False if the kernel is unavailable or the shapes were never uploaded.
	 */
	bool computeWorldAabbs(const std::vector<b3RigidBodyData>& bodies,
						   std::vector<b3IrrAabb>& aabbs, float margin = 0.f);

	/**
	 * @brief Sphere/sphere contacts for a broadphase pair list. Non-sphere pairs are skipped.
	 * @param bodies Bodies indexed by the pair list.
	 * @param pairs Broadphase output; body-index pairs.
	 * @param contacts Receives one b3Contact4Data per touching sphere pair.
	 * @param maxContacts Append capacity; overflow truncates rather than corrupting.
	 * @param collisionMargin Extra separation still counted as touching.
	 * @param overflowed Set when the GPU produced more contacts than maxContacts.
	 * @return False if the kernel is unavailable or the shapes were never uploaded.
	 */
	bool computeSphereContacts(const std::vector<b3RigidBodyData>& bodies,
							   const std::vector<std::pair<unsigned int, unsigned int> >& pairs,
							   std::vector<b3Contact4Data>& contacts,
							   unsigned int maxContacts = 65536,
							   float collisionMargin = 0.f,
							   bool* overflowed = 0);

	/**
	 * @brief Sets noise2.hlsl's five `params` constants for the planet kernel.
	 *
	 * Leaving these at zero makes texsize 0, which divides to NaN inside the noise - and a NaN
	 * separation is not rejected by a `>` test, so every sample becomes a spurious contact.
	 *
	 * @param mTimer Planet seed.
	 * @param plates Non-zero to enable plate tectonics shaping.
	 * @param rivers Non-zero to enable river carving.
	 * @param atmosphereDensity Drives the final height scaling.
	 * @param texsize Heightmap resolution the noise was authored against; must be non-zero.
	 */
	void setPlanetNoiseParams(float mTimer, float plates, float rivers, float atmosphereDensity,
							  float texsize);

	/// Planet parameters, layout-identical to the HLSL PlanetData (32 bytes).
	struct b3IrrPlanet
	{
		float center[3];
		float baseRadius;
		float heightScale;
		float pad0;
		float pad1;
		float pad2;
	};

	/**
	 * @brief Sphere-vs-planet contacts, evaluating the terrain noise FUNCTION per sample.
	 *
	 * Deliberately does not read the generated heightmap RTT: that is one patch's render target at
	 * whatever LOD the camera warranted, so it would sample an unrelated patch for most positions.
	 *
	 * @param bodies Bodies indexed by the pair list.
	 * @param planets Planets indexed by each pair's second element.
	 * @param pairs (body index, planet index) pairs to test.
	 * @param contacts Receives one contact per touching pair.
	 * @param maxContacts Append capacity; overflow truncates rather than corrupting.
	 * @param overflowed Set when the GPU produced more contacts than maxContacts.
	 * @return False if the kernel is unavailable or the shapes were never uploaded.
	 */
	bool computePlanetContacts(const std::vector<b3RigidBodyData>& bodies,
							   const std::vector<b3IrrPlanet>& planets,
							   const std::vector<std::pair<unsigned int, unsigned int> >& pairs,
							   std::vector<b3Contact4Data>& contacts,
							   unsigned int maxContacts = 65536,
							   bool* overflowed = 0);

	/**
	 * @brief Unpacks the 16-bit fixed-point restitution the contact kernel writes.
	 * @param contact Contact to read.
	 * @return Restitution in [0,1].
	 */
	static float getRestitution(const b3Contact4Data& contact);

	/**
	 * @brief Unpacks the 16-bit fixed-point friction the contact kernel writes.
	 * @param contact Contact to read.
	 * @return Friction in [0,1].
	 */
	static float getFriction(const b3Contact4Data& contact);

	int getNumCollidables() const { return (int)m_cpuCollidables.size(); }

	/**
	 * @brief Whether the device-buffer entry points below can run.
	 * @return True once B3GpuResident.hlsl compiled.
	 */
	bool isResidentPathAvailable() const;

	/**
	 * @brief Seeds the persistent body buffer the resident phases read and write.
	 * @param bodies Bodies to upload; sizes the buffer for every later resident call.
	 * @return False if there is nothing to upload.
	 */
	bool uploadBodiesResident(const std::vector<b3RigidBodyData>& bodies);

	/**
	 * @brief Seeds the persistent body buffer with df64 bodies, for a double-single instance.
	 * @param bodies Bodies whose position carries a hi and a lo half.
	 * @return False if there is nothing to upload or this instance is not double-single.
	 */
	bool uploadBodiesResidentDS(const std::vector<b3IrrGpu::b3IrrRigidBodyDataDS>& bodies);

	/// Persistent body buffer - the one the whole resident chain reads and the solver writes.
	irr::scene::IComputeBuffer* getBodyBuffer() const { return m_bodyBuffer; }
	/// World AABBs left by computeWorldAabbs/computeWorldAabbsResident, ready for the broadphase.
	irr::scene::IComputeBuffer* getWorldAabbBuffer() const { return m_worldAabbBuffer; }
	/// Manifolds left by the last contact call.
	irr::scene::IComputeBuffer* getContactBuffer() const { return m_contactBuffer; }
	/// Appended contact count, as EHBF_DRAW_INDIRECT_ARGS - feeds an indirect solve dispatch.
	irr::scene::IComputeBuffer* getContactCountBuffer() const { return m_contactCountBuffer; }

	/// Shape registry as uploaded by writeShapesToGpu, for kernels outside this class that resolve
	/// a body's shape (b3IrrlichtQueries). Hull buffers are null while no hull is registered.
	irr::scene::IComputeBuffer* getCollidableBuffer() const { return m_collidableBuffer; }
	irr::scene::IComputeBuffer* getHullBuffer() const { return m_hullBuffer; }
	irr::scene::IComputeBuffer* getHullFaceBuffer() const { return m_faceBuffer; }
	irr::scene::IComputeBuffer* getHullVertexBuffer() const { return m_vertexBuffer; }
	irr::scene::IComputeBuffer* getHullFaceIndexBuffer() const { return m_indexBuffer; }
	irr::scene::IComputeBuffer* getHullEdgeBuffer() const { return m_edgeBuffer; }

	/**
	 * @brief computeWorldAabbs straight out of a device-resident body buffer.
	 * @param bodies Device-resident bodies.
	 * @param numBodies Bodies in that buffer.
	 * @param margin Collision margin added to every bound.
	 * @param sleepState Optional previous-step sleep bits; sleeping bodies keep their existing bound.
	 * @return False if the kernel is unavailable or the shapes were never uploaded.
	 */
	bool computeWorldAabbsResident(irr::scene::IComputeBuffer* bodies, unsigned int numBodies,
								   float margin = 0.f,
								   irr::scene::IComputeBuffer* sleepState = 0);

	/**
	 * @brief computeConvexContacts against a device-resident pair list of GPU-decided length.
	 *
	 * The pair count never comes back: it is patched into the kernel's params and into the
	 * dispatch arguments on the device.
	 *
	 * @param bodies Device-resident bodies.
	 * @param numBodies Bodies in that buffer.
	 * @param pairs Device-resident (bodyA, bodyB) pairs.
	 * @param pairCount EHBF_DRAW_INDIRECT_ARGS buffer holding the appended pair count.
	 * @param maxPairs Capacity of the pair buffer; the count is clamped to it.
	 * @param maxContacts Capacity for the contact buffer.
	 * @return False if the kernels are unavailable or the shapes were never uploaded.
	 */
	bool computeConvexContactsResident(irr::scene::IComputeBuffer* bodies, unsigned int numBodies,
									   irr::scene::IComputeBuffer* pairs,
									   irr::scene::IComputeBuffer* pairCount,
									   unsigned int maxPairs, unsigned int maxContacts = 65536);

	/**
	 * @brief Expands device-resident body pairs into leaf pairs (bodyA, bodyB, childA, childB).
	 *
	 * This is the seam for every multi-part shape: compounds append one leaf per child pair
	 * here; a triangle-mesh path adds its own branch in CSExpandPairs and its own leaf kind.
	 *
	 * @param bodies Device-resident bodies.
	 * @param pairs Device-resident (bodyA, bodyB) pairs.
	 * @param pairCount EHBF_DRAW_INDIRECT_ARGS buffer with the appended pair count; 0 to use numPairs.
	 * @param numPairs CPU-known pair count when pairCount is 0, else the capacity cap.
	 * @return False if the kernel is unavailable or the shapes were never uploaded.
	 */
	bool expandPairsResident(irr::scene::IComputeBuffer* bodies, irr::scene::IComputeBuffer* pairs,
							 irr::scene::IComputeBuffer* pairCount, unsigned int numPairs);

	/// Leaf pairs (int4) left by the last expandPairsResident call.
	irr::scene::IComputeBuffer* getLeafPairBuffer() const { return m_leafPairBuffer; }
	/// Appended leaf-pair count, as EHBF_DRAW_INDIRECT_ARGS.
	irr::scene::IComputeBuffer* getLeafPairCountBuffer() const { return m_leafPairCountBuffer; }

private:
	/// Clips the current leaf-pair buffer into m_contactBuffer by indirect dispatch.
	bool clipLeafPairs(irr::scene::IComputeBuffer* bodies, unsigned int maxContacts);
	/// Reads back the append count and the contacts the last clip dispatch produced.
	bool downloadContacts(std::vector<b3Contact4Data>& contacts, unsigned int maxContacts,
						  bool* overflowed);
	b3IrrlichtNarrowphase(const b3IrrlichtNarrowphase&);
	b3IrrlichtNarrowphase& operator=(const b3IrrlichtNarrowphase&);

	bool uploadBodies(const std::vector<b3RigidBodyData>& bodies);

	irr::video::IVideoDriver* m_driver;
	bool m_doubleSingle;

	b3IrrGpu::DispatchHelper* m_dispatch;

	int m_aabbMaterial;
	int m_sphereContactMaterial;
	int m_satMaterial;
	int m_clipMaterial;
	int m_planetMaterial;
	int m_expandMaterial;
	int m_clipLeafMaterial;

	unsigned int m_maxLeafPairs;

	irr::scene::IComputeBuffer* m_expandParamBuffer;
	irr::scene::IComputeBuffer* m_childShapeBuffer;
	irr::scene::IComputeBuffer* m_leafPairBuffer;
	irr::scene::IComputeBuffer* m_leafPairCountBuffer;

	irr::scene::IComputeBuffer* m_aabbParamBuffer;
	irr::scene::IComputeBuffer* m_contactParamBuffer;
	irr::scene::IComputeBuffer* m_satParamBuffer;
	irr::scene::IComputeBuffer* m_hullBuffer;
	irr::scene::IComputeBuffer* m_vertexBuffer;
	irr::scene::IComputeBuffer* m_faceBuffer;
	irr::scene::IComputeBuffer* m_indexBuffer;
	irr::scene::IComputeBuffer* m_edgeBuffer;
	irr::scene::IComputeBuffer* m_satResultBuffer;
	irr::scene::IComputeBuffer* m_planetBuffer;

	// Owned by the material renderer once handed to addComputeShaderFromFile; kept to update the
	// values between dispatches without recompiling the shader.
	class PlanetNoiseParams* m_planetNoiseParams;
	irr::scene::IComputeBuffer* m_bodyBuffer;
	irr::scene::IComputeBuffer* m_collidableBuffer;
	irr::scene::IComputeBuffer* m_localAabbBuffer;
	irr::scene::IComputeBuffer* m_worldAabbBuffer;
	irr::scene::IComputeBuffer* m_pairBuffer;
	irr::scene::IComputeBuffer* m_contactBuffer;
	irr::scene::IComputeBuffer* m_contactCountBuffer;

	std::vector<b3Collidable> m_cpuCollidables;
	std::vector<b3IrrAabb> m_cpuLocalAabbs;

	std::vector<b3ConvexPolyhedronData> m_cpuHulls;
	std::vector<float> m_cpuVertices;   // 4 floats per vertex
	std::vector<b3GpuFace> m_cpuFaces;
	std::vector<int> m_cpuIndices;      // face vertex indices, per b3GpuFace::m_indexOffset
	std::vector<float> m_cpuEdges;      // 4 floats per unique edge direction
	std::vector<b3GpuChildShape> m_cpuChildShapes;

	/// Parallel to m_cpuCollidables, like the local AABBs: the solver indexes it by collidable.
	std::vector<float> m_cpuRollingFriction;
	irr::scene::IComputeBuffer* m_rollingFrictionBuffer;
	float m_speculativeDt;
	float m_ccdMotionThreshold;
};

#endif  //B3_IRRLICHT_NARROWPHASE_H
