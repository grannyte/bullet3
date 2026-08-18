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

private:
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
};

#endif  //B3_IRRLICHT_NARROWPHASE_H
