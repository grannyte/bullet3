#ifndef B3_IRRLICHT_QUERIES_H
#define B3_IRRLICHT_QUERIES_H

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

class b3IrrlichtLbvh;
class b3IrrlichtNarrowphase;
class b3IrrPlanetNoiseParams;

/// Batched closest-hit rays and sphere sweeps over the resident GPU world (B3Queries.hlsl): the GPU
/// stand-in for CBulletThread::RayTest / sphere casts, one dispatch and at most one readback per tick.
///
/// Hits stay resident (getHitBuffer) so a GPU consumer needs no readback at all; fetch() returns the
/// batch BEFORE the most recent submit, fetchBlocking() the most recent (blocking copy if no async).
///
/// Exact vs conservative per shape is documented at the top of B3QueriesBody.hlsli.
class b3IrrlichtQueries
{
public:
	enum Kind
	{
		QUERY_RAY = 0,
		QUERY_SPHERE = 1
	};

	enum : int
	{
		/// Owner root that removes a body from every query (a shape answered elsewhere).
		EXCLUDED_ROOT = -2
	};

	enum : unsigned int
	{
		/// ignoreOffset of a query with no ignore list.
		NO_IGNORE = 0xFFFFFFFFu,
		/// Longest per-query ignore list the kernel walks.
		MAX_IGNORE = 256
	};

	/// Layout-identical to the HLSL b3Query (48 bytes).
	struct b3IrrQuery
	{
		float from[4];
		float to[4];
		float radius;      // 0 for a ray
		int ownerRoot;     // -1 filters nothing
		unsigned int kind; // Kind
		unsigned int ignoreOffset; // NO_IGNORE, or [count, body...] in the batch's ignore list
	};

	/// Layout-identical to the HLSL b3QueryHit (48 bytes).
	struct b3IrrQueryHit
	{
		float point[3];    // world hit point
		float fraction;    // along from->to; 1 when nothing was hit
		float normal[3];   // world normal at the hit
		float pad;
		int bodyIndex;     // -1 when nothing was hit
		int hit;           // 0/1
		int pad0;
		int pad1;
	};

	/// Planet-body-frame segment for castPlanetRays, hi + lo per axis (64 bytes, B3PlanetQueries.hlsl).
	struct b3IrrPlanetRay
	{
		float fromHi[4];
		float fromLo[4];
		float toHi[4];
		float toLo[4];
	};

	/// One castPlanetRays answer (32 bytes).
	struct b3IrrPlanetHit
	{
		float normal[3];   // facet outward normal, planet body frame
		float fraction;    // along the segment; 1 on a miss
		int hit;           // 0/1
		int status;        // PLANET_ANSWERED, or PLANET_BUDGET when the walk stopped short of the end
		int triangles;     // lattice triangles walked
		int pad;
	};

	enum : int
	{
		PLANET_ANSWERED = 0,
		PLANET_BUDGET = 1
	};

	b3IrrlichtQueries(irr::video::IVideoDriver* driver);
	~b3IrrlichtQueries();

	/**
	 * @brief Compiles the optional planet kernel, a host-side file that includes the host's terrain noise.
	 * @param fileSystem Device filesystem; without it a missing shader binds an unrelated material.
	 * @return False when the host ships no media/shaders/B3PlanetQueries.hlsl or it failed to compile.
	 */
	bool initPlanet(irr::io::IFileSystem* fileSystem = 0);

	bool isPlanetQueryAvailable() const { return m_planetMaterial >= 0; }

	/**
	 * @brief The five noise constants the planet kernel runs with (b3IrrlichtNarrowphase::setPlanetNoiseParams').
	 */
	void setPlanetNoiseParams(float mTimer, float plates, float rivers, float atmosphereDensity, float texsize);

	/**
	 * @brief Closest front-face hit of each segment on a planet's contact lattice; blocking.
	 * @param planets b3IrrlichtNarrowphase::getPlanetBuffer().
	 * @param planetEntry Entry of that buffer to query.
	 * @param rays Segments in the planet body's frame.
	 * @param maxTriangles Lattice triangles one segment may walk before it reports PLANET_BUDGET.
	 * @param out One answer per ray, in order.
	 * @return False when the kernel or the planet buffer is unavailable.
	 */
	bool castPlanetRays(irr::scene::IComputeBuffer* planets, unsigned int planetEntry,
						const std::vector<b3IrrPlanetRay>& rays, unsigned int maxTriangles,
						std::vector<b3IrrPlanetHit>& out);

	/**
	 * @brief Compiles the query kernel.
	 * @param fileSystem Device filesystem; without it a missing shader binds an unrelated material.
	 * @param doubleSingle Match the LBVH/narrowphase instances: their df64 buffers have wider strides.
	 * @return False if the shader is missing or failed to compile.
	 */
	bool init(irr::io::IFileSystem* fileSystem = 0, bool doubleSingle = false);

	bool isDoubleSingle() const { return m_doubleSingle; }

	/**
	 * @brief Owner root per body: the hull's body index for a welded child, itself otherwise.
	 *
	 * A query whose ownerRoot equals a body's root skips that body - the whole welded tree of the
	 * shooter, never a stranger. Empty (the default) means identity, i.e. a query skips only itself.
	 *
	 * @param roots One entry per body, in body order; re-uploaded on the next submit.
	 */
	void setBodyOwnerRoots(const std::vector<int>& roots);

	/**
	 * @brief Per-body collision margin a convex hull is rounded by (Bullet's hull ray semantics).
	 * @param margins One entry per body, in body order; empty (the default) rounds nothing.
	 */
	void setBodyMargins(const std::vector<float>& margins);

	/// Discards any queries added since the last submit.
	void beginBatch();

	/**
	 * @brief Adds a closest-hit ray.
	 * @param from Segment start, world space.
	 * @param to Segment end, world space.
	 * @param ownerRoot Root id whose bodies the ray ignores; -1 ignores nothing.
	 * @param ignoreBodies Body indices this query alone skips; may be null.
	 * @param ignoreCount Entries in ignoreBodies, at most MAX_IGNORE.
	 * @return Index of the query inside this batch - its slot in the result array.
	 */
	unsigned int addRay(const float from[3], const float to[3], int ownerRoot = -1,
						const int* ignoreBodies = 0, unsigned int ignoreCount = 0);

	/**
	 * @brief Adds a sphere sweep: the sphere's centre travels from->to.
	 * @param from Centre at the start, world space.
	 * @param to Centre at the end, world space.
	 * @param radius Sphere radius.
	 * @param ownerRoot Root id whose bodies the sweep ignores; -1 ignores nothing.
	 * @param ignoreBodies Body indices this query alone skips; may be null.
	 * @param ignoreCount Entries in ignoreBodies, at most MAX_IGNORE.
	 * @return Index of the query inside this batch.
	 */
	unsigned int addSphereSweep(const float from[3], const float to[3], float radius, int ownerRoot = -1,
								const int* ignoreBodies = 0, unsigned int ignoreCount = 0);

	/// Queries added since beginBatch.
	unsigned int getBatchSize() const { return (unsigned int)m_batch.size(); }

	/**
	 * @brief Uploads the batch and dispatches it against the tree the last resident broadphase left.
	 *
	 * Call after b3IrrlichtLbvh::calculateOverlappingPairsResident for the same step, so tree, world
	 * AABBs and bodies describe one consistent world. An empty batch is legal.
	 *
	 * @param lbvh Broadphase holding the resident tree.
	 * @param narrowphase Shape registry and the world-AABB buffer it computed this step.
	 * @param bodies Device-resident bodies the tree was built over.
	 * @param numBodies Bodies in that buffer.
	 * @return False if the kernel is unavailable or the broadphase has no resident tree.
	 */
	bool submit(const b3IrrlichtLbvh& lbvh, const b3IrrlichtNarrowphase& narrowphase,
				irr::scene::IComputeBuffer* bodies, unsigned int numBodies);

	/**
	 * @brief Results of the batch submitted BEFORE the most recent one - one tick latent.
	 *
	 * Normally complete, so no stall; it waits rather than returning stale data. A hit-buffer growth
	 * between the two submits loses the older staging copy and returns false for that one tick.
	 *
	 * @param out Receives one hit per query of that batch, in addRay/addSphereSweep order.
	 * @return False when no such batch exists yet or its copy is unavailable.
	 */
	bool fetch(std::vector<b3IrrQueryHit>& out);

	/**
	 * @brief Results of the most recent submit; stalls until the GPU has produced them.
	 * @param out Receives one hit per query, in addRay/addSphereSweep order.
	 * @return False if nothing has been submitted.
	 */
	bool fetchBlocking(std::vector<b3IrrQueryHit>& out);

	/// Resident results of the most recent submit, one b3IrrQueryHit per query, for GPU consumers.
	irr::scene::IComputeBuffer* getHitBuffer() const { return m_hitBuffer; }
	/// Resident copy of the most recent submit's queries, parallel to getHitBuffer().
	irr::scene::IComputeBuffer* getQueryBuffer() const { return m_queryBuffer; }
	/// Query count of the most recent submit.
	unsigned int getSubmittedCount() const;

	/// Whether the driver took the last submit's readback asynchronously; false means submit()
	/// already paid a blocking copy and fetch()/fetchBlocking() only hand it out.
	bool isAsyncReadbackAvailable() const { return m_asyncSupported; }

private:
	b3IrrlichtQueries(const b3IrrlichtQueries&);
	b3IrrlichtQueries& operator=(const b3IrrlichtQueries&);

	/**
	 * @brief Copies one readback slot's results out, through the staging copy or the CPU fallback.
	 * @param slot Readback slot of the batch wanted.
	 * @param out Receives that batch's hits.
	 * @return False if the slot holds no completed batch.
	 */
	bool readSlot(unsigned int slot, std::vector<b3IrrQueryHit>& out);

	enum
	{
		READBACK_SLOTS = 2
	};

	struct Pending
	{
		unsigned int count;
		bool valid;
		bool async;
	};

	irr::video::IVideoDriver* m_driver;
	bool m_doubleSingle;
	bool m_asyncSupported;
	int m_material;

	std::vector<b3IrrQuery> m_batch;
	std::vector<int> m_ignore;
	std::vector<int> m_ownerRoots;
	bool m_ownerRootsDirty;
	unsigned int m_ownerRootsUploaded;
	std::vector<float> m_margins;
	bool m_marginsDirty;

	irr::scene::IComputeBuffer* m_paramBuffer;
	irr::scene::IComputeBuffer* m_queryBuffer;
	irr::scene::IComputeBuffer* m_hitBuffer;
	irr::scene::IComputeBuffer* m_ownerRootBuffer;
	irr::scene::IComputeBuffer* m_marginBuffer;
	irr::scene::IComputeBuffer* m_ignoreBuffer;

	int m_planetMaterial;
	b3IrrPlanetNoiseParams* m_planetNoiseParams;  // owned by the planet material
	irr::scene::IComputeBuffer* m_planetParamBuffer;
	irr::scene::IComputeBuffer* m_planetRayBuffer;
	irr::scene::IComputeBuffer* m_planetHitBuffer;

	unsigned int m_submitCount;
	Pending m_pending[READBACK_SLOTS];
	std::vector<b3IrrQueryHit> m_fallback[READBACK_SLOTS];
};

#endif  //B3_IRRLICHT_QUERIES_H
