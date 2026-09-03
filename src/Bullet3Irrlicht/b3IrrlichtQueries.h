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

	/// Layout-identical to the HLSL b3Query (48 bytes).
	struct b3IrrQuery
	{
		float from[4];
		float to[4];
		float radius;      // 0 for a ray
		int ownerRoot;     // -1 filters nothing
		unsigned int kind; // Kind
		unsigned int pad;
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

	b3IrrlichtQueries(irr::video::IVideoDriver* driver);
	~b3IrrlichtQueries();

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

	/// Discards any queries added since the last submit.
	void beginBatch();

	/**
	 * @brief Adds a closest-hit ray.
	 * @param from Segment start, world space.
	 * @param to Segment end, world space.
	 * @param ownerRoot Root id whose bodies the ray ignores; -1 ignores nothing.
	 * @return Index of the query inside this batch - its slot in the result array.
	 */
	unsigned int addRay(const float from[3], const float to[3], int ownerRoot = -1);

	/**
	 * @brief Adds a sphere sweep: the sphere's centre travels from->to.
	 * @param from Centre at the start, world space.
	 * @param to Centre at the end, world space.
	 * @param radius Sphere radius.
	 * @param ownerRoot Root id whose bodies the sweep ignores; -1 ignores nothing.
	 * @return Index of the query inside this batch.
	 */
	unsigned int addSphereSweep(const float from[3], const float to[3], float radius, int ownerRoot = -1);

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
	std::vector<int> m_ownerRoots;
	bool m_ownerRootsDirty;
	unsigned int m_ownerRootsUploaded;

	irr::scene::IComputeBuffer* m_paramBuffer;
	irr::scene::IComputeBuffer* m_queryBuffer;
	irr::scene::IComputeBuffer* m_hitBuffer;
	irr::scene::IComputeBuffer* m_ownerRootBuffer;

	unsigned int m_submitCount;
	Pending m_pending[READBACK_SLOTS];
	std::vector<b3IrrQueryHit> m_fallback[READBACK_SLOTS];
};

#endif  //B3_IRRLICHT_QUERIES_H
