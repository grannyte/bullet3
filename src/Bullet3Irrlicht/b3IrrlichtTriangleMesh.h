#ifndef B3_IRRLICHT_TRIANGLE_MESH_H
#define B3_IRRLICHT_TRIANGLE_MESH_H

#include "b3IrrlichtLbvh.h"
#include "Bullet3Collision/NarrowPhaseCollision/shared/b3Contact4Data.h"
#include "Bullet3Collision/NarrowPhaseCollision/shared/b3RigidBodyData.h"

#include <vector>
#include <utility>

class b3IrrlichtNarrowphase;

namespace b3IrrGpu
{
class DispatchHelper;
}

/// Per-mesh GPU record, 32 bytes, layout-identical to b3IrrMeshHeader in
/// B3TriangleMeshContactsBody.hlsli. Offsets index the concatenated all-mesh buffers.
struct b3IrrMeshHeader
{
	unsigned int triangleOffset;
	unsigned int numTriangles;
	unsigned int nodeOffset;
	int rootIndex;
	unsigned int leafOffset;
	unsigned int aabbOffset;
	unsigned int numLeaves;
	unsigned int flags;
};

/**
 * @brief Concave triangle meshes for the Irrlicht-compute narrowphase: a resident per-mesh triangle
 *        BVH (same Karras LBVH as the body broadphase) and ONE manifold per body/mesh pair.
 *
 * One manifold per pair, not per triangle: the Jacobi solver applies each manifold in full, so a
 * body straddling a seam would otherwise be corrected twice. Collidable = SHAPE_CONCAVE_TRIMESH.
 */
class b3IrrlichtTriangleMesh
{
public:
	enum MeshFlags
	{
		/// Reject bodies behind the winding-defined front face instead of flipping the normal.
		FLAG_ONE_SIDED = 1
	};

	/// Per-edge classification, 2 bits per edge in the adjacency record: edge e at bits [2e+1:2e].
	enum EdgeClass
	{
		EDGE_BOUNDARY = 0,
		EDGE_CONVEX = 1,
		EDGE_FLAT = 2,
		EDGE_CONCAVE = 3
	};

	b3IrrlichtTriangleMesh(irr::video::IVideoDriver* driver);
	~b3IrrlichtTriangleMesh();

	/**
	 * @brief Compiles the contact kernel and the f32 LBVH used to build each mesh's tree.
	 * @param fileSystem Device filesystem; without it a missing shader binds an unrelated material.
	 * @param doubleSingle Compile the emulated-double (df64) kernel, for a df64 body buffer.
	 * @return False if a required shader is missing or failed to compile.
	 */
	bool init(irr::io::IFileSystem* fileSystem = 0, bool doubleSingle = false);

	bool isDoubleSingle() const { return m_doubleSingle; }

	/**
	 * @brief Registers a triangle mesh in its own local frame; the tree is built on writeMeshesToGpu.
	 *
	 * Adjacency is derived by vertex INDEX, so an unwelded mesh (duplicated vertices along seams)
	 * gets boundary edges everywhere and no seam smoothing - weld before registering.
	 *
	 * @param xyz numVertices * 3 floats.
	 * @param numVertices Vertex count.
	 * @param indices numTriangles * 3 vertex indices, counter-clockwise seen from the front.
	 * @param numTriangles Triangle count; at least 1.
	 * @param flags MeshFlags.
	 * @return Mesh index, or -1 on an out-of-range index.
	 */
	int registerTriangleMesh(const float* xyz, unsigned int numVertices, const unsigned int* indices,
							 unsigned int numTriangles, unsigned int flags = 0);

	int getNumMeshes() const { return (int)m_headers.size(); }

	/// Local-space bounds of a registered mesh.
	const b3IrrAabb& getLocalAabb(int mesh) const { return m_localAabbs[mesh]; }

	/// Edge classes of one triangle (EdgeClass packed 2 bits per edge), as uploaded.
	unsigned int getEdgeClassBits(int mesh, unsigned int triangle) const;

	/**
	 * @brief Registers the narrowphase collidable for a mesh: SHAPE_CONCAVE_TRIMESH, shapeIndex = mesh.
	 * @param narrowphase Registry to add to; call its writeShapesToGpu afterwards as usual.
	 * @param mesh Index from registerTriangleMesh.
	 * @return Collidable index to store in b3RigidBodyData::m_collidableIdx.
	 */
	int registerCollidable(b3IrrlichtNarrowphase& narrowphase, int mesh) const;

	/**
	 * @brief Builds every not-yet-built tree and uploads all mesh data. Call after the last register.
	 * @return False if a tree could not be built or nothing is registered.
	 */
	bool writeMeshesToGpu();

	/**
	 * @brief Appends body-vs-mesh manifolds onto the narrowphase's resident contact list.
	 *
	 * Runs over the ORIGINAL pair list after computeConvexContactsResident, appending into the same
	 * contact buffer without resetting its counter; the count is re-copied so the solver sees both.
	 *
	 * @param narrowphase Owner of the shape registry and the contact buffers.
	 * @param bodies Device-resident bodies.
	 * @param numBodies Bodies in that buffer.
	 * @param pairs Device-resident (bodyA, bodyB) pairs.
	 * @param pairCount EHBF_DRAW_INDIRECT_ARGS buffer holding the appended pair count.
	 * @param maxPairs Capacity of the pair buffer; the count is clamped to it.
	 * @param collisionMargin Extra separation still counted as touching.
	 * @param resetContacts Start the contact list afresh instead of appending (no convex stage ran).
	 * @return False if the kernel is unavailable, no mesh is uploaded, or the contact buffer is missing.
	 */
	bool appendContactsResident(const b3IrrlichtNarrowphase& narrowphase,
								irr::scene::IComputeBuffer* bodies, unsigned int numBodies,
								irr::scene::IComputeBuffer* pairs, irr::scene::IComputeBuffer* pairCount,
								unsigned int maxPairs, float collisionMargin = 0.f,
								bool resetContacts = false);

	/**
	 * @brief std::vector form for tests: body-vs-mesh manifolds for a CPU pair list.
	 * @param narrowphase Owner of the shape registry; its body buffer is refilled from bodies.
	 * @param bodies Bodies indexed by the pair list.
	 * @param pairs Body-index pairs; non-mesh pairs are skipped.
	 * @param contacts Receives one manifold per touching body/mesh pair.
	 * @param maxContacts Append capacity; overflow truncates rather than corrupting.
	 * @param collisionMargin Extra separation still counted as touching.
	 * @param overflowed Set when the GPU produced more contacts than maxContacts.
	 * @return False if the kernel is unavailable or the meshes were never uploaded.
	 */
	bool computeContacts(b3IrrlichtNarrowphase& narrowphase, const std::vector<b3RigidBodyData>& bodies,
						 const std::vector<std::pair<unsigned int, unsigned int> >& pairs,
						 std::vector<b3Contact4Data>& contacts, unsigned int maxContacts = 65536,
						 float collisionMargin = 0.f, bool* overflowed = 0);

private:
	b3IrrlichtTriangleMesh(const b3IrrlichtTriangleMesh&);
	b3IrrlichtTriangleMesh& operator=(const b3IrrlichtTriangleMesh&);

	/// CPU-side description kept until the tree is built and packed into the shared arrays.
	struct PendingMesh
	{
		unsigned int firstVertex;
		unsigned int numVertices;
		unsigned int firstTriangle;
		unsigned int numTriangles;
		unsigned int flags;
		bool built;
	};

	/**
	 * @brief Builds one mesh's tree and appends its nodes to the packed arrays.
	 * @param mesh Index into m_pending / m_headers.
	 * @return False if the LBVH build failed.
	 */
	bool buildMesh(int mesh);

	/**
	 * @brief Binds everything but the pair list and contact buffer and dispatches the kernel.
	 * @param narrowphase Shape registry owner.
	 * @param bodies Body buffer to read.
	 * @param pairs Pair buffer to read.
	 * @param contacts Append buffer at u0.
	 * @param resetContacts Zero the append counter after binding instead of appending.
	 * @param indirectArgs Dispatch args, or 0 to dispatch groups directly.
	 * @param groups Group count when indirectArgs is 0.
	 */
	void dispatch(const b3IrrlichtNarrowphase& narrowphase, irr::scene::IComputeBuffer* bodies,
				  irr::scene::IComputeBuffer* pairs, irr::scene::IComputeBuffer* contacts,
				  bool resetContacts, irr::scene::IComputeBuffer* indirectArgs, unsigned int groups);

	irr::video::IVideoDriver* m_driver;
	bool m_doubleSingle;
	int m_material;
	b3IrrlichtLbvh* m_builder;
	b3IrrGpu::DispatchHelper* m_dispatch;
	bool m_uploaded;

	std::vector<PendingMesh> m_pending;
	std::vector<b3IrrMeshHeader> m_headers;
	std::vector<b3IrrAabb> m_localAabbs;

	// Packed, all meshes. Vertices are 4 floats each (float4 stride), triangles 2 x 4 uints.
	std::vector<float> m_vertices;
	std::vector<unsigned int> m_triangles;
	std::vector<int> m_childNodes;
	std::vector<unsigned int> m_leafTriangles;
	std::vector<b3IrrAabb> m_aabbs;

	irr::scene::IComputeBuffer* m_paramBuffer;
	irr::scene::IComputeBuffer* m_headerBuffer;
	irr::scene::IComputeBuffer* m_vertexBuffer;
	irr::scene::IComputeBuffer* m_triangleBuffer;
	irr::scene::IComputeBuffer* m_childNodeBuffer;
	irr::scene::IComputeBuffer* m_leafTriangleBuffer;
	irr::scene::IComputeBuffer* m_aabbBuffer;

	// std::vector path only.
	irr::scene::IComputeBuffer* m_pairBuffer;
	irr::scene::IComputeBuffer* m_contactBuffer;
	irr::scene::IComputeBuffer* m_contactCountBuffer;
};

#endif  //B3_IRRLICHT_TRIANGLE_MESH_H
