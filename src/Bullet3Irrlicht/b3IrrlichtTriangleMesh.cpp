#include "b3IrrlichtTriangleMesh.h"
#include "b3IrrlichtNarrowphase.h"
#include "b3IrrlichtRadixSort.h"
#include "b3IrrlichtGpuBuffers.h"

#include <irrlicht.h>
#include "ComputeBuffer.h"

#include <algorithm>
#include <cstring>
#include <cmath>
#include <map>

namespace
{
struct MeshContactParams
{
	unsigned int numPairs;
	float collisionMargin;
	unsigned int pad0;
	unsigned int pad1;
};

struct IrrUint2
{
	unsigned int x;
	unsigned int y;
};

/// The buffer's element stride IS the shader's indexing unit: a float4 SRV must be uploaded from a
/// 16-byte-stride type, never a flat float array.
struct IrrFloat4
{
	float x, y, z, w;
};

struct IrrUint4
{
	unsigned int x, y, z, w;
};

static_assert(sizeof(MeshContactParams) == 16, "MeshContactParams must match the HLSL ContactParams stride");
static_assert(sizeof(b3IrrMeshHeader) == 32, "b3IrrMeshHeader must match the HLSL struct stride");
static_assert(sizeof(IrrFloat4) == 16, "IrrFloat4 must be a float4 stride");
static_assert(sizeof(IrrUint4) == 16, "IrrUint4 must be a uint4 stride");
static_assert(sizeof(b3Contact4Data) == 112, "b3Contact4Data must match the HLSL struct stride");

// Neighbours within this dihedral cosine continue the surface: ~1.8 degrees.
const float FLAT_EDGE_COS = 0.9995f;
const unsigned int NO_NEIGHBOUR = 0xffffffffu;

using b3IrrGpu::dropBuffer;
using b3IrrGpu::ensureBuffer;
using b3IrrGpu::uploadBuffer;

void sub3(const float* a, const float* b, float* out)
{
	out[0] = a[0] - b[0];
	out[1] = a[1] - b[1];
	out[2] = a[2] - b[2];
}

void cross3(const float* a, const float* b, float* out)
{
	out[0] = a[1] * b[2] - a[2] * b[1];
	out[1] = a[2] * b[0] - a[0] * b[2];
	out[2] = a[0] * b[1] - a[1] * b[0];
}

float dot3(const float* a, const float* b)
{
	return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}
}  // namespace

b3IrrlichtTriangleMesh::b3IrrlichtTriangleMesh(irr::video::IVideoDriver* driver)
	: m_driver(driver), m_doubleSingle(false), m_material(-1), m_builder(0), m_dispatch(0),
	  m_uploaded(false), m_paramBuffer(0), m_headerBuffer(0), m_vertexBuffer(0), m_triangleBuffer(0),
	  m_childNodeBuffer(0), m_leafTriangleBuffer(0), m_aabbBuffer(0), m_pairBuffer(0),
	  m_contactBuffer(0), m_contactCountBuffer(0)
{
}

b3IrrlichtTriangleMesh::~b3IrrlichtTriangleMesh()
{
	dropBuffer(m_paramBuffer);
	dropBuffer(m_headerBuffer);
	dropBuffer(m_vertexBuffer);
	dropBuffer(m_triangleBuffer);
	dropBuffer(m_childNodeBuffer);
	dropBuffer(m_leafTriangleBuffer);
	dropBuffer(m_aabbBuffer);
	dropBuffer(m_pairBuffer);
	dropBuffer(m_contactBuffer);
	dropBuffer(m_contactCountBuffer);
	delete m_builder;
	delete m_dispatch;
}

bool b3IrrlichtTriangleMesh::init(irr::io::IFileSystem* fileSystem, bool doubleSingle)
{
	m_doubleSingle = doubleSingle;
	if (!m_driver)
		return false;

	irr::video::IGPUProgrammingServices* gpu = m_driver->getGPUProgrammingServices();
	if (!gpu)
		return false;

	const irr::io::path path = m_doubleSingle ? "media/shaders/B3TriangleMeshContactsDS.hlsl"
											  : "media/shaders/B3TriangleMeshContacts.hlsl";
	if (fileSystem && !fileSystem->existFile(path))
		return false;

	m_material = gpu->addComputeShaderFromFile(path, "CSTriangleMeshContacts", irr::video::ECST_CS_5_0, 0);

	// Always the f32 tree builder: triangle AABBs are mesh-local, and the std::vector buildTree
	// entry point refuses in df64 mode.
	if (!m_builder)
		m_builder = new b3IrrlichtLbvh(m_driver);
	if (!m_builder->init(fileSystem, false))
		return false;

	if (!m_dispatch)
		m_dispatch = new b3IrrGpu::DispatchHelper(m_driver);
	m_dispatch->init(fileSystem);

	return m_material >= 0;
}

int b3IrrlichtTriangleMesh::registerTriangleMesh(const float* xyz, unsigned int numVertices,
												 const unsigned int* indices, unsigned int numTriangles,
												 unsigned int flags)
{
	if (!xyz || !indices || numVertices < 3 || numTriangles < 1)
		return -1;
	for (unsigned int i = 0; i < numTriangles * 3; ++i)
		if (indices[i] >= numVertices)
			return -1;

	PendingMesh pending;
	pending.firstVertex = (unsigned int)(m_vertices.size() / 4);
	pending.numVertices = numVertices;
	pending.firstTriangle = (unsigned int)(m_triangles.size() / 8);
	pending.numTriangles = numTriangles;
	pending.flags = flags;
	pending.built = false;

	b3IrrAabb local;
	for (int a = 0; a < 3; ++a)
	{
		local.minVec[a] = xyz[a];
		local.maxVec[a] = xyz[a];
	}
	local.minVec[3] = local.maxVec[3] = 0.f;

	for (unsigned int v = 0; v < numVertices; ++v)
	{
		for (int a = 0; a < 3; ++a)
		{
			const float c = xyz[v * 3 + a];
			m_vertices.push_back(c);
			if (c < local.minVec[a]) local.minVec[a] = c;
			if (c > local.maxVec[a]) local.maxVec[a] = c;
		}
		m_vertices.push_back(0.f);
	}

	// Face normals and centroids, for the edge classification below.
	std::vector<float> normals(numTriangles * 3, 0.f);
	std::vector<float> centroids(numTriangles * 3, 0.f);
	for (unsigned int t = 0; t < numTriangles; ++t)
	{
		const float* v0 = xyz + indices[t * 3 + 0] * 3;
		const float* v1 = xyz + indices[t * 3 + 1] * 3;
		const float* v2 = xyz + indices[t * 3 + 2] * 3;
		float e0[3], e1[3], n[3];
		sub3(v1, v0, e0);
		sub3(v2, v0, e1);
		cross3(e0, e1, n);
		const float len = std::sqrt(dot3(n, n));
		for (int a = 0; a < 3; ++a)
		{
			normals[t * 3 + a] = (len > 1e-12f) ? n[a] / len : 0.f;
			centroids[t * 3 + a] = (v0[a] + v1[a] + v2[a]) / 3.f;
		}
	}

	// Undirected edge -> triangles using it. Non-manifold edges keep the first other triangle.
	std::map<std::pair<unsigned int, unsigned int>, std::vector<unsigned int> > edgeUsers;
	for (unsigned int t = 0; t < numTriangles; ++t)
		for (int e = 0; e < 3; ++e)
		{
			unsigned int ia = indices[t * 3 + e];
			unsigned int ib = indices[t * 3 + (e + 1) % 3];
			if (ia > ib) std::swap(ia, ib);
			edgeUsers[std::make_pair(ia, ib)].push_back(t);
		}

	for (unsigned int t = 0; t < numTriangles; ++t)
	{
		unsigned int neighbour[3];
		unsigned int classBits = 0;
		for (int e = 0; e < 3; ++e)
		{
			unsigned int ia = indices[t * 3 + e];
			unsigned int ib = indices[t * 3 + (e + 1) % 3];
			const float* edgeStart = xyz + ia * 3;
			if (ia > ib) std::swap(ia, ib);
			const std::vector<unsigned int>& users = edgeUsers[std::make_pair(ia, ib)];

			neighbour[e] = NO_NEIGHBOUR;
			for (size_t u = 0; u < users.size(); ++u)
				if (users[u] != t)
				{
					neighbour[e] = users[u];
					break;
				}

			unsigned int cls = EDGE_BOUNDARY;
			if (neighbour[e] != NO_NEIGHBOUR)
			{
				const float* nt = &normals[t * 3];
				const float* nn = &normals[neighbour[e] * 3];
				float toNeighbour[3];
				sub3(&centroids[neighbour[e] * 3], edgeStart, toNeighbour);
				// fabs: an inconsistently wound neighbour still continues the same surface.
				if (std::fabs(dot3(nt, nn)) > FLAT_EDGE_COS)
					cls = EDGE_FLAT;
				else if (dot3(toNeighbour, nt) < 0.f)
					cls = EDGE_CONVEX;   // neighbour folds away below this face
				else
					cls = EDGE_CONCAVE;
			}
			classBits |= cls << (2 * e);
		}

		m_triangles.push_back(pending.firstVertex + indices[t * 3 + 0]);
		m_triangles.push_back(pending.firstVertex + indices[t * 3 + 1]);
		m_triangles.push_back(pending.firstVertex + indices[t * 3 + 2]);
		m_triangles.push_back(classBits);
		for (int e = 0; e < 3; ++e)
			m_triangles.push_back(neighbour[e] == NO_NEIGHBOUR ? NO_NEIGHBOUR
															   : pending.firstTriangle + neighbour[e]);
		m_triangles.push_back(0u);
	}

	b3IrrMeshHeader header;
	memset(&header, 0, sizeof(header));
	header.triangleOffset = pending.firstTriangle;
	header.numTriangles = numTriangles;
	header.flags = flags;

	m_pending.push_back(pending);
	m_headers.push_back(header);
	m_localAabbs.push_back(local);
	m_uploaded = false;
	return (int)m_headers.size() - 1;
}

unsigned int b3IrrlichtTriangleMesh::getEdgeClassBits(int mesh, unsigned int triangle) const
{
	if (mesh < 0 || mesh >= (int)m_pending.size() || triangle >= m_pending[mesh].numTriangles)
		return 0;
	return m_triangles[(m_pending[mesh].firstTriangle + triangle) * 8 + 3];
}

int b3IrrlichtTriangleMesh::registerCollidable(b3IrrlichtNarrowphase& narrowphase, int mesh) const
{
	if (mesh < 0 || mesh >= (int)m_headers.size())
		return -1;
	return narrowphase.registerShapeWithLocalAabb(m_localAabbs[mesh], SHAPE_CONCAVE_TRIMESH, mesh);
}

bool b3IrrlichtTriangleMesh::buildMesh(int mesh)
{
	PendingMesh& pending = m_pending[mesh];
	b3IrrMeshHeader& header = m_headers[mesh];

	// A Karras tree needs two leaves; a lone triangle is simply listed twice.
	const unsigned int numLeaves = pending.numTriangles < 2 ? 2 : pending.numTriangles;
	std::vector<unsigned int> leafToTriangle(numLeaves);
	std::vector<b3IrrAabb> triAabbs(numLeaves);
	for (unsigned int l = 0; l < numLeaves; ++l)
	{
		const unsigned int t = (l < pending.numTriangles) ? l : 0;
		leafToTriangle[l] = pending.firstTriangle + t;

		const unsigned int* rec = &m_triangles[(pending.firstTriangle + t) * 8];
		b3IrrAabb& box = triAabbs[l];
		for (int a = 0; a < 3; ++a)
		{
			box.minVec[a] = 1e30f;
			box.maxVec[a] = -1e30f;
		}
		box.minVec[3] = box.maxVec[3] = 0.f;
		for (int k = 0; k < 3; ++k)
		{
			const float* v = &m_vertices[rec[k] * 4];
			for (int a = 0; a < 3; ++a)
			{
				if (v[a] < box.minVec[a]) box.minVec[a] = v[a];
				if (v[a] > box.maxVec[a]) box.maxVec[a] = v[a];
			}
		}
	}

	b3IrrlichtLbvh::TreeResult tree;
	if (!m_builder || !m_builder->buildTree(triAabbs, tree))
		return false;

	const unsigned int numInternal = numLeaves - 1;
	if (tree.childNodes.size() != numInternal * 2 || tree.internalAabbs.size() != numInternal ||
		tree.sortedCodes.size() != numLeaves)
		return false;

	header.nodeOffset = (unsigned int)(m_childNodes.size() / 2);
	header.rootIndex = tree.rootIndex;
	header.leafOffset = (unsigned int)m_leafTriangles.size();
	header.aabbOffset = (unsigned int)m_aabbs.size();
	header.numLeaves = numLeaves;

	// Local child encoding is kept verbatim; the kernel adds nodeOffset/leafOffset itself.
	m_childNodes.insert(m_childNodes.end(), tree.childNodes.begin(), tree.childNodes.end());
	for (unsigned int i = 0; i < numInternal; ++i)
		m_aabbs.push_back(tree.internalAabbs[i]);
	for (unsigned int s = 0; s < numLeaves; ++s)
	{
		const unsigned int leaf = tree.sortedCodes[s].value;
		if (leaf >= numLeaves)
			return false;
		m_leafTriangles.push_back(leafToTriangle[leaf]);
		m_aabbs.push_back(triAabbs[leaf]);
	}

	pending.built = true;
	return true;
}

bool b3IrrlichtTriangleMesh::writeMeshesToGpu()
{
	if (m_headers.empty())
		return false;

	for (size_t m = 0; m < m_pending.size(); ++m)
		if (!m_pending[m].built && !buildMesh((int)m))
			return false;

	uploadBuffer<b3IrrMeshHeader>(m_headerBuffer, &m_headers[0], (irr::u32)m_headers.size());
	uploadBuffer<IrrFloat4>(m_vertexBuffer, (const IrrFloat4*)&m_vertices[0],
							(irr::u32)(m_vertices.size() / 4));
	uploadBuffer<IrrUint4>(m_triangleBuffer, (const IrrUint4*)&m_triangles[0],
						   (irr::u32)(m_triangles.size() / 4));
	uploadBuffer<int>(m_childNodeBuffer, &m_childNodes[0], (irr::u32)m_childNodes.size());
	uploadBuffer<unsigned int>(m_leafTriangleBuffer, &m_leafTriangles[0], (irr::u32)m_leafTriangles.size());
	uploadBuffer<b3IrrAabb>(m_aabbBuffer, &m_aabbs[0], (irr::u32)m_aabbs.size());

	m_uploaded = m_headerBuffer && m_vertexBuffer && m_triangleBuffer && m_childNodeBuffer &&
				 m_leafTriangleBuffer && m_aabbBuffer;
	return m_uploaded;
}

void b3IrrlichtTriangleMesh::dispatch(const b3IrrlichtNarrowphase& narrowphase,
									  irr::scene::IComputeBuffer* bodies,
									  irr::scene::IComputeBuffer* pairs,
									  irr::scene::IComputeBuffer* contacts, bool resetContacts,
									  irr::scene::IComputeBuffer* indirectArgs, unsigned int groups)
{
	irr::video::SMaterial mat;
	mat.MaterialType = (irr::video::E_MATERIAL_TYPE)m_material;
	m_driver->setMaterial(mat);
	m_driver->bindComputeBuffer(0, m_paramBuffer, irr::video::EHBT_SHADER_RESOURCE);
	m_driver->bindComputeBuffer(1, bodies, irr::video::EHBT_SHADER_RESOURCE);
	m_driver->bindComputeBuffer(2, narrowphase.getCollidableBuffer(), irr::video::EHBT_SHADER_RESOURCE);
	// Hull buffers are null while only spheres are registered; an unbound SRV reads as zero.
	if (narrowphase.getHullBuffer())
	{
		m_driver->bindComputeBuffer(3, narrowphase.getHullBuffer(), irr::video::EHBT_SHADER_RESOURCE);
		m_driver->bindComputeBuffer(4, narrowphase.getHullVertexBuffer(), irr::video::EHBT_SHADER_RESOURCE);
		m_driver->bindComputeBuffer(5, narrowphase.getHullFaceBuffer(), irr::video::EHBT_SHADER_RESOURCE);
		m_driver->bindComputeBuffer(6, narrowphase.getHullEdgeBuffer(), irr::video::EHBT_SHADER_RESOURCE);
		m_driver->bindComputeBuffer(8, narrowphase.getHullFaceIndexBuffer(), irr::video::EHBT_SHADER_RESOURCE);
	}
	m_driver->bindComputeBuffer(7, pairs, irr::video::EHBT_SHADER_RESOURCE);
	m_driver->bindComputeBuffer(9, m_headerBuffer, irr::video::EHBT_SHADER_RESOURCE);
	m_driver->bindComputeBuffer(10, m_vertexBuffer, irr::video::EHBT_SHADER_RESOURCE);
	m_driver->bindComputeBuffer(11, m_triangleBuffer, irr::video::EHBT_SHADER_RESOURCE);
	m_driver->bindComputeBuffer(12, m_childNodeBuffer, irr::video::EHBT_SHADER_RESOURCE);
	m_driver->bindComputeBuffer(13, m_leafTriangleBuffer, irr::video::EHBT_SHADER_RESOURCE);
	m_driver->bindComputeBuffer(14, m_aabbBuffer, irr::video::EHBT_SHADER_RESOURCE);
	m_driver->bindComputeBuffer(0, contacts, irr::video::EHBT_COMPUTE);
	// Only valid once the buffer is bound; an unreset bind keeps the counter (append mode).
	if (resetContacts)
		m_driver->resetStructureCount(contacts, 0);
	if (indirectArgs)
		m_driver->dispatchComputeShaderIndirect(indirectArgs, 0);
	else
		m_driver->dispatchComputeShaderBound(irr::core::vector3d<irr::u32>(groups, 1, 1));
	m_driver->unbindComputeResources();
}

bool b3IrrlichtTriangleMesh::appendContactsResident(const b3IrrlichtNarrowphase& narrowphase,
													irr::scene::IComputeBuffer* bodies,
													unsigned int numBodies,
													irr::scene::IComputeBuffer* pairs,
													irr::scene::IComputeBuffer* pairCount,
													unsigned int maxPairs, float collisionMargin,
													bool resetContacts)
{
	if (m_material < 0 || !m_uploaded || !bodies || !pairs || !pairCount || numBodies == 0)
		return false;
	if (!m_dispatch || !m_dispatch->isAvailable())
		return false;

	irr::scene::IComputeBuffer* contacts = narrowphase.getContactBuffer();
	irr::scene::IComputeBuffer* contactCount = narrowphase.getContactCountBuffer();
	if (!contacts || !contactCount || !narrowphase.getCollidableBuffer())
		return false;

	ensureBuffer<MeshContactParams>(m_paramBuffer, 1);
	MeshContactParams p;
	p.numPairs = 0;   // patched on the device from the broadphase counter
	p.collisionMargin = collisionMargin;
	p.pad0 = p.pad1 = 0;
	memcpy(m_paramBuffer->getBufferPointer(), &p, sizeof(p));
	m_paramBuffer->setDirty();

	if (!m_dispatch->prepareIndirect(pairCount, m_paramBuffer, 64, maxPairs))
		return false;

	// The counter is left alone by default: the convex stage's manifolds are already in the list.
	dispatch(narrowphase, bodies, pairs, contacts, resetContacts, m_dispatch->getArgsBuffer(), 0);
	m_driver->copyStructureCount(contactCount, 0, contacts);
	m_driver->computeBarrier(contacts);
	return true;
}

bool b3IrrlichtTriangleMesh::computeContacts(b3IrrlichtNarrowphase& narrowphase,
											 const std::vector<b3RigidBodyData>& bodies,
											 const std::vector<std::pair<unsigned int, unsigned int> >& pairs,
											 std::vector<b3Contact4Data>& contacts, unsigned int maxContacts,
											 float collisionMargin, bool* overflowed)
{
	if (overflowed)
		*overflowed = false;
	contacts.clear();

	if (m_material < 0 || !m_uploaded || m_doubleSingle || bodies.empty())
		return false;
	if (!narrowphase.getCollidableBuffer())
		return false;
	if (pairs.empty())
		return true;

	if (!narrowphase.uploadBodiesResident(bodies))
		return false;

	ensureBuffer<IrrUint2>(m_pairBuffer, (irr::u32)pairs.size());
	IrrUint2* dst = (IrrUint2*)m_pairBuffer->getBufferPointer();
	for (size_t i = 0; i < pairs.size(); ++i)
	{
		dst[i].x = pairs[i].first;
		dst[i].y = pairs[i].second;
	}
	m_pairBuffer->setDirty();

	ensureBuffer<b3Contact4Data>(m_contactBuffer, maxContacts, irr::video::EHBF_COMPUTE_APPEND);
	ensureBuffer<unsigned int>(m_contactCountBuffer, 4, irr::video::EHBF_DRAW_INDIRECT_ARGS);
	ensureBuffer<MeshContactParams>(m_paramBuffer, 1);

	MeshContactParams p;
	p.numPairs = (unsigned int)pairs.size();
	p.collisionMargin = collisionMargin;
	p.pad0 = p.pad1 = 0;
	memcpy(m_paramBuffer->getBufferPointer(), &p, sizeof(p));
	m_paramBuffer->setDirty();

	dispatch(narrowphase, narrowphase.getBodyBuffer(), m_pairBuffer, m_contactBuffer, true, 0,
			 ((unsigned int)pairs.size() + 63) / 64);
	m_driver->copyStructureCount(m_contactCountBuffer, 0, m_contactBuffer);

	m_contactCountBuffer->downloadFromGPU();
	unsigned int count = 0;
	memcpy(&count, m_contactCountBuffer->getBufferPointer(), sizeof(unsigned int));
	if (count > maxContacts)
	{
		// Appends past capacity are dropped but still advance the counter.
		if (overflowed)
			*overflowed = true;
		count = maxContacts;
	}
	if (count == 0)
		return true;

	m_contactBuffer->downloadFromGPU();
	contacts.resize(count);
	memcpy(&contacts[0], m_contactBuffer->getBufferPointer(), count * sizeof(b3Contact4Data));
	return true;
}
