#include "b3IrrlichtNarrowphase.h"
#include "b3IrrlichtGpuBuffers.h"

#include <irrlicht.h>
#include "ComputeBuffer.h"

#include <algorithm>
#include <cfloat>
#include <cstring>
#include <cmath>

namespace
{
struct ExpandParams
{
	unsigned int numPairs;
	unsigned int pad0;
	unsigned int pad1;
	unsigned int pad2;
};

/// Leaf pair (bodyA, bodyB, childA, childB), layout-identical to the HLSL int4 / b3CompoundOverlappingPair.
struct IrrInt4
{
	int x, y, z, w;
};

struct AabbParams
{
	unsigned int numBodies;
	float margin;
	unsigned int useSleep;
	float sweepDt;   // > 0 grows each bound to cover the step's motion
};

/// Mirrors ContactParams / ClipParams in the contact kernels.
struct ContactParams
{
	unsigned int numPairs;
	float collisionMargin;
	float speculativeDt;
	float ccdMotionThreshold;
};

struct IrrUint2
{
	unsigned int x;
	unsigned int y;
};

struct SatParams
{
	unsigned int numPairs;
	unsigned int pad0;
	unsigned int pad1;
	unsigned int pad2;
};

/// The buffer's element stride IS the shader's indexing unit, so a float4 SRV must be uploaded
/// from a 16-byte-stride buffer - a flat float buffer makes Vertices[i] read floats[i..i+3].
struct IrrFloat4
{
	float x, y, z, w;
};
}  // namespace

/// Pushes noise2.hlsl's five `params` constants for the planet kernel.
class PlanetNoiseParams : public irr::video::IShaderConstantSetCallBack
{
public:
	float mTimer, plates, rivers, atmosphereDensity, texsize;

	PlanetNoiseParams()
		: mTimer(0.f), plates(1.f), rivers(1.f), atmosphereDensity(1.4f), texsize(512.f)
	{
	}

	virtual void OnSetConstants(irr::video::IMaterialRendererServices* services, irr::s32 userData)
	{
		set(services, "mTimer", mTimer);
		set(services, "Plates", plates);
		set(services, "rivers", rivers);
		set(services, "AtmosphereDensity", atmosphereDensity);
		set(services, "texsize", texsize);
	}

private:
	static void set(irr::video::IMaterialRendererServices* services, const char* name, float& value)
	{
		const irr::s32 id = services->getComputeShaderConstantID(name);
		if (id >= 0)
			services->setComputeShaderConstant(id, &value, 1);
	}
};

namespace
{

static_assert(sizeof(AabbParams) == 16, "AabbParams must match the HLSL struct stride");
static_assert(sizeof(ContactParams) == 16, "ContactParams must match the HLSL struct stride");
static_assert(sizeof(b3Collidable) == 16, "b3Collidable must match the HLSL struct stride");
static_assert(sizeof(b3RigidBodyData) == 80, "b3RigidBodyData must match the HLSL struct stride");
static_assert(sizeof(b3IrrGpu::b3IrrRigidBodyDataDS) == 96, "b3IrrGpu::b3IrrRigidBodyDataDS must match the OS_DS HLSL stride");
static_assert(sizeof(b3Contact4Data) == 112, "b3Contact4Data must match the HLSL struct stride");
static_assert(sizeof(SatParams) == 16, "SatParams must match the HLSL struct stride");
static_assert(sizeof(b3IrrSatResult) == 32, "b3IrrSatResult must match the HLSL SatResult stride");
static_assert(sizeof(b3ConvexPolyhedronData) == 96, "b3ConvexPolyhedronData must match the HLSL struct stride");
static_assert(sizeof(b3GpuFace) == 32, "b3GpuFace must match the HLSL struct stride");
static_assert(sizeof(b3IrrlichtNarrowphase::b3IrrPlanet) == 32, "b3IrrPlanet must match the HLSL PlanetData stride");
static_assert(sizeof(ExpandParams) == 16, "ExpandParams must match the HLSL struct stride");
static_assert(sizeof(IrrInt4) == 16, "IrrInt4 must match the HLSL int4 stride");
static_assert(sizeof(b3GpuChildShape) == 48, "b3GpuChildShape must match the HLSL struct stride");

using b3IrrGpu::dropBuffer;
using b3IrrGpu::ensureBuffer;
using b3IrrGpu::uploadBuffer;

/// Largest face the clip kernel can hold - its MAX_POLY, which is also its unroll factor.
const int kMaxFaceVertices = 8;

/// One hull face: vertex indices ordered around the face, plus its outward plane n.p = d.
struct HullFace
{
	std::vector<int> indices;
	b3Vector3 normal;
	float d;
};

/// Faces of the convex hull of pts: every plane through 3 points with all points behind it,
/// deduplicated, then its coplanar points sorted by angle. O(n^4) - registration time only.
bool buildConvexHullFaces(const std::vector<b3Vector3>& pts, std::vector<HullFace>& faces)
{
	const int n = (int)pts.size();
	if (n < 4)
		return false;

	b3Vector3 centroid = b3MakeVector3(0.f, 0.f, 0.f);
	for (int i = 0; i < n; ++i)
		centroid += pts[i];
	centroid /= (float)n;

	float extent = 0.f;
	for (int i = 0; i < n; ++i)
		extent = (std::max)(extent, (float)(pts[i] - centroid).length());
	if (extent <= 0.f)
		return false;
	const float tol = 1e-5f * (std::max)(extent, 1.f);

	std::vector<HullFace> planes;
	for (int i = 0; i < n; ++i)
		for (int j = i + 1; j < n; ++j)
			for (int k = j + 1; k < n; ++k)
			{
				b3Vector3 nrm = (pts[j] - pts[i]).cross(pts[k] - pts[i]);
				const float len = (float)nrm.length();
				if (len <= tol * extent)
					continue;
				nrm /= len;
				float d = (float)nrm.dot(pts[i]);

				// Orient outward; a centroid on the plane means the cloud is flat.
				const float side = (float)nrm.dot(centroid) - d;
				if (std::fabs(side) < tol)
					continue;
				if (side > 0.f)
				{
					nrm = -nrm;
					d = -d;
				}

				bool duplicate = false;
				for (size_t p = 0; p < planes.size() && !duplicate; ++p)
					duplicate = planes[p].normal.dot(nrm) > 1.f - 1e-5f && std::fabs(planes[p].d - d) < tol;
				if (duplicate)
					continue;

				bool supporting = true;
				for (int m = 0; m < n && supporting; ++m)
					supporting = (float)nrm.dot(pts[m]) - d <= tol;
				if (!supporting)
					continue;

				HullFace f;
				f.normal = nrm;
				f.d = d;
				planes.push_back(f);
			}

	if (planes.size() < 4)
		return false;

	for (size_t p = 0; p < planes.size(); ++p)
	{
		HullFace& f = planes[p];
		b3Vector3 faceCenter = b3MakeVector3(0.f, 0.f, 0.f);
		for (int m = 0; m < n; ++m)
			if (std::fabs((float)f.normal.dot(pts[m]) - f.d) <= tol)
			{
				f.indices.push_back(m);
				faceCenter += pts[m];
			}
		if (f.indices.size() < 3)
			continue;
		faceCenter /= (float)f.indices.size();

		b3Vector3 u = std::fabs((float)f.normal.getX()) < 0.9f ? b3MakeVector3(1.f, 0.f, 0.f)
																: b3MakeVector3(0.f, 1.f, 0.f);
		u = u - f.normal * u.dot(f.normal);
		u.normalize();
		const b3Vector3 v = f.normal.cross(u);

		struct AngleLess
		{
			const std::vector<b3Vector3>* pts;
			b3Vector3 c, u, v;
			float angle(int m) const
			{
				const b3Vector3 r = (*pts)[m] - c;
				return std::atan2((float)r.dot(v), (float)r.dot(u));
			}
			bool operator()(int a, int b) const { return angle(a) < angle(b); }
		};
		AngleLess less = {&pts, faceCenter, u, v};
		std::sort(f.indices.begin(), f.indices.end(), less);
		faces.push_back(f);
	}

	return faces.size() >= 4;
}

/// Fills the pair buffer from the caller's std::vector, growing it in place.
irr::scene::IComputeBuffer* uploadPairs(irr::scene::IComputeBuffer*& buffer,
										const std::vector<std::pair<unsigned int, unsigned int> >& pairs)
{
	ensureBuffer<IrrUint2>(buffer, (irr::u32)pairs.size());
	IrrUint2* dst = (IrrUint2*)buffer->getBufferPointer();
	for (size_t i = 0; i < pairs.size(); ++i)
	{
		dst[i].x = pairs[i].first;
		dst[i].y = pairs[i].second;
	}
	buffer->setDirty();
	return buffer;
}
}  // namespace

b3IrrlichtNarrowphase::b3IrrlichtNarrowphase(irr::video::IVideoDriver* driver)
	: m_driver(driver), m_doubleSingle(false), m_dispatch(0), m_aabbMaterial(-1), m_sphereContactMaterial(-1), m_satMaterial(-1),
	  m_clipMaterial(-1), m_planetMaterial(-1), m_expandMaterial(-1), m_clipLeafMaterial(-1),
	  m_maxLeafPairs(65536),
	  m_expandParamBuffer(0), m_childShapeBuffer(0), m_leafPairBuffer(0), m_leafPairCountBuffer(0),
	  m_aabbParamBuffer(0), m_contactParamBuffer(0), m_satParamBuffer(0), m_hullBuffer(0),
	  m_vertexBuffer(0), m_faceBuffer(0), m_indexBuffer(0), m_edgeBuffer(0), m_satResultBuffer(0),
	  m_planetBuffer(0), m_planetNoiseParams(0),
	  m_bodyBuffer(0), m_collidableBuffer(0),
	  m_localAabbBuffer(0), m_worldAabbBuffer(0), m_pairBuffer(0), m_contactBuffer(0),
	  m_contactCountBuffer(0), m_rollingFrictionBuffer(0), m_speculativeDt(0.f),
	  m_ccdMotionThreshold(0.f)
{
}

b3IrrlichtNarrowphase::~b3IrrlichtNarrowphase()
{
	dropBuffer(m_aabbParamBuffer);
	dropBuffer(m_contactParamBuffer);
	dropBuffer(m_satParamBuffer);
	dropBuffer(m_hullBuffer);
	dropBuffer(m_vertexBuffer);
	dropBuffer(m_faceBuffer);
	dropBuffer(m_indexBuffer);
	dropBuffer(m_edgeBuffer);
	dropBuffer(m_satResultBuffer);
	dropBuffer(m_planetBuffer);
	dropBuffer(m_bodyBuffer);
	dropBuffer(m_collidableBuffer);
	dropBuffer(m_localAabbBuffer);
	dropBuffer(m_worldAabbBuffer);
	dropBuffer(m_pairBuffer);
	dropBuffer(m_contactBuffer);
	dropBuffer(m_contactCountBuffer);
	dropBuffer(m_expandParamBuffer);
	dropBuffer(m_childShapeBuffer);
	dropBuffer(m_leafPairBuffer);
	dropBuffer(m_leafPairCountBuffer);
	dropBuffer(m_rollingFrictionBuffer);
	delete m_dispatch;
}

void b3IrrlichtNarrowphase::setRollingFriction(int collidableIdx, float coefficient)
{
	if (collidableIdx < 0 || collidableIdx >= (int)m_cpuCollidables.size())
		return;
	if (m_cpuRollingFriction.size() < m_cpuCollidables.size())
		m_cpuRollingFriction.resize(m_cpuCollidables.size(), 0.f);
	m_cpuRollingFriction[collidableIdx] = coefficient > 0.f ? coefficient : 0.f;
}

bool b3IrrlichtNarrowphase::init(irr::io::IFileSystem* fileSystem, bool doubleSingle)
{
	m_doubleSingle = doubleSingle;
	if (!m_driver)
		return false;

	irr::video::IGPUProgrammingServices* gpu = m_driver->getGPUProgrammingServices();
	if (!gpu)
		return false;

	const irr::io::path aabbPath = m_doubleSingle ? "media/shaders/B3NarrowphaseDS.hlsl" : "media/shaders/B3Narrowphase.hlsl";
	const irr::io::path contactPath = "media/shaders/B3NarrowphaseContacts.hlsl";
	if (fileSystem && (!fileSystem->existFile(aabbPath) || !fileSystem->existFile(contactPath)))
		return false;

	const irr::io::path satPath = "media/shaders/B3NarrowphaseSat.hlsl";
	if (fileSystem && !fileSystem->existFile(satPath))
		return false;

	m_aabbMaterial = gpu->addComputeShaderFromFile(aabbPath, "CSComputeWorldAabbs", irr::video::ECST_CS_5_0, 0);
	m_sphereContactMaterial = gpu->addComputeShaderFromFile(contactPath, "CSSphereSphereContacts", irr::video::ECST_CS_5_0, 0);
	const irr::io::path clipPath = m_doubleSingle ? "media/shaders/B3NarrowphaseClipDS.hlsl" : "media/shaders/B3NarrowphaseClip.hlsl";
	if (fileSystem && !fileSystem->existFile(clipPath))
		return false;

	m_satMaterial = gpu->addComputeShaderFromFile(satPath, "CSFindSeparatingAxis", irr::video::ECST_CS_5_0, 0);
	m_clipMaterial = gpu->addComputeShaderFromFile(clipPath, "CSClipContacts", irr::video::ECST_CS_5_0, 0);

	// Leaf-pair kernels are optional: only compound (and later mesh) contacts need them.
	const irr::io::path expandPath = m_doubleSingle ? "media/shaders/B3NarrowphasePairExpandDS.hlsl" : "media/shaders/B3NarrowphasePairExpand.hlsl";
	const irr::io::path clipLeafPath = m_doubleSingle ? "media/shaders/B3NarrowphaseClipLeafDS.hlsl" : "media/shaders/B3NarrowphaseClipLeaf.hlsl";
	if (!fileSystem || (fileSystem->existFile(expandPath) && fileSystem->existFile(clipLeafPath)))
	{
		m_expandMaterial = gpu->addComputeShaderFromFile(expandPath, "CSExpandPairs", irr::video::ECST_CS_5_0, 0);
		m_clipLeafMaterial = gpu->addComputeShaderFromFile(clipLeafPath, "CSClipContacts", irr::video::ECST_CS_5_0, 0);
	}

	// PlanetShape is OPTIONAL: it #includes noise2.hlsl, whose FBM loops fail to unroll under
	// Release shader flags (X3511). A caller that only collides boxes must not lose the whole
	// pipeline because a kernel it never calls would not build.
	const irr::io::path planetPath = "media/shaders/B3PlanetContacts.hlsl";
	if (!fileSystem || fileSystem->existFile(planetPath))
	{
		m_planetNoiseParams = new PlanetNoiseParams();
		m_planetMaterial = gpu->addComputeShaderFromFile(planetPath, "CSPlanetSphereContacts",
														 irr::video::ECST_CS_5_0, m_planetNoiseParams);
		m_planetNoiseParams->drop();
		if (m_planetMaterial < 0)
			m_planetNoiseParams = 0;
	}

	if (!m_dispatch)
		m_dispatch = new b3IrrGpu::DispatchHelper(m_driver);
	m_dispatch->init(fileSystem);

	return m_aabbMaterial >= 0 && m_sphereContactMaterial >= 0 && m_satMaterial >= 0 &&
		   m_clipMaterial >= 0;
}

bool b3IrrlichtNarrowphase::isResidentPathAvailable() const
{
	return m_dispatch && m_dispatch->isAvailable() && m_aabbMaterial >= 0 && m_clipMaterial >= 0;
}

int b3IrrlichtNarrowphase::registerSphereShape(float radius)
{
	b3Collidable col;
	memset(&col, 0, sizeof(col));
	col.m_shapeType = SHAPE_SPHERE;
	col.m_radius = radius;
	col.m_shapeIndex = -1;
	m_cpuCollidables.push_back(col);

	// Never read for a sphere, but the buffer is indexed by collidable index, so it must stay
	// parallel to m_cpuCollidables rather than only holding the non-sphere entries.
	b3IrrAabb local;
	for (int i = 0; i < 3; ++i)
	{
		local.minVec[i] = -radius;
		local.maxVec[i] = radius;
	}
	local.minVec[3] = local.maxVec[3] = 0.f;
	m_cpuLocalAabbs.push_back(local);

	return (int)m_cpuCollidables.size() - 1;
}

int b3IrrlichtNarrowphase::registerShapeWithLocalAabb(const b3IrrAabb& localAabb, int shapeType,
													  int shapeIndex)
{
	b3Collidable col;
	memset(&col, 0, sizeof(col));
	col.m_shapeType = shapeType;
	col.m_shapeIndex = shapeIndex;
	m_cpuCollidables.push_back(col);
	m_cpuLocalAabbs.push_back(localAabb);

	return (int)m_cpuCollidables.size() - 1;
}

int b3IrrlichtNarrowphase::registerBoxShape(const float halfExtents[3])
{
	const int hullIndex = (int)m_cpuHulls.size();
	const int vertexOffset = (int)(m_cpuVertices.size() / 4);
	const int faceOffset = (int)m_cpuFaces.size();
	const int edgeOffset = (int)(m_cpuEdges.size() / 4);

	for (int c = 0; c < 8; ++c)
	{
		m_cpuVertices.push_back((c & 1) ? halfExtents[0] : -halfExtents[0]);
		m_cpuVertices.push_back((c & 2) ? halfExtents[1] : -halfExtents[1]);
		m_cpuVertices.push_back((c & 4) ? halfExtents[2] : -halfExtents[2]);
		m_cpuVertices.push_back(0.f);
	}

	// Vertex c encodes sign per axis in its bits, so a face is the 4 vertices whose bit for this
	// axis matches the face's side. Ordered around the quad, not by index - clipping walks edges.
	for (int axis = 0; axis < 3; ++axis)
	{
		for (int sign = 0; sign <= 1; ++sign)
		{
			const int bit = 1 << axis;
			const int u = (axis + 1) % 3;
			const int v = (axis + 2) % 3;
			const int ubit = 1 << u;
			const int vbit = 1 << v;
			const int base = sign ? bit : 0;

			const int indexOffset = (int)m_cpuIndices.size();
			m_cpuIndices.push_back(base);
			m_cpuIndices.push_back(base | ubit);
			m_cpuIndices.push_back(base | ubit | vbit);
			m_cpuIndices.push_back(base | vbit);

			b3GpuFace face;
			memset(&face, 0, sizeof(face));
			float n[3] = {0.f, 0.f, 0.f};
			n[axis] = sign ? 1.f : -1.f;
			face.m_plane = b3MakeVector3(n[0], n[1], n[2]);
			face.m_plane.w = -halfExtents[axis];
			face.m_indexOffset = indexOffset;
			face.m_numIndices = 4;
			m_cpuFaces.push_back(face);
		}
	}

	// A box has 3 unique edge DIRECTIONS, not 12 edges - the other 9 are parallel duplicates and
	// would only produce degenerate cross products.
	for (int axis = 0; axis < 3; ++axis)
	{
		m_cpuEdges.push_back(axis == 0 ? 1.f : 0.f);
		m_cpuEdges.push_back(axis == 1 ? 1.f : 0.f);
		m_cpuEdges.push_back(axis == 2 ? 1.f : 0.f);
		m_cpuEdges.push_back(0.f);
	}

	b3ConvexPolyhedronData hull;
	memset(&hull, 0, sizeof(hull));
	hull.m_localCenter = b3MakeVector3(0.f, 0.f, 0.f);
	hull.m_extents = b3MakeVector3(halfExtents[0], halfExtents[1], halfExtents[2]);
	hull.m_radius = std::sqrt(halfExtents[0] * halfExtents[0] + halfExtents[1] * halfExtents[1] +
							  halfExtents[2] * halfExtents[2]);
	hull.m_faceOffset = faceOffset;
	hull.m_numFaces = 6;
	hull.m_numVertices = 8;
	hull.m_vertexOffset = vertexOffset;
	hull.m_uniqueEdgesOffset = edgeOffset;
	hull.m_numUniqueEdges = 3;
	m_cpuHulls.push_back(hull);

	b3IrrAabb local;
	for (int i = 0; i < 3; ++i)
	{
		local.minVec[i] = -halfExtents[i];
		local.maxVec[i] = halfExtents[i];
	}
	local.minVec[3] = local.maxVec[3] = 0.f;

	return registerShapeWithLocalAabb(local, SHAPE_CONVEX_HULL, hullIndex);
}

int b3IrrlichtNarrowphase::registerConvexHullShape(const float* xyz, int numVertices)
{
	if (!xyz || numVertices < 4)
		return -1;

	std::vector<b3Vector3> pts((size_t)numVertices);
	for (int i = 0; i < numVertices; ++i)
		pts[i] = b3MakeVector3(xyz[i * 3 + 0], xyz[i * 3 + 1], xyz[i * 3 + 2]);

	std::vector<HullFace> faces;
	if (!buildConvexHullFaces(pts, faces))
		return -1;

	const int hullIndex = (int)m_cpuHulls.size();
	const int vertexOffset = (int)(m_cpuVertices.size() / 4);
	const int faceOffset = (int)m_cpuFaces.size();
	const int edgeOffset = (int)(m_cpuEdges.size() / 4);

	b3Vector3 centroid = b3MakeVector3(0.f, 0.f, 0.f);
	b3IrrAabb local;
	for (int i = 0; i < 3; ++i)
	{
		local.minVec[i] = FLT_MAX;
		local.maxVec[i] = -FLT_MAX;
	}
	local.minVec[3] = local.maxVec[3] = 0.f;
	float radius = 0.f;
	for (int i = 0; i < numVertices; ++i)
	{
		for (int c = 0; c < 3; ++c)
		{
			m_cpuVertices.push_back((float)pts[i][c]);
			local.minVec[c] = (std::min)(local.minVec[c], (float)pts[i][c]);
			local.maxVec[c] = (std::max)(local.maxVec[c], (float)pts[i][c]);
		}
		m_cpuVertices.push_back(0.f);
		centroid += pts[i];
		radius = (std::max)(radius, (float)pts[i].length());
	}
	centroid /= (float)numVertices;

	int numFaces = 0;
	int numEdges = 0;
	for (size_t f = 0; f < faces.size(); ++f)
	{
		const HullFace& face = faces[f];
		const int n = (int)face.indices.size();

		// A face wider than the clip kernel's MAX_POLY is split into coplanar fans sharing
		// vertex 0 and one vertex with the next chunk. Same normal, so SAT is unaffected.
		int start = 1;
		do
		{
			const int indexOffset = (int)m_cpuIndices.size();
			m_cpuIndices.push_back(face.indices[0]);
			int count = 1;
			for (int k = start; k < n && count < kMaxFaceVertices; ++k, ++count)
				m_cpuIndices.push_back(face.indices[k]);

			b3GpuFace gf;
			memset(&gf, 0, sizeof(gf));
			gf.m_plane = face.normal;
			gf.m_plane.w = -face.d;
			gf.m_indexOffset = indexOffset;
			gf.m_numIndices = count;
			m_cpuFaces.push_back(gf);
			++numFaces;

			start += kMaxFaceVertices - 2;
		} while (start < n - 1);

		// Edge directions from the original polygon; parallel ones only give degenerate cross products.
		for (int k = 0; k < n; ++k)
		{
			b3Vector3 dir = pts[face.indices[(k + 1) % n]] - pts[face.indices[k]];
			const float len = (float)dir.length();
			if (len <= 1e-7f)
				continue;
			dir /= len;

			bool duplicate = false;
			for (int e = edgeOffset; e < edgeOffset + numEdges && !duplicate; ++e)
			{
				const float d = m_cpuEdges[e * 4 + 0] * (float)dir.getX() + m_cpuEdges[e * 4 + 1] * (float)dir.getY()
							  + m_cpuEdges[e * 4 + 2] * (float)dir.getZ();
				duplicate = std::fabs(d) > 1.f - 1e-6f;
			}
			if (duplicate)
				continue;

			m_cpuEdges.push_back((float)dir.getX());
			m_cpuEdges.push_back((float)dir.getY());
			m_cpuEdges.push_back((float)dir.getZ());
			m_cpuEdges.push_back(0.f);
			++numEdges;
		}
	}

	b3ConvexPolyhedronData hull;
	memset(&hull, 0, sizeof(hull));
	hull.m_localCenter = centroid;
	hull.m_extents = b3MakeVector3(0.5f * (local.maxVec[0] - local.minVec[0]),
								   0.5f * (local.maxVec[1] - local.minVec[1]),
								   0.5f * (local.maxVec[2] - local.minVec[2]));
	hull.m_radius = radius;
	hull.m_faceOffset = faceOffset;
	hull.m_numFaces = numFaces;
	hull.m_numVertices = numVertices;
	hull.m_vertexOffset = vertexOffset;
	hull.m_uniqueEdgesOffset = edgeOffset;
	hull.m_numUniqueEdges = numEdges;
	m_cpuHulls.push_back(hull);

	return registerShapeWithLocalAabb(local, SHAPE_CONVEX_HULL, hullIndex);
}

int b3IrrlichtNarrowphase::registerCylinderShape(float radius, float halfHeight)
{
	const int segments = 8;
	std::vector<float> xyz;
	for (int s = 0; s < segments; ++s)
	{
		const float a = 6.2831853f * (float)s / (float)segments;
		const float x = radius * std::cos(a);
		const float z = radius * std::sin(a);
		for (int end = -1; end <= 1; end += 2)
		{
			xyz.push_back(x);
			xyz.push_back((float)end * halfHeight);
			xyz.push_back(z);
		}
	}
	return registerConvexHullShape(&xyz[0], (int)(xyz.size() / 3));
}

int b3IrrlichtNarrowphase::registerCapsuleShape(float radius, float halfHeight)
{
	const int segments = 12;
	const int rings = 2;   // latitude rings per end, beyond the equator ring at +-halfHeight
	std::vector<float> xyz;
	for (int end = -1; end <= 1; end += 2)
	{
		for (int ring = 0; ring <= rings; ++ring)
		{
			const float lat = 1.5707963f * (float)ring / (float)(rings + 1);
			const float rr = radius * std::cos(lat);
			const float y = (float)end * (halfHeight + radius * std::sin(lat));
			for (int s = 0; s < segments; ++s)
			{
				const float a = 6.2831853f * (float)s / (float)segments;
				xyz.push_back(rr * std::cos(a));
				xyz.push_back(y);
				xyz.push_back(rr * std::sin(a));
			}
		}
		xyz.push_back(0.f);
		xyz.push_back((float)end * (halfHeight + radius));
		xyz.push_back(0.f);
	}
	return registerConvexHullShape(&xyz[0], (int)(xyz.size() / 3));
}

int b3IrrlichtNarrowphase::registerCompoundShape(const b3IrrCompoundChild* children, int numChildren)
{
	if (!children || numChildren <= 0)
		return -1;

	for (int c = 0; c < numChildren; ++c)
	{
		const int ci = children[c].collidableIndex;
		if (ci < 0 || ci >= (int)m_cpuCollidables.size())
			return -1;
		if (m_cpuCollidables[ci].m_shapeType == SHAPE_COMPOUND_OF_CONVEX_HULLS)
			return -1;
	}

	const int childOffset = (int)m_cpuChildShapes.size();

	b3IrrAabb merged;
	for (int i = 0; i < 3; ++i)
	{
		merged.minVec[i] = FLT_MAX;
		merged.maxVec[i] = -FLT_MAX;
	}
	merged.minVec[3] = merged.maxVec[3] = 0.f;

	for (int c = 0; c < numChildren; ++c)
	{
		const b3IrrCompoundChild& in = children[c];
		const b3Collidable& col = m_cpuCollidables[in.collidableIndex];
		const b3Quaternion q(in.orientation[0], in.orientation[1], in.orientation[2], in.orientation[3]);
		const b3Vector3 pos = b3MakeVector3(in.position[0], in.position[1], in.position[2]);

		b3GpuChildShape cs;
		memset(&cs, 0, sizeof(cs));
		cs.m_childPosition = pos;
		cs.m_childOrientation = q;
		cs.m_shapeIndex = col.m_shapeIndex;
		cs.m_numChildShapes = 0;
		cs.m_collidableShapeIndex = in.collidableIndex;
		cs.m_shapeType = col.m_shapeType;
		m_cpuChildShapes.push_back(cs);

		// Same abs-matrix bound the world-AABB kernel applies, so the parent bound encloses
		// whatever that kernel would have produced per child.
		const b3IrrAabb& la = m_cpuLocalAabbs[in.collidableIndex];
		const b3Vector3 center = b3MakeVector3(0.5f * (la.maxVec[0] + la.minVec[0]),
											   0.5f * (la.maxVec[1] + la.minVec[1]),
											   0.5f * (la.maxVec[2] + la.minVec[2]));
		const b3Vector3 ext = b3MakeVector3(0.5f * (la.maxVec[0] - la.minVec[0]),
											0.5f * (la.maxVec[1] - la.minVec[1]),
											0.5f * (la.maxVec[2] - la.minVec[2]));
		const b3Vector3 bx = b3QuatRotate(q, b3MakeVector3(1.f, 0.f, 0.f));
		const b3Vector3 by = b3QuatRotate(q, b3MakeVector3(0.f, 1.f, 0.f));
		const b3Vector3 bz = b3QuatRotate(q, b3MakeVector3(0.f, 0.f, 1.f));
		const b3Vector3 wc = pos + b3QuatRotate(q, center);
		for (int i = 0; i < 3; ++i)
		{
			const float we = std::fabs((float)bx[i]) * (float)ext[0] + std::fabs((float)by[i]) * (float)ext[1]
						   + std::fabs((float)bz[i]) * (float)ext[2];
			merged.minVec[i] = (std::min)(merged.minVec[i], (float)wc[i] - we);
			merged.maxVec[i] = (std::max)(merged.maxVec[i], (float)wc[i] + we);
		}
	}

	b3Collidable col;
	memset(&col, 0, sizeof(col));
	col.m_shapeType = SHAPE_COMPOUND_OF_CONVEX_HULLS;
	col.m_shapeIndex = childOffset;
	col.m_numChildShapes = numChildren;
	m_cpuCollidables.push_back(col);
	m_cpuLocalAabbs.push_back(merged);

	return (int)m_cpuCollidables.size() - 1;
}

bool b3IrrlichtNarrowphase::writeShapesToGpu()
{
	if (m_cpuCollidables.empty())
		return false;

	const irr::u32 count = (irr::u32)m_cpuCollidables.size();
	uploadBuffer<b3Collidable>(m_collidableBuffer, &m_cpuCollidables[0], count);
	uploadBuffer<b3IrrAabb>(m_localAabbBuffer, &m_cpuLocalAabbs[0], count);

	if (!m_cpuHulls.empty())
	{
		uploadBuffer<b3ConvexPolyhedronData>(m_hullBuffer, &m_cpuHulls[0],
											 (irr::u32)m_cpuHulls.size());
		uploadBuffer<IrrFloat4>(m_vertexBuffer, (const IrrFloat4*)&m_cpuVertices[0],
								(irr::u32)(m_cpuVertices.size() / 4));
		uploadBuffer<b3GpuFace>(m_faceBuffer, &m_cpuFaces[0], (irr::u32)m_cpuFaces.size());
		uploadBuffer<int>(m_indexBuffer, &m_cpuIndices[0], (irr::u32)m_cpuIndices.size());
		uploadBuffer<IrrFloat4>(m_edgeBuffer, (const IrrFloat4*)&m_cpuEdges[0],
								(irr::u32)(m_cpuEdges.size() / 4));
	}

	if (!m_cpuChildShapes.empty())
		uploadBuffer<b3GpuChildShape>(m_childShapeBuffer, &m_cpuChildShapes[0],
									  (irr::u32)m_cpuChildShapes.size());

	// Always full length: the solver indexes it by collidable, so a short buffer would misindex.
	if (m_cpuRollingFriction.size() < count)
		m_cpuRollingFriction.resize(count, 0.f);
	uploadBuffer<float>(m_rollingFrictionBuffer, &m_cpuRollingFriction[0], count);

	return true;
}

bool b3IrrlichtNarrowphase::expandPairsResident(irr::scene::IComputeBuffer* bodies,
												irr::scene::IComputeBuffer* pairs,
												irr::scene::IComputeBuffer* pairCount,
												unsigned int numPairs)
{
	if (m_expandMaterial < 0 || !bodies || !pairs || !m_collidableBuffer || numPairs == 0)
		return false;
	if (pairCount && !(m_dispatch && m_dispatch->isAvailable()))
		return false;

	ensureBuffer<IrrInt4>(m_leafPairBuffer, m_maxLeafPairs, irr::video::EHBF_COMPUTE_APPEND);
	ensureBuffer<unsigned int>(m_leafPairCountBuffer, 4, irr::video::EHBF_DRAW_INDIRECT_ARGS);
	ensureBuffer<ExpandParams>(m_expandParamBuffer, 1);

	// With a device-side count, numPairs is patched in below from the broadphase's counter.
	ExpandParams p;
	p.numPairs = pairCount ? 0 : numPairs;
	p.pad0 = p.pad1 = p.pad2 = 0;
	memcpy(m_expandParamBuffer->getBufferPointer(), &p, sizeof(p));
	m_expandParamBuffer->setDirty();

	if (pairCount && !m_dispatch->prepareIndirect(pairCount, m_expandParamBuffer, 64, numPairs))
		return false;

	irr::video::SMaterial mat;
	mat.MaterialType = (irr::video::E_MATERIAL_TYPE)m_expandMaterial;
	m_driver->setMaterial(mat);
	m_driver->bindComputeBuffer(0, m_expandParamBuffer, irr::video::EHBT_SHADER_RESOURCE);
	m_driver->bindComputeBuffer(1, bodies, irr::video::EHBT_SHADER_RESOURCE);
	m_driver->bindComputeBuffer(2, m_collidableBuffer, irr::video::EHBT_SHADER_RESOURCE);
	m_driver->bindComputeBuffer(3, pairs, irr::video::EHBT_SHADER_RESOURCE);
	m_driver->bindComputeBuffer(0, m_leafPairBuffer, irr::video::EHBT_COMPUTE);
	m_driver->resetStructureCount(m_leafPairBuffer, 0);
	if (pairCount)
		m_driver->dispatchComputeShaderIndirect(m_dispatch->getArgsBuffer(), 0);
	else
		m_driver->dispatchComputeShaderBound(irr::core::vector3d<irr::u32>((numPairs + 63) / 64, 1, 1));
	m_driver->copyStructureCount(m_leafPairCountBuffer, 0, m_leafPairBuffer);
	m_driver->unbindComputeResources();

	// Written as a UAV here, read as an SRV by the leaf clip.
	m_driver->computeBarrier(m_leafPairBuffer);
	return true;
}

bool b3IrrlichtNarrowphase::clipLeafPairs(irr::scene::IComputeBuffer* bodies, unsigned int maxContacts)
{
	if (m_clipLeafMaterial < 0 || !bodies || !m_leafPairBuffer || !m_leafPairCountBuffer)
		return false;
	if (!m_hullBuffer || !m_childShapeBuffer || !m_dispatch || !m_dispatch->isAvailable())
		return false;

	ensureBuffer<b3Contact4Data>(m_contactBuffer, maxContacts, irr::video::EHBF_COMPUTE_APPEND);
	ensureBuffer<unsigned int>(m_contactCountBuffer, 4, irr::video::EHBF_DRAW_INDIRECT_ARGS);
	ensureBuffer<ContactParams>(m_contactParamBuffer, 1);

	ContactParams p;
	p.numPairs = 0;
	p.collisionMargin = 0.f;
	p.speculativeDt = m_speculativeDt;
	p.ccdMotionThreshold = m_ccdMotionThreshold;
	memcpy(m_contactParamBuffer->getBufferPointer(), &p, sizeof(p));
	m_contactParamBuffer->setDirty();

	if (!m_dispatch->prepareIndirect(m_leafPairCountBuffer, m_contactParamBuffer, 64, m_maxLeafPairs))
		return false;

	irr::video::SMaterial mat;
	mat.MaterialType = (irr::video::E_MATERIAL_TYPE)m_clipLeafMaterial;
	m_driver->setMaterial(mat);
	m_driver->bindComputeBuffer(0, m_contactParamBuffer, irr::video::EHBT_SHADER_RESOURCE);
	m_driver->bindComputeBuffer(1, bodies, irr::video::EHBT_SHADER_RESOURCE);
	m_driver->bindComputeBuffer(2, m_collidableBuffer, irr::video::EHBT_SHADER_RESOURCE);
	m_driver->bindComputeBuffer(3, m_hullBuffer, irr::video::EHBT_SHADER_RESOURCE);
	m_driver->bindComputeBuffer(4, m_vertexBuffer, irr::video::EHBT_SHADER_RESOURCE);
	m_driver->bindComputeBuffer(5, m_faceBuffer, irr::video::EHBT_SHADER_RESOURCE);
	m_driver->bindComputeBuffer(6, m_edgeBuffer, irr::video::EHBT_SHADER_RESOURCE);
	m_driver->bindComputeBuffer(7, m_leafPairBuffer, irr::video::EHBT_SHADER_RESOURCE);
	m_driver->bindComputeBuffer(8, m_indexBuffer, irr::video::EHBT_SHADER_RESOURCE);
	m_driver->bindComputeBuffer(9, m_childShapeBuffer, irr::video::EHBT_SHADER_RESOURCE);
	m_driver->bindComputeBuffer(0, m_contactBuffer, irr::video::EHBT_COMPUTE);
	m_driver->resetStructureCount(m_contactBuffer, 0);
	m_driver->dispatchComputeShaderIndirect(m_dispatch->getArgsBuffer(), 0);
	m_driver->copyStructureCount(m_contactCountBuffer, 0, m_contactBuffer);
	m_driver->unbindComputeResources();

	m_driver->computeBarrier(m_contactBuffer);
	return true;
}

bool b3IrrlichtNarrowphase::downloadContacts(std::vector<b3Contact4Data>& contacts,
											 unsigned int maxContacts, bool* overflowed)
{
	m_contactCountBuffer->downloadFromGPU();
	unsigned int count = 0;
	memcpy(&count, m_contactCountBuffer->getBufferPointer(), sizeof(unsigned int));

	if (count > maxContacts)
	{
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

bool b3IrrlichtNarrowphase::findSeparatingAxis(
	const std::vector<b3RigidBodyData>& bodies,
	const std::vector<std::pair<unsigned int, unsigned int> >& pairs,
	std::vector<b3IrrSatResult>& results)
{
	if (m_satMaterial < 0 || !m_collidableBuffer || !m_hullBuffer || bodies.empty())
		return false;

	results.clear();
	if (pairs.empty())
		return true;

	if (!uploadBodies(bodies))
		return false;

	uploadPairs(m_pairBuffer, pairs);
	ensureBuffer<b3IrrSatResult>(m_satResultBuffer, (irr::u32)pairs.size());
	ensureBuffer<SatParams>(m_satParamBuffer, 1);

	SatParams p;
	p.numPairs = (unsigned int)pairs.size();
	p.pad0 = p.pad1 = p.pad2 = 0;
	memcpy(m_satParamBuffer->getBufferPointer(), &p, sizeof(p));
	m_satParamBuffer->setDirty();

	irr::video::SMaterial mat;
	mat.MaterialType = (irr::video::E_MATERIAL_TYPE)m_satMaterial;
	m_driver->setMaterial(mat);
	m_driver->bindComputeBuffer(0, m_satParamBuffer, irr::video::EHBT_SHADER_RESOURCE);
	m_driver->bindComputeBuffer(1, m_bodyBuffer, irr::video::EHBT_SHADER_RESOURCE);
	m_driver->bindComputeBuffer(2, m_collidableBuffer, irr::video::EHBT_SHADER_RESOURCE);
	m_driver->bindComputeBuffer(3, m_hullBuffer, irr::video::EHBT_SHADER_RESOURCE);
	m_driver->bindComputeBuffer(4, m_vertexBuffer, irr::video::EHBT_SHADER_RESOURCE);
	m_driver->bindComputeBuffer(5, m_faceBuffer, irr::video::EHBT_SHADER_RESOURCE);
	m_driver->bindComputeBuffer(6, m_edgeBuffer, irr::video::EHBT_SHADER_RESOURCE);
	m_driver->bindComputeBuffer(7, m_pairBuffer, irr::video::EHBT_SHADER_RESOURCE);
	m_driver->bindComputeBuffer(0, m_satResultBuffer, irr::video::EHBT_COMPUTE);
	m_driver->dispatchComputeShaderBound(
		irr::core::vector3d<irr::u32>(((irr::u32)pairs.size() + 63) / 64, 1, 1));
	m_driver->unbindComputeResources();

	m_satResultBuffer->downloadFromGPU();
	results.resize(pairs.size());
	memcpy(&results[0], m_satResultBuffer->getBufferPointer(),
		   pairs.size() * sizeof(b3IrrSatResult));
	return true;
}

bool b3IrrlichtNarrowphase::uploadBodies(const std::vector<b3RigidBodyData>& bodies)
{
	// The std::vector entry points move 80-byte bodies; the df64 kernels index a 96-byte stride.
	if (m_doubleSingle)
		return false;

	uploadBuffer<b3RigidBodyData>(m_bodyBuffer, &bodies[0], (irr::u32)bodies.size());
	return true;
}

bool b3IrrlichtNarrowphase::uploadBodiesResidentDS(const std::vector<b3IrrGpu::b3IrrRigidBodyDataDS>& bodies)
{
	if (bodies.empty() || !m_doubleSingle)
		return false;

	uploadBuffer<b3IrrGpu::b3IrrRigidBodyDataDS>(m_bodyBuffer, &bodies[0], (irr::u32)bodies.size());
	return true;
}

bool b3IrrlichtNarrowphase::uploadBodiesResident(const std::vector<b3RigidBodyData>& bodies)
{
	if (bodies.empty() || m_doubleSingle)
		return false;

	uploadBuffer<b3RigidBodyData>(m_bodyBuffer, &bodies[0], (irr::u32)bodies.size());
	return true;
}

bool b3IrrlichtNarrowphase::computeWorldAabbs(const std::vector<b3RigidBodyData>& bodies,
											  std::vector<b3IrrAabb>& aabbs, float margin)
{
	if (m_aabbMaterial < 0 || bodies.empty() || !m_collidableBuffer)
		return false;

	if (!uploadBodies(bodies))
		return false;

	ensureBuffer<b3IrrAabb>(m_worldAabbBuffer, (irr::u32)bodies.size());
	ensureBuffer<AabbParams>(m_aabbParamBuffer, 1);

	AabbParams p;
	p.numBodies = (unsigned int)bodies.size();
	p.margin = margin;
	// The vector path has no sleep state, so every body is recomputed.
	p.useSleep = 0;
	p.sweepDt = m_speculativeDt;
	memcpy(m_aabbParamBuffer->getBufferPointer(), &p, sizeof(p));
	m_aabbParamBuffer->setDirty();

	irr::video::SMaterial mat;
	mat.MaterialType = (irr::video::E_MATERIAL_TYPE)m_aabbMaterial;
	m_driver->setMaterial(mat);
	m_driver->bindComputeBuffer(0, m_aabbParamBuffer, irr::video::EHBT_SHADER_RESOURCE);
	m_driver->bindComputeBuffer(1, m_bodyBuffer, irr::video::EHBT_SHADER_RESOURCE);
	m_driver->bindComputeBuffer(2, m_collidableBuffer, irr::video::EHBT_SHADER_RESOURCE);
	m_driver->bindComputeBuffer(3, m_localAabbBuffer, irr::video::EHBT_SHADER_RESOURCE);
	m_driver->bindComputeBuffer(0, m_worldAabbBuffer, irr::video::EHBT_COMPUTE);
	m_driver->dispatchComputeShaderBound(
		irr::core::vector3d<irr::u32>(((irr::u32)bodies.size() + 63) / 64, 1, 1));
	m_driver->unbindComputeResources();

	m_worldAabbBuffer->downloadFromGPU();
	aabbs.resize(bodies.size());
	memcpy(&aabbs[0], m_worldAabbBuffer->getBufferPointer(), bodies.size() * sizeof(b3IrrAabb));
	return true;
}

bool b3IrrlichtNarrowphase::computeWorldAabbsResident(irr::scene::IComputeBuffer* bodies,
													  unsigned int numBodies, float margin,
													  irr::scene::IComputeBuffer* sleepState)
{
	if (m_aabbMaterial < 0 || !bodies || numBodies == 0 || !m_collidableBuffer)
		return false;

	// A grown buffer's new entries are undefined, so the skip below would leave them that way.
	const bool grew = !m_worldAabbBuffer || m_worldAabbBuffer->getStructureCount() < numBodies;
	if (m_doubleSingle)
		ensureBuffer<b3IrrAabbDS>(m_worldAabbBuffer, numBodies);
	else
		ensureBuffer<b3IrrAabb>(m_worldAabbBuffer, numBodies);
	ensureBuffer<AabbParams>(m_aabbParamBuffer, 1);

	AabbParams p;
	p.numBodies = numBodies;
	p.margin = margin;
	p.useSleep = (sleepState && !grew) ? 1u : 0u;
	p.sweepDt = m_speculativeDt;
	memcpy(m_aabbParamBuffer->getBufferPointer(), &p, sizeof(p));
	m_aabbParamBuffer->setDirty();

	irr::video::SMaterial mat;
	mat.MaterialType = (irr::video::E_MATERIAL_TYPE)m_aabbMaterial;
	m_driver->setMaterial(mat);
	m_driver->bindComputeBuffer(0, m_aabbParamBuffer, irr::video::EHBT_SHADER_RESOURCE);
	m_driver->bindComputeBuffer(1, bodies, irr::video::EHBT_SHADER_RESOURCE);
	m_driver->bindComputeBuffer(2, m_collidableBuffer, irr::video::EHBT_SHADER_RESOURCE);
	m_driver->bindComputeBuffer(3, m_localAabbBuffer, irr::video::EHBT_SHADER_RESOURCE);
	if (p.useSleep)
		m_driver->bindComputeBuffer(4, sleepState, irr::video::EHBT_SHADER_RESOURCE);
	m_driver->bindComputeBuffer(0, m_worldAabbBuffer, irr::video::EHBT_COMPUTE);
	m_driver->dispatchComputeShaderBound(irr::core::vector3d<irr::u32>((numBodies + 63) / 64, 1, 1));
	m_driver->unbindComputeResources();

	// Written as a UAV here, read as an SRV by the broadphase.
	m_driver->computeBarrier(m_worldAabbBuffer);
	return true;
}

bool b3IrrlichtNarrowphase::computeConvexContactsResident(irr::scene::IComputeBuffer* bodies,
														  unsigned int numBodies,
														  irr::scene::IComputeBuffer* pairs,
														  irr::scene::IComputeBuffer* pairCount,
														  unsigned int maxPairs,
														  unsigned int maxContacts)
{
	if (!isResidentPathAvailable() || !bodies || !pairs || !pairCount || numBodies == 0)
		return false;

	if (!m_collidableBuffer || !m_hullBuffer)
		return false;

	if (needsPairExpansion())
	{
		if (!expandPairsResident(bodies, pairs, pairCount, maxPairs))
			return false;
		return clipLeafPairs(bodies, maxContacts);
	}

	ensureBuffer<b3Contact4Data>(m_contactBuffer, maxContacts, irr::video::EHBF_COMPUTE_APPEND);
	ensureBuffer<unsigned int>(m_contactCountBuffer, 4, irr::video::EHBF_DRAW_INDIRECT_ARGS);
	ensureBuffer<ContactParams>(m_contactParamBuffer, 1);

	// numPairs is filled in by the GPU below, from the broadphase's own append counter.
	ContactParams p;
	p.numPairs = 0;
	p.collisionMargin = 0.f;
	p.speculativeDt = m_speculativeDt;
	p.ccdMotionThreshold = m_ccdMotionThreshold;
	memcpy(m_contactParamBuffer->getBufferPointer(), &p, sizeof(p));
	m_contactParamBuffer->setDirty();

	if (!m_dispatch->prepareIndirect(pairCount, m_contactParamBuffer, 64, maxPairs))
		return false;

	irr::video::SMaterial mat;
	mat.MaterialType = (irr::video::E_MATERIAL_TYPE)m_clipMaterial;
	m_driver->setMaterial(mat);
	m_driver->bindComputeBuffer(0, m_contactParamBuffer, irr::video::EHBT_SHADER_RESOURCE);
	m_driver->bindComputeBuffer(1, bodies, irr::video::EHBT_SHADER_RESOURCE);
	m_driver->bindComputeBuffer(2, m_collidableBuffer, irr::video::EHBT_SHADER_RESOURCE);
	m_driver->bindComputeBuffer(3, m_hullBuffer, irr::video::EHBT_SHADER_RESOURCE);
	m_driver->bindComputeBuffer(4, m_vertexBuffer, irr::video::EHBT_SHADER_RESOURCE);
	m_driver->bindComputeBuffer(5, m_faceBuffer, irr::video::EHBT_SHADER_RESOURCE);
	m_driver->bindComputeBuffer(6, m_edgeBuffer, irr::video::EHBT_SHADER_RESOURCE);
	m_driver->bindComputeBuffer(7, pairs, irr::video::EHBT_SHADER_RESOURCE);
	m_driver->bindComputeBuffer(8, m_indexBuffer, irr::video::EHBT_SHADER_RESOURCE);
	m_driver->bindComputeBuffer(0, m_contactBuffer, irr::video::EHBT_COMPUTE);
	m_driver->resetStructureCount(m_contactBuffer, 0);
	m_driver->dispatchComputeShaderIndirect(m_dispatch->getArgsBuffer(), 0);
	m_driver->copyStructureCount(m_contactCountBuffer, 0, m_contactBuffer);
	m_driver->unbindComputeResources();

	m_driver->computeBarrier(m_contactBuffer);
	return true;
}

bool b3IrrlichtNarrowphase::computeSphereContacts(
	const std::vector<b3RigidBodyData>& bodies,
	const std::vector<std::pair<unsigned int, unsigned int> >& pairs,
	std::vector<b3Contact4Data>& contacts, unsigned int maxContacts, float collisionMargin,
	bool* overflowed)
{
	if (overflowed)
		*overflowed = false;

	if (m_sphereContactMaterial < 0 || !m_collidableBuffer || bodies.empty())
		return false;

	contacts.clear();
	if (pairs.empty())
		return true;

	if (!uploadBodies(bodies))
		return false;

	uploadPairs(m_pairBuffer, pairs);
	ensureBuffer<b3Contact4Data>(m_contactBuffer, maxContacts, irr::video::EHBF_COMPUTE_APPEND);
	ensureBuffer<unsigned int>(m_contactCountBuffer, 4, irr::video::EHBF_DRAW_INDIRECT_ARGS);
	ensureBuffer<ContactParams>(m_contactParamBuffer, 1);

	ContactParams p;
	p.numPairs = (unsigned int)pairs.size();
	p.collisionMargin = collisionMargin;
	p.speculativeDt = m_speculativeDt;
	p.ccdMotionThreshold = m_ccdMotionThreshold;
	memcpy(m_contactParamBuffer->getBufferPointer(), &p, sizeof(p));
	m_contactParamBuffer->setDirty();

	irr::video::SMaterial mat;
	mat.MaterialType = (irr::video::E_MATERIAL_TYPE)m_sphereContactMaterial;
	m_driver->setMaterial(mat);
	m_driver->bindComputeBuffer(0, m_contactParamBuffer, irr::video::EHBT_SHADER_RESOURCE);
	m_driver->bindComputeBuffer(1, m_bodyBuffer, irr::video::EHBT_SHADER_RESOURCE);
	m_driver->bindComputeBuffer(2, m_collidableBuffer, irr::video::EHBT_SHADER_RESOURCE);
	m_driver->bindComputeBuffer(3, m_pairBuffer, irr::video::EHBT_SHADER_RESOURCE);
	m_driver->bindComputeBuffer(0, m_contactBuffer, irr::video::EHBT_COMPUTE);
	m_driver->resetStructureCount(m_contactBuffer, 0);
	m_driver->dispatchComputeShaderBound(
		irr::core::vector3d<irr::u32>(((irr::u32)pairs.size() + 63) / 64, 1, 1));
	m_driver->copyStructureCount(m_contactCountBuffer, 0, m_contactBuffer);
	m_driver->unbindComputeResources();

	m_contactCountBuffer->downloadFromGPU();
	unsigned int count = 0;
	memcpy(&count, m_contactCountBuffer->getBufferPointer(), sizeof(unsigned int));

	if (count > maxContacts)
	{
		// Appends past capacity are dropped but still advance the counter - which is exactly how
		// overflow stays visible instead of silently truncating.
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

bool b3IrrlichtNarrowphase::computeConvexContacts(
	const std::vector<b3RigidBodyData>& bodies,
	const std::vector<std::pair<unsigned int, unsigned int> >& pairs,
	std::vector<b3Contact4Data>& contacts, unsigned int maxContacts, bool* overflowed)
{
	if (overflowed)
		*overflowed = false;

	if (m_clipMaterial < 0 || !m_collidableBuffer || !m_hullBuffer || bodies.empty())
		return false;

	contacts.clear();
	if (pairs.empty())
		return true;

	if (!uploadBodies(bodies))
		return false;

	uploadPairs(m_pairBuffer, pairs);

	if (needsPairExpansion())
	{
		if (!expandPairsResident(m_bodyBuffer, m_pairBuffer, 0, (unsigned int)pairs.size()))
			return false;
		if (!clipLeafPairs(m_bodyBuffer, maxContacts))
			return false;
		return downloadContacts(contacts, maxContacts, overflowed);
	}

	ensureBuffer<b3Contact4Data>(m_contactBuffer, maxContacts, irr::video::EHBF_COMPUTE_APPEND);
	ensureBuffer<unsigned int>(m_contactCountBuffer, 4, irr::video::EHBF_DRAW_INDIRECT_ARGS);
	ensureBuffer<ContactParams>(m_contactParamBuffer, 1);

	ContactParams p;
	p.numPairs = (unsigned int)pairs.size();
	p.collisionMargin = 0.f;
	p.speculativeDt = m_speculativeDt;
	p.ccdMotionThreshold = m_ccdMotionThreshold;
	memcpy(m_contactParamBuffer->getBufferPointer(), &p, sizeof(p));
	m_contactParamBuffer->setDirty();

	irr::video::SMaterial mat;
	mat.MaterialType = (irr::video::E_MATERIAL_TYPE)m_clipMaterial;
	m_driver->setMaterial(mat);
	m_driver->bindComputeBuffer(0, m_contactParamBuffer, irr::video::EHBT_SHADER_RESOURCE);
	m_driver->bindComputeBuffer(1, m_bodyBuffer, irr::video::EHBT_SHADER_RESOURCE);
	m_driver->bindComputeBuffer(2, m_collidableBuffer, irr::video::EHBT_SHADER_RESOURCE);
	m_driver->bindComputeBuffer(3, m_hullBuffer, irr::video::EHBT_SHADER_RESOURCE);
	m_driver->bindComputeBuffer(4, m_vertexBuffer, irr::video::EHBT_SHADER_RESOURCE);
	m_driver->bindComputeBuffer(5, m_faceBuffer, irr::video::EHBT_SHADER_RESOURCE);
	m_driver->bindComputeBuffer(6, m_edgeBuffer, irr::video::EHBT_SHADER_RESOURCE);
	m_driver->bindComputeBuffer(7, m_pairBuffer, irr::video::EHBT_SHADER_RESOURCE);
	m_driver->bindComputeBuffer(8, m_indexBuffer, irr::video::EHBT_SHADER_RESOURCE);
	m_driver->bindComputeBuffer(0, m_contactBuffer, irr::video::EHBT_COMPUTE);
	m_driver->resetStructureCount(m_contactBuffer, 0);
	m_driver->dispatchComputeShaderBound(
		irr::core::vector3d<irr::u32>(((irr::u32)pairs.size() + 63) / 64, 1, 1));
	m_driver->copyStructureCount(m_contactCountBuffer, 0, m_contactBuffer);
	m_driver->unbindComputeResources();

	m_contactCountBuffer->downloadFromGPU();
	unsigned int count = 0;
	memcpy(&count, m_contactCountBuffer->getBufferPointer(), sizeof(unsigned int));

	if (count > maxContacts)
	{
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

void b3IrrlichtNarrowphase::setPlanetNoiseParams(float mTimer, float plates, float rivers,
												 float atmosphereDensity, float texsize)
{
	if (!m_planetNoiseParams)
		return;
	m_planetNoiseParams->mTimer = mTimer;
	m_planetNoiseParams->plates = plates;
	m_planetNoiseParams->rivers = rivers;
	m_planetNoiseParams->atmosphereDensity = atmosphereDensity;
	m_planetNoiseParams->texsize = texsize;
}

bool b3IrrlichtNarrowphase::computePlanetContacts(
	const std::vector<b3RigidBodyData>& bodies, const std::vector<b3IrrPlanet>& planets,
	const std::vector<std::pair<unsigned int, unsigned int> >& pairs,
	std::vector<b3Contact4Data>& contacts, unsigned int maxContacts, bool* overflowed)
{
	if (overflowed)
		*overflowed = false;

	if (m_planetMaterial < 0 || !m_collidableBuffer || bodies.empty() || planets.empty())
		return false;

	contacts.clear();
	if (pairs.empty())
		return true;

	if (!uploadBodies(bodies))
		return false;

	uploadBuffer<b3IrrPlanet>(m_planetBuffer, &planets[0], (irr::u32)planets.size());
	uploadPairs(m_pairBuffer, pairs);
	ensureBuffer<b3Contact4Data>(m_contactBuffer, maxContacts, irr::video::EHBF_COMPUTE_APPEND);
	ensureBuffer<unsigned int>(m_contactCountBuffer, 4, irr::video::EHBF_DRAW_INDIRECT_ARGS);
	ensureBuffer<ContactParams>(m_contactParamBuffer, 1);

	ContactParams p;
	p.numPairs = (unsigned int)pairs.size();
	p.collisionMargin = 0.f;
	// The planet kernel has no speculative path; the fields are inert there.
	p.speculativeDt = 0.f;
	p.ccdMotionThreshold = 0.f;
	memcpy(m_contactParamBuffer->getBufferPointer(), &p, sizeof(p));
	m_contactParamBuffer->setDirty();

	irr::video::SMaterial mat;
	mat.MaterialType = (irr::video::E_MATERIAL_TYPE)m_planetMaterial;
	m_driver->setMaterial(mat);
	// t0 is left free: noise2.hlsl declares its texfix sampler there.
	m_driver->bindComputeBuffer(1, m_contactParamBuffer, irr::video::EHBT_SHADER_RESOURCE);
	m_driver->bindComputeBuffer(2, m_bodyBuffer, irr::video::EHBT_SHADER_RESOURCE);
	m_driver->bindComputeBuffer(3, m_collidableBuffer, irr::video::EHBT_SHADER_RESOURCE);
	m_driver->bindComputeBuffer(4, m_planetBuffer, irr::video::EHBT_SHADER_RESOURCE);
	m_driver->bindComputeBuffer(5, m_pairBuffer, irr::video::EHBT_SHADER_RESOURCE);
	m_driver->bindComputeBuffer(0, m_contactBuffer, irr::video::EHBT_COMPUTE);
	m_driver->resetStructureCount(m_contactBuffer, 0);
	m_driver->dispatchComputeShaderBound(
		irr::core::vector3d<irr::u32>(((irr::u32)pairs.size() + 63) / 64, 1, 1));
	m_driver->copyStructureCount(m_contactCountBuffer, 0, m_contactBuffer);
	m_driver->unbindComputeResources();

	m_contactCountBuffer->downloadFromGPU();
	unsigned int count = 0;
	memcpy(&count, m_contactCountBuffer->getBufferPointer(), sizeof(unsigned int));

	if (count > maxContacts)
	{
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

float b3IrrlichtNarrowphase::getRestitution(const b3Contact4Data& contact)
{
	return contact.m_restituitionCoeffCmp / 65535.f;
}

float b3IrrlichtNarrowphase::getFriction(const b3Contact4Data& contact)
{
	return contact.m_frictionCoeffCmp / 65535.f;
}
