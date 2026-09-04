#ifndef B3_IRRLICHT_PREFIX_SCAN_H
#define B3_IRRLICHT_PREFIX_SCAN_H

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

/// Irrlicht-compute analog of b3PrefixScanCL. Exclusive scan over u32.
/// SM5.0 groupshared implementation - see B3PrefixScan.hlsl for why not wave intrinsics.
class b3IrrlichtPrefixScan
{
public:
	/// Elements one workgroup scans. Fixed at compile time in the shader (plan rule R5):
	/// changing it changes reduction order, so it is not runtime-tunable.
	enum
	{
		ELEMS_PER_BLOCK = 256
	};

	b3IrrlichtPrefixScan(irr::video::IVideoDriver* driver);
	~b3IrrlichtPrefixScan();

	bool init(irr::io::IFileSystem* fileSystem = 0);

	/// Exclusive-scans src into dst. Returns false if the size needs more blocks than the
	/// single-group top-level scan can handle (numBlocks > ELEMS_PER_BLOCK).
	bool execute(const std::vector<unsigned int>& src, std::vector<unsigned int>& dst);

	/// GPU-resident form: no upload or readback, for use inside a larger GPU pipeline.
	/// src and dst must be distinct - D3D11 cannot bind one resource as SRV and UAV at once.
	/// blockSums must hold at least ELEMS_PER_BLOCK entries.
	bool scanBuffers(irr::scene::IComputeBuffer* src, irr::scene::IComputeBuffer* dst,
					 irr::scene::IComputeBuffer* blockSums, unsigned int numElems);

	/// Largest element count scanBuffers can handle before it needs to recurse.
	static unsigned int maxElements() { return ELEMS_PER_BLOCK * ELEMS_PER_BLOCK; }

private:
	b3IrrlichtPrefixScan(const b3IrrlichtPrefixScan&);
	b3IrrlichtPrefixScan& operator=(const b3IrrlichtPrefixScan&);

	irr::video::IVideoDriver* m_driver;
	irr::scene::IComputeBuffer* m_paramBuffer;
	irr::scene::IComputeBuffer* m_srcBuffer;
	irr::scene::IComputeBuffer* m_dstBuffer;
	irr::scene::IComputeBuffer* m_blockSumBuffer;

	int m_localScanMaterial;
	int m_topLevelScanMaterial;
	int m_addOffsetMaterial;
};

#endif  //B3_IRRLICHT_PREFIX_SCAN_H
