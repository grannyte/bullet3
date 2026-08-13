#ifndef B3_IRRLICHT_RADIX_SORT_H
#define B3_IRRLICHT_RADIX_SORT_H

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

class b3IrrlichtPrefixScan;

/// Key/value pair, layout-identical to the HLSL uint2 and to Bullet's b3SortData.
struct b3IrrSortData
{
	unsigned int key;
	unsigned int value;
};

/// Irrlicht-compute analog of b3RadixSort32CL. Stable LSD radix sort, 4 bits per pass.
/// Stability matters twice over: multi-pass LSD depends on it, and Bullet's contact-determinism
/// pass is four stable sorts keyed on (bodyA, bodyB, childShapeA, childShapeB).
class b3IrrlichtRadixSort
{
public:
	enum
	{
		ELEMS_PER_BLOCK = 256,
		BITS_PER_PASS = 4,
		NUM_BUCKETS = 1 << BITS_PER_PASS
	};

	b3IrrlichtRadixSort(irr::video::IVideoDriver* driver);
	~b3IrrlichtRadixSort();

	bool init(irr::io::IFileSystem* fileSystem = 0);

	/// Sorts ascending by key. keyBits lets a caller sort fewer than 32 bits (e.g. 30-bit
	/// Morton codes) and skip the passes that would do nothing.
	bool execute(std::vector<b3IrrSortData>& data, unsigned int keyBits = 32);

	/**
	 * @brief Sorts a device-resident key/value buffer without any CPU round trip.
	 *
	 * The input is never written; the result lands in one of the two internal ping-pong buffers.
	 *
	 * @param input Device-resident (key, value) pairs.
	 * @param numElems Elements to sort.
	 * @param keyBits Significant key bits; fewer bits skip whole passes.
	 * @param result Receives the buffer holding the sorted result. Valid until the next call.
	 * @return False if the kernels are unavailable or the histogram exceeds the scan's limit.
	 */
	bool executeResident(irr::scene::IComputeBuffer* input, unsigned int numElems,
						 unsigned int keyBits, irr::scene::IComputeBuffer** result);

private:
	b3IrrlichtRadixSort(const b3IrrlichtRadixSort&);
	b3IrrlichtRadixSort& operator=(const b3IrrlichtRadixSort&);

	void releaseBuffers();

	/**
	 * @brief Grows the work buffers to cover one sort. Grow-only, so a steady-state sort
	 *        allocates nothing.
	 * @param numElems Elements to sort.
	 * @param histSize Histogram entries needed.
	 * @return False if a buffer could not be created.
	 */
	bool ensureWorkBuffers(unsigned int numElems, unsigned int histSize);

	/**
	 * @brief Runs the LSD passes, ping-ponging between the work buffers.
	 * @param input Buffer holding the unsorted pairs; read-only after the first pass.
	 * @param numElems Elements to sort.
	 * @param keyBits Significant key bits.
	 * @param result Receives whichever work buffer ended up holding the sorted data.
	 * @return False if the histogram scan failed.
	 */
	bool sortPasses(irr::scene::IComputeBuffer* input, unsigned int numElems, unsigned int keyBits,
					irr::scene::IComputeBuffer** result);

	irr::video::IVideoDriver* m_driver;
	b3IrrlichtPrefixScan* m_scan;

	irr::scene::IComputeBuffer* m_paramBuffer;
	irr::scene::IComputeBuffer* m_srcBuffer;
	irr::scene::IComputeBuffer* m_dstBuffer;
	irr::scene::IComputeBuffer* m_histogram;
	irr::scene::IComputeBuffer* m_scannedHistogram;
	irr::scene::IComputeBuffer* m_histBlockSums;

	int m_streamCountMaterial;
	int m_sortScatterMaterial;
};

#endif  //B3_IRRLICHT_RADIX_SORT_H
