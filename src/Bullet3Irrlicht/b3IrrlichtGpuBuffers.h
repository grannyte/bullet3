#ifndef B3_IRRLICHT_GPU_BUFFERS_H
#define B3_IRRLICHT_GPU_BUFFERS_H

#include <irrlicht.h>
#include "ComputeBuffer.h"

#include <cstring>

/// Persistent-buffer plumbing shared by every Irrlicht-compute physics phase. Header-only so a
/// new translation unit never has to be added to a .vcxproj.
namespace b3IrrGpu
{
/**
 * @brief Drops a compute buffer and nulls the pointer.
 * @param buffer Buffer pointer, cleared on return.
 */
inline void dropBuffer(irr::scene::IComputeBuffer*& buffer)
{
	if (buffer)
	{
		buffer->drop();
		buffer = 0;
	}
}

/**
 * @brief Keeps a buffer if it can already hold count elements, otherwise recreates it.
 *
 * Grow-only: a smaller call never shrinks the allocation, which is what takes buffer
 * creation off the per-step path. Contents are undefined after a growth.
 *
 * @param buffer Buffer slot, replaced only when it must grow or its shape changed.
 * @param count Elements needed this call.
 * @param flags E_HARDWARE_BUFFER_FLAGS the buffer must have been created with.
 * @return The usable buffer, or 0 when count is 0.
 */
template <typename T>
inline irr::scene::IComputeBuffer* ensureBuffer(irr::scene::IComputeBuffer*& buffer, irr::u32 count,
												irr::u32 flags = 0)
{
	if (count == 0)
		return buffer;

	if (buffer)
	{
		if (buffer->getStructureStride() == sizeof(T) && buffer->getBufferFlags() == flags &&
			buffer->getStructureCount() >= count)
			return buffer;
		dropBuffer(buffer);
	}

	irr::scene::ComputeBuffer<T>* fresh = new irr::scene::ComputeBuffer<T>();
	fresh->setHardwareMappingHint(irr::scene::EHM_DYNAMIC);
	if (flags)
		fresh->setBufferFlags(flags);
	fresh->set_used(count);
	buffer = fresh;
	return buffer;
}

/**
 * @brief ensureBuffer plus a CPU-side upload of the whole array.
 * @param buffer Buffer slot.
 * @param data First element.
 * @param count Element count.
 * @param flags E_HARDWARE_BUFFER_FLAGS for creation.
 * @return The uploaded buffer.
 */
template <typename T>
inline irr::scene::IComputeBuffer* uploadBuffer(irr::scene::IComputeBuffer*& buffer, const T* data,
												irr::u32 count, irr::u32 flags = 0)
{
	ensureBuffer<T>(buffer, count, flags);
	if (buffer && data && count)
	{
		memcpy(buffer->getBufferPointer(), data, count * sizeof(T));
		buffer->setDirty();
	}
	return buffer;
}

/// b3RigidBodyData with a low half for the position: the emulated-double (df64) body layout,
/// 96 bytes, matching B3Precision.hlsli under OS_DS. Only the position is emulated.
struct b3IrrRigidBodyDataDS
{
	float pos[4];
	float posLo[4];
	float quat[4];
	float linVel[4];
	float angVel[4];
	int collidableIdx;
	float invMass;
	float restituitionCoeff;
	float frictionCoeff;
};

/// Renderer-facing df64 transform, 40 bytes, matching B3IntegrateTransformsBody.hlsli under OS_DS.
struct b3IrrBodyTransformDS
{
	float position[3];
	float positionLo[3];
	float orientation[4];   // xyzw
};

static_assert(sizeof(b3IrrRigidBodyDataDS) == 96, "b3IrrRigidBodyDataDS must match the OS_DS HLSL b3RigidBodyData stride");

/// Element stride of a body buffer in each precision mode (b3RigidBodyData / b3IrrRigidBodyDataDS).
inline unsigned int bodyStride(bool doubleSingle) { return doubleSingle ? 96u : 80u; }
/// Element stride of a world-AABB buffer in each precision mode (b3IrrAabb / b3IrrAabbDS).
inline unsigned int aabbStride(bool doubleSingle) { return doubleSingle ? 64u : 32u; }

/**
 * @brief Whether a device buffer carries the stride a stage's kernels index by.
 *
 * A df64 buffer read through f32 kernels (or the reverse) misindexes every element silently,
 * so resident entry points refuse on a mismatch instead of dispatching.
 *
 * @param buffer Buffer about to be bound.
 * @param expectedStride Stride the kernel set was compiled for.
 * @return True when the buffer exists and its stride matches.
 */
inline bool strideMatches(const irr::scene::IComputeBuffer* buffer, unsigned int expectedStride)
{
	return buffer && buffer->getStructureStride() == expectedStride;
}

/**
 * @brief Splits a hardware double into the (hi, lo) float pair the df64 kernels consume.
 * @param value Value to split.
 * @param hi Receives the nearest float.
 * @param lo Receives the exact remainder.
 */
inline void b3DsSplit(double value, float& hi, float& lo)
{
	hi = (float)value;
	lo = (float)(value - (double)hi);
}

/**
 * @brief Recombines a df64 pair into a hardware double.
 * @param hi High half.
 * @param lo Low half.
 * @return hi + lo in double precision.
 */
inline double b3DsCombine(float hi, float lo)
{
	return (double)hi + (double)lo;
}

/// 16-byte params block, first field the element count. Every phase's params struct has this
/// shape, which is what lets one GPU kernel patch all of them.
struct CountParams
{
	unsigned int count;
	unsigned int cap;
	unsigned int reduceCount;
	unsigned int numBodies;
};

/**
 * @brief Owns B3GpuResident.hlsl: count-to-dispatch-args patching and int max-reduction.
 *
 * Held by value inside each phase so the phase can size its own dispatches from a count that
 * only exists on the device.
 */
class DispatchHelper
{
public:
	DispatchHelper(irr::video::IVideoDriver* driver)
		: m_driver(driver), m_patchMaterial(-1), m_reduceMaterial(-1),
		  m_paramBuffer(0), m_argsBuffer(0), m_reduceBuffer(0)
	{
	}

	~DispatchHelper()
	{
		dropBuffer(m_paramBuffer);
		dropBuffer(m_argsBuffer);
		dropBuffer(m_reduceBuffer);
	}

	/**
	 * @brief Compiles the resident glue kernels. Optional: a caller that only uses the
	 *        std::vector entry points stays fully functional without them.
	 * @param fileSystem Device filesystem; without it a missing shader binds an unrelated material.
	 * @return True when the resident path is usable.
	 */
	bool init(irr::io::IFileSystem* fileSystem)
	{
		if (!m_driver)
			return false;

		irr::video::IGPUProgrammingServices* gpu = m_driver->getGPUProgrammingServices();
		if (!gpu)
			return false;

		const irr::io::path path = "media/shaders/B3GpuResident.hlsl";
		if (fileSystem && !fileSystem->existFile(path))
			return false;

		m_patchMaterial = gpu->addComputeShaderFromFile(path, "CSPatchCountParams", irr::video::ECST_CS_5_0, 0);
		m_reduceMaterial = gpu->addComputeShaderFromFile(path, "CSMaxInt", irr::video::ECST_CS_5_0, 0);

		return isAvailable();
	}

	bool isAvailable() const
	{
		return m_patchMaterial >= 0 && m_reduceMaterial >= 0;
	}

	/// Indirect dispatch arguments written by the last prepareIndirect call.
	irr::scene::IComputeBuffer* getArgsBuffer() const { return m_argsBuffer; }

	/**
	 * @brief Writes a device-side count into a params struct and into indirect dispatch args.
	 * @param countBuffer EHBF_DRAW_INDIRECT_ARGS buffer holding the count at byte 0.
	 * @param targetParams 16-byte params struct whose first field is the count.
	 * @param threadsPerGroup Group size of the kernel that will consume the args.
	 * @param maxCount Cap; an append past capacity still advances the counter.
	 * @return False if the kernels are unavailable.
	 */
	bool prepareIndirect(irr::scene::IComputeBuffer* countBuffer,
						 irr::scene::IComputeBuffer* targetParams, unsigned int threadsPerGroup,
						 unsigned int maxCount)
	{
		if (!isAvailable() || !countBuffer || !targetParams)
			return false;

		CountParams p;
		p.count = threadsPerGroup;
		p.cap = maxCount;
		p.reduceCount = 0;
		p.numBodies = 0;
		uploadBuffer<CountParams>(m_paramBuffer, &p, 1);
		ensureBuffer<unsigned int>(m_argsBuffer, 4, irr::video::EHBF_DRAW_INDIRECT_ARGS);

		irr::video::SMaterial mat;
		mat.MaterialType = (irr::video::E_MATERIAL_TYPE)m_patchMaterial;
		m_driver->setMaterial(mat);
		m_driver->bindComputeBuffer(0, m_paramBuffer, irr::video::EHBT_SHADER_RESOURCE);
		m_driver->bindComputeBuffer(0, countBuffer, irr::video::EHBT_COMPUTE);
		m_driver->bindComputeBuffer(1, targetParams, irr::video::EHBT_COMPUTE);
		m_driver->bindComputeBuffer(2, m_argsBuffer, irr::video::EHBT_COMPUTE);
		m_driver->dispatchComputeShaderBound(irr::core::vector3d<irr::u32>(1, 1, 1));
		m_driver->unbindComputeResources();
		return true;
	}

	/**
	 * @brief Maximum of an int buffer, reduced on the GPU and read back as 4 bytes.
	 * @param source Ints to reduce.
	 * @param count Elements to consider.
	 * @param result Receives the maximum.
	 * @return False if the kernels are unavailable.
	 */
	bool reduceMax(irr::scene::IComputeBuffer* source, unsigned int count, int& result)
	{
		if (!isAvailable() || !source || count == 0)
			return false;

		CountParams p;
		p.count = 0;
		p.cap = 0;
		p.reduceCount = count;
		p.numBodies = 0;
		uploadBuffer<CountParams>(m_paramBuffer, &p, 1);
		ensureBuffer<int>(m_reduceBuffer, 1);

		irr::video::SMaterial mat;
		mat.MaterialType = (irr::video::E_MATERIAL_TYPE)m_reduceMaterial;
		m_driver->setMaterial(mat);
		m_driver->bindComputeBuffer(0, m_paramBuffer, irr::video::EHBT_SHADER_RESOURCE);
		m_driver->bindComputeBuffer(1, source, irr::video::EHBT_SHADER_RESOURCE);
		m_driver->bindComputeBuffer(3, m_reduceBuffer, irr::video::EHBT_COMPUTE);
		m_driver->dispatchComputeShaderBound(irr::core::vector3d<irr::u32>(1, 1, 1));
		m_driver->unbindComputeResources();

		m_reduceBuffer->downloadFromGPU();
		memcpy(&result, m_reduceBuffer->getBufferPointer(), sizeof(int));
		return true;
	}

private:
	DispatchHelper(const DispatchHelper&);
	DispatchHelper& operator=(const DispatchHelper&);

	irr::video::IVideoDriver* m_driver;

	int m_patchMaterial;
	int m_reduceMaterial;

	irr::scene::IComputeBuffer* m_paramBuffer;
	irr::scene::IComputeBuffer* m_argsBuffer;
	irr::scene::IComputeBuffer* m_reduceBuffer;
};
}  // namespace b3IrrGpu

#endif  //B3_IRRLICHT_GPU_BUFFERS_H
