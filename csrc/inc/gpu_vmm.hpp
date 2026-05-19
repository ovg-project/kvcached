// SPDX-FileCopyrightText: Copyright contributors to the kvcached project
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Thin GPU-VMM compatibility shim (ROCm port, Phase 2).
//
// torch's hipify pass mechanically renames ~all of kvcached's CUDA driver
// surface (cu* -> hip*, CU* enums -> hip* enums). This header covers the two
// places hipify CANNOT: a symbol missing from its CUDA->HIP map, and an API
// signature that genuinely differs between the two VMM ABIs. The CUDA build
// (un-hipified original) takes the #else branch; the HIP build (hipify output,
// compiled with -D__HIP_PLATFORM_AMD__=1) takes the #if branch.

#include <cstddef>

#if defined(__HIP_PLATFORM_AMD__)

#include <hip/hip_runtime.h>

// hipify's CUDA2HIP table has no entry for this device attribute, so the
// literal token survives untranslated and fails to compile. Map it here.
// (Phase 0 spike confirmed this attribute reads 1 on the gfx90a MI250X.)
#define KVCACHED_GPU_ATTR_VMM_SUPPORTED                                        \
  hipDeviceAttributeVirtualMemoryManagementSupported

// CUDA's cuMemAddressReserve/Map/Unmap/SetAccess take an integer CUdeviceptr;
// HIP's hipMem* counterparts take a void*. A textual rename can't bridge a
// signature difference, so the fixed-address hint must be a real void* on HIP.
static inline void *kvcached_vmm_addr(size_t addr) {
  return reinterpret_cast<void *>(addr);
}

#else

#include <cuda.h>

#define KVCACHED_GPU_ATTR_VMM_SUPPORTED                                        \
  CU_DEVICE_ATTRIBUTE_VIRTUAL_ADDRESS_MANAGEMENT_SUPPORTED

static inline CUdeviceptr kvcached_vmm_addr(size_t addr) {
  return static_cast<CUdeviceptr>(addr);
}

#endif
