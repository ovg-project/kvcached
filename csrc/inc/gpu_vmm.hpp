// SPDX-FileCopyrightText: Copyright contributors to the kvcached project
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdlib>
#include <iostream>

#if defined(KVCACHED_USE_HIP)
#include <hip/hip_runtime.h>
#elif defined(KVCACHED_USE_CUDA)
#include <cuda.h>
#include <cuda_runtime.h>
#else
#error "kvcached requires one of KVCACHED_USE_HIP or KVCACHED_USE_CUDA."
#endif

namespace kvcached {
namespace gpu_vmm {

#if defined(KVCACHED_USE_HIP)

using status_t = hipError_t;
using allocation_handle_t = hipMemGenericAllocationHandle_t;
using allocation_prop_t = hipMemAllocationProp;
using access_desc_t = hipMemAccessDesc;

inline const char *backend_name() { return "HIP"; }

inline const char *error_string(status_t status) {
  return hipGetErrorString(status);
}

inline bool is_success(status_t status) { return status == hipSuccess; }

inline void check(status_t status, const char *tok, const char *file,
                  unsigned line) {
  if (!is_success(status)) {
    std::cerr << file << ':' << line << ' ' << tok << " failed in HIP runtime ("
              << static_cast<unsigned>(status) << "): " << error_string(status)
              << std::endl;
    std::abort();
  }
}

inline status_t initialize_runtime() { return hipInit(0); }

inline status_t set_device(int dev_idx) { return hipSetDevice(dev_idx); }

inline int current_device() {
  int dev_idx = -1;
  check(hipGetDevice(&dev_idx), "hipGetDevice(&dev_idx)", __FILE__, __LINE__);
  return dev_idx;
}

inline status_t get_vmm_support(int *supports_vmm, int dev_idx) {
  return hipDeviceGetAttribute(
      supports_vmm, hipDeviceAttributeVirtualMemoryManagementSupported,
      dev_idx);
}

inline allocation_prop_t make_pinned_device_allocation_prop(int dev_idx) {
  allocation_prop_t prop{};
  prop.type = hipMemAllocationTypePinned;
  prop.requestedHandleType = hipMemHandleTypeNone;
  prop.location.type = hipMemLocationTypeDevice;
  prop.location.id = dev_idx;
  return prop;
}

inline access_desc_t make_device_rw_access_desc(int dev_idx) {
  access_desc_t desc{};
  desc.location.type = hipMemLocationTypeDevice;
  desc.location.id = dev_idx;
  desc.flags = hipMemAccessFlagsProtReadWrite;
  return desc;
}

inline status_t get_allocation_granularity(size_t *granularity,
                                           const allocation_prop_t *prop) {
  return hipMemGetAllocationGranularity(granularity, prop,
                                        hipMemAllocationGranularityMinimum);
}

inline status_t mem_create(allocation_handle_t *handle, size_t size,
                           const allocation_prop_t *prop) {
  return hipMemCreate(handle, size, prop, 0ULL);
}

inline status_t mem_release(allocation_handle_t handle) {
  return hipMemRelease(handle);
}

inline status_t address_reserve(void **ptr, size_t size, size_t alignment,
                                void *preferred_addr = nullptr) {
  return hipMemAddressReserve(ptr, size, alignment, preferred_addr, 0ULL);
}

inline status_t address_free(void *ptr, size_t size) {
  return hipMemAddressFree(ptr, size);
}

inline status_t mem_map(void *ptr, size_t size, size_t offset,
                        allocation_handle_t handle) {
  return hipMemMap(ptr, size, offset, handle, 0ULL);
}

inline status_t mem_unmap(void *ptr, size_t size) {
  return hipMemUnmap(ptr, size);
}

inline status_t set_access(void *ptr, size_t size, const access_desc_t *desc,
                           size_t count) {
  return hipMemSetAccess(ptr, size, desc, count);
}

#elif defined(KVCACHED_USE_CUDA)

using drv_status_t = CUresult;
using rt_status_t = cudaError_t;
using allocation_handle_t = CUmemGenericAllocationHandle;
using allocation_prop_t = CUmemAllocationProp;
using access_desc_t = CUmemAccessDesc;

inline const char *backend_name() { return "CUDA"; }

inline const char *error_string(drv_status_t status) {
  const char *err = nullptr;
  (void)cuGetErrorString(status, &err);
  return err ? err : "unknown CUDA driver error";
}

inline const char *error_string(rt_status_t status) {
  return cudaGetErrorString(status);
}

inline bool is_success(drv_status_t status) { return status == CUDA_SUCCESS; }

inline bool is_success(rt_status_t status) { return status == cudaSuccess; }

inline void check(drv_status_t status, const char *tok, const char *file,
                  unsigned line) {
  if (!is_success(status)) {
    std::cerr << file << ':' << line << ' ' << tok << " failed in CUDA driver ("
              << static_cast<unsigned>(status) << "): " << error_string(status)
              << std::endl;
    std::abort();
  }
}

inline void check(rt_status_t status, const char *tok, const char *file,
                  unsigned line) {
  if (!is_success(status)) {
    std::cerr << file << ':' << line << ' ' << tok
              << " failed in CUDA runtime (" << static_cast<unsigned>(status)
              << "): " << error_string(status) << std::endl;
    std::abort();
  }
}

inline rt_status_t initialize_runtime() { return cudaFree(0); }

inline rt_status_t set_device(int dev_idx) { return cudaSetDevice(dev_idx); }

inline int current_device() {
  int dev_idx = -1;
  check(cudaGetDevice(&dev_idx), "cudaGetDevice(&dev_idx)", __FILE__, __LINE__);
  return dev_idx;
}

inline drv_status_t get_vmm_support(int *supports_vmm, int dev_idx) {
#if defined(CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED)
  constexpr auto attr = CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED;
#else
  constexpr auto attr =
      CU_DEVICE_ATTRIBUTE_VIRTUAL_ADDRESS_MANAGEMENT_SUPPORTED;
#endif
  return cuDeviceGetAttribute(supports_vmm, attr,
                              static_cast<CUdevice>(dev_idx));
}

inline allocation_prop_t make_pinned_device_allocation_prop(int dev_idx) {
  allocation_prop_t prop{};
  prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
  prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  prop.location.id = dev_idx;
  return prop;
}

inline access_desc_t make_device_rw_access_desc(int dev_idx) {
  access_desc_t desc{};
  desc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  desc.location.id = dev_idx;
  desc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
  return desc;
}

inline drv_status_t get_allocation_granularity(size_t *granularity,
                                               const allocation_prop_t *prop) {
  return cuMemGetAllocationGranularity(granularity, prop,
                                       CU_MEM_ALLOC_GRANULARITY_MINIMUM);
}

inline drv_status_t mem_create(allocation_handle_t *handle, size_t size,
                               const allocation_prop_t *prop) {
  return cuMemCreate(handle, size, prop, 0ULL);
}

inline drv_status_t mem_release(allocation_handle_t handle) {
  return cuMemRelease(handle);
}

inline drv_status_t address_reserve(void **ptr, size_t size, size_t alignment,
                                    void *preferred_addr = nullptr) {
  return cuMemAddressReserve(
      reinterpret_cast<CUdeviceptr *>(ptr), size, alignment,
      reinterpret_cast<CUdeviceptr>(preferred_addr), 0ULL);
}

inline drv_status_t address_free(void *ptr, size_t size) {
  return cuMemAddressFree(reinterpret_cast<CUdeviceptr>(ptr), size);
}

inline drv_status_t mem_map(void *ptr, size_t size, size_t offset,
                            allocation_handle_t handle) {
  return cuMemMap(reinterpret_cast<CUdeviceptr>(ptr), size, offset, handle,
                  0ULL);
}

inline drv_status_t mem_unmap(void *ptr, size_t size) {
  return cuMemUnmap(reinterpret_cast<CUdeviceptr>(ptr), size);
}

inline drv_status_t set_access(void *ptr, size_t size,
                               const access_desc_t *desc, size_t count) {
  return cuMemSetAccess(reinterpret_cast<CUdeviceptr>(ptr), size, desc, count);
}

#endif

} // namespace gpu_vmm
} // namespace kvcached
