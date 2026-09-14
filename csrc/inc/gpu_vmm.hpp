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
#elif defined(KVCACHED_USE_XPU)
#include <cstdint>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>

#include "xpu_runtime.hpp"

#include <sycl/ext/oneapi/virtual_mem/physical_mem.hpp>
#include <sycl/ext/oneapi/virtual_mem/virtual_mem.hpp>
#else
#error                                                                         \
    "kvcached requires one of KVCACHED_USE_HIP, KVCACHED_USE_CUDA or KVCACHED_USE_XPU."
#endif

namespace kvcached {
namespace gpu_vmm {

#if defined(KVCACHED_USE_HIP)

using status_t = hipError_t;
using allocation_handle_t = hipMemGenericAllocationHandle_t;
using allocation_prop_t = hipMemAllocationProp;
using access_desc_t = hipMemAccessDesc;

inline const char *backend_name() { return "HIP"; }

// See the CUDA arm for what this guarantees; hipMemMap matches cuMemMap here.
inline constexpr bool supports_shared_page_mapping() { return true; }

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

inline status_t mem_get_info(size_t *free_bytes, size_t *total_bytes) {
  return hipMemGetInfo(free_bytes, total_bytes);
}

inline status_t device_synchronize() { return hipDeviceSynchronize(); }

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
                        allocation_handle_t handle, bool accessible = true) {
  // hipMemMap never grants access on its own; set_access() does that, so the
  // range is already inaccessible when the caller defers it.
  (void)accessible;
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

// True when one physical allocation may be mapped into several virtual ranges
// at once. cuMemMap supports it, which is what lets FTensor back every virtual
// page of a reservation with a single shared zero page (see
// FTensor::init_with_zero_) so that reads of never-allocated regions return
// zeros instead of faulting.
inline constexpr bool supports_shared_page_mapping() { return true; }

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

inline rt_status_t mem_get_info(size_t *free_bytes, size_t *total_bytes) {
  return cudaMemGetInfo(free_bytes, total_bytes);
}

inline rt_status_t device_synchronize() { return cudaDeviceSynchronize(); }

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
                            allocation_handle_t handle,
                            bool accessible = true) {
  // cuMemMap never grants access on its own; cuMemSetAccess does that, so the
  // range is already inaccessible when the caller defers it.
  (void)accessible;
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

#elif defined(KVCACHED_USE_XPU)

// Intel XPU backend.
//
// VMM comes from the sycl_ext_oneapi_virtual_mem extension, which the SYCL
// runtime forwards to Level Zero (zeVirtualMemReserve / zePhysicalMemCreate /
// zeVirtualMemMap / zeVirtualMemSetAccessAttribute and friends) on the Level
// Zero backend. Going through SYCL rather than calling ze* directly is what
// lets kvcached share PyTorch's context: see xpu_runtime.hpp for why that is a
// correctness requirement and not a convenience.
//
// Differences from the CUDA/HIP arms, all forced by the underlying API:
//   * failures are reported by exception, not by return code, so every entry
//     point funnels through detail::guard() to restore the status-based
//     contract the rest of kvcached expects;
//   * there is no allocation-property struct: the device is passed directly to
//     each call, so allocation_prop_t/access_desc_t only carry a device index;
//   * reservations take no alignment argument (see address_reserve);
//   * access mode is an argument to map() rather than a separate step, and
//     applies to a whole range rather than a per-peer descriptor array.

namespace se = sycl::ext::oneapi::experimental;

using status_t = int;
// physical_mem has no default constructor and is not an integral handle, but
// GPUPage stores allocation_handle_t by value and default-constructs it. Hand
// out opaque tokens into a registry instead; 0 is never issued, so a
// default-constructed handle is recognizably invalid.
using allocation_handle_t = uint64_t;

struct allocation_prop_t {
  int dev_idx;
};

struct access_desc_t {
  int dev_idx;
  se::address_access_mode mode;
};

namespace detail {

constexpr status_t kOk = 0;
constexpr status_t kErr = -1;

// Last failure on this thread. check() aborts and DRV_CALL_RET warns
// immediately after a non-zero status, so the message is always the one that
// belongs to the status being reported.
inline std::string &last_error() {
  static thread_local std::string msg;
  return msg;
}

// Converts the extension's exception-based failures into kvcached's
// status-based seam, keeping the SYCL diagnostic for error_string().
template <typename Fn> inline status_t guard(const char *op, Fn &&fn) {
  try {
    fn();
    return kOk;
  } catch (const std::exception &e) {
    last_error() = std::string(op) + ": " + e.what();
    return kErr;
  } catch (...) {
    last_error() = std::string(op) + ": unknown XPU error";
    return kErr;
  }
}

// Owns the physical allocations behind allocation_handle_t. Guarded because
// PageAllocator maps and frees pages from its prealloc thread as well as from
// the caller's thread.
class PhysicalMemRegistry {
public:
  allocation_handle_t create(const sycl::device &dev, const sycl::context &ctx,
                             size_t size) {
    // physical_mem(device, context, bytes) allocates device memory; there is
    // deliberately no host-memory fallback here.
    se::physical_mem phys(dev, ctx, size);
    std::lock_guard<std::mutex> lock(mtx_);
    const allocation_handle_t handle = ++next_handle_;
    pages_.emplace(handle, std::move(phys));
    return handle;
  }

  // Returned by value: physical_mem is a reference-counted handle, so a copy
  // keeps the allocation alive even if another thread releases the token.
  se::physical_mem get(allocation_handle_t handle) {
    std::lock_guard<std::mutex> lock(mtx_);
    const auto it = pages_.find(handle);
    if (it == pages_.end())
      throw std::runtime_error("unknown XPU allocation handle " +
                               std::to_string(handle));
    return it->second;
  }

  void destroy(allocation_handle_t handle) {
    std::lock_guard<std::mutex> lock(mtx_);
    if (pages_.erase(handle) == 0)
      throw std::runtime_error("release of unknown XPU allocation handle " +
                               std::to_string(handle));
  }

private:
  std::mutex mtx_;
  allocation_handle_t next_handle_ = 0;
  std::unordered_map<allocation_handle_t, se::physical_mem> pages_;
};

// Deliberately leaked. The registry is created on the first mem_create(), so
// necessarily after FTensorAllocator's namespace-scope singletons; a
// function-local static would therefore be destroyed *before* them, and any
// GPUPage still alive at exit would run ~GPUPage -> mem_release() ->
// registry().destroy() against a destroyed mutex and map. Leaking also keeps
// se::physical_mem destructors out of static teardown, where the SYCL runtime
// may already be gone. The pages it holds are freed by the driver at exit.
inline PhysicalMemRegistry &registry() {
  static PhysicalMemRegistry *reg = new PhysicalMemRegistry();
  return *reg;
}

} // namespace detail

inline const char *backend_name() { return "XPU"; }

// False, unlike CUDA and HIP: mapping one physical page into a second virtual
// range silently does not work here. zeVirtualMemMap accepts the second range
// and returns ZE_RESULT_SUCCESS, but the range is not aliased to the first --
// reading it does not observe the first range's writes. From there the device
// is lost either way:
//
//   write through the second range -> ZE_RESULT_ERROR_DEVICE_LOST
//   or just unmap it again         -> reports SUCCESS, but the device is gone
//                                     regardless: the next read of the *first*
//                                     range fails with DEVICE_LOST, a write to
//                                     it fails with OUT_OF_DEVICE_MEMORY, and
//                                     unmapping it segfaults in the driver
//
// So a read-only second mapping is no safer than a written one -- creating it
// at all is enough, and the unmap that reports success is what loses the
// device.
//
// Verified against the Level Zero API directly (no SYCL, no UR): same failure
// whether the two ranges come from one reservation or two, on three separate
// cards; the identical sequence with one physical page per range completes
// cleanly, so it is the sharing and not the call sequence. Arc Pro B60,
// level-zero 1.28.0 / intel-opencl-icd 26.18.38308.1. The spec neither permits
// nor forbids the sharing, so this is driver behavior and could change; the
// flag is the single place to flip if it does.
//
// FTensor therefore leaves XPU reservations unbacked instead of installing a
// shared zero page.
inline constexpr bool supports_shared_page_mapping() { return false; }

// Unlike the CUDA and HIP arms, this is not a pure function of `status`: SYCL
// reports failures as exception messages, not codes, so the text comes from the
// calling thread's last_error(). The returned pointer is only valid until the
// next failing seam call on this thread -- copy it if you need to keep it.
inline const char *error_string(status_t status) {
  if (status == detail::kOk)
    return "success";
  const std::string &msg = detail::last_error();
  return msg.empty() ? "unknown XPU error" : msg.c_str();
}

inline bool is_success(status_t status) { return status == detail::kOk; }

inline void check(status_t status, const char *tok, const char *file,
                  unsigned line) {
  if (!is_success(status)) {
    std::cerr << file << ':' << line << ' ' << tok << " failed in XPU runtime ("
              << status << "): " << error_string(status) << std::endl;
    std::abort();
  }
}

inline status_t initialize_runtime() {
  return detail::guard("initialize_runtime", [] {
    // Forces PyTorch's XPU device pool (and with it the SYCL/Level Zero
    // runtime) to come up before any VMM call, and fails here rather than
    // deeper in if no XPU is visible.
    (void)xpu_runtime::device_count();
    (void)xpu_runtime::context();
  });
}

inline status_t set_device(int dev_idx) {
  return detail::guard("set_device",
                       [dev_idx] { xpu_runtime::set_current_device(dev_idx); });
}

inline int current_device() {
  int dev_idx = -1;
  check(detail::guard("current_device",
                      [&dev_idx] { dev_idx = xpu_runtime::current_device(); }),
        "xpu_runtime::current_device()", __FILE__, __LINE__);
  return dev_idx;
}

inline status_t mem_get_info(size_t *free_bytes, size_t *total_bytes) {
  return detail::guard("mem_get_info", [&] {
    xpu_runtime::mem_get_info(-1, free_bytes, total_bytes);
  });
}

inline status_t device_synchronize() {
  return detail::guard("device_synchronize",
                       [] { xpu_runtime::synchronize_device(-1); });
}

inline status_t get_allocation_granularity(size_t *granularity,
                                           const allocation_prop_t *prop) {
  return detail::guard("get_allocation_granularity", [&] {
    // granularity_mode::minimum is the counterpart of
    // CU_MEM_ALLOC_GRANULARITY_MINIMUM.
    *granularity = se::get_mem_granularity(xpu_runtime::device(prop->dev_idx),
                                           xpu_runtime::context(),
                                           se::granularity_mode::minimum);
    if (*granularity == 0)
      throw std::runtime_error("device reported a zero allocation granularity");
  });
}

inline status_t get_vmm_support(int *supports_vmm, int dev_idx) {
  *supports_vmm = 0;

  // Level Zero exposes no counterpart to
  // CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED, so probe
  // behaviorally with one full reserve -> create -> map -> unmap -> release
  // cycle at minimum granularity.
  const status_t probe = detail::guard("get_vmm_support", [&] {
    const sycl::context &ctx = xpu_runtime::context();
    const sycl::device &dev = xpu_runtime::device(dev_idx);
    const size_t size =
        se::get_mem_granularity(dev, ctx, se::granularity_mode::minimum);
    if (size == 0)
      return;

    const uintptr_t vaddr = se::reserve_virtual_mem(0, size, ctx);
    try {
      se::physical_mem phys(dev, ctx, size);
      phys.map(vaddr, size, se::address_access_mode::read_write, 0);
      se::unmap(reinterpret_cast<void *>(vaddr), size, ctx);
    } catch (...) {
      // Never leak the probe reservation, whichever step failed -- but do not
      // let cleanup replace the exception that explains why the probe failed.
      try {
        se::free_virtual_mem(vaddr, size, ctx);
      } catch (...) {
      }
      throw;
    }
    se::free_virtual_mem(vaddr, size, ctx);
    *supports_vmm = 1;
  });

  // A throwing probe means "this device cannot do VMM", not "the query failed".
  // Report success either way so FTensorAllocator::init_gpu_() can emit its own
  // diagnostic, matching how CUDA's attribute query behaves. But the probe
  // allocates real memory, so "no VMM", "bad device index" and "transient OOM"
  // all reach the caller as the same bare "not supported" -- print the SYCL
  // message here, because error_string() is unreachable once we return kOk.
  if (!is_success(probe))
    std::cerr << "kvcached: XPU VMM probe failed on device " << dev_idx << ": "
              << detail::last_error() << std::endl;
  return detail::kOk;
}

inline allocation_prop_t make_pinned_device_allocation_prop(int dev_idx) {
  allocation_prop_t prop{};
  prop.dev_idx = dev_idx;
  return prop;
}

inline access_desc_t make_device_rw_access_desc(int dev_idx) {
  access_desc_t desc{};
  desc.dev_idx = dev_idx;
  desc.mode = se::address_access_mode::read_write;
  return desc;
}

inline status_t mem_create(allocation_handle_t *handle, size_t size,
                           const allocation_prop_t *prop) {
  return detail::guard("mem_create", [&] {
    *handle = detail::registry().create(xpu_runtime::device(prop->dev_idx),
                                        xpu_runtime::context(), size);
  });
}

inline status_t mem_release(allocation_handle_t handle) {
  return detail::guard("mem_release",
                       [&] { detail::registry().destroy(handle); });
}

inline status_t address_reserve(void **ptr, size_t size, size_t alignment,
                                void *preferred_addr = nullptr) {
  return detail::guard("address_reserve", [&] {
    const sycl::context &ctx = xpu_runtime::context();
    const uintptr_t vaddr = se::reserve_virtual_mem(
        reinterpret_cast<uintptr_t>(preferred_addr), size, ctx);

    // reserve_virtual_mem() has no alignment argument (nor does
    // zeVirtualMemReserve underneath it) -- it aligns to the device page size.
    // kvcached asks for 2MB, so verify rather than silently ignore the request.
    if (alignment != 0 && (vaddr % alignment) != 0) {
      se::free_virtual_mem(vaddr, size, ctx);
      std::ostringstream oss;
      oss << "reservation 0x" << std::hex << vaddr << std::dec
          << " is not aligned to " << alignment << " bytes";
      throw std::runtime_error(oss.str());
    }
    *ptr = reinterpret_cast<void *>(vaddr);
  });
}

inline status_t address_free(void *ptr, size_t size) {
  return detail::guard("address_free", [&] {
    se::free_virtual_mem(reinterpret_cast<uintptr_t>(ptr), size,
                         xpu_runtime::context());
  });
}

inline status_t mem_map(void *ptr, size_t size, size_t offset,
                        allocation_handle_t handle, bool accessible = true) {
  return detail::guard("mem_map", [&] {
    // Access mode is an argument to map() here rather than a separate step, so
    // `accessible` has to be honored at map time to match cuMemMap, which
    // leaves a range faulting until cuMemSetAccess runs. A later set_access()
    // call still overrides whatever is chosen here.
    const auto mode = accessible ? se::address_access_mode::read_write
                                 : se::address_access_mode::none;
    void *mapped = detail::registry().get(handle).map(
        reinterpret_cast<uintptr_t>(ptr), size, mode, offset);
    if (mapped != ptr) {
      std::ostringstream oss;
      oss << "mapped at " << mapped << ", not the requested " << ptr;
      throw std::runtime_error(oss.str());
    }
  });
}

inline status_t mem_unmap(void *ptr, size_t size) {
  return detail::guard("mem_unmap",
                       [&] { se::unmap(ptr, size, xpu_runtime::context()); });
}

inline status_t set_access(void *ptr, size_t size, const access_desc_t *desc,
                           size_t count) {
  return detail::guard("set_access", [&] {
    // One access mode covers the whole range; there is no equivalent of CUDA's
    // per-peer-device descriptor array. kvcached only ever passes one.
    if (count != 1)
      throw std::runtime_error(
          "XPU set_access expects exactly one descriptor, got " +
          std::to_string(count));
    se::set_access_mode(ptr, size, desc->mode, xpu_runtime::context());
  });
}

#endif

} // namespace gpu_vmm
} // namespace kvcached
