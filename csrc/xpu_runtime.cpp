// SPDX-FileCopyrightText: Copyright contributors to the kvcached project
// SPDX-License-Identifier: Apache-2.0

// The whole translation unit is XPU-only: setup.py compiles every csrc/*.cpp
// for all backends, so CUDA and HIP builds see an empty object file here and
// never link against libsycl or libc10_xpu.
#if defined(KVCACHED_USE_XPU)

#include "xpu_runtime.hpp"

#include <sstream>
#include <stdexcept>
#include <string>

// The rest of csrc targets the PyTorch stable ABI, and c10/xpu/* refuses to
// compile while either gate setup.py sets -- TORCH_TARGET_VERSION or
// TORCH_STABLE_ONLY -- is defined. This one file has to opt out:
// the stable ABI exposes no way to reach the SYCL context PyTorch launches
// kernels in, and Level Zero memory is only addressable from the context it was
// mapped in, so borrowing that context is what makes kvcached's pages readable
// by engine kernels at all. syncStreamsOnDevice() has no stable equivalent
// either, and it is the barrier that keeps an in-flight kernel from reading
// pages being unmapped -- on this driver a stray access costs the device.
//
// Safe to confine here: nothing below crosses the ABI boundary. This file
// defines no shared type and exposes only the xpu_runtime signatures in
// xpu_runtime.hpp, which name SYCL types, not c10 ones, so no type is compiled
// two ways in the extension. XPU wheels do link libc10_xpu and therefore still
// need a rebuild per PyTorch version; CUDA and HIP keep the stable ABI's
// cross-version property, since they reach their drivers without PyTorch.
// tests/test_stable_abi.py allowlists exactly the c10::xpu symbols this file
// needs, so any further unstable dependency still fails the suite.
#undef TORCH_TARGET_VERSION
#undef TORCH_STABLE_ONLY

#include <c10/xpu/XPUFunctions.h>
#include <c10/xpu/XPUStream.h>

// Declares sycl::ext::intel::info::device::free_memory
// (ext_intel_device_traits.def).
#include <sycl/info/info_desc.hpp>

namespace kvcached {
namespace xpu_runtime {

namespace {

c10::DeviceIndex resolve(int dev_idx) {
  if (dev_idx < 0)
    return c10::xpu::current_device();
  const int count = static_cast<int>(c10::xpu::device_count());
  if (dev_idx >= count) {
    std::ostringstream oss;
    oss << "XPU device index " << dev_idx << " is out of range; " << count
        << " XPU device(s) visible";
    throw std::runtime_error(oss.str());
  }
  return static_cast<c10::DeviceIndex>(dev_idx);
}

} // namespace

const sycl::context &context() { return c10::xpu::get_device_context(); }

const sycl::device &device(int dev_idx) {
  return c10::xpu::get_raw_device(resolve(dev_idx));
}

int device_count() {
  // Fails loudly when no XPU is visible instead of letting a later VMM call
  // abort with an opaque SYCL error.
  return static_cast<int>(c10::xpu::device_count_ensure_non_zero());
}

int current_device() { return static_cast<int>(c10::xpu::current_device()); }

void set_current_device(int dev_idx) { c10::xpu::set_device(resolve(dev_idx)); }

void synchronize_device(int dev_idx) {
  c10::xpu::syncStreamsOnDevice(resolve(dev_idx));
}

void mem_get_info(int dev_idx, size_t *free_bytes, size_t *total_bytes) {
  const sycl::device &dev = device(dev_idx);

  const size_t total =
      static_cast<size_t>(dev.get_info<sycl::info::device::global_mem_size>());

  if (!dev.has(sycl::aspect::ext_intel_free_memory)) {
    // Reporting total-as-free would make PageAllocator believe the whole device
    // is available and OOM under load; host memory statistics would be worse
    // still. Fail with an actionable message instead.
    throw std::runtime_error(
        "XPU device does not report free memory (sycl::aspect::"
        "ext_intel_free_memory unavailable). Export ZES_ENABLE_SYSMAN=1 before "
        "starting the process so the Level Zero sysman layer is enabled.");
  }

  const size_t free = static_cast<size_t>(
      dev.get_info<sycl::ext::intel::info::device::free_memory>());

  if (total_bytes != nullptr)
    *total_bytes = total;
  if (free_bytes != nullptr)
    *free_bytes = free;
}

} // namespace xpu_runtime
} // namespace kvcached

#endif // KVCACHED_USE_XPU
