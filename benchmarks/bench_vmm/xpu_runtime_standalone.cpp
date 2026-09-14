// SPDX-FileCopyrightText: Copyright contributors to the kvcached project
// SPDX-License-Identifier: Apache-2.0

// Standalone (torch-free) implementation of kvcached::xpu_runtime for the VMM
// microbenchmark.
//
// csrc/xpu_runtime.cpp borrows PyTorch's SYCL context, because the extension
// hands its reserved VA range to at::from_blob() and engine kernels must be
// able to address the mappings. bench_vmm links no torch and launches no
// kernels, so it owns a context of its own instead.
//
// That difference is why this file lives here and not in csrc/: a context
// created here is NOT interoperable with PyTorch allocations. Never link it
// into kvcached.vmm_ops -- doing so would produce mappings that engine kernels
// cannot read, which is the exact failure mode the design took care to avoid.
#if defined(KVCACHED_USE_XPU)

#include "xpu_runtime.hpp"

#include <memory>
#include <sstream>
#include <stdexcept>
#include <vector>

// Narrow includes rather than <sycl/sycl.hpp>: the umbrella header drags in
// hundreds of deprecation warnings from device_traits.def under a plain host
// compiler.
#include <sycl/platform.hpp>
#include <sycl/queue.hpp>

// Declares sycl::ext::intel::info::device::free_memory
// (ext_intel_device_traits.def).
#include <sycl/info/info_desc.hpp>

namespace kvcached {
namespace xpu_runtime {

namespace {

// All GPUs of a single platform, plus one context spanning them. A Level Zero
// context cannot span platforms, and cross-context memory is not addressable,
// so the benchmark confines itself to the first platform that exposes a GPU.
struct Runtime {
  std::vector<sycl::device> devices;
  std::unique_ptr<sycl::context> ctx;
  std::vector<sycl::queue> queues;
  int current = 0;

  Runtime() {
    for (const auto &platform : sycl::platform::get_platforms()) {
      auto gpus = platform.get_devices(sycl::info::device_type::gpu);
      if (!gpus.empty()) {
        devices = std::move(gpus);
        break;
      }
    }
    if (devices.empty()) {
      throw std::runtime_error("No XPU (GPU) devices are available.");
    }
    ctx = std::make_unique<sycl::context>(devices);
    queues.reserve(devices.size());
    for (const auto &dev : devices) {
      queues.emplace_back(*ctx, dev);
    }
  }
};

// Constructed on first use. If construction throws (no GPU), the standard
// retries it on the next call, so the error surfaces at each entry point rather
// than being latched.
Runtime &runtime() {
  static Runtime rt;
  return rt;
}

int resolve(int dev_idx) {
  auto &rt = runtime();
  if (dev_idx < 0)
    return rt.current;
  const int count = static_cast<int>(rt.devices.size());
  if (dev_idx >= count) {
    std::ostringstream oss;
    oss << "XPU device index " << dev_idx << " is out of range; " << count
        << " XPU device(s) visible";
    throw std::runtime_error(oss.str());
  }
  return dev_idx;
}

} // namespace

const sycl::context &context() { return *runtime().ctx; }

const sycl::device &device(int dev_idx) {
  return runtime().devices[resolve(dev_idx)];
}

int device_count() { return static_cast<int>(runtime().devices.size()); }

int current_device() { return runtime().current; }

void set_current_device(int dev_idx) { runtime().current = resolve(dev_idx); }

void synchronize_device(int dev_idx) {
  // The benchmark submits no kernels, so this only drains the queue this
  // translation unit created. csrc/xpu_runtime.cpp instead waits on PyTorch's
  // streams, which is what actually matters in the extension.
  runtime().queues[resolve(dev_idx)].wait_and_throw();
}

void mem_get_info(int dev_idx, size_t *free_bytes, size_t *total_bytes) {
  const sycl::device &dev = device(dev_idx);

  const size_t total =
      static_cast<size_t>(dev.get_info<sycl::info::device::global_mem_size>());

  if (!dev.has(sycl::aspect::ext_intel_free_memory)) {
    throw std::runtime_error(
        "XPU device does not report free memory (sycl::aspect::"
        "ext_intel_free_memory unavailable). Export ZES_ENABLE_SYSMAN=1 before "
        "running the benchmark so the Level Zero sysman layer is enabled.");
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
