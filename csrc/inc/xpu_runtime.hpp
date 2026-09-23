// SPDX-FileCopyrightText: Copyright contributors to the kvcached project
// SPDX-License-Identifier: Apache-2.0

#pragma once

#if defined(KVCACHED_USE_XPU)

#include <cstddef>

// Including <sycl/sycl.hpp> without -fsycl is correct for host-only code (we
// never emit device kernels), but the header warns about it. kvcached compiles
// with a plain host compiler, matching how the HIP backend avoids hipcc.
#ifndef SYCL_DISABLE_FSYCL_SYCLHPP_WARNING
#define SYCL_DISABLE_FSYCL_SYCLHPP_WARNING 1
#endif

#include <sycl/context.hpp>
#include <sycl/device.hpp>

namespace kvcached {
namespace xpu_runtime {

// Runtime/device concerns that Level Zero itself does not own, isolated here so
// that gpu_vmm.hpp stays free of PyTorch dependencies (benchmarks/bench_vmm
// links gpu_vmm.hpp without torch).
//
// Ownership: every handle returned below is owned by PyTorch's XPU device pool.
// kvcached borrows them and must never create or destroy a SYCL context. This
// is load-bearing rather than cosmetic: unlike CUDA/HIP, which map memory into
// a device's implicit primary context, Level Zero memory belongs to an explicit
// context and is not addressable from another one. kvcached hands its reserved
// VA range to from_blob() for engine kernels to read, so the reservation
// and every mapping must live in the same context PyTorch launches kernels in.

// PyTorch's per-process SYCL context. Borrowed; never destroyed.
const sycl::context &context();

// PyTorch's device handle. dev_idx < 0 means the current device.
const sycl::device &device(int dev_idx);

int device_count();
int current_device();
void set_current_device(int dev_idx);

// Waits for all PyTorch XPU streams on the device to drain. Level Zero has no
// device-wide synchronize, so this is the closest equivalent to
// cudaDeviceSynchronize() for kvcached's purposes: it guarantees no in-flight
// kernel still reads pages that are about to be unmapped.
void synchronize_device(int dev_idx);

// Level Zero core exposes no free-memory query, so this reports the SYCL
// device's total global memory and, when the device supports the Intel
// free-memory extension, its actual free memory. Throws if unavailable rather
// than substituting host memory statistics.
void mem_get_info(int dev_idx, size_t *free_bytes, size_t *total_bytes);

} // namespace xpu_runtime
} // namespace kvcached

#endif // KVCACHED_USE_XPU
