// SPDX-FileCopyrightText: Copyright contributors to the kvcached project
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <c10/core/Device.h>

#include "gpu_vmm.hpp"

namespace kvcached {

// True when `dev` is the accelerator this build targets, i.e. the device whose
// memory is managed through gpu_vmm rather than through host mmap().
//
// Deliberately not a bare dev.is_cuda(): PyTorch's ROCm build reports AMD GPUs
// as c10::DeviceType::CUDA, which is why adding the HIP backend needed no
// dispatch changes at all. A real Intel GPU is c10::DeviceType::XPU, so an
// is_cuda() test would send every XPU allocation down the CPU path and quietly
// serve the KV cache out of host memory instead of failing.
inline bool is_accelerator(const c10::Device &dev) {
#if defined(KVCACHED_USE_XPU)
  return dev.is_xpu();
#else
  return dev.is_cuda();
#endif
}

// True when an FTensor on `dev` will actually map the shared zero page, and so
// when there is any point allocating one.
//
// The zero page is what makes a read of never-allocated KV return zeros instead
// of faulting, and it works by mapping one physical page into every virtual
// page of the reservation. A backend that cannot alias a page that way never
// maps it (see FTensor::init_with_zero_), so allocating one there would pin
// device memory that nothing can ever read -- a whole compound page, sized for
// every layer and KV buffer, under the contiguous layout. Both the allocation
// and the mapping ask this, so the two cannot disagree.
inline bool uses_zero_page(const c10::Device &dev) {
  return !is_accelerator(dev) || gpu_vmm::supports_shared_page_mapping();
}

} // namespace kvcached
