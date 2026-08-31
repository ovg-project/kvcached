// SPDX-FileCopyrightText: Copyright contributors to the kvcached project
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <c10/core/Device.h>

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

} // namespace kvcached
