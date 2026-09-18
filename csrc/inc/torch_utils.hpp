// SPDX-FileCopyrightText: Copyright contributors to the kvcached project
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <stdexcept>
#include <string>

#ifndef TORCH_TARGET_VERSION
#error "kvcached requires the PyTorch stable ABI"
#endif

#include <torch/csrc/stable/device.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/headeronly/core/ScalarType.h>

// Fail if the target is newer than the libtorch headers
// TODO: drop torch>=2.15, PyTorch ships this guard (pytorch/pytorch#193962).
#if TORCH_TARGET_VERSION > TORCH_ABI_VERSION
#error "TORCH_TARGET_VERSION is newer than TORCH_ABI_VERSION"
#endif

namespace kvcached {

// Map a raw element size to a ScalarType. Only the width matters, so any type
// of the right size works.
static inline torch::headeronly::ScalarType
torch_dtype_from_size(size_t dtype_size) {
  using ST = torch::headeronly::ScalarType;
  switch (dtype_size) {
  case 1:
    return ST::Char;
  case 2:
    return ST::Short;
  case 4:
    return ST::Int;
  case 8:
    return ST::Long;
  default:
    throw std::runtime_error("Unsupported dtype size: " +
                             std::to_string(dtype_size));
  }
}

// Element size of a bare ScalarType, needed before any tensor exists (to size
// the mapping). The stable ABI has no free function for it, so compute
// directly.
static inline size_t element_size(torch::headeronly::ScalarType dtype) {
  using ST = torch::headeronly::ScalarType;
  switch (dtype) {
  case ST::Byte:
  case ST::Char:
  case ST::Bool:
    return 1;
  case ST::Short:
  case ST::Half:
  case ST::BFloat16:
    return 2;
  case ST::Int:
  case ST::Float:
    return 4;
  case ST::Long:
  case ST::Double:
    return 8;
  default:
    throw std::runtime_error("Unsupported dtype for element_size");
  }
}

} // namespace kvcached
