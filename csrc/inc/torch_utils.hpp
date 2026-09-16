// SPDX-FileCopyrightText: Copyright contributors to the kvcached project
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <stdexcept>
#include <string>

#include <torch/headeronly/core/ScalarType.h>

namespace kvcached {

// Map a raw element size (in bytes) to a stable-ABI ScalarType. kvcached only
// cares about the element width, not the semantic type, so any integer type of
// the right size works.
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

// Element size of a stable-ABI ScalarType. The stable Tensor exposes
// element_size(), but FTensor needs the size of a bare ScalarType (before any
// tensor exists), which the stable ABI does not provide as a free function.
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
