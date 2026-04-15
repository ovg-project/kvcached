// SPDX-FileCopyrightText: Copyright contributors to the kvcached project
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <stdexcept>
#include <string>

#include <torch/headeronly/core/ScalarType.h>

namespace kvcached {

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
