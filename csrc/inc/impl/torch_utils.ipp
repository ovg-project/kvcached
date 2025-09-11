// SPDX-FileCopyrightText: Copyright contributors to the kvcached project
// SPDX-License-Identifier: Apache-2.0

#pragma once

namespace kvcached {

static inline torch::Dtype torch_dtype_cast(const py::object &dtype) {
  if (dtype.is(py::module_::import("torch").attr("float32")))
    return torch::kFloat32;
  if (dtype.is(py::module_::import("torch").attr("float64")))
    return torch::kFloat64;
  if (dtype.is(py::module_::import("torch").attr("float16")))
    return torch::kFloat16;
  if (dtype.is(py::module_::import("torch").attr("int32")))
    return torch::kInt32;
  if (dtype.is(py::module_::import("torch").attr("int64")))
    return torch::kInt64;
  if (dtype.is(py::module_::import("torch").attr("int16")))
    return torch::kInt16;
  if (dtype.is(py::module_::import("torch").attr("int8")))
    return torch::kInt8;
  if (dtype.is(py::module_::import("torch").attr("uint8")))
    return torch::kUInt8;
  if (dtype.is(py::module_::import("torch").attr("bool")))
    return torch::kBool;

  throw std::runtime_error("Unsupported dtype: " +
                           py::str(dtype).cast<std::string>());
}

static inline torch::Dtype torch_dtype_from_size(size_t dtype_size) {
  switch (dtype_size) {
  case 1:
    return torch::kInt8;
  case 2:
    return torch::kInt16;
  case 4:
    return torch::kInt32;
  case 8:
    return torch::kInt64;
  default:
    throw std::runtime_error("Unsupported dtype size: " +
                             std::to_string(dtype_size));
  }
}

} // namespace kvcached
