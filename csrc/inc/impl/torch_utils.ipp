// SPDX-FileCopyrightText: Copyright contributors to the kvcached project
// SPDX-License-Identifier: Apache-2.0

#pragma once

namespace kvcached {

static inline c10::ScalarType torch_dtype_cast(const py::object &dtype) {
  if (dtype.is(py::module_::import("torch").attr("float32")))
    return c10::ScalarType::Float;
  if (dtype.is(py::module_::import("torch").attr("float64")))
    return c10::ScalarType::Double;
  if (dtype.is(py::module_::import("torch").attr("float16")))
    return c10::ScalarType::Half;
  if (dtype.is(py::module_::import("torch").attr("int32")))
    return c10::ScalarType::Int;
  if (dtype.is(py::module_::import("torch").attr("int64")))
    return c10::ScalarType::Long;
  if (dtype.is(py::module_::import("torch").attr("int16")))
    return c10::ScalarType::Short;
  if (dtype.is(py::module_::import("torch").attr("int8")))
    return c10::ScalarType::Char;
  if (dtype.is(py::module_::import("torch").attr("uint8")))
    return c10::ScalarType::Byte;
  if (dtype.is(py::module_::import("torch").attr("bool")))
    return c10::ScalarType::Bool;

  throw std::runtime_error("Unsupported dtype: " +
                           py::str(dtype).cast<std::string>());
}

static inline c10::ScalarType torch_dtype_from_size(size_t dtype_size) {
  switch (dtype_size) {
  case 1:
    return c10::ScalarType::Char;
  case 2:
    return c10::ScalarType::Short;
  case 4:
    return c10::ScalarType::Int;
  case 8:
    return c10::ScalarType::Long;
  default:
    throw std::runtime_error("Unsupported dtype size: " +
                             std::to_string(dtype_size));
  }
}

} // namespace kvcached
