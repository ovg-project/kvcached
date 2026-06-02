// SPDX-FileCopyrightText: Copyright contributors to the kvcached project
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <c10/core/ScalarType.h>
#include <string>

namespace kvcached {

namespace py = pybind11;
static inline c10::ScalarType torch_dtype_cast(const py::object &dtype);
static inline c10::ScalarType torch_dtype_from_size(size_t dtype_size);

} // namespace kvcached

#include "impl/torch_utils.ipp"
