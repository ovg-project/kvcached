// SPDX-FileCopyrightText: Copyright contributors to the kvcached project
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <pybind11/pytypes.h>
#include <string>
#include <torch/extension.h>

namespace kvcached {

namespace py = pybind11;
static inline torch::Dtype torch_dtype_cast(const py::object &dtype);
static inline torch::Dtype torch_dtype_from_size(size_t dtype_size);

} // namespace kvcached

#include "impl/torch_utils.ipp"
