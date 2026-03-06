// SPDX-FileCopyrightText: Copyright contributors to the kvcached project
// SPDX-License-Identifier: Apache-2.0

#include <pybind11/pybind11.h>
#include <string>
#include <torch/extension.h>
#include <vector>

#include "allocator.hpp"
#include "constants.hpp"
#include "torch_utils.hpp"

namespace kvcached {

void init_kvcached(const std::string &dev_str, size_t page_size = 0,
                   bool contiguous_layout = false) {
  py::gil_scoped_release release;
  FTensorAllocator::init(dev_str, page_size, contiguous_layout);
}

void shutdown_kvcached() {
  py::gil_scoped_release release;
  FTensorAllocator::shutdown();
}

std::vector<torch::Tensor> create_kv_tensors(size_t size, size_t dtype_size,
                                             const std::string &dev_str,
                                             int64_t num_layers,
                                             int64_t num_kv_buffers = 2,
                                             int64_t group_id = 0) {
  py::gil_scoped_release release;
  auto allocator = FTensorAllocator::global_allocator();
  auto dtype_ = torch_dtype_from_size(dtype_size);
  return allocator->create_kv_tensors(size, dtype_, dev_str, num_layers,
                                      num_kv_buffers, group_id);
}

bool kv_tensors_created(int64_t group_id = 0) {
  py::gil_scoped_release release;
  auto allocator = FTensorAllocator::global_allocator();
  return allocator->kv_tensors_created(group_id);
}

bool map_to_kv_tensors(const std::vector<offset_t> &offsets,
                       int64_t group_id = 0) {
  py::gil_scoped_release release;
  auto allocator = FTensorAllocator::global_allocator();
  return allocator->map_to_kv_tensors(offsets, group_id);
}

bool unmap_from_kv_tensors(const std::vector<offset_t> &offsets,
                           int64_t group_id = 0) {
  py::gil_scoped_release release;
  auto allocator = FTensorAllocator::global_allocator();
  return allocator->unmap_from_kv_tensors(offsets, group_id);
}

} // namespace kvcached

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.doc() = "kvcached VMM plugin";

  m.def("init_kvcached", &kvcached::init_kvcached, "Initialize kvcached",
        py::arg("dev_str"), py::arg("page_size") = 0,
        py::arg("contiguous_layout") = true);
  m.def("shutdown_kvcached", &kvcached::shutdown_kvcached, "Shutdown kvcached");
  m.def("create_kv_tensors", &kvcached::create_kv_tensors, "create_kv_tensors",
        py::arg("size"), py::arg("dtype_size"), py::arg("dev_str"),
        py::arg("num_layers"), py::arg("num_kv_buffers") = 2,
        py::arg("group_id") = 0);
  m.def("kv_tensors_created", &kvcached::kv_tensors_created,
        "kv_tensors_created", py::arg("group_id") = 0);
  m.def("map_to_kv_tensors", &kvcached::map_to_kv_tensors, "map_to_kv_tensors",
        py::arg("offsets"), py::arg("group_id") = 0);
  m.def("unmap_from_kv_tensors", &kvcached::unmap_from_kv_tensors,
        "unmap_from_kv_tensors", py::arg("offsets"), py::arg("group_id") = 0);
}
