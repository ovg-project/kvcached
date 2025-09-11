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
                                             int64_t num_layers) {
  py::gil_scoped_release release;
  auto allocator = FTensorAllocator::global_allocator();
  auto dtype_ = torch_dtype_from_size(dtype_size);
  return allocator->create_kv_tensors(size, dtype_, dev_str, num_layers);
}

bool kv_tensors_created() {
  py::gil_scoped_release release;
  auto allocator = FTensorAllocator::global_allocator();
  return allocator->kv_tensors_created();
}

bool map_to_kv_tensors(const std::vector<offset_t> &offsets) {
  py::gil_scoped_release release;
  auto allocator = FTensorAllocator::global_allocator();
  return allocator->map_to_kv_tensors(offsets);
}

bool unmap_from_kv_tensors(const std::vector<offset_t> &offsets) {
  py::gil_scoped_release release;
  auto allocator = FTensorAllocator::global_allocator();
  return allocator->unmap_from_kv_tensors(offsets);
}

} // namespace kvcached

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.doc() = "kvcached VMM plugin";

  m.def("init_kvcached", &kvcached::init_kvcached, "Initialize kvcached",
        py::arg("dev_str"), py::arg("page_size") = 0,
        py::arg("contiguous_layout") = true);
  m.def("shutdown_kvcached", &kvcached::shutdown_kvcached, "Shutdown kvcached");
  m.def("create_kv_tensors", &kvcached::create_kv_tensors, "create_kv_tensors");
  m.def("kv_tensors_created", &kvcached::kv_tensors_created,
        "kv_tensors_created");
  m.def("map_to_kv_tensors", &kvcached::map_to_kv_tensors, "map_to_kv_tensors");
  m.def("unmap_from_kv_tensors", &kvcached::unmap_from_kv_tensors,
        "unmap_from_kv_tensors");
}
