// SPDX-FileCopyrightText: Copyright contributors to the kvcached project
// SPDX-License-Identifier: Apache-2.0

#include <Python.h>

#include <string>
#include <vector>

#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/tensor.h>

#include "allocator.hpp"
#include "constants.hpp"
#include "torch_utils.hpp"

namespace kvcached {

void init_kvcached(std::string dev_str, int64_t page_size,
                   bool contiguous_layout) {
  FTensorAllocator::init(dev_str, static_cast<size_t>(page_size),
                         contiguous_layout);
}

void shutdown_kvcached() { FTensorAllocator::shutdown(); }

std::vector<torch::stable::Tensor>
create_kv_tensors(int64_t size, int64_t dtype_size, std::string dev_str,
                  int64_t num_layers, int64_t num_kv_buffers,
                  int64_t group_id) {
  auto allocator = FTensorAllocator::global_allocator(group_id);
  auto dtype_ = torch_dtype_from_size(static_cast<size_t>(dtype_size));
  return allocator->create_kv_tensors(static_cast<size_t>(size), dtype_,
                                      dev_str, num_layers, num_kv_buffers);
}

bool kv_tensors_created(int64_t group_id) {
  auto allocator = FTensorAllocator::global_allocator(group_id);
  return allocator->kv_tensors_created();
}

bool map_to_kv_tensors(std::vector<int64_t> offsets, int64_t group_id) {
  auto allocator = FTensorAllocator::global_allocator(group_id);
  return allocator->map_to_kv_tensors(offsets);
}

bool unmap_from_kv_tensors(std::vector<int64_t> offsets, int64_t group_id) {
  auto allocator = FTensorAllocator::global_allocator(group_id);
  return allocator->unmap_from_kv_tensors(offsets);
}

} // namespace kvcached

STABLE_TORCH_LIBRARY(kvcached, m) {
  m.def("init_kvcached(str dev_str, int page_size=0, bool "
        "contiguous_layout=False) -> ()");
  m.def("shutdown_kvcached() -> ()");
  m.def("create_kv_tensors(int size, int dtype_size, str dev_str, int "
        "num_layers, int num_kv_buffers=2, int group_id=0) -> Tensor[]");
  m.def("kv_tensors_created(int group_id=0) -> bool");
  m.def("map_to_kv_tensors(int[] offsets, int group_id=0) -> bool");
  m.def("unmap_from_kv_tensors(int[] offsets, int group_id=0) -> bool");
}

STABLE_TORCH_LIBRARY_IMPL(kvcached, CompositeExplicitAutograd, m) {
  m.impl("init_kvcached", TORCH_BOX(&kvcached::init_kvcached));
  m.impl("shutdown_kvcached", TORCH_BOX(&kvcached::shutdown_kvcached));
  m.impl("create_kv_tensors", TORCH_BOX(&kvcached::create_kv_tensors));
  m.impl("kv_tensors_created", TORCH_BOX(&kvcached::kv_tensors_created));
  m.impl("map_to_kv_tensors", TORCH_BOX(&kvcached::map_to_kv_tensors));
  m.impl("unmap_from_kv_tensors", TORCH_BOX(&kvcached::unmap_from_kv_tensors));
}

// Minimal Python module init.
// Importing this module triggers the STABLE_TORCH_LIBRARY static constructors
// above, which register the ops in PyTorch's dispatcher.
static PyModuleDef _vmm_ops_lib_module = {
    PyModuleDef_HEAD_INIT,
    "_vmm_ops_lib",
    "kvcached VMM operations (stable ABI)",
    -1,
};

extern "C" PyMODINIT_FUNC PyInit__vmm_ops_lib(void) {
  return PyModule_Create(&_vmm_ops_lib_module);
}
