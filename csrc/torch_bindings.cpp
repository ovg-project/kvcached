// SPDX-FileCopyrightText: Copyright contributors to the kvcached project
// SPDX-License-Identifier: Apache-2.0

#include <memory>
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <string>
#include <torch/extension.h>
#include <vector>

#include "allocator.hpp"
#include "constants.hpp"
#include "page_allocator.hpp"
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

std::vector<torch::Tensor>
create_kv_tensors(size_t size, size_t dtype_size, const std::string &dev_str,
                  int64_t num_layers, int64_t num_kv_buffers = 2,
                  int64_t group_id = 0, bool unified_pool = false) {
  py::gil_scoped_release release;
  auto allocator = FTensorAllocator::global_allocator(group_id);
  auto dtype_ = torch_dtype_from_size(dtype_size);
  return allocator->create_kv_tensors(size, dtype_, dev_str, num_layers,
                                      num_kv_buffers, unified_pool);
}

bool kv_tensors_created(int64_t group_id = 0) {
  py::gil_scoped_release release;
  auto allocator = FTensorAllocator::global_allocator(group_id);
  return allocator->kv_tensors_created();
}

bool map_to_kv_tensors(const std::vector<offset_t> &offsets,
                       int64_t group_id = 0) {
  py::gil_scoped_release release;
  auto allocator = FTensorAllocator::global_allocator(group_id);
  return allocator->map_to_kv_tensors(offsets);
}

bool unmap_from_kv_tensors(const std::vector<offset_t> &offsets,
                           int64_t group_id = 0) {
  py::gil_scoped_release release;
  auto allocator = FTensorAllocator::global_allocator(group_id);
  return allocator->unmap_from_kv_tensors(offsets);
}

// PageAllocator bindings
std::shared_ptr<PageAllocator> create_page_allocator(
    int64_t num_layers, int64_t mem_size_per_layer, int64_t page_size,
    int64_t world_size = 1, int64_t pp_rank = 0, bool async_sched = false,
    bool contiguous_layout = true, bool enable_page_prealloc = true,
    int64_t num_kv_buffers = 2, int64_t group_id = 0) {

  return std::make_shared<PageAllocator>(
      num_layers, mem_size_per_layer, page_size, world_size, pp_rank,
      async_sched, contiguous_layout, enable_page_prealloc, num_kv_buffers,
      group_id);
}

// PageAllocator method bindings
void page_allocator_start_prealloc_thread(
    std::shared_ptr<PageAllocator> allocator) {
  allocator->start_prealloc_thread();
}

void page_allocator_stop_prealloc_thread(
    std::shared_ptr<PageAllocator> allocator) {
  allocator->stop_prealloc_thread();
}

std::shared_ptr<InternalPage>
page_allocator_alloc_page(std::shared_ptr<PageAllocator> allocator) {
  return allocator->alloc_page();
}

void page_allocator_free_page(std::shared_ptr<PageAllocator> allocator,
                              page_id_t page_id) {
  allocator->free_page(page_id);
}

void page_allocator_free_pages(std::shared_ptr<PageAllocator> allocator,
                               const std::vector<page_id_t> &page_ids) {
  allocator->free_pages(page_ids);
}

bool page_allocator_resize(std::shared_ptr<PageAllocator> allocator,
                           int64_t new_mem_size) {
  return allocator->resize(new_mem_size);
}

void page_allocator_trim(std::shared_ptr<PageAllocator> allocator) {
  allocator->trim();
}

void page_allocator_reset_free_page_order(
    std::shared_ptr<PageAllocator> allocator) {
  allocator->reset_free_page_order();
}

int64_t
page_allocator_get_num_free_pages(std::shared_ptr<PageAllocator> allocator) {
  return allocator->get_num_free_pages();
}

int64_t
page_allocator_get_num_inuse_pages(std::shared_ptr<PageAllocator> allocator) {
  return allocator->get_num_inuse_pages();
}

int64_t
page_allocator_get_num_total_pages(std::shared_ptr<PageAllocator> allocator) {
  return allocator->get_num_total_pages();
}

int64_t page_allocator_get_num_reserved_pages(
    std::shared_ptr<PageAllocator> allocator) {
  return allocator->get_num_reserved_pages();
}

int64_t page_allocator_get_avail_physical_pages(
    std::shared_ptr<PageAllocator> allocator) {
  return allocator->get_avail_physical_pages();
}

void page_allocator_set_broadcast_map_callback(
    std::shared_ptr<PageAllocator> allocator, BroadcastMapCallback callback) {
  allocator->set_broadcast_map_callback(callback);
}

void page_allocator_set_broadcast_unmap_callback(
    std::shared_ptr<PageAllocator> allocator, BroadcastUnmapCallback callback) {
  allocator->set_broadcast_unmap_callback(callback);
}

void page_allocator_set_should_use_worker_ipc_callback(
    std::shared_ptr<PageAllocator> allocator,
    ShouldUseWorkerIpcCallback callback) {
  allocator->set_should_use_worker_ipc_callback(callback);
}

page_id_t page_allocator_get_page_id(std::shared_ptr<PageAllocator> allocator,
                                     int64_t block_id, int64_t block_mem_size) {
  return allocator->get_page_id(block_id, block_mem_size);
}

// New function for grouping indices by page
std::unordered_map<page_id_t, std::vector<int64_t>>
page_allocator_group_indices_by_page(std::shared_ptr<PageAllocator> allocator,
                                     const std::vector<int64_t> &indices,
                                     int64_t block_mem_size) {
  return allocator->group_indices_by_page(indices, block_mem_size);
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
        py::arg("group_id") = 0, py::arg("unified_pool") = false);
  m.def("kv_tensors_created", &kvcached::kv_tensors_created,
        "kv_tensors_created", py::arg("group_id") = 0);
  m.def("map_to_kv_tensors", &kvcached::map_to_kv_tensors, "map_to_kv_tensors",
        py::arg("offsets"), py::arg("group_id") = 0);
  m.def("unmap_from_kv_tensors", &kvcached::unmap_from_kv_tensors,
        "unmap_from_kv_tensors", py::arg("offsets"), py::arg("group_id") = 0);

  // PageAllocator bindings
  py::class_<kvcached::PageAllocator, std::shared_ptr<kvcached::PageAllocator>>(
      m, "PageAllocator")
      .def(py::init(&kvcached::create_page_allocator), py::arg("num_layers"),
           py::arg("mem_size_per_layer"), py::arg("page_size"),
           py::arg("world_size") = 1, py::arg("pp_rank") = 0,
           py::arg("async_sched") = false, py::arg("contiguous_layout") = true,
           py::arg("enable_page_prealloc") = true,
           py::arg("num_kv_buffers") = 2, py::arg("group_id") = 0)
      .def("start_prealloc_thread",
           &kvcached::page_allocator_start_prealloc_thread)
      .def("stop_prealloc_thread",
           &kvcached::page_allocator_stop_prealloc_thread)
      .def("alloc_page", &kvcached::page_allocator_alloc_page)
      .def("free_page", &kvcached::page_allocator_free_page)
      .def("free_pages", &kvcached::page_allocator_free_pages)
      .def("resize", &kvcached::page_allocator_resize)
      .def("trim", &kvcached::page_allocator_trim)
      .def("reset_free_page_order",
           &kvcached::page_allocator_reset_free_page_order)
      .def("get_num_free_pages", &kvcached::page_allocator_get_num_free_pages)
      .def("get_num_inuse_pages", &kvcached::page_allocator_get_num_inuse_pages)
      .def("get_num_total_pages", &kvcached::page_allocator_get_num_total_pages)
      .def("get_num_reserved_pages",
           &kvcached::page_allocator_get_num_reserved_pages)
      .def("get_avail_physical_pages",
           &kvcached::page_allocator_get_avail_physical_pages)
      .def("get_page_id", &kvcached::page_allocator_get_page_id)
      .def("group_indices_by_page",
           &kvcached::page_allocator_group_indices_by_page)
      .def("set_broadcast_map_callback",
           &kvcached::page_allocator_set_broadcast_map_callback)
      .def("set_broadcast_unmap_callback",
           &kvcached::page_allocator_set_broadcast_unmap_callback)
      .def("set_should_use_worker_ipc_callback",
           &kvcached::page_allocator_set_should_use_worker_ipc_callback);

  // InternalPage bindings (now as independent class)
  py::class_<kvcached::InternalPage, std::shared_ptr<kvcached::InternalPage>>(
      m, "InternalPage")
      .def(py::init<kvcached::page_id_t, int64_t>(), py::arg("page_id"),
           py::arg("page_size"))
      .def_readonly("page_id", &kvcached::InternalPage::page_id)
      .def_readonly("page_size", &kvcached::InternalPage::page_size)
      .def("init", &kvcached::InternalPage::init)
      .def("alloc", &kvcached::InternalPage::alloc)
      .def("free", &kvcached::InternalPage::free)
      .def("free_batch", &kvcached::InternalPage::free_batch)
      .def("empty", &kvcached::InternalPage::empty)
      .def("full", &kvcached::InternalPage::full)
      .def("num_free_blocks", &kvcached::InternalPage::num_free_blocks)
      .def("get_free_blocks", &kvcached::InternalPage::get_free_blocks)
      .def_static("get_block_range", &kvcached::InternalPage::get_block_range)
      .def_static("get_num_blocks", &kvcached::InternalPage::get_num_blocks);
}
