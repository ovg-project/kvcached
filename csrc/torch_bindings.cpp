// SPDX-FileCopyrightText: Copyright contributors to the kvcached project
// SPDX-License-Identifier: Apache-2.0

#include <memory>
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <string>
#include <utility>
#include <vector>

#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/tensor.h>

#include "allocator.hpp"
#include "constants.hpp"
#include "page_allocator.hpp"
#include "torch_utils.hpp"
#include "transaction_error.hpp"

namespace py = pybind11;

namespace kvcached {

// ---------------------------------------------------------------------------
// KV tensor ops.
//
// These are the only bindings that touch torch::Tensor, so they are registered
// through the PyTorch stable ABI (STABLE_TORCH_LIBRARY) instead of pybind11.
// This decouples them from libtorch's unstable C++ ABI. They are reached from
// Python via torch.ops.kvcached.* (re-exported by kvcached/vmm_ops.py).
//
// The dispatcher schema uses int64_t/bool/str/int[]/Tensor[]; sizes that are
// logically size_t are passed as int64_t and cast at the boundary.
// ---------------------------------------------------------------------------

void init_kvcached(std::string dev_str, int64_t page_size,
                   bool contiguous_layout) {
  FTensorAllocator::init(dev_str, static_cast<size_t>(page_size),
                         contiguous_layout);
}

void shutdown_kvcached() { FTensorAllocator::shutdown(); }

std::vector<torch::stable::Tensor>
create_kv_tensors(int64_t size, int64_t dtype_size, std::string dev_str,
                  int64_t num_layers, int64_t num_kv_buffers, int64_t group_id,
                  bool unified_pool) {
  auto allocator = FTensorAllocator::global_allocator(group_id);
  auto dtype_ = torch_dtype_from_size(static_cast<size_t>(dtype_size));
  return allocator->create_kv_tensors(static_cast<size_t>(size), dtype_,
                                      dev_str, num_layers, num_kv_buffers,
                                      unified_pool);
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

// ---------------------------------------------------------------------------
// Transactional map/unmap ops. These are torch-free (int[]/str/bool args, and a
// (bool, int[]) return), so they stay on pybind11 rather than the stable
// dispatcher, and release the GIL while they block in the allocator.
// ---------------------------------------------------------------------------

std::pair<bool, std::vector<offset_t>>
map_to_kv_tensors_with_result(const std::vector<offset_t> &offsets,
                              int64_t group_id = 0) {
  py::gil_scoped_release release;
  auto allocator = FTensorAllocator::global_allocator(group_id);
  return allocator->map_to_kv_tensors_with_result(offsets);
}

bool prepare_unmap_from_kv_tensors(const std::vector<offset_t> &offsets,
                                   const std::string &transaction_id,
                                   int64_t group_id = 0) {
  py::gil_scoped_release release;
  auto allocator = FTensorAllocator::global_allocator(group_id);
  return allocator->prepare_unmap_from_kv_tensors(offsets, transaction_id);
}

bool commit_unmap_from_kv_tensors(const std::string &transaction_id,
                                  int64_t group_id = 0) {
  py::gil_scoped_release release;
  auto allocator = FTensorAllocator::global_allocator(group_id);
  return allocator->commit_unmap_from_kv_tensors(transaction_id);
}

bool abort_unmap_from_kv_tensors(const std::string &transaction_id,
                                 int64_t group_id = 0) {
  py::gil_scoped_release release;
  auto allocator = FTensorAllocator::global_allocator(group_id);
  return allocator->abort_unmap_from_kv_tensors(transaction_id);
}

// ---------------------------------------------------------------------------
// PageAllocator / InternalPage bindings.
//
// These classes contain no torch types (only ints, vectors, dicts, callbacks),
// so they do not couple to libtorch's C++ ABI and stay on pybind11, which keeps
// their class-based API, Python callbacks, and dict returns intact.
// ---------------------------------------------------------------------------
std::shared_ptr<PageAllocator> create_page_allocator(
    int64_t num_layers, int64_t mem_size_per_layer, int64_t page_size,
    int64_t world_size = 1, int64_t pp_rank = 0, bool async_sched = false,
    bool contiguous_layout = true, bool enable_page_prealloc = true,
    int64_t num_kv_buffers = 2, int64_t group_id = 0,
    const std::string &ipc_name = "") {

  return std::make_shared<PageAllocator>(
      num_layers, mem_size_per_layer, page_size, world_size, pp_rank,
      async_sched, contiguous_layout, enable_page_prealloc, num_kv_buffers,
      group_id, ipc_name);
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

py::dict
page_allocator_get_page_state(std::shared_ptr<PageAllocator> allocator) {
  PageState state;
  {
    py::gil_scoped_release release;
    state = allocator->get_page_state();
  }
  py::dict result;
  result["total_pages"] = state.total_pages;
  result["free_pages"] = state.free_pages;
  result["inuse_pages"] = state.inuse_pages;
  result["reserved_pages"] = state.reserved_pages;
  return result;
}

int64_t page_allocator_get_avail_physical_pages(
    std::shared_ptr<PageAllocator> allocator) {
  return allocator->get_avail_physical_pages();
}

int64_t page_allocator_check_and_get_resize_target(
    std::shared_ptr<PageAllocator> allocator, int64_t current_mem_size) {
  return allocator->check_and_get_resize_target(current_mem_size);
}

int64_t
page_allocator_get_resize_target(std::shared_ptr<PageAllocator> allocator) {
  return allocator->get_resize_target();
}

void page_allocator_set_broadcast_map_callback(
    std::shared_ptr<PageAllocator> allocator, BroadcastMapCallback callback) {
  allocator->set_broadcast_map_callback(
      [callback = std::move(callback)](int64_t size,
                                       const std::vector<offset_t> &offsets) {
        try {
          callback(size, offsets);
        } catch (py::error_already_set &error) {
          py::gil_scoped_acquire acquire;
          auto errors = py::module_::import("kvcached.errors");
          if (error.matches(errors.attr("StateConsistencyError").ptr())) {
            throw StateConsistencyError(error.what());
          }
          if (error.matches(errors.attr("MapQuarantinedError").ptr())) {
            throw MapQuarantinedError(error.what());
          }
          throw;
        }
      });
}

void page_allocator_set_broadcast_unmap_callback(
    std::shared_ptr<PageAllocator> allocator, BroadcastUnmapCallback callback) {
  allocator->set_broadcast_unmap_callback(callback);
}

void page_allocator_set_use_worker_ipc(std::shared_ptr<PageAllocator> allocator,
                                       bool use_worker_ipc) {
  allocator->set_use_worker_ipc(use_worker_ipc);
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

// Register the KV tensor ops in the "kvcached" dispatcher namespace.
STABLE_TORCH_LIBRARY(kvcached, m) {
  m.def("init_kvcached(str dev_str, int page_size=0, bool "
        "contiguous_layout=False) -> ()");
  m.def("shutdown_kvcached() -> ()");
  m.def("create_kv_tensors(int size, int dtype_size, str dev_str, int "
        "num_layers, int num_kv_buffers=2, int group_id=0, bool "
        "unified_pool=False) -> Tensor[]");
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

// The pybind11 module hosts only the torch-free PageAllocator / InternalPage
// classes. TORCH_EXTENSION_NAME resolves to the compiled extension name (_C);
// importing it also runs the STABLE_TORCH_LIBRARY static initializers above,
// registering the KV tensor ops.
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.doc() = "kvcached VMM plugin";
  auto errors = py::module_::import("kvcached.errors");
  py::register_exception<kvcached::MapQuarantinedError>(
      m, "MapQuarantinedError", errors.attr("MapQuarantinedError").ptr());
  py::register_exception<kvcached::StateConsistencyError>(
      m, "StateConsistencyError", errors.attr("StateConsistencyError").ptr());
  py::register_exception<kvcached::QuarantinedResizeError>(
      m, "QuarantinedResizeError", errors.attr("QuarantinedResizeError").ptr());

  // Torch-free transactional ops (the six core ops are on the stable
  // dispatcher; see STABLE_TORCH_LIBRARY above).
  m.def("map_to_kv_tensors_with_result",
        &kvcached::map_to_kv_tensors_with_result,
        "map_to_kv_tensors_with_result", py::arg("offsets"),
        py::arg("group_id") = 0);
  m.def("prepare_unmap_from_kv_tensors",
        &kvcached::prepare_unmap_from_kv_tensors,
        "prepare_unmap_from_kv_tensors", py::arg("offsets"),
        py::arg("transaction_id"), py::arg("group_id") = 0);
  m.def("commit_unmap_from_kv_tensors", &kvcached::commit_unmap_from_kv_tensors,
        "commit_unmap_from_kv_tensors", py::arg("transaction_id"),
        py::arg("group_id") = 0);
  m.def("abort_unmap_from_kv_tensors", &kvcached::abort_unmap_from_kv_tensors,
        "abort_unmap_from_kv_tensors", py::arg("transaction_id"),
        py::arg("group_id") = 0);

  // PageAllocator bindings
  py::class_<kvcached::PageAllocator, std::shared_ptr<kvcached::PageAllocator>>(
      m, "PageAllocator")
      .def(py::init(&kvcached::create_page_allocator), py::arg("num_layers"),
           py::arg("mem_size_per_layer"), py::arg("page_size"),
           py::arg("world_size") = 1, py::arg("pp_rank") = 0,
           py::arg("async_sched") = false, py::arg("contiguous_layout") = true,
           py::arg("enable_page_prealloc") = true,
           py::arg("num_kv_buffers") = 2, py::arg("group_id") = 0,
           py::arg("ipc_name") = "")
      .def("get_transaction_state",
           [](const kvcached::PageAllocator &allocator) {
             auto state = [&allocator] {
               py::gil_scoped_release release;
               return allocator.get_transaction_state();
             }();
             py::dict result;
             result["state"] =
                 state.failed
                     ? "FAILED"
                     : (state.quarantined_page_ids.empty() ? "HEALTHY"
                                                           : "DEGRADED");
             result["quarantined_page_ids"] = state.quarantined_page_ids;
             result["quarantined_pages"] = state.quarantined_page_ids.size();
             result["retained_bytes_upper_bound"] =
                 state.retained_bytes_upper_bound;
             result["last_error"] = state.last_error;
             return result;
           })
      .def("start_prealloc_thread",
           &kvcached::page_allocator_start_prealloc_thread)
      // The bindings below can block inside the allocator (alloc_page waits on
      // a condition variable for the prealloc worker; free/resize/trim unmap
      // pages; stop joins the worker). They must not hold the GIL while
      // blocked: the prealloc worker needs the GIL to run the Python broadcast
      // callback, and a caller parked in here with the GIL held deadlocks the
      // whole process (issue #371).
      .def("stop_prealloc_thread",
           &kvcached::page_allocator_stop_prealloc_thread,
           py::call_guard<py::gil_scoped_release>())
      .def("alloc_page", &kvcached::page_allocator_alloc_page,
           py::call_guard<py::gil_scoped_release>())
      .def("free_page", &kvcached::page_allocator_free_page,
           py::call_guard<py::gil_scoped_release>())
      .def("free_pages", &kvcached::page_allocator_free_pages,
           py::call_guard<py::gil_scoped_release>())
      .def("resize", &kvcached::page_allocator_resize,
           py::call_guard<py::gil_scoped_release>())
      .def("trim", &kvcached::page_allocator_trim,
           py::call_guard<py::gil_scoped_release>())
      .def("reset_free_page_order",
           &kvcached::page_allocator_reset_free_page_order)
      .def("get_num_free_pages", &kvcached::page_allocator_get_num_free_pages)
      .def("get_num_inuse_pages", &kvcached::page_allocator_get_num_inuse_pages)
      .def("get_num_total_pages", &kvcached::page_allocator_get_num_total_pages)
      .def("get_num_reserved_pages",
           &kvcached::page_allocator_get_num_reserved_pages)
      .def("get_page_state", &kvcached::page_allocator_get_page_state)
      .def("get_avail_physical_pages",
           &kvcached::page_allocator_get_avail_physical_pages)
      .def("check_and_get_resize_target",
           &kvcached::page_allocator_check_and_get_resize_target)
      .def("get_resize_target", &kvcached::page_allocator_get_resize_target)
      .def("get_page_id", &kvcached::page_allocator_get_page_id)
      .def("group_indices_by_page",
           &kvcached::page_allocator_group_indices_by_page)
      .def("set_broadcast_map_callback",
           &kvcached::page_allocator_set_broadcast_map_callback)
      .def("set_broadcast_unmap_callback",
           &kvcached::page_allocator_set_broadcast_unmap_callback)
      .def("set_use_worker_ipc", &kvcached::page_allocator_set_use_worker_ipc);

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
