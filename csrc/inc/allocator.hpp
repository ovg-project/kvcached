// SPDX-FileCopyrightText: Copyright contributors to the kvcached project
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include <ATen/core/Tensor.h>
#include <c10/core/Device.h>
#include <c10/core/ScalarType.h>

#include "constants.hpp"
#include "ftensor.hpp"
#include "page.hpp"

namespace kvcached {

class FTensorAllocator {
public:
  FTensorAllocator(const c10::Device &device, bool contiguous_layout);
  ~FTensorAllocator();

  // KV cache interfaces.
  std::vector<at::Tensor> create_kv_tensors(size_t size, c10::ScalarType dtype,
                                            const std::string &dev_str,
                                            int64_t num_layers,
                                            int64_t num_kv_buffers = 2,
                                            bool unified_pool = false);
  bool kv_tensors_created();
  // Best-effort: attempts every offset even if some fail to map. This is the
  // legacy single-call map path (the TP-worker IPC handler is its only
  // remaining caller, and it reports success unconditionally regardless of
  // this return value), so it keeps the pre-existing best-effort behavior
  // rather than adopting prepare_/commit_'s short-circuit -- see the
  // definition for why short-circuiting here would be strictly worse for
  // that caller. Returns false if any offset failed to map, or if KV tensors
  // were never created.
  bool map_to_kv_tensors(const std::vector<offset_t> &offsets);
  // Best-effort: attempts every offset even if some fail, since this is
  // teardown and a skipped unmap would leave a page mapped that PageAllocator
  // is about to treat as free. Returns false if any offset failed to unmap
  // (all are still attempted), or if KV tensors were never created.
  bool unmap_from_kv_tensors(const std::vector<offset_t> &offsets);
  // Split of map_to_kv_tensors(): prepare_ allocates physical pages only (safe
  // off the main thread); commit_ does the VA edits (must be GPU-idle). See
  // FTensor::prepare/commit. Unlike map_/unmap_, both short-circuit on the
  // first offset that fails (see for_each_mapping_'s contract below) --
  // their only caller (PageAllocator::prepare_pages/commit_pages) converts a
  // false straight to a thrown exception, so there's no best-effort case to
  // preserve, and stopping early avoids wasted GPU work and (for commit_)
  // widening the partially-mapped window.
  bool prepare_kv_tensors(const std::vector<offset_t> &offsets);
  bool commit_kv_tensors(const std::vector<offset_t> &offsets);

  // Global status interfaces.
  // init() creates the default allocator (group_id=0).
  // global_allocator(group_id) returns the allocator for the given group,
  // lazily creating one if it doesn't exist yet.
  static void init(const std::string &dev_str, size_t page_size = 0,
                   bool contiguous_layout = false);
  static void shutdown();
  static FTensorAllocator *global_allocator(int64_t group_id = 0);
  void destroy();

private:
  // Raw FTensor interfaces. Must call with lock.
  static std::string get_anon_tensor_name_();
  std::vector<at::Tensor>
  create_kv_tensors_per_layer_(std::string_view prefix, size_t size,
                               c10::ScalarType dtype,
                               const std::string &dev_str, int64_t num_layers);
  std::vector<at::Tensor>
  create_kv_tensors_contiguous_(size_t size, c10::ScalarType dtype,
                                const std::string &dev_str, int64_t num_layers,
                                size_t compound_page_size);
  at::Tensor create_ftensor_(size_t size, c10::ScalarType dtype,
                             const std::string &dev_str, std::string name = "");
  void free_ftensor_(at::Tensor &ftensor);
  // Apply `op` (FTensor::map/unmap/prepare/commit) to every (ftensor, offset)
  // slot implied by the current layout. Must be called with mtx_ held.
  // Contract: returns false and stops at the first offset where `op` returns
  // false (offsets after that point are never visited), or false immediately
  // if KV tensors haven't been created. This short-circuit is only safe
  // because every `op` this function is called with treats a `false` as a
  // real failure it needs to see, not a signal to skip -- FTensor::prepare/
  // commit/unmap's only false paths are all invariant violations on the
  // caller's part (never "nothing to do here, move on"), so stopping never
  // drops work that still needed to happen *for that op*. Whether stopping
  // early is still the right call for the *batch as a whole* is a separate,
  // per-caller question -- map_to_kv_tensors and unmap_from_kv_tensors both
  // need every offset attempted regardless (see their own comments for why),
  // so they wrap `op` to always return true here and track real failures
  // themselves; only prepare_/commit_ pass `op` straight through and rely on
  // this function's short-circuit. See csrc/inc/mapping_iteration.hpp for
  // the pure iteration core and its CPU-only test.
  bool for_each_mapping_(const std::vector<offset_t> &offsets,
                         const std::function<bool(FTensor *, offset_t)> &op);

  // GPU VMM util functions.
  void init_gpu_();

  // Multiton: one allocator per group_id.
  static std::unordered_map<int64_t, std::unique_ptr<FTensorAllocator>>
      g_allocators_;
  static std::mutex g_allocator_mutex_;
  // Device and layout from init(), used to create new group allocators.
  static c10::Device g_device_;
  static bool g_contiguous_layout_;

  c10::Device dev_;

  int64_t num_layers_;
  bool contiguous_layout_;
  bool unified_pool_;
  size_t kv_tensor_size_per_layer_;

  mutable std::mutex mtx_;
  // For per-layer layout: one tensor per layer
  std::unordered_map<std::string, std::unique_ptr<FTensor>> ftensors_;
  // For contiguous layout: single tensor containing all layers
  std::unique_ptr<FTensor> contiguous_kv_tensor_;
  std::shared_ptr<Page> zero_page_;
};

} // namespace kvcached
