// SPDX-FileCopyrightText: Copyright contributors to the kvcached project
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <functional>
#include <vector>

#include "constants.hpp"

namespace kvcached {

// Pure, GPU/torch-free iteration core shared by FTensorAllocator's
// map/unmap/prepare/commit entry points (csrc/allocator.cpp). Kept separate
// from FTensorAllocator so the failure-propagation contract below can be
// unit-tested without the compiled extension or a GPU -- see
// tests/cpp/test_mapping_iteration.cpp.
//
// Applies `op` to every (handle, offset) pair implied by the layout, in
// iteration order (contiguous: one call per offset; unified_pool:
// layer-major then offset; per-layer: layer-major, then offset then
// offset+v_base_offset per layer). Returns false and stops at the FIRST call
// where `op` returns false -- pairs after that point are never visited.
// Callers that must attempt every pair regardless of a false (e.g.
// best-effort teardown) should wrap `op` to always return true and track
// failures themselves; see FTensorAllocator::unmap_from_kv_tensors. An
// exception thrown by `op` propagates uncaught.
template <typename Handle>
bool for_each_mapping_impl(
    bool contiguous_layout, bool unified_pool, int64_t num_layers,
    const std::vector<offset_t> &offsets, Handle contiguous_handle,
    const std::function<Handle(int64_t layer)> &layer_handle,
    const std::function<offset_t(Handle)> &v_base_offset,
    const std::function<bool(Handle, offset_t)> &op) {
  if (contiguous_layout) {
    // Single contiguous tensor; each offset covers all layers.
    for (auto offset : offsets) {
      if (!op(contiguous_handle, offset)) return false;
    }
  } else if (unified_pool) {
    // Unified pool: one block-interleaved handle per layer, one page per
    // pid.
    for (int64_t i = 0; i < num_layers; i++) {
      Handle h = layer_handle(i);
      for (auto offset : offsets) {
        if (!op(h, offset)) return false;
      }
    }
  } else {
    // Original per-layer layout: K and V are stacked at the 1st dim.
    for (int64_t i = 0; i < num_layers; i++) {
      Handle h = layer_handle(i);
      offset_t vbo = v_base_offset(h);
      for (auto offset : offsets) {
        if (!op(h, offset)) return false;
        if (!op(h, offset + vbo)) return false;
      }
    }
  }
  return true;
}

} // namespace kvcached
