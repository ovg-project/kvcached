// SPDX-FileCopyrightText: Copyright contributors to the kvcached project
// SPDX-License-Identifier: Apache-2.0

// GPU/torch-free coverage of the failure-propagation contract documented on
// for_each_mapping_impl() (csrc/inc/mapping_iteration.hpp): a false from `op`
// must short-circuit the remaining offsets/layers, and its return value must
// not be silently discarded by the loop. This is the regression test for the
// PR #465 review comment on FTensorAllocator::for_each_mapping_, run without
// the compiled extension or a GPU by faking `Handle` as a plain int and `op`
// as a lambda that never touches real FTensor/CUDA state.

#include <cassert>
#include <cstdio>
#include <functional>
#include <stdexcept>
#include <vector>

#include "mapping_iteration.hpp"

using kvcached::for_each_mapping_impl;
using kvcached::offset_t;

namespace {

// A fake per-layer handle: just the layer index. v_base_offset is fixed so
// per-layer K/V pairs are easy to predict.
constexpr offset_t kVBaseOffset = 1000;

std::function<int(int64_t)> layer_handle() {
  return [](int64_t i) { return static_cast<int>(i); };
}

std::function<offset_t(int)> v_base_offset() {
  return [](int) { return kVBaseOffset; };
}

// Records every (handle, offset) pair `op` is actually invoked with, so a
// test can assert exactly what was and wasn't visited.
struct Recorder {
  std::vector<std::pair<int, offset_t>> visited;

  std::function<bool(int, offset_t)> always_true() {
    return [this](int h, offset_t off) {
      visited.emplace_back(h, off);
      return true;
    };
  }

  // Fails from the Nth call onward (1-indexed), counting all calls made to
  // any closure this Recorder has returned since construction; succeeds
  // before that. Use one fresh Recorder per test/scenario -- the count is
  // cumulative across calls, not local to a single fails_from() closure.
  std::function<bool(int, offset_t)> fails_from(int n) {
    return [this, n](int h, offset_t off) {
      visited.emplace_back(h, off);
      return static_cast<int>(visited.size()) < n;
    };
  }
};

void test_contiguous_visits_all_offsets_on_success() {
  Recorder rec;
  std::vector<offset_t> offsets = {0, 16, 32};
  bool ok = for_each_mapping_impl<int>(
      /*contiguous_layout=*/true, /*unified_pool=*/false, /*num_layers=*/1,
      offsets, /*contiguous_handle=*/-1, layer_handle(), v_base_offset(),
      rec.always_true());
  assert(ok);
  assert(rec.visited.size() == 3);
  assert(rec.visited[0] == (std::pair<int, offset_t>{-1, 0}));
  assert(rec.visited[2] == (std::pair<int, offset_t>{-1, 32}));
}

// This is the exact defect the PR #465 review flagged: before the fix,
// for_each_mapping_() called op() in a loop, discarded the bool, and
// unconditionally returned true. Assert both halves of the fix: the false
// return propagates, AND the offsets after the failure are never visited
// (i.e. it stops, it doesn't just report failure after doing everything).
void test_contiguous_short_circuits_and_propagates_false() {
  Recorder rec;
  std::vector<offset_t> offsets = {0, 16, 32, 48};
  bool ok = for_each_mapping_impl<int>(true, false, 1, offsets, -1,
                                       layer_handle(), v_base_offset(),
                                       rec.fails_from(2));
  assert(!ok); // false must propagate, not be swallowed as true
  assert(rec.visited.size() == 2); // stopped at the failing call
  assert(rec.visited[1] == (std::pair<int, offset_t>{-1, 16}));
}

void test_unified_pool_short_circuits_across_layers() {
  Recorder rec;
  std::vector<offset_t> offsets = {0, 16};
  // 2 offsets/layer; fail on the 3rd call overall (1st offset of layer 1).
  bool ok = for_each_mapping_impl<int>(false, true, 2, offsets, -1,
                                       layer_handle(), v_base_offset(),
                                       rec.fails_from(3));
  assert(!ok);
  assert(rec.visited.size() == 3);
  assert(rec.visited[2] == (std::pair<int, offset_t>{1, 0}));
}

void test_per_layer_visits_k_then_v_and_short_circuits_on_v() {
  Recorder rec;
  std::vector<offset_t> offsets = {0};
  // Single offset, single layer: op() is called for K (offset) then V
  // (offset + v_base_offset). Fail on the V call.
  bool ok = for_each_mapping_impl<int>(false, false, 1, offsets, -1,
                                       layer_handle(), v_base_offset(),
                                       rec.fails_from(2));
  assert(!ok);
  assert(rec.visited.size() == 2);
  assert(rec.visited[0] == (std::pair<int, offset_t>{0, 0}));
  assert(rec.visited[1] == (std::pair<int, offset_t>{0, kVBaseOffset}));
}

void test_per_layer_success_visits_every_k_and_v_pair() {
  Recorder rec;
  std::vector<offset_t> offsets = {0, 16};
  bool ok = for_each_mapping_impl<int>(false, false, 2, offsets, -1,
                                       layer_handle(), v_base_offset(),
                                       rec.always_true());
  assert(ok);
  // 2 layers * 2 offsets * (K + V) = 8 calls.
  assert(rec.visited.size() == 8);
}


void test_exception_from_op_propagates_uncaught() {
  std::vector<offset_t> offsets = {0, 16};
  std::function<bool(int, offset_t)> throws = [](int, offset_t) -> bool {
    throw std::runtime_error("op failed");
  };
  bool threw = false;
  try {
    for_each_mapping_impl<int>(true, false, 1, offsets, -1, layer_handle(),
                               v_base_offset(), throws);
  } catch (const std::runtime_error &) {
    threw = true;
  }
  assert(threw);
}

} // namespace

int main() {
  test_contiguous_visits_all_offsets_on_success();
  test_contiguous_short_circuits_and_propagates_false();
  test_unified_pool_short_circuits_across_layers();
  test_per_layer_visits_k_then_v_and_short_circuits_on_v();
  test_per_layer_success_visits_every_k_and_v_pair();
  test_exception_from_op_propagates_uncaught();
  std::puts("test_mapping_iteration: all checks passed");
  return 0;
}
