// SPDX-FileCopyrightText: Copyright contributors to the kvcached project
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <unordered_set>
#include <vector>

namespace kvcached {

// Claim exactly these free block IDs, preserving the order of the remaining
// free list. A conflict is a miss; invalid requests throw before any mutation.
inline bool claim_exact_blocks(std::vector<int64_t> &free_blocks,
                               const std::vector<int64_t> &requested) {
  std::unordered_set<int64_t> wanted;
  wanted.reserve(requested.size());
  for (const auto id : requested) {
    if (id < 0 || !wanted.insert(id).second) {
      throw std::invalid_argument(
          "Exact block IDs must be nonnegative and unique");
    }
  }
  const std::unordered_set<int64_t> available(free_blocks.begin(),
                                              free_blocks.end());
  if (available.size() != free_blocks.size()) {
    throw std::logic_error("Duplicate block IDs in page free list");
  }
  for (const auto id : wanted) {
    if (!available.count(id)) {
      return false;
    }
  }
  // Allocate the replacement before modifying the original vector, including
  // when allocation of the temporary storage itself throws.
  std::vector<int64_t> remaining;
  remaining.reserve(free_blocks.size() - requested.size());
  for (const auto id : free_blocks) {
    if (!wanted.count(id)) {
      remaining.push_back(id);
    }
  }
  free_blocks.swap(remaining);
  return true;
}

} // namespace kvcached
