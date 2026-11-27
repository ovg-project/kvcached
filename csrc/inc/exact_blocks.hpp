// SPDX-FileCopyrightText: Copyright contributors to the kvcached project
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <cstdint>
#include <deque>
#include <limits>
#include <optional>
#include <stdexcept>
#include <tuple>
#include <unordered_set>
#include <utility>
#include <vector>

namespace kvcached {

// Quote one exact claim assuming unused reservations can be returned to the
// driver. Shrink permits swapping reclaimed IDs into free active slots.
using ExactRecoveryQuote = std::tuple<uint64_t, std::vector<int64_t>, size_t>;

inline std::optional<ExactRecoveryQuote> quote_exact_pages(
    const std::vector<int64_t> &pages, const std::deque<int64_t> &free,
    const std::deque<int64_t> &reclaimed, const std::deque<int64_t> &reserved,
    uint64_t bundle, const std::deque<std::vector<int64_t>> &retained) {
  std::unordered_set<int64_t> wanted(pages.begin(), pages.end());
  if (!bundle || wanted.size() != pages.size())
    throw std::invalid_argument("Invalid exact recovery request");
  std::vector<int64_t> idle;
  for (const auto id : reserved) {
    if (!wanted.erase(id))
      idle.push_back(id);
  }
  // Retried IDs can reuse existing backing, subject to peer admission.
  // Unrequested members of these batches are never available to trim.
  size_t spare_after_retry = 0;
  for (const auto &batch : retained) {
    size_t matched = 0;
    for (const auto id : batch)
      matched += wanted.erase(id);
    if (matched)
      spare_after_retry += batch.size() - matched;
  }
  // Exact allocation adopts matching batches first. Its other requested IDs
  // may then use surplus backing from those batches within this pool.
  const auto fresh = wanted.size() - std::min(wanted.size(), spare_after_retry);
  if (fresh > free.size() + idle.size() || fresh > UINT64_MAX / bundle)
    return std::nullopt;
  if (!wanted.empty()) {
    for (const auto id : free) {
      if (wanted.erase(id) && wanted.empty())
        break;
    }
    if (!wanted.empty()) {
      for (const auto id : reclaimed) {
        if (wanted.erase(id) && wanted.empty())
          break;
      }
    }
    if (!wanted.empty())
      return std::nullopt;
  }
  const size_t local = fresh > free.size() ? fresh - free.size() : 0;
  return std::make_tuple(fresh * bundle, std::move(idle), local);
}

// Select only still-idle entries from a recovery quote. A matching reserve
// or a page claimed since the quote must never be unmapped.
inline std::vector<int64_t>
extract_reserved_pages(std::deque<int64_t> &reserved,
                       const std::vector<int64_t> &eligible) {
  const std::unordered_set<int64_t> wanted(eligible.begin(), eligible.end());
  std::vector<int64_t> selected;
  selected.reserve(std::min(wanted.size(), reserved.size()));
  auto keep = std::remove_if(reserved.begin(), reserved.end(), [&](auto id) {
    if (!wanted.count(id))
      return false;
    selected.push_back(id);
    return true;
  });
  reserved.erase(keep, reserved.end());
  return selected;
}

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
