// SPDX-FileCopyrightText: Copyright contributors to the kvcached project
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace kvcached {

constexpr size_t saturating_subtract(size_t value, size_t amount) {
  return value > amount ? value - amount : size_t{0};
}

// A negative balance is still owed after physical reserves are unmapped.
constexpr int64_t signed_headroom(uint64_t capacity, uint64_t charged) {
  constexpr uint64_t max = std::numeric_limits<int64_t>::max();
  return capacity >= charged
             ? static_cast<int64_t>(std::min(capacity - charged, max))
             : -static_cast<int64_t>(std::min(charged - capacity, max));
}

} // namespace kvcached
