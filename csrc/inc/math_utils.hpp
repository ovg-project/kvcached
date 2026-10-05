// SPDX-FileCopyrightText: Copyright contributors to the kvcached project
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>

namespace kvcached {

constexpr size_t saturating_subtract(size_t value, size_t amount) {
  return value > amount ? value - amount : size_t{0};
}

} // namespace kvcached
