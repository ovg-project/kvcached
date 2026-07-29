// SPDX-FileCopyrightText: Copyright contributors to the kvcached project
// SPDX-License-Identifier: Apache-2.0

#include <cstddef>

#include "math_utils.hpp"

namespace {

constexpr size_t kHeadroom = 4;

static_assert(kvcached::saturating_subtract(10, kHeadroom) == 6,
              "available memory above headroom must return the difference");
static_assert(kvcached::saturating_subtract(kHeadroom, kHeadroom) == 0,
              "available memory equal to headroom must return zero");
static_assert(kvcached::saturating_subtract(2, kHeadroom) == 0,
              "available memory below headroom must return zero");

} // namespace

int main() { return 0; }
