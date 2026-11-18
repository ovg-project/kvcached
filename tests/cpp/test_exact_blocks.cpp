// SPDX-FileCopyrightText: Copyright contributors to the kvcached project
// SPDX-License-Identifier: Apache-2.0

#include "exact_blocks.hpp"
#include <cassert>
#include <iostream>

int main() {
  using kvcached::claim_exact_blocks;
  std::vector<int64_t> free{1, 4, 7, 8};
  const auto before = free;
  assert(!claim_exact_blocks(free, {4, 5}));
  assert(free == before);
  for (const std::vector<int64_t> &bad : {std::vector<int64_t>{4, 4}, {-1}}) {
    bool rejected = false;
    try {
      claim_exact_blocks(free, bad);
    } catch (const std::invalid_argument &) {
      rejected = true;
    }
    assert(rejected && free == before);
  }
  assert(claim_exact_blocks(free, {8, 4}));
  assert((free == std::vector<int64_t>{1, 7}));
  assert(claim_exact_blocks(free, {}));
  assert(claim_exact_blocks(free, {7, 1}));
  assert(free.empty());
  std::cout << "exact block selection contracts passed\n";
}
