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
  // One unrelated reserve, one free driver bundle: map the requested ID,
  // leaving no driver bytes for a sibling. If the driver is already full,
  // native trims the unrelated reserve first. A matching ID is free to use.
  const auto quote =
      kvcached::quote_exact_pages({2, 7}, {2}, {}, {7, 8}, 20, {});
  assert(quote && std::get<0>(*quote) == 20 &&
         std::get<1>(*quote) == std::vector<int64_t>({8}) &&
         std::get<2>(*quote) == 0);
  const auto local =
      kvcached::quote_exact_pages({2, 7}, {}, {2}, {7, 8}, 20, {});
  assert(local && std::get<2>(*local) == 1 &&
         std::get<1>(*local) == std::vector<int64_t>({8}));
  assert(!kvcached::quote_exact_pages({9}, {2}, {}, {7}, 20, {}));
  const auto retry =
      kvcached::quote_exact_pages({1, 4}, {4}, {}, {5}, 20, {{0, 1}});
  assert(retry && std::get<0>(*retry) == 0 &&
         std::get<1>(*retry) == std::vector<int64_t>({5}));
  const auto unmatched =
      kvcached::quote_exact_pages({4}, {4}, {}, {5}, 20, {{0, 1}});
  assert(unmatched && std::get<0>(*unmatched) == 20);
  std::deque<int64_t> reserved{5, 9, 10};
  assert(kvcached::extract_reserved_pages(reserved, {9, 17}) ==
         std::vector<int64_t>({9}));
  assert(reserved == std::deque<int64_t>({5, 10}));
  std::cout << "exact block selection contracts passed\n";
}
