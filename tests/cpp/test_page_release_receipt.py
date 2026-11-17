# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

"""Exercise the native release body with controlled preallocation completion."""

import os
import subprocess
import sys
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    text = (root / "csrc/page_allocator.cpp").read_text()
    start = text.index("void PageAllocator::free_pages(")
    body = text[start:text.index("\nbool PageAllocator::resize(", start)]
    source = r'''
#include <algorithm>
#include <atomic>
#include <cassert>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <type_traits>
#include <vector>
#define LOGGER(level, fmt, count, elapsed) ((void)(count), (void)(elapsed))
using page_id_t = int64_t;
class PageAllocator {
public:
  std::mutex lock_;
  std::condition_variable cond_;
  std::atomic<int64_t> num_free_pages_{0};
  int64_t max_reserved_pages_ = 1;
  std::deque<page_id_t> reserved_page_list_, free_page_list_;
  std::vector<page_id_t> unmapped;
  bool fail_unmap = false, failed = false;
  bool interleave = false, unmapping = false, completed = false;
  void throw_if_failed() {
    if (failed) throw std::runtime_error("failed pool");
  }
  void update_memory_usage_unlocked() {}
  void unmap_pages(const std::vector<page_id_t> &ids) {
    if (interleave) {
      std::unique_lock<std::mutex> guard(lock_);
      unmapping = true;
      cond_.notify_all();
      cond_.wait(guard, [&] { return completed; });
    }
    if (fail_unmap) {
      failed = true;
      throw std::runtime_error("injected unmap failure");
    }
    unmapped.insert(unmapped.end(), ids.begin(), ids.end());
  }
  void free_pages(const std::vector<page_id_t> &ids);
  int64_t free_pages_with_receipt(const std::vector<page_id_t> &ids);
};
'''
    source += body
    source += r'''
int main() {
  static_assert(std::is_same_v<decltype(PageAllocator().free_pages({})), void>);
  // Normal reserve, overflow, empty input and the legacy entry point.
  {
    PageAllocator p;
    assert(p.free_pages_with_receipt({1}) == 1);
    assert(p.free_pages_with_receipt({2, 3}) == 0);
    assert(p.free_pages_with_receipt({}) == 0);
    p.free_pages({4});
    assert((p.unmapped == std::vector<page_id_t>{2, 3, 4}));
  }
  for (bool full : {false, true}) {
    PageAllocator p;
    if (full) p.reserved_page_list_.push_back(1);
    p.interleave = true;
    std::thread preallocator([&] {
      std::unique_lock<std::mutex> guard(p.lock_);
      p.cond_.wait(guard, [&] { return p.unmapping; });
      // These pages were mapped and charged before the release began.
      // Completing preallocation only publishes them to the reserve.
      p.reserved_page_list_.push_back(10);
      p.reserved_page_list_.push_back(11);
      p.completed = true;
      p.cond_.notify_all();
    });
    auto retained = p.free_pages_with_receipt({2, 3});
    preallocator.join();
    assert(retained == (full ? 0 : 1));
    assert(p.reserved_page_list_.size() == 3);
    assert(retained + p.unmapped.size() == 2);
  }
  {
    PageAllocator p;
    p.fail_unmap = true;
    int64_t receipt = -1;
    try {
      receipt = p.free_pages_with_receipt({1, 2});
      assert(false);
    } catch (const std::runtime_error &) {}
    assert(receipt == -1);  // Never publish a successful partial receipt.
    assert(p.reserved_page_list_.size() == 1 && p.unmapped.empty());
    try {
      p.free_pages({1, 2});
      assert(false);
    } catch (const std::runtime_error &) {}
    assert(p.reserved_page_list_.size() == 1);
  }
}
'''
    build = Path(sys.argv[1])
    cpp = build / "test_page_release_receipt.cpp"
    binary = build / "test_page_release_receipt"
    cpp.write_text(source)
    subprocess.run([os.environ.get("CXX", "c++"), "-std=c++17", "-Wall", "-Wextra",
                    "-Werror", "-pthread", str(cpp), "-o", str(binary)], check=True)
    subprocess.run([str(binary)], check=True, timeout=10)


if __name__ == "__main__":
    main()
