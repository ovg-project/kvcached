// SPDX-FileCopyrightText: Copyright contributors to the kvcached project
// SPDX-License-Identifier: Apache-2.0

#include "page_allocator.hpp"

#include <Python.h>

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <stdexcept>

#include "allocator.hpp"
#include "gpu_utils.hpp"
#include "math_utils.hpp"
#include "mem_info_tracker.hpp"

namespace kvcached {

// Constants
constexpr double PREALLOC_THREAD_TIMEOUT = 2.0; // seconds

// Environment variable based constants
const int64_t MIN_RESERVED_PAGES = []() {
  const char *env_val = std::getenv("KVCACHED_MIN_RESERVED_PAGES");
  return env_val ? std::atoi(env_val) : 5;
}();

const int64_t MAX_RESERVED_PAGES = []() {
  const char *env_val = std::getenv("KVCACHED_MAX_RESERVED_PAGES");
  return env_val ? std::atoi(env_val) : 10;
}();

const double GPU_UTILIZATION = []() {
  const char *env_val = std::getenv("KVCACHED_GPU_UTILIZATION");
  return env_val ? std::atof(env_val) : 0.95;
}();

// InternalPage implementation
InternalPage::InternalPage(page_id_t id, int64_t size)
    : page_id(id), page_size(size), start_block(0), end_block(0),
      num_kv_blocks(0) {}

void InternalPage::init(int64_t block_mem_size) {
  auto range = get_block_range(page_id, page_size, block_mem_size);
  start_block = range.first;
  end_block = range.second;
  num_kv_blocks = end_block - start_block;
  free_list.clear();
  for (int64_t i = start_block; i < end_block; ++i) {
    free_list.push_back(i);
  }
}

std::vector<int64_t> InternalPage::alloc(int64_t num_blocks) {
  if (free_list.size() < static_cast<size_t>(num_blocks)) {
    throw std::runtime_error("Not enough free blocks in page");
  }

  std::vector<int64_t> block_ids;
  block_ids.reserve(num_blocks);
  for (int64_t i = 0; i < num_blocks; ++i) {
    block_ids.push_back(free_list[i]);
  }
  free_list.erase(free_list.begin(), free_list.begin() + num_blocks);
  return block_ids;
}

void InternalPage::free(int64_t block_id) { free_list.push_back(block_id); }

void InternalPage::free_batch(const std::vector<int64_t> &block_ids) {
  free_list.insert(free_list.end(), block_ids.begin(), block_ids.end());
}

bool InternalPage::empty() const {
  return free_list.size() == static_cast<size_t>(num_kv_blocks);
}

bool InternalPage::full() const { return free_list.empty(); }

int64_t InternalPage::num_free_blocks() const {
  return static_cast<int64_t>(free_list.size());
}

std::vector<int64_t> InternalPage::get_free_blocks() const { return free_list; }

std::pair<int64_t, int64_t>
InternalPage::get_block_range(page_id_t page_id, int64_t page_size,
                              int64_t block_mem_size) {

  int64_t start_block =
      (page_id * page_size + block_mem_size - 1) / block_mem_size;
  int64_t end_block = ((page_id + 1) * page_size) / block_mem_size;
  return {start_block, end_block};
}

int64_t InternalPage::get_num_blocks(int64_t page_size,
                                     int64_t block_mem_size) {
  return page_size / block_mem_size;
}

// PageAllocator implementation
PageAllocator::PageAllocator(int64_t num_layers, int64_t mem_size_per_layer,
                             int64_t page_size, int64_t world_size,
                             int64_t pp_rank, bool async_sched,
                             bool contiguous_layout, bool enable_page_prealloc,
                             int64_t num_kv_buffers, int64_t group_id,
                             const std::string &ipc_name)
    : num_layers_(num_layers), mem_size_per_layer_(mem_size_per_layer),
      page_size_(page_size), world_size_(world_size), pp_rank_(pp_rank),
      num_kv_buffers_(num_kv_buffers), group_id_(group_id),
      async_sched_(async_sched), contiguous_layout_(contiguous_layout),
      enable_page_prealloc_(enable_page_prealloc),
      gpu_utilization_(GPU_UTILIZATION),
      num_free_pages_(mem_size_per_layer / page_size),
      num_total_pages_(mem_size_per_layer / page_size),
      min_reserved_pages_(std::min(
          num_free_pages_.load(std::memory_order_relaxed), MIN_RESERVED_PAGES)),
      max_reserved_pages_(std::min(
          num_free_pages_.load(std::memory_order_relaxed), MAX_RESERVED_PAGES)),
      prealloc_running_(false), prealloc_needed_(false),
      total_memory_size_(mem_size_per_layer * num_layers * num_kv_buffers) {

  // Initialize free page list
  const int64_t initial_free_pages =
      num_free_pages_.load(std::memory_order_relaxed);
  for (int64_t i = 0; i < initial_free_pages; ++i) {
    free_page_list_.push_back(i);
  }

  // Initialize memory info tracker
  mem_info_tracker_ =
      std::make_unique<MemInfoTracker>(total_memory_size_, group_id_, ipc_name);

  std::cout << "Init C++ PageAllocator: "
            << "num_layers=" << num_layers << ", "
            << "mem_size_per_layer=" << mem_size_per_layer / (1024 * 1024)
            << "MB, "
            << "total_mem_size="
            << (num_kv_buffers * num_layers * mem_size_per_layer) /
                   (1024 * 1024)
            << "MB, "
            << "page_size=" << page_size / (1024 * 1024) << "MB, "
            << "world_size=" << world_size << ", "
            << "pp_rank=" << pp_rank << ", "
            << "async_sched=" << async_sched << ", "
            << "contiguous_layout=" << contiguous_layout << ", "
            << "enable_prealloc=" << enable_page_prealloc << ", "
            << "num_kv_buffers=" << num_kv_buffers << ", "
            << "group_id=" << group_id << ", "
            << "min_reserved_pages=" << min_reserved_pages_ << ", "
            << "max_reserved_pages=" << max_reserved_pages_ << std::endl;
}

PageAllocator::~PageAllocator() {
  try {
    if (enable_page_prealloc_ && prealloc_thread_) {
      stop_prealloc_thread_internal();
    }
  } catch (...) {
    // Silently ignore exceptions during cleanup
  }
}

std::shared_ptr<InternalPage> PageAllocator::alloc_page() {
  auto start_time = std::chrono::steady_clock::now();

  std::unique_lock<std::mutex> lock(lock_);
  page_id_t page_id = -1;

  while (page_id == -1) {
    // Fast path: allocate from reserved pages
    if (!reserved_page_list_.empty()) {
      page_id = reserved_page_list_.front();
      reserved_page_list_.pop_front();
      num_free_pages_.fetch_sub(1, std::memory_order_relaxed);

      // Trigger preallocation to refill reserved pool if getting low
      if (reserved_page_list_.size() <
          static_cast<size_t>(min_reserved_pages_)) {
        prealloc_needed_ = true;
        cond_.notify_all();
      }

      update_memory_usage_unlocked();
      auto end_time = std::chrono::steady_clock::now();
      auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
          end_time - start_time);
      LOGGER(DEBUG, "alloc 1 page fast path cost %lu us", duration.count());
      // std::cout << "alloc 1 page fast path cost " << duration.count() << "
      // us" << std::endl;

      return std::make_shared<InternalPage>(page_id, page_size_);
    }

    // Slow path: allocate from free pages
    if (!free_page_list_.empty()) {
      page_id = free_page_list_.front();
      free_page_list_.pop_front();
      num_free_pages_.fetch_sub(1, std::memory_order_relaxed);
      break;
    }

    if (num_free_pages_.load(std::memory_order_relaxed) <= 0) {
      throw std::runtime_error("No free pages left");
    }

    if (!enable_page_prealloc_) {
      throw std::runtime_error(
          "Inconsistent page allocator state: no free pages available");
    }

    // Wait for background preallocation
    cond_.wait(lock);
  }

  lock.unlock();

  try {
    map_pages({page_id});
  } catch (const std::exception &e) {
    std::lock_guard<std::mutex> guard(lock_);
    free_page_list_.push_front(page_id);
    num_free_pages_.fetch_add(1, std::memory_order_relaxed);
    cond_.notify_all();
    throw std::runtime_error("Failed to map page " + std::to_string(page_id) +
                             ": " + e.what());
  }

  if (enable_page_prealloc_) {
    trigger_preallocation();
  }

  {
    std::lock_guard<std::mutex> lock(lock_);
    update_memory_usage_unlocked();
  }
  auto end_time = std::chrono::steady_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
      end_time - start_time);
  LOGGER(DEBUG, "alloc 1 page slow path cost %lu us", duration.count());

  return std::make_shared<InternalPage>(page_id, page_size_);
}

void PageAllocator::free_page(page_id_t page_id) {
  {
    std::lock_guard<std::mutex> lock(lock_);
    num_free_pages_.fetch_add(1, std::memory_order_relaxed);

    if (reserved_page_list_.size() < static_cast<size_t>(max_reserved_pages_)) {
      // Fast path: reserve page
      reserved_page_list_.push_back(page_id);
      update_memory_usage_unlocked();
      cond_.notify_all();
      return;
    }
  }

  // Slow path: free page and unmap (lock released, exception-safe)
  unmap_pages({page_id});

  {
    std::lock_guard<std::mutex> lock(lock_);
    free_page_list_.push_back(page_id);
    update_memory_usage_unlocked();
    cond_.notify_all();
  }
}

void PageAllocator::free_pages(const std::vector<page_id_t> &page_ids) {
  auto start_time = std::chrono::steady_clock::now();

  std::vector<page_id_t> pages_to_unmap;

  {
    std::lock_guard<std::mutex> lock(lock_);
    num_free_pages_.fetch_add(static_cast<int64_t>(page_ids.size()),
                              std::memory_order_relaxed);
    int64_t num_to_reserve = max_reserved_pages_ - reserved_page_list_.size();

    if (num_to_reserve > 0) {
      // Fast path: reserve pages
      auto reserve_end =
          page_ids.begin() +
          std::min(static_cast<size_t>(num_to_reserve), page_ids.size());
      reserved_page_list_.insert(reserved_page_list_.end(), page_ids.begin(),
                                 reserve_end);

      pages_to_unmap.assign(reserve_end, page_ids.end());

      if (pages_to_unmap.empty()) {
        update_memory_usage_unlocked();
        cond_.notify_all();
        return;
      }
    } else {
      pages_to_unmap = page_ids;
    }
  }

  // Slow path: unmap pages (lock released, exception-safe)
  unmap_pages(pages_to_unmap);

  {
    std::lock_guard<std::mutex> lock(lock_);
    free_page_list_.insert(free_page_list_.end(), pages_to_unmap.begin(),
                           pages_to_unmap.end());
    update_memory_usage_unlocked();
    cond_.notify_all();
  }

  auto end_time = std::chrono::steady_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
      end_time - start_time);
  LOGGER(DEBUG, "free %ld pages cost %lu us", page_ids.size(),
         duration.count());
}

bool PageAllocator::resize(int64_t new_mem_size) {
  int64_t new_num_pages = new_mem_size / page_size_;

  std::vector<page_id_t> pages_to_unmap;

  {
    std::lock_guard<std::mutex> lock(lock_);

    if (new_num_pages < get_num_inuse_pages_unlocked()) {
      return false;
    }

    const int64_t current_total_pages =
        num_total_pages_.load(std::memory_order_relaxed);
    if (new_num_pages == current_total_pages) {
      return true;
    } else if (new_num_pages > current_total_pages) {
      int64_t num_to_expand = new_num_pages - current_total_pages;

      // Reuse previously reclaimed pages first
      int64_t num_to_reuse = std::min(
          static_cast<int64_t>(reclaimed_page_list_.size()), num_to_expand);
      if (num_to_reuse > 0) {
        for (int64_t i = 0; i < num_to_reuse; ++i) {
          free_page_list_.push_back(reclaimed_page_list_.front());
          reclaimed_page_list_.pop_front();
        }
        num_to_expand -= num_to_reuse;
        num_free_pages_.fetch_add(num_to_reuse, std::memory_order_relaxed);
      }

      // Allocate new pages if needed
      if (num_to_expand > 0) {
        for (int64_t i = current_total_pages;
             i < current_total_pages + num_to_expand; ++i) {
          free_page_list_.push_back(i);
        }
        num_free_pages_.fetch_add(num_to_expand, std::memory_order_relaxed);
      }
      num_total_pages_.store(new_num_pages, std::memory_order_relaxed);
      update_memory_usage_unlocked();
      return true;
    } else {
      // Shrink path
      int64_t num_to_reclaim = current_total_pages - new_num_pages;

      if (free_page_list_.size() < static_cast<size_t>(num_to_reclaim)) {
        // Need to trim reserved pages first
        if (!reserved_page_list_.empty()) {
          pages_to_unmap.assign(reserved_page_list_.begin(),
                                reserved_page_list_.end());
          reserved_page_list_.clear();
        } else {
          return false;
        }
      } else {
        // Enough free pages, reclaim directly
        for (int64_t i = 0; i < num_to_reclaim; ++i) {
          reclaimed_page_list_.push_back(free_page_list_.back());
          free_page_list_.pop_back();
        }
        num_free_pages_.fetch_sub(num_to_reclaim, std::memory_order_relaxed);
        num_total_pages_.store(new_num_pages, std::memory_order_relaxed);
        return true;
      }
    }
  }

  // Unmap pages outside the lock (exception-safe)
  unmap_pages(pages_to_unmap);

  {
    std::lock_guard<std::mutex> lock(lock_);
    int64_t num_to_reclaim =
        num_total_pages_.load(std::memory_order_relaxed) - new_num_pages;

    free_page_list_.insert(free_page_list_.end(), pages_to_unmap.begin(),
                           pages_to_unmap.end());
    update_memory_usage_unlocked();

    if (free_page_list_.size() < static_cast<size_t>(num_to_reclaim)) {
      return false;
    }

    for (int64_t i = 0; i < num_to_reclaim; ++i) {
      reclaimed_page_list_.push_back(free_page_list_.back());
      free_page_list_.pop_back();
    }
    num_free_pages_.fetch_sub(num_to_reclaim, std::memory_order_relaxed);
    num_total_pages_.store(new_num_pages, std::memory_order_relaxed);
  }
  return true;
}

void PageAllocator::trim() {
  std::vector<page_id_t> pages_to_unmap;

  {
    std::lock_guard<std::mutex> lock(lock_);
    pages_to_unmap.assign(reserved_page_list_.begin(),
                          reserved_page_list_.end());
    reserved_page_list_.clear();

    if (pages_to_unmap.empty()) {
      update_memory_usage_unlocked();
      return;
    }
  }

  // Unmap pages outside the lock (exception-safe)
  unmap_pages(pages_to_unmap);

  {
    std::lock_guard<std::mutex> lock(lock_);
    free_page_list_.insert(free_page_list_.end(), pages_to_unmap.begin(),
                           pages_to_unmap.end());
    update_memory_usage_unlocked();
  }
}

int64_t PageAllocator::get_num_free_pages() const {
  return num_free_pages_.load(std::memory_order_relaxed);
}

int64_t PageAllocator::get_num_inuse_pages() const {
  const int64_t total = num_total_pages_.load(std::memory_order_relaxed);
  const int64_t free = num_free_pages_.load(std::memory_order_relaxed);
  return total - free;
}

int64_t PageAllocator::get_num_total_pages() const {
  return num_total_pages_.load(std::memory_order_relaxed);
}

int64_t PageAllocator::get_num_reserved_pages() const {
  std::lock_guard<std::mutex> lock(lock_);
  return reserved_page_list_.size();
}

PageState PageAllocator::get_page_state() const {
  std::lock_guard<std::mutex> lock(lock_);
  return get_page_state_unlocked();
}

int64_t PageAllocator::get_num_inuse_pages_unlocked() const {
  return num_total_pages_.load(std::memory_order_relaxed) -
         num_free_pages_.load(std::memory_order_relaxed);
}

PageState PageAllocator::get_page_state_unlocked() const {
  const int64_t total = num_total_pages_.load(std::memory_order_relaxed);
  const int64_t free = num_free_pages_.load(std::memory_order_relaxed);
  return PageState{total, free, total - free,
                   static_cast<int64_t>(reserved_page_list_.size())};
}

int64_t PageAllocator::get_avail_physical_pages() const {
  size_t avail_phy_mem_size = 0, total_phy_mem_size = 0;
  CHECK_GPU(gpu_vmm::mem_get_info(&avail_phy_mem_size, &total_phy_mem_size));

  const size_t headroom =
      static_cast<size_t>(total_phy_mem_size * (1.0 - gpu_utilization_));
  const size_t usable_phy_mem_size =
      saturating_subtract(avail_phy_mem_size, headroom);

  // Calculate available pages considering layers and KV buffers
  int64_t avail_phy_pages = usable_phy_mem_size / page_size_;
  int64_t avail_pages_per_layer =
      avail_phy_pages / num_layers_ / num_kv_buffers_;
  return avail_pages_per_layer;
}

page_id_t PageAllocator::get_page_id(int64_t block_id,
                                     int64_t block_mem_size) const {
  return block_id * block_mem_size / page_size_;
}

int64_t
PageAllocator::check_and_get_resize_target(int64_t current_mem_size) const {
  if (!mem_info_tracker_) {
    return -1;
  }
  return mem_info_tracker_->check_and_get_resize_target(
      current_mem_size, num_layers_, num_kv_buffers_);
}

std::unordered_map<page_id_t, std::vector<int64_t>>
PageAllocator::group_indices_by_page(const std::vector<int64_t> &indices,
                                     int64_t block_mem_size) const {

  auto start_time = std::chrono::steady_clock::now();

  std::unordered_map<page_id_t, std::vector<int64_t>> result;

  // Pre-calculate constants for efficiency
  int64_t blocks_per_page = page_size_ / block_mem_size;

  // Reserve space for efficiency
  result.reserve(indices.size() / blocks_per_page + 1);

  // Group indices by page_id
  for (int64_t idx : indices) {
    page_id_t page_id = get_page_id(idx, block_mem_size);
    result[page_id].push_back(idx);
  }

  auto end_time = std::chrono::steady_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
      end_time - start_time);
  LOGGER(DEBUG, "C++ group_indices_by_page processed %zu indices in %ld us",
         indices.size(), duration.count());

  return result;
}

// Callback function setters
//
// The swap-then-assign shape in the two setters below is deliberate: these
// callbacks wrap Python callables, and destroying one acquires the GIL. With
// the blocking bindings releasing the GIL, destroying a callback while holding
// lock_ would invert the lock/GIL order against a thread that holds lock_ and
// needs the GIL -- so the old callback must be destroyed after lock_ is
// released (when `old` goes out of scope).
void PageAllocator::set_broadcast_map_callback(BroadcastMapCallback callback) {
  BroadcastMapCallback old;
  {
    std::lock_guard<std::mutex> lock(lock_);
    old = std::move(broadcast_map_callback_);
    broadcast_map_callback_ = std::move(callback);
  }
  LOGGER(INFO, "Broadcast map callback set for PageAllocator (world_size=%ld)",
         world_size_);
}

void PageAllocator::set_broadcast_unmap_callback(
    BroadcastUnmapCallback callback) {
  BroadcastUnmapCallback old;
  {
    std::lock_guard<std::mutex> lock(lock_);
    old = std::move(broadcast_unmap_callback_);
    broadcast_unmap_callback_ = std::move(callback);
  }
  LOGGER(INFO,
         "Broadcast unmap callback set for PageAllocator (world_size=%ld)",
         world_size_);
}

void PageAllocator::set_use_worker_ipc(bool use_worker_ipc) {
  use_worker_ipc_.store(use_worker_ipc, std::memory_order_release);
  LOGGER(INFO, "use_worker_ipc set to %d for PageAllocator",
         static_cast<int>(use_worker_ipc));
}

void PageAllocator::start_prealloc_thread() {
  if (enable_page_prealloc_) {
    start_prealloc_thread_internal();
  }
}

void PageAllocator::stop_prealloc_thread() {
  if (enable_page_prealloc_) {
    stop_prealloc_thread_internal();
  }
}

void PageAllocator::prealloc_worker() {
  auto start_time = std::chrono::steady_clock::now();

  while (prealloc_running_) {
    std::unique_lock<std::mutex> lock(lock_);

    // Wait until preallocation is needed or thread is stopped
    while (!prealloc_needed_ && prealloc_running_) {
      cond_.wait(lock);
    }

    LOGGER(INFO, "prealloc worker triggered...");
    if (!prealloc_running_) {
      break;
    }

    start_time = std::chrono::steady_clock::now();
    prealloc_needed_ = false;

    int64_t current_reserved = reserved_page_list_.size();
    int64_t to_reserve = std::max(0L, min_reserved_pages_ - current_reserved);
    // Only try to reserve up to the available free pages and physical memory
    to_reserve =
        std::min({to_reserve, static_cast<int64_t>(free_page_list_.size()),
                  get_avail_physical_pages()});

    LOGGER(INFO,
           "max_reserved_pages: %ld, min_reserved_pages: %ld, "
           "current_reserved: %ld, to_reserve: %ld, len(free_page_list): %zu",
           max_reserved_pages_, min_reserved_pages_, current_reserved,
           to_reserve, free_page_list_.size());

    if (to_reserve <= 0) {
      auto end_time = std::chrono::steady_clock::now();
      auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
          end_time - start_time);
      LOGGER(INFO,
             "prealloc cost: %ld us, no need to preallocate(to_reserve: %ld).",
             duration.count(), to_reserve);
      continue;
    }

    std::vector<page_id_t> pages_to_reserve;
    pages_to_reserve.reserve(to_reserve);

    // Get pages from free list
    for (int64_t i = 0; i < to_reserve && !free_page_list_.empty(); ++i) {
      pages_to_reserve.push_back(free_page_list_.front());
      free_page_list_.pop_front();
    }

    lock.unlock();

    if (!pages_to_reserve.empty()) {
      try {
        map_pages(pages_to_reserve);
        lock.lock();
        reserved_page_list_.insert(reserved_page_list_.end(),
                                   pages_to_reserve.begin(),
                                   pages_to_reserve.end());
        update_memory_usage_unlocked();
        cond_.notify_all();
        LOGGER(INFO, "Preallocated %ld pages, reserved=%ld",
               pages_to_reserve.size(), reserved_page_list_.size());
      } catch (const std::exception &e) {
        lock.lock();
        free_page_list_.insert(free_page_list_.begin(),
                               pages_to_reserve.begin(),
                               pages_to_reserve.end());
        cond_.notify_all();
        LOGGER(ERROR, "Failed to preallocate %ld pages: %s",
               pages_to_reserve.size(), e.what());
      }

      auto end_time = std::chrono::steady_clock::now();
      auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
          end_time - start_time);
      LOGGER(INFO, "prealloc cost: %ld us, prealloc %ld pages.",
             duration.count(), pages_to_reserve.size());
    }
  }
}

void PageAllocator::map_pages(const std::vector<page_id_t> &page_ids) {
  std::vector<offset_t> offsets;
  offsets.reserve(page_ids.size());

  if (contiguous_layout_) {
    for (page_id_t pid : page_ids) {
      offsets.push_back(pid * page_size_ * num_layers_ * num_kv_buffers_);
    }
  } else {
    for (page_id_t pid : page_ids) {
      offsets.push_back(pid * page_size_);
    }
  }

  if ((world_size_ > 1 || should_use_worker_ipc()) && broadcast_map_callback_) {
    // Multi-process mode: execute map on all TP workers via broadcast callback
    broadcast_map_callback_(world_size_, offsets);
  } else {
    // Single-process mode: directly call FTensorAllocator
    auto allocator = FTensorAllocator::global_allocator(group_id_);
    bool success = allocator->map_to_kv_tensors(offsets);
    if (!success) {
      throw std::runtime_error("Failed to map pages to KV tensors");
    }
  }

  LOGGER(INFO, "Mapped %zu pages to KV tensors", page_ids.size());
}

void PageAllocator::unmap_pages(const std::vector<page_id_t> &page_ids) {
  auto start_time = std::chrono::steady_clock::now();

  std::vector<offset_t> offsets;
  offsets.reserve(page_ids.size());

  if (contiguous_layout_) {
    for (page_id_t pid : page_ids) {
      offsets.push_back(pid * page_size_ * num_layers_ * num_kv_buffers_);
    }
  } else {
    for (page_id_t pid : page_ids) {
      offsets.push_back(pid * page_size_);
    }
  }

  if ((world_size_ > 1 || should_use_worker_ipc()) &&
      broadcast_unmap_callback_) {
    // Multi-process mode: execute unmap on all TP workers via broadcast
    // callback
    broadcast_unmap_callback_(world_size_, offsets);
  } else {
    // Need to synchronize first in async scheduling mode
    if (async_sched_) {
      CHECK_GPU(gpu_vmm::device_synchronize());
    }
    auto allocator = FTensorAllocator::global_allocator(group_id_);
    bool success = allocator->unmap_from_kv_tensors(offsets);
    if (!success) {
      throw std::runtime_error("Failed to unmap pages from KV tensors");
    }
  }

  auto end_time = std::chrono::steady_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
      end_time - start_time);
  LOGGER(INFO, "Unmapped %zu pages from KV tensors, cost: %lu us",
         page_ids.size(), duration.count());
}

void PageAllocator::update_memory_usage_unlocked() {
  // Calculate currently used physical memory (excluding preallocated pages)
  int64_t used_phy_mem_size = get_num_inuse_pages_unlocked() * num_layers_ *
                              page_size_ * num_kv_buffers_;
  // Calculate physical memory occupied by preallocated pages
  int64_t prealloc_phy_mem_size =
      static_cast<int64_t>(reserved_page_list_.size()) * num_layers_ *
      page_size_ * num_kv_buffers_;

  if (mem_info_tracker_) {
    mem_info_tracker_->update_memory_usage(used_phy_mem_size,
                                           prealloc_phy_mem_size);
  }
}

void PageAllocator::reset_free_page_order() {
  std::lock_guard<std::mutex> lock(lock_);
  std::vector<page_id_t> sorted_pages(free_page_list_.begin(),
                                      free_page_list_.end());
  std::sort(sorted_pages.begin(), sorted_pages.end());
  free_page_list_.assign(sorted_pages.begin(), sorted_pages.end());
}

void PageAllocator::trigger_preallocation() {
  std::lock_guard<std::mutex> lock(lock_);
  prealloc_needed_ = true;
  cond_.notify_all();
}

void PageAllocator::start_prealloc_thread_internal() {
  std::lock_guard<std::mutex> ctl(thread_ctl_lock_);
  if (!prealloc_thread_) {
    prealloc_running_ = true;
    prealloc_thread_ =
        std::make_unique<std::thread>(&PageAllocator::prealloc_worker, this);

    // Initial preallocation trigger
    trigger_preallocation();
  }

  // Start resize watcher thread alongside prealloc thread
  if (!resize_watcher_thread_) {
    resize_watcher_running_ = true;
    resize_watcher_thread_ =
        std::make_unique<std::thread>(&PageAllocator::resize_watcher, this);
  }
}

namespace {
// Join a thread without holding the GIL. The prealloc worker may be inside a
// Python broadcast callback waiting for the GIL; joining it while holding the
// GIL would deadlock. The stop_prealloc_thread binding already releases the
// GIL, but the destructor runs from Python garbage collection with the GIL
// held, so release conditionally here.
void join_without_gil(std::thread &t) {
  if (PyGILState_Check()) {
    PyThreadState *state = PyEval_SaveThread();
    t.join();
    PyEval_RestoreThread(state);
  } else {
    t.join();
  }
}
} // namespace

void PageAllocator::stop_prealloc_thread_internal() {
  std::lock_guard<std::mutex> ctl(thread_ctl_lock_);
  if (prealloc_thread_) {
    {
      std::lock_guard<std::mutex> lock(lock_);
      prealloc_running_ = false;
      cond_.notify_all();
    }

    join_without_gil(*prealloc_thread_);
    prealloc_thread_.reset();
    LOGGER(DEBUG, "Stopped page preallocation thread");
  }

  // Stop resize watcher thread
  if (resize_watcher_thread_) {
    resize_watcher_running_ = false;
    join_without_gil(*resize_watcher_thread_);
    resize_watcher_thread_.reset();
    LOGGER(DEBUG, "Stopped resize watcher thread");
  }
}

bool PageAllocator::should_use_worker_ipc() const {
  // A plain pushed value: no thread ever needs Python (or the GIL) to answer
  // this. The decision is fixed once Python calls set_use_worker_ipc() during
  // KVCacheManager init, before the prealloc thread starts.
  return use_worker_ipc_.load(std::memory_order_acquire);
}

void PageAllocator::resize_watcher() {
  LOGGER(INFO, "Resize watcher thread started (poll interval: 100ms)");
  while (resize_watcher_running_) {
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    if (!resize_watcher_running_) {
      break;
    }
    if (mem_info_tracker_) {
      int64_t target = mem_info_tracker_->check_and_get_resize_target(
          mem_size_per_layer_, num_layers_, num_kv_buffers_);
      resize_target_.store(target, std::memory_order_relaxed);
    }
  }
  LOGGER(INFO, "Resize watcher thread stopped");
}

int64_t PageAllocator::get_resize_target() const {
  return resize_target_.load(std::memory_order_relaxed);
}

} // namespace kvcached
