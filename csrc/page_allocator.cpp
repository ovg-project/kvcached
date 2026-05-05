// SPDX-FileCopyrightText: Copyright contributors to the kvcached project
// SPDX-License-Identifier: Apache-2.0

#include "page_allocator.hpp"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <stdexcept>

#include "allocator.hpp"
#include "cuda_utils.hpp"
#include "mem_info_tracker.hpp"
#include "torch_utils.hpp"

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
                             int64_t num_kv_buffers, int64_t group_id)
    : num_layers_(num_layers), mem_size_per_layer_(mem_size_per_layer),
      page_size_(page_size), world_size_(world_size), pp_rank_(pp_rank),
      num_kv_buffers_(num_kv_buffers), group_id_(group_id),
      async_sched_(async_sched), contiguous_layout_(contiguous_layout),
      enable_page_prealloc_(enable_page_prealloc),
      gpu_utilization_(GPU_UTILIZATION),
      num_free_pages_(mem_size_per_layer / page_size),
      num_total_pages_(mem_size_per_layer / page_size),
      min_reserved_pages_(std::min(num_free_pages_, MIN_RESERVED_PAGES)),
      max_reserved_pages_(std::min(num_free_pages_, MAX_RESERVED_PAGES)),
      prealloc_running_(false), prealloc_needed_(false),
      total_memory_size_(mem_size_per_layer * num_layers * num_kv_buffers) {

  // Initialize free page list
  for (int64_t i = 0; i < num_free_pages_; ++i) {
    free_page_list_.push_back(i);
  }

  // Initialize memory info tracker
  mem_info_tracker_ =
      std::make_unique<MemInfoTracker>(total_memory_size_, group_id_);

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
      num_free_pages_--;

      // Trigger preallocation if reserved pool is getting low
      if (free_page_list_.size() != 0 &&
          reserved_page_list_.size() <
              static_cast<size_t>(min_reserved_pages_)) {
        max_reserved_pages_ =
            std::max(max_reserved_pages_ * 4, num_total_pages_);
        LOGGER(INFO, "Increase max_reserved_pages_ %ld -> %ld",
               max_reserved_pages_ / 2, max_reserved_pages_);
        prealloc_needed_ = true;
        cond_.notify_all();
        // pass
      }

      update_memory_usage();
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
      num_free_pages_--;
      break;
    }

    if (num_free_pages_ <= 0) {
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
    num_free_pages_++;
    cond_.notify_all();
    throw std::runtime_error("Failed to map page " + std::to_string(page_id) +
                             ": " + e.what());
  }

  if (enable_page_prealloc_) {
    trigger_preallocation();
  }

  update_memory_usage();
  auto end_time = std::chrono::steady_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
      end_time - start_time);
  LOGGER(DEBUG, "alloc 1 page slow path cost %lu us", duration.count());

  return std::make_shared<InternalPage>(page_id, page_size_);
}

void PageAllocator::free_page(page_id_t page_id) {
  std::lock_guard<std::mutex> lock(lock_);

  num_free_pages_++;

  if (reserved_page_list_.size() < static_cast<size_t>(max_reserved_pages_)) {
    // Fast path: reserve page
    reserved_page_list_.push_back(page_id);
    update_memory_usage();
    cond_.notify_all();
    return;
  }

  lock_.unlock();

  // Slow path: free page and unmap
  unmap_pages({page_id});

  lock_.lock();
  free_page_list_.push_back(page_id);
  update_memory_usage();
  cond_.notify_all();
}

void PageAllocator::free_pages(const std::vector<page_id_t> &page_ids) {
  auto start_time = std::chrono::steady_clock::now();

  std::lock_guard<std::mutex> lock(lock_);

  num_free_pages_ += page_ids.size();
  int64_t num_to_reserve = max_reserved_pages_ - reserved_page_list_.size();

  if (num_to_reserve > 0) {
    // Fast path: reserve pages
    auto reserve_end =
        page_ids.begin() +
        std::min(static_cast<size_t>(num_to_reserve), page_ids.size());
    reserved_page_list_.insert(reserved_page_list_.end(), page_ids.begin(),
                               reserve_end);

    std::vector<page_id_t> remaining_pages(reserve_end, page_ids.end());

    if (remaining_pages.empty()) {
      update_memory_usage();
      cond_.notify_all();
      return;
    }

    lock_.unlock();

    // Slow path: free remaining pages
    unmap_pages(remaining_pages);

    lock_.lock();
    free_page_list_.insert(free_page_list_.end(), remaining_pages.begin(),
                           remaining_pages.end());
  } else {
    lock_.unlock();
    unmap_pages(page_ids);
    lock_.lock();
    free_page_list_.insert(free_page_list_.end(), page_ids.begin(),
                           page_ids.end());
  }

  update_memory_usage();
  cond_.notify_all();

  auto end_time = std::chrono::steady_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
      end_time - start_time);
  LOGGER(DEBUG, "free %ld pages cost %lu us", page_ids.size(),
         duration.count());
}

bool PageAllocator::resize(int64_t new_mem_size) {
  int64_t new_num_pages = new_mem_size / page_size_;
  std::lock_guard<std::mutex> lock(lock_);

  if (new_num_pages < get_num_inuse_pages()) {
    return false;
  }

  if (new_num_pages == num_total_pages_) {
    return true;
  } else if (new_num_pages > num_total_pages_) {
    int64_t num_to_expand = new_num_pages - num_total_pages_;

    // Reuse previously reclaimed pages first
    int64_t num_to_reuse = std::min(
        static_cast<int64_t>(reclaimed_page_list_.size()), num_to_expand);
    if (num_to_reuse > 0) {
      for (int64_t i = 0; i < num_to_reuse; ++i) {
        free_page_list_.push_back(reclaimed_page_list_.front());
        reclaimed_page_list_.pop_front();
      }
      num_to_expand -= num_to_reuse;
      num_free_pages_ += num_to_reuse;
    }

    // Allocate new pages if needed
    if (num_to_expand > 0) {
      for (int64_t i = num_total_pages_; i < num_total_pages_ + num_to_expand;
           ++i) {
        free_page_list_.push_back(i);
      }
      num_free_pages_ += num_to_expand;
    }
    num_total_pages_ = new_num_pages;
    update_memory_usage();
  } else {
    int64_t num_to_reclaim = num_total_pages_ - new_num_pages;

    if (free_page_list_.size() < static_cast<size_t>(num_to_reclaim)) {
      // Need to trim reserved pages first
      if (!reserved_page_list_.empty()) {
        std::vector<page_id_t> pages_to_unmap(reserved_page_list_.begin(),
                                              reserved_page_list_.end());
        reserved_page_list_.clear();

        lock_.unlock();
        unmap_pages(pages_to_unmap);
        lock_.lock();

        free_page_list_.insert(free_page_list_.end(), pages_to_unmap.begin(),
                               pages_to_unmap.end());
        update_memory_usage();
      }
    }

    if (free_page_list_.size() < static_cast<size_t>(num_to_reclaim)) {
      return false;
    }

    for (int64_t i = 0; i < num_to_reclaim; ++i) {
      reclaimed_page_list_.push_back(free_page_list_.back());
      free_page_list_.pop_back();
    }
    num_free_pages_ -= num_to_reclaim;
    num_total_pages_ = new_num_pages;
  }
  return true;
}

void PageAllocator::trim() {
  std::lock_guard<std::mutex> lock(lock_);
  std::vector<page_id_t> pages_to_unmap(reserved_page_list_.begin(),
                                        reserved_page_list_.end());
  reserved_page_list_.clear();

  if (pages_to_unmap.empty()) {
    update_memory_usage();
    return;
  }

  lock_.unlock();
  unmap_pages(pages_to_unmap);
  lock_.lock();

  free_page_list_.insert(free_page_list_.end(), pages_to_unmap.begin(),
                         pages_to_unmap.end());
  update_memory_usage();
}

int64_t PageAllocator::get_num_free_pages() const { return num_free_pages_; }

int64_t PageAllocator::get_num_inuse_pages() const {
  return num_total_pages_ - num_free_pages_;
}

int64_t PageAllocator::get_num_total_pages() const { return num_total_pages_; }

int64_t PageAllocator::get_num_reserved_pages() const {
  std::lock_guard<std::mutex> lock(lock_);
  return reserved_page_list_.size();
}

int64_t PageAllocator::get_avail_physical_pages() const {
  size_t avail_phy_mem_size, total_phy_mem_size;
  cudaMemGetInfo(&avail_phy_mem_size, &total_phy_mem_size);

  size_t headroom = total_phy_mem_size * (1.0 - gpu_utilization_);
  avail_phy_mem_size =
      std::max(avail_phy_mem_size - headroom, static_cast<size_t>(0));

  // Calculate available pages considering layers and KV buffers
  int64_t avail_phy_pages = avail_phy_mem_size / page_size_;
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
void PageAllocator::set_broadcast_map_callback(BroadcastMapCallback callback) {
  std::lock_guard<std::mutex> lock(lock_);
  broadcast_map_callback_ = callback;
  LOGGER(INFO, "Broadcast map callback set for PageAllocator (world_size=%ld)",
         world_size_);
}

void PageAllocator::set_broadcast_unmap_callback(
    BroadcastUnmapCallback callback) {
  std::lock_guard<std::mutex> lock(lock_);
  broadcast_unmap_callback_ = callback;
  LOGGER(INFO,
         "Broadcast unmap callback set for PageAllocator (world_size=%ld)",
         world_size_);
}

void PageAllocator::set_should_use_worker_ipc_callback(
    ShouldUseWorkerIpcCallback callback) {
  std::lock_guard<std::mutex> lock(lock_);
  should_use_worker_ipc_callback_ = callback;
  LOGGER(INFO, "Should-use-worker-ipc callback set for PageAllocator");
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
        update_memory_usage();
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
    // Need to synchronize CUDA first in async scheduling mode
    if (async_sched_) {
      CHECK_RT(cudaDeviceSynchronize());
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

void PageAllocator::update_memory_usage() {
  // Calculate currently used physical memory (excluding preallocated pages)
  int64_t used_phy_mem_size =
      get_num_inuse_pages() * num_layers_ * page_size_ * num_kv_buffers_;
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
  if (!prealloc_thread_) {
    prealloc_running_ = true;
    prealloc_thread_ =
        std::make_unique<std::thread>(&PageAllocator::prealloc_worker, this);

    // Initial preallocation trigger
    trigger_preallocation();
  }
}

void PageAllocator::stop_prealloc_thread_internal() {
  if (prealloc_thread_) {
    {
      std::lock_guard<std::mutex> lock(lock_);
      prealloc_running_ = false;
      cond_.notify_all();
    }

    prealloc_thread_->join();
    prealloc_thread_.reset();
    LOGGER(DEBUG, "Stopped page preallocation thread");
  }
}

bool PageAllocator::should_use_worker_ipc() const {
  if (should_use_worker_ipc_callback_) {
    return should_use_worker_ipc_callback_();
  }
  return false;
}

} // namespace kvcached