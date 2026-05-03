// SPDX-FileCopyrightText: Copyright contributors to the kvcached project
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <vector>

#include <cuda_runtime.h>
#include <torch/extension.h>

#include "constants.hpp"
#include "mem_info_tracker.hpp"
#include "page.hpp"

namespace kvcached {

// Callback function types for multi-process support
using BroadcastMapCallback =
    std::function<void(int64_t, const std::vector<offset_t> &)>;
using BroadcastUnmapCallback =
    std::function<void(int64_t, const std::vector<offset_t> &)>;
using ShouldUseWorkerIpcCallback = std::function<bool()>;

// Independent InternalPage class
class InternalPage {
public:
  page_id_t page_id;
  int64_t page_size;
  int64_t start_block;
  int64_t end_block;
  int64_t num_kv_blocks;
  std::vector<int64_t> free_list;

  InternalPage(page_id_t id, int64_t size);
  void init(int64_t block_mem_size);
  std::vector<int64_t> alloc(int64_t num_blocks = 1);
  void free(int64_t block_id);
  void free_batch(const std::vector<int64_t> &block_ids);
  bool empty() const;
  bool full() const;
  int64_t num_free_blocks() const;
  std::vector<int64_t> get_free_blocks() const;

  static std::pair<int64_t, int64_t>
  get_block_range(page_id_t page_id, int64_t page_size, int64_t block_mem_size);
  static int64_t get_num_blocks(int64_t page_size, int64_t block_mem_size);
};

class PageAllocator {
public:
  PageAllocator(int64_t num_layers, int64_t mem_size_per_layer,
                int64_t page_size, int64_t world_size = 1, int64_t pp_rank = 0,
                bool async_sched = false, bool contiguous_layout = true,
                bool enable_page_prealloc = true, int64_t num_kv_buffers = 2,
                int64_t group_id = 0);

  ~PageAllocator();

  // Page allocation and deallocation
  std::shared_ptr<InternalPage> alloc_page();
  void free_page(page_id_t page_id);
  void free_pages(const std::vector<page_id_t> &page_ids);

  // Memory management
  bool resize(int64_t new_mem_size);
  void trim();

  // Status queries
  int64_t get_num_free_pages() const;
  int64_t get_num_inuse_pages() const;
  int64_t get_num_total_pages() const;
  int64_t get_num_reserved_pages() const;
  int64_t get_avail_physical_pages() const;

  // Poll the shared-memory MemInfoStruct to see if an external controller
  // (e.g. `kvctl limit`) has written a new total_size. Returns the new
  // per-layer mem_size if it differs from current_mem_size, otherwise -1.
  int64_t check_and_get_resize_target(int64_t current_mem_size) const;

  // Utility functions
  page_id_t get_page_id(int64_t block_id, int64_t block_mem_size) const;

  // New method for efficient index grouping
  std::unordered_map<page_id_t, std::vector<int64_t>>
  group_indices_by_page(const std::vector<int64_t> &indices,
                        int64_t block_mem_size) const;

  // Page list management
  void reset_free_page_order();

  // Thread management
  void start_prealloc_thread();
  void stop_prealloc_thread();

  // Callback function setters for multi-process support
  void set_broadcast_map_callback(BroadcastMapCallback callback);
  void set_broadcast_unmap_callback(BroadcastUnmapCallback callback);
  void set_should_use_worker_ipc_callback(ShouldUseWorkerIpcCallback callback);

private:
  // Preallocation thread worker
  void prealloc_worker();

  // Internal methods
  void map_pages(const std::vector<page_id_t> &page_ids);
  void unmap_pages(const std::vector<page_id_t> &page_ids);
  void update_memory_usage();
  void trigger_preallocation();
  void start_prealloc_thread_internal();
  void stop_prealloc_thread_internal();
  bool should_use_worker_ipc() const;

  // Configuration
  int64_t num_layers_;
  int64_t mem_size_per_layer_;
  int64_t page_size_;
  int64_t world_size_;
  int64_t pp_rank_;
  int64_t num_kv_buffers_;
  int64_t group_id_;
  bool async_sched_;
  bool contiguous_layout_;
  bool enable_page_prealloc_;
  double gpu_utilization_;

  // Memory tracking
  int64_t num_free_pages_;
  int64_t num_total_pages_;

  // Page lists
  std::deque<page_id_t> free_page_list_;
  std::deque<page_id_t> reserved_page_list_;
  std::deque<page_id_t> reclaimed_page_list_;

  // Preallocation settings
  int64_t min_reserved_pages_;
  int64_t max_reserved_pages_;

  // Thread management
  mutable std::mutex lock_;
  std::condition_variable cond_;
  std::atomic<bool> prealloc_running_;
  std::atomic<bool> prealloc_needed_;
  std::unique_ptr<std::thread> prealloc_thread_;

  // Memory info tracker
  int64_t total_memory_size_;
  std::unique_ptr<MemInfoTracker> mem_info_tracker_;

  // Callback functions for multi-process support
  BroadcastMapCallback broadcast_map_callback_;
  BroadcastUnmapCallback broadcast_unmap_callback_;
  ShouldUseWorkerIpcCallback should_use_worker_ipc_callback_;
};

} // namespace kvcached
