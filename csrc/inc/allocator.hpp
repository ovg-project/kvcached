// SPDX-FileCopyrightText: Copyright contributors to the kvcached project
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include <cuda_runtime.h>
#include <torch/extension.h>

#include "constants.hpp"
#include "ftensor.hpp"
#include "page.hpp"

namespace kvcached {

class FTensorAllocator {
public:
  FTensorAllocator(const torch::Device &device, bool contiguous_layout);
  ~FTensorAllocator();

  // KV cache interfaces.
  // group_id allows multiple independent KV cache pools (e.g., for hybrid
  // attention models with separate full-attention and sliding-window groups).
  // Default group_id=0 preserves backward compatibility.
  std::vector<torch::Tensor> create_kv_tensors(size_t size, torch::Dtype dtype,
                                               const std::string &dev_str,
                                               int64_t num_layers,
                                               int64_t num_kv_buffers = 2,
                                               int64_t group_id = 0);
  bool kv_tensors_created(int64_t group_id = 0);
  bool map_to_kv_tensors(const std::vector<offset_t> &offsets,
                         int64_t group_id = 0);
  bool unmap_from_kv_tensors(const std::vector<offset_t> &offsets,
                             int64_t group_id = 0);

  // Global status interfaces.
  static void init(const std::string &dev_str, size_t page_size = 0,
                   bool contiguous_layout = false);
  static void shutdown();
  static FTensorAllocator *global_allocator();
  void destroy();

private:
  // Per-group state that holds FTensors and metadata for one KV cache pool.
  struct KVGroup {
    int64_t num_layers = 0;
    size_t kv_tensor_size_per_layer = 0;
    // For per-layer layout: one tensor per layer
    std::unordered_map<std::string, std::unique_ptr<FTensor>> ftensors;
    // For contiguous layout: single tensor containing all layers
    std::unique_ptr<FTensor> contiguous_kv_tensor;
    std::shared_ptr<Page> zero_page;
  };

  // Get or create a KVGroup for the given group_id.  Must be called with
  // mtx_ held.
  KVGroup &get_or_create_group_(int64_t group_id);

  // Raw FTensor interfaces. Must call with lock.
  static std::string get_anon_tensor_name_();
  std::vector<torch::Tensor>
  create_kv_tensors_per_layer_(KVGroup &group, std::string_view prefix,
                               size_t size, torch::Dtype dtype,
                               const std::string &dev_str, int64_t num_layers);
  std::vector<torch::Tensor>
  create_kv_tensors_contiguous_(KVGroup &group, size_t size, torch::Dtype dtype,
                                const std::string &dev_str, int64_t num_layers,
                                size_t compound_page_size);
  torch::Tensor create_ftensor_(KVGroup &group, size_t size, torch::Dtype dtype,
                                const std::string &dev_str,
                                std::string name = "");
  void free_ftensor_(KVGroup &group, torch::Tensor &ftensor);

  // CUDA util functions.
  void init_cuda_();

  static std::unique_ptr<FTensorAllocator> g_allocator_;
  static std::mutex g_allocator_mutex_;

  torch::Device dev_;
  bool contiguous_layout_;

  mutable std::mutex mtx_;
  // Map from group_id to per-group KV cache state.
  std::unordered_map<int64_t, KVGroup> kv_groups_;
};

} // namespace kvcached
