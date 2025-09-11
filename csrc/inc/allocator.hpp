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
  std::vector<torch::Tensor> create_kv_tensors(size_t size, torch::Dtype dtype,
                                               const std::string &dev_str,
                                               int64_t num_layers);
  bool kv_tensors_created();
  bool map_to_kv_tensors(const std::vector<offset_t> &offsets);
  bool unmap_from_kv_tensors(const std::vector<offset_t> &offsets);

  // Global status interfaces.
  static void init(const std::string &dev_str, size_t page_size = 0,
                   bool contiguous_layout = false);
  static void shutdown();
  static FTensorAllocator *global_allocator();
  void destroy();

private:
  // Raw FTensor interfaces. Must call with lock.
  static std::string get_anon_tensor_name_();
  std::vector<torch::Tensor>
  create_kv_tensors_per_layer_(std::string_view prefix, size_t size,
                               torch::Dtype dtype, const std::string &dev_str,
                               int64_t num_layers);
  std::vector<torch::Tensor>
  create_kv_tensors_contiguous_(size_t size, torch::Dtype dtype,
                                const std::string &dev_str, int64_t num_layers);
  torch::Tensor create_ftensor_(size_t size, torch::Dtype dtype,
                                const std::string &dev_str,
                                std::string name = "");
  void free_ftensor_(torch::Tensor &ftensor);

  // CUDA util functions.
  void init_cuda_();

  static std::unique_ptr<FTensorAllocator> g_allocator_;
  static std::mutex g_allocator_mutex_;

  torch::Device dev_;

  int64_t num_layers_;
  bool contiguous_layout_;
  size_t kv_tensor_size_per_layer_;

  mutable std::mutex mtx_;
  // For per-layer layout: one tensor per layer
  std::unordered_map<std::string, std::unique_ptr<FTensor>> ftensors_;
  // For contiguous layout: single tensor containing all layers
  std::unique_ptr<FTensor> contiguous_kv_tensor_;
  std::shared_ptr<Page> zero_page_;
};

} // namespace kvcached
