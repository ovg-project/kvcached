// SPDX-FileCopyrightText: Copyright contributors to the kvcached project
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include <ATen/core/Tensor.h>
#include <c10/core/Device.h>
#include <c10/core/ScalarType.h>

#include "constants.hpp"
#include "ftensor.hpp"
#include "page.hpp"

namespace kvcached {

class FTensorAllocator {
public:
  FTensorAllocator(const c10::Device &device, bool contiguous_layout);
  ~FTensorAllocator();

  // KV cache interfaces.
  std::vector<at::Tensor> create_kv_tensors(size_t size, c10::ScalarType dtype,
                                            const std::string &dev_str,
                                            int64_t num_layers,
                                            int64_t num_kv_buffers = 2);
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
  std::vector<at::Tensor>
  create_kv_tensors_per_layer_(std::string_view prefix, size_t size,
                               c10::ScalarType dtype,
                               const std::string &dev_str,
                               int64_t num_layers);
  std::vector<at::Tensor>
  create_kv_tensors_contiguous_(size_t size, c10::ScalarType dtype,
                                const std::string &dev_str, int64_t num_layers,
                                size_t compound_page_size);
  at::Tensor create_ftensor_(size_t size, c10::ScalarType dtype,
                             const std::string &dev_str,
                             std::string name = "");
  void free_ftensor_(at::Tensor &ftensor);

  // GPU VMM util functions.
  void init_gpu_();

  static std::unique_ptr<FTensorAllocator> g_allocator_;
  static std::mutex g_allocator_mutex_;

  c10::Device dev_;

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
