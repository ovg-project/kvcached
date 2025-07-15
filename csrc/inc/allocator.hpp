#pragma once

#include <cstddef>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include <cuda_runtime.h>
#include <torch/extension.h>

#include "constants.hpp"
#include "ftensor.hpp"
#include "page.hpp"

namespace kvcached {

class FTensorAllocator {
public:
  FTensorAllocator(size_t total_mem_size, const torch::Device &device,
                   bool single_vaddr = false);
  ~FTensorAllocator();

  // Raw FTensor interfaces.
  torch::Tensor create_ftensor(size_t size, torch::Dtype dtype,
                               const std::string &dev_str,
                               std::string name = "", void *vaddr = nullptr);
  void free_ftensor(torch::Tensor &ftensor);
  void destroy();

  // KV cache interfaces.
  std::vector<torch::Tensor> create_kv_tensors(size_t size, torch::Dtype dtype,
                                               const std::string &dev_str,
                                               int64_t num_layers);
  void free_kv_tensors();
  bool map_to_kv_tensors(const std::vector<offset_t> &offsets);
  bool unmap_from_kv_tensors(const std::vector<offset_t> &offsets);

  // Global status interfaces.
  static void init(size_t total_mem_size, const std::string &dev_str,
                   bool single_vaddr);
  static void shutdown();
  static FTensorAllocator *global_allocator();

private:
  static std::string get_anon_tensor_name_();
  std::vector<torch::Tensor> create_kv_tensors_impl_(std::string_view prefix,
                                                     size_t size,
                                                     torch::Dtype dtype,
                                                     const std::string &dev_str,
                                                     int64_t num_layers);
  std::vector<torch::Tensor>
  create_kv_tensors_impl_(std::string_view prefix, size_t size,
                          torch::Dtype dtype, const std::string &dev_str,
                          int64_t num_layers, generic_ptr_t vaddr);

  void init_cuda_();

  static std::unique_ptr<FTensorAllocator> g_allocator_;

  size_t total_mem_size_;
  torch::Device dev_;

  int64_t num_layers_;

  std::mutex mtx_;
  std::unordered_map<std::string, std::unique_ptr<FTensor>> ftensors_;
  std::shared_ptr<Page> zero_page_;
  bool single_vaddr_;
  generic_ptr_t vaddr_;
};

} // namespace kvcached
