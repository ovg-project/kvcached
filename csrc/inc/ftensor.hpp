// SPDX-FileCopyrightText: Copyright contributors to the kvcached project
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <memory>
#include <string>
#include <unordered_map>

#include <torch/csrc/stable/device.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/headeronly/core/ScalarType.h>

#include "constants.hpp"
#include "page.hpp"

namespace kvcached {

/* NOTE: FTensorAllocator is thread-safe but FTensor is not. */
class __attribute__((visibility("hidden"))) FTensor {
public:
  FTensor(const std::string &name, size_t size,
          torch::headeronly::ScalarType dtype, torch::stable::Device dev,
          std::shared_ptr<Page> zero_page, size_t page_size = 0);
  ~FTensor();
  bool map(offset_t offset);
  bool unmap(offset_t offset);

  inline torch::stable::Tensor get_tensor() noexcept { return tensor_; }

private:
  bool map_(Page *page, offset_t offset, bool set_access = true);
  bool set_access_(generic_ptr_t addr, size_t size);
  bool init_with_zero_();

  std::string name_;
  generic_ptr_t vaddr_;
  size_t size_;
  size_t page_size_;
  torch::headeronly::ScalarType dtype_;
  torch::stable::Device dev_;
  std::shared_ptr<Page> zero_page_;

  torch::stable::Tensor tensor_;
  std::unordered_map<page_id_t, std::unique_ptr<Page>> mapping_;
};

} // namespace kvcached
