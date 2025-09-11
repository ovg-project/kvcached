// SPDX-FileCopyrightText: Copyright contributors to the kvcached project
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <memory>
#include <unordered_map>

#include <torch/extension.h>

#include "constants.hpp"
#include "page.hpp"

namespace kvcached {

/* NOTE: FTensorAllocator is thread-safe but FTensor is not. */
class FTensor {
public:
  FTensor(const std::string &name, size_t size, torch::Dtype dtype,
          torch::Device dev, std::shared_ptr<Page> zero_page);
  ~FTensor();
  bool map(offset_t offset);
  bool unmap(offset_t offset);

  inline torch::Tensor get_tensor() noexcept { return tensor_; }

private:
  bool map_(Page *page, offset_t offset, bool set_access = true);
  bool set_access_(generic_ptr_t addr, size_t size);
  bool init_with_zero_();

  std::string name_;
  generic_ptr_t vaddr_;
  size_t size_;
  torch::Dtype dtype_;
  torch::Device dev_;
  std::shared_ptr<Page> zero_page_;

  torch::Tensor tensor_;
  std::unordered_map<page_id_t, std::unique_ptr<Page>> mapping_;
};

} // namespace kvcached
