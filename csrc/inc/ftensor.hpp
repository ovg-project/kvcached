// SPDX-FileCopyrightText: Copyright contributors to the kvcached project
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <memory>
#include <unordered_map>

#include <ATen/core/Tensor.h>
#include <c10/core/Device.h>
#include <c10/core/ScalarType.h>

#include "constants.hpp"
#include "page.hpp"

namespace kvcached {

/* NOTE: FTensorAllocator is thread-safe but FTensor is not. */
class FTensor {
public:
  FTensor(const std::string &name, size_t size, c10::ScalarType dtype,
          c10::Device dev, std::shared_ptr<Page> zero_page,
          size_t page_size = 0);
  ~FTensor();
  bool map(offset_t offset);
  bool unmap(offset_t offset);

  inline at::Tensor get_tensor() noexcept { return tensor_; }

private:
  bool map_(Page *page, offset_t offset, bool set_access = true);
  bool set_access_(generic_ptr_t addr, size_t size);
  bool init_with_zero_();
  bool install_anchor_page_();
  bool release_anchor_page_();

  std::string name_;
  generic_ptr_t vaddr_;
  size_t size_;
  size_t page_size_;
  c10::ScalarType dtype_;
  c10::Device dev_;
  std::shared_ptr<Page> zero_page_;
  // Set by init_with_zero_() when the shared zero page was actually installed
  // across the reservation. False on backends that cannot map one physical page
  // into several virtual ranges (see gpu_vmm::supports_shared_page_mapping),
  // where unmapped virtual pages have no backing at all and must not be read.
  bool zero_page_backed_ = false;
  // Backs virtual page 0 whenever zero_page_backed_ is false and the allocator
  // has not claimed that page, so the reservation stays a recognizable device
  // pointer for at::from_blob(). See install_anchor_page_().
  std::unique_ptr<Page> anchor_page_;

  at::Tensor tensor_;
  std::unordered_map<page_id_t, std::unique_ptr<Page>> mapping_;
};

} // namespace kvcached
