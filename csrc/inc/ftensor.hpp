// SPDX-FileCopyrightText: Copyright contributors to the kvcached project
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "constants.hpp"
#include "page.hpp"
#include "torch_utils.hpp"

namespace kvcached {

/* NOTE: FTensorAllocator is thread-safe but FTensor is not. */
class KVCACHED_HIDDEN FTensor {
public:
  FTensor(const std::string &name, size_t size,
          torch::headeronly::ScalarType dtype, torch::stable::Device dev,
          std::shared_ptr<Page> zero_page, size_t page_size = 0);
  ~FTensor();
  bool map(offset_t offset);
  bool unmap(offset_t offset);

  inline torch::stable::Tensor get_tensor() noexcept { return tensor_; }

private:
  friend class FTensorAllocator;

  bool is_mapped_(offset_t offset) const;
  bool unmap_retain_(offset_t offset, std::unique_ptr<Page> &retained_page);
  bool restore_mapping_(offset_t offset, std::unique_ptr<Page> &retained_page);
  bool map_(Page *page, offset_t offset, bool set_access = true);
  void validate_offset_(offset_t offset) const;
  bool set_access_(generic_ptr_t addr, size_t size);
  bool init_with_zero_();
  bool install_anchor_page_();
  bool release_anchor_page_();

  std::string name_;
  generic_ptr_t vaddr_;
  size_t size_;
  size_t page_size_;
  torch::headeronly::ScalarType dtype_;
  torch::stable::Device dev_;
  std::shared_ptr<Page> zero_page_;
  // Set by init_with_zero_() when the shared zero page was actually installed
  // across the reservation. False on backends that cannot map one physical page
  // into several virtual ranges (see gpu_vmm::supports_shared_page_mapping),
  // where unmapped virtual pages have no backing at all and must not be read.
  bool zero_page_backed_ = false;
  // Backs virtual page 0 whenever zero_page_backed_ is false and the allocator
  // has not claimed that page, so the reservation stays a recognizable device
  // pointer for from_blob(). See install_anchor_page_().
  //
  // The physical page is allocated once and held for the lifetime of the
  // FTensor, even while the allocator owns virtual page 0 and the anchor is
  // unmapped, so anchor_mapped_ rather than anchor_page_ says whether anything
  // is mapped at vaddr_.
  std::unique_ptr<Page> anchor_page_;
  bool anchor_mapped_ = false;

  torch::stable::Tensor tensor_;
  std::unordered_map<page_id_t, std::unique_ptr<Page>> mapping_;
  std::vector<std::unique_ptr<Page>> failed_pages_;
};

} // namespace kvcached
