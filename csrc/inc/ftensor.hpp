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
  // Split of map() into a create half and a VA-edit half, so the two can run
  // on different threads. prepare() does only the physical allocation
  // (cuMemCreate) and stashes the page; it touches no mapped VA and is safe to
  // run concurrently with in-flight kernels. commit() does the VA page-table
  // edit (unmap zero page -> map the prepared page) and must run at a GPU-idle
  // point. map() = prepare() + commit().
  bool prepare(offset_t offset);
  bool commit(offset_t offset);
  bool unmap(offset_t offset);

  inline at::Tensor get_tensor() noexcept { return tensor_; }

private:
  bool map_(Page *page, offset_t offset, bool set_access = true);
  bool set_access_(generic_ptr_t addr, size_t size);
  bool init_with_zero_();

  std::string name_;
  generic_ptr_t vaddr_;
  size_t size_;
  size_t page_size_;
  c10::ScalarType dtype_;
  c10::Device dev_;
  std::shared_ptr<Page> zero_page_;

  at::Tensor tensor_;
  std::unordered_map<page_id_t, std::unique_ptr<Page>> mapping_;
  // Pages created by prepare() but not yet committed (VA still on the zero
  // page). commit() moves an entry from here to mapping_; unmap() discards one
  // from here if the page was never committed.
  std::unordered_map<page_id_t, std::unique_ptr<Page>> prepared_;
};

} // namespace kvcached
