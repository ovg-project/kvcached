// SPDX-FileCopyrightText: Copyright contributors to the kvcached project
// SPDX-License-Identifier: Apache-2.0

#include "page.hpp"
#include "constants.hpp"
#include "gpu_utils.hpp"

namespace kvcached {

GPUPage::GPUPage(page_id_t page_id, int dev_idx, size_t page_size)
    : page_id_(page_id), dev_idx_(dev_idx),
      page_size_(page_size > 0 ? page_size : kPageSize), handle_() {
  auto prop = gpu_vmm::make_pinned_device_allocation_prop(dev_idx_);
  CHECK_GPU(gpu_vmm::mem_create(&handle_, page_size_, &prop));
}

GPUPage::~GPUPage() { CHECK_GPU(gpu_vmm::mem_release(handle_)); }

bool GPUPage::map(generic_ptr_t vaddr, bool set_access) {
  auto access_desc = gpu_vmm::make_device_rw_access_desc(dev_idx_);
  CHECK_GPU(gpu_vmm::mem_map(vaddr, page_size_, 0, handle_));
  if (set_access)
    CHECK_GPU(gpu_vmm::set_access(vaddr, page_size_, &access_desc, 1));
  return true;
}

// TODO: finish CPUPage impl.
CPUPage::CPUPage(page_id_t page_id, size_t page_size)
    : page_id_(page_id), page_size_(page_size > 0 ? page_size : kPageSize),
      mapped_addr_(nullptr) {}

CPUPage::~CPUPage() {}

bool CPUPage::map(void *vaddr, bool set_access) {
  mapped_addr_ = vaddr;
  return true;
}

} // namespace kvcached
