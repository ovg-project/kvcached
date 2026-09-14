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

GPUPage::~GPUPage() {
  if constexpr (gpu_vmm::release_failure_is_fatal()) {
    CHECK_GPU(gpu_vmm::mem_release(handle_));
  } else {
    // Log and carry on rather than abort: on backends where a release can fail
    // during teardown, aborting from a destructor turns a page leaked at
    // process exit into a crash and skips the rest of teardown. See
    // gpu_vmm::release_failure_is_fatal().
    auto res = gpu_vmm::mem_release(handle_);
    if (!gpu_vmm::is_success(res)) {
      LOGE("mem_release during GPUPage cleanup failed: %s",
           gpu_vmm::error_string(res));
    }
  }
}

bool GPUPage::map(generic_ptr_t vaddr, bool set_access) {
  auto access_desc = gpu_vmm::make_device_rw_access_desc(dev_idx_);
  CHECK_GPU(gpu_vmm::mem_map(vaddr, page_size_, 0, handle_, set_access));
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
