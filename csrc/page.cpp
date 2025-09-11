// SPDX-FileCopyrightText: Copyright contributors to the kvcached project
// SPDX-License-Identifier: Apache-2.0

#include <cuda_runtime.h>

#include "constants.hpp"
#include "cuda_utils.hpp"
#include "page.hpp"

namespace kvcached {

GPUPage::GPUPage(page_id_t page_id, int dev_idx)
    : page_id_(page_id), dev_(dev_idx), handle_(0) {
  // CHECK_DRV(cuCtxGetDevice(&dev_));

  CUmemAllocationProp prop = {
      .type = CU_MEM_ALLOCATION_TYPE_PINNED,
      .location =
          {
              .type = CU_MEM_LOCATION_TYPE_DEVICE,
              .id = dev_,
          },
  };
  CHECK_DRV(cuMemCreate(&handle_, kPageSize, &prop, 0));
}

GPUPage::~GPUPage() { CHECK_DRV(cuMemRelease(handle_)); }

bool GPUPage::map(generic_ptr_t vaddr, bool set_access) {
  CUmemAccessDesc accessDesc_{
      .location =
          {
              .type = CU_MEM_LOCATION_TYPE_DEVICE,
              .id = dev_,
          },
      .flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE,
  };
  CHECK_DRV(
      cuMemMap(reinterpret_cast<CUdeviceptr>(vaddr), kPageSize, 0, handle_, 0));
  if (set_access)
    CHECK_DRV(cuMemSetAccess(reinterpret_cast<CUdeviceptr>(vaddr), kPageSize,
                             &accessDesc_, 1));
  return true;
}

// TODO: finish CPUPage impl.
CPUPage::CPUPage(page_id_t page_id)
    : page_id_(page_id), mapped_addr_(nullptr) {}

CPUPage::~CPUPage() {}

bool CPUPage::map(void *vaddr, bool set_access) {
  mapped_addr_ = vaddr;
  return true;
}

} // namespace kvcached
