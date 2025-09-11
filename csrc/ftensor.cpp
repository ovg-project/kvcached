// SPDX-FileCopyrightText: Copyright contributors to the kvcached project
// SPDX-License-Identifier: Apache-2.0

#include <fcntl.h>
#include <sys/mman.h>

#include "constants.hpp"
#include "cuda_utils.hpp"
#include "ftensor.hpp"
#include "page.hpp"

namespace kvcached {

static std::atomic<size_t> g_vaddr_allocated_offset = 0;

static inline generic_ptr_t alloc_virtual_mem(const torch::Device &dev,
                                              size_t size) {
  size_t alignment_2mb = 2 * 1024 * 1024;
  ASSERT(size % alignment_2mb == 0,
         "alloc size not aligned."); // Ensure alignment.

  generic_ptr_t vaddr;
  size_t offset = g_vaddr_allocated_offset.fetch_add(size);
  if (dev.is_cuda()) {
    CHECK_DRV(cuMemAddressReserve(reinterpret_cast<CUdeviceptr *>(&vaddr), size,
                                  alignment_2mb, kStartAddr + offset, 0ULL));
  } else {
    vaddr = mmap(reinterpret_cast<void *>(kStartAddr + offset), size,
                 PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    ASSERT(vaddr != MAP_FAILED, "mmap failed.");
  }
  // LOGE("Allocated virtual memory at %p", vaddr);
  return vaddr;
}

static inline std::unique_ptr<Page> make_unique_page(const torch::Device &dev,
                                                     page_id_t page_id) {
  if (dev.is_cuda()) {
    return std::make_unique<GPUPage>(page_id, dev.index());
  } else if (dev.is_cpu()) {
    return std::make_unique<CPUPage>(page_id);
  }
  ASSERT(false, "Unsupported device type.");
  return nullptr;
}

FTensor::FTensor(const std::string &name, size_t size, torch::Dtype dtype,
                 torch::Device dev, std::shared_ptr<Page> zero_page)
    : name_(name), vaddr_(nullptr), size_(size), dtype_(dtype), dev_(dev),
      zero_page_(zero_page) {
  vaddr_ = alloc_virtual_mem(dev_, size_);
  init_with_zero_();

  auto num_elems = static_cast<int64_t>(size / torch::elementSize(dtype_));
  auto options =
      torch::TensorOptions().dtype(dtype_).device(dev_).requires_grad(false);
  tensor_ =
      torch::from_blob(reinterpret_cast<void *>(vaddr_), {num_elems}, options);
}

FTensor::~FTensor() {
  mapping_.clear(); // Free all physical pages directly.
  zero_page_.reset();
  if (vaddr_) {
    CHECK_DRV(cuMemUnmap(reinterpret_cast<CUdeviceptr>(vaddr_), size_));
    CHECK_DRV(cuMemAddressFree(reinterpret_cast<CUdeviceptr>(vaddr_), size_));
  }
}

bool FTensor::map(offset_t offset) {
  assert(offset % kPageSize == 0); // Ensure alignment.

  page_id_t page_id = offset / kPageSize;
  if (mapping_.find(page_id) != mapping_.end()) {
    LOGE("Page %ld is already mapped.", page_id);
    return false;
  }

  auto vaddr = reinterpret_cast<generic_ptr_t>(
      reinterpret_cast<uintptr_t>(vaddr_) + offset);
  CHECK_DRV(cuMemUnmap(reinterpret_cast<CUdeviceptr>(vaddr), kPageSize));

  mapping_[page_id] = make_unique_page(dev_, page_id);
  mapping_[page_id]->map(vaddr);
  return true;
}

bool FTensor::unmap(offset_t offset) {
  assert(offset % kPageSize == 0); // Ensure alignment.

  page_id_t page_id = offset / kPageSize;
  if (mapping_.find(page_id) == mapping_.end()) {
    LOGE("Page %ld is not mapped.", page_id);
    return false;
  }

  auto vaddr = reinterpret_cast<generic_ptr_t>(
      reinterpret_cast<uintptr_t>(vaddr_) + offset);
  CHECK_DRV(cuMemUnmap(reinterpret_cast<CUdeviceptr>(vaddr), kPageSize));

  // Map the zero page instead to ensure memory integrity.
  map_(zero_page_.get(), offset);

  mapping_.erase(page_id);
  return true;
}

bool FTensor::map_(Page *page, offset_t offset, bool set_access) {
  assert(offset % kPageSize == 0); // Ensure alignment.
  assert(page);
  auto vaddr =
      reinterpret_cast<void *>(reinterpret_cast<uintptr_t>(vaddr_) + offset);
  return page->map(vaddr, set_access);
}

bool FTensor::set_access_(generic_ptr_t addr, size_t size) {
  CUmemAccessDesc accessDesc_{
      .location =
          {
              .type = CU_MEM_LOCATION_TYPE_DEVICE,
              .id = dev_.index(),
          },
      .flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE,
  };
  CHECK_DRV(cuMemSetAccess(reinterpret_cast<CUdeviceptr>(addr), size,
                           &accessDesc_, 1));
  return true;
}

bool FTensor::init_with_zero_() {
  assert(vaddr_ % kPageSize == 0); // Ensure alignment.
  assert(size_ % kPageSize == 0);  // Ensure alignment.

  bool succ = true;
  for (size_t offset = 0; offset < size_; offset += kPageSize) {
    if (!map_(zero_page_.get(), offset, /* set_access = */ true)) {
      succ = false;
      break;
    }
  }
  // if (succ)
  //   set_access_(vaddr_, size_);

  return succ;
}

} // namespace kvcached
