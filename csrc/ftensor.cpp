// SPDX-FileCopyrightText: Copyright contributors to the kvcached project
// SPDX-License-Identifier: Apache-2.0

#include <fcntl.h>
#include <sys/mman.h>

#include <ATen/ops/from_blob.h>
#include <c10/core/ScalarType.h>

#include "constants.hpp"
#include "ftensor.hpp"
#include "gpu_utils.hpp"
#include "page.hpp"

namespace kvcached {

static std::atomic<size_t> g_vaddr_allocated_offset = 0;

static inline int resolve_device_index(const c10::Device &dev) {
  if (dev.index() >= 0) {
    return dev.index();
  }
  return gpu_vmm::current_device();
}

static inline generic_ptr_t alloc_virtual_mem(const c10::Device &dev,
                                              size_t size) {
  size_t alignment_2mb = 2 * 1024 * 1024;
  ASSERT(size % alignment_2mb == 0,
         "alloc size not aligned."); // Ensure alignment.

  generic_ptr_t vaddr;
  size_t offset = g_vaddr_allocated_offset.fetch_add(size);
  // is_cuda() returns true for both NVIDIA (CUDA) and AMD (HIP/ROCm) devices,
  // because PyTorch's ROCm build masquerades HIP devices as CUDA.
  if (dev.is_cuda()) {
    CHECK_GPU(gpu_vmm::address_reserve(
        reinterpret_cast<void **>(&vaddr), size, alignment_2mb,
        reinterpret_cast<void *>(kStartAddr + offset)));
  } else {
    vaddr = mmap(reinterpret_cast<void *>(kStartAddr + offset), size,
                 PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    ASSERT(vaddr != MAP_FAILED, "mmap failed.");
  }
  // LOGE("Allocated virtual memory at %p", vaddr);
  return vaddr;
}

static inline std::unique_ptr<Page> make_unique_page(const c10::Device &dev,
                                                     page_id_t page_id,
                                                     size_t page_size = 0) {
  if (dev.is_cuda()) {
    return std::make_unique<GPUPage>(page_id, resolve_device_index(dev),
                                     page_size);
  } else if (dev.is_cpu()) {
    return std::make_unique<CPUPage>(page_id, page_size);
  }
  ASSERT(false, "Unsupported device type.");
  return nullptr;
}

FTensor::FTensor(const std::string &name, size_t size, c10::ScalarType dtype,
                 c10::Device dev, std::shared_ptr<Page> zero_page,
                 size_t page_size)
    : name_(name), vaddr_(nullptr), size_(size),
      page_size_(page_size > 0 ? page_size : kPageSize), dtype_(dtype),
      dev_(dev), zero_page_(zero_page) {
  vaddr_ = alloc_virtual_mem(dev_, size_);
  init_with_zero_();

  auto num_elems = static_cast<int64_t>(size / c10::elementSize(dtype_));
  auto options =
      at::TensorOptions().dtype(dtype_).device(dev_).requires_grad(false);
  tensor_ =
      at::from_blob(reinterpret_cast<void *>(vaddr_), {num_elems}, options);
}

FTensor::~FTensor() {
  mapping_.clear(); // Free all physical pages directly.
  zero_page_.reset();
  if (vaddr_) {
    if (dev_.is_cuda()) {
      CHECK_GPU(gpu_vmm::mem_unmap(vaddr_, size_));
      CHECK_GPU(gpu_vmm::address_free(vaddr_, size_));
    } else if (dev_.is_cpu()) {
      ASSERT(munmap(vaddr_, size_) == 0, "munmap failed.");
    }
  }
}

bool FTensor::map(offset_t offset) {
  assert(offset % page_size_ == 0); // Ensure alignment.

  page_id_t page_id = offset / page_size_;
  if (mapping_.find(page_id) != mapping_.end()) {
    LOGE("Page %ld is already mapped.", page_id);
    return false;
  }

  auto vaddr = reinterpret_cast<generic_ptr_t>(
      reinterpret_cast<uintptr_t>(vaddr_) + offset);
  if (dev_.is_cuda()) {
    CHECK_GPU(gpu_vmm::mem_unmap(vaddr, page_size_));
  }

  mapping_[page_id] = make_unique_page(dev_, page_id, page_size_);
  mapping_[page_id]->map(vaddr);
  return true;
}

bool FTensor::unmap(offset_t offset) {
  assert(offset % page_size_ == 0); // Ensure alignment.

  page_id_t page_id = offset / page_size_;
  if (mapping_.find(page_id) == mapping_.end()) {
    LOGE("Page %ld is not mapped.", page_id);
    return false;
  }

  auto vaddr = reinterpret_cast<generic_ptr_t>(
      reinterpret_cast<uintptr_t>(vaddr_) + offset);
  if (dev_.is_cuda()) {
    CHECK_GPU(gpu_vmm::mem_unmap(vaddr, page_size_));
  }

  // Map the zero page instead to ensure memory integrity.
  map_(zero_page_.get(), offset);

  mapping_.erase(page_id);
  return true;
}

bool FTensor::map_(Page *page, offset_t offset, bool set_access) {
  assert(offset % page_size_ == 0); // Ensure alignment.
  assert(page);
  auto vaddr =
      reinterpret_cast<void *>(reinterpret_cast<uintptr_t>(vaddr_) + offset);
  return page->map(vaddr, set_access);
}

bool FTensor::set_access_(generic_ptr_t addr, size_t size) {
  if (!dev_.is_cuda()) {
    return true;
  }
  auto access_desc =
      gpu_vmm::make_device_rw_access_desc(resolve_device_index(dev_));
  CHECK_GPU(gpu_vmm::set_access(addr, size, &access_desc, 1));
  return true;
}

bool FTensor::init_with_zero_() {
  assert(reinterpret_cast<uintptr_t>(vaddr_) % page_size_ ==
         0);                       // Ensure alignment.
  assert(size_ % page_size_ == 0); // Ensure alignment.

  bool succ = true;
  for (size_t offset = 0; offset < size_; offset += page_size_) {
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
