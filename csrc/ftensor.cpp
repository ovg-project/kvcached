// SPDX-FileCopyrightText: Copyright contributors to the kvcached project
// SPDX-License-Identifier: Apache-2.0

#include <fcntl.h>
#include <sys/mman.h>

#include <stdexcept>
#include <string>

#include <ATen/ops/from_blob.h>
#include <c10/core/ScalarType.h>

#include "constants.hpp"
#include "ftensor.hpp"
#include "gpu_utils.hpp"
#include "page.hpp"

namespace kvcached {

namespace {

template <typename Status>
void throw_on_gpu_error(Status status, const char *operation) {
  if (!gpu_vmm::is_success(status)) {
    throw std::runtime_error(std::string(operation) + " failed in " +
                             gpu_vmm::backend_name() + ": " +
                             gpu_vmm::error_string(status));
  }
}

[[noreturn]] void throw_rollback_error(const char *operation,
                                       const std::string &original_error,
                                       const std::string &rollback_error) {
  throw std::runtime_error(std::string("state_inconsistency: ") + operation +
                           " failed: " + original_error +
                           "; rollback failed: " + rollback_error);
}

} // namespace

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
  if (vaddr_) {
    if (dev_.is_cuda()) {
      // Tolerate stale VMM mappings during teardown: log, do not abort.
      auto res = gpu_vmm::mem_unmap(vaddr_, size_);
      if (!gpu_vmm::is_success(res)) {
        LOGGER(ERROR, "mem_unmap during FTensor cleanup failed: %s",
               gpu_vmm::error_string(res));
      }
      res = gpu_vmm::address_free(vaddr_, size_);
      if (!gpu_vmm::is_success(res)) {
        LOGGER(ERROR, "address_free during FTensor cleanup failed: %s",
               gpu_vmm::error_string(res));
      }
    } else if (dev_.is_cpu()) {
      ASSERT(munmap(vaddr_, size_) == 0, "munmap failed.");
    }
  }
  mapping_.clear(); // Free physical page handles after their mappings are gone.
  zero_page_.reset();
}

bool FTensor::map(offset_t offset) {
  validate_offset_(offset);
  assert(offset % page_size_ == 0); // Ensure alignment.

  page_id_t page_id = offset / page_size_;
  if (is_mapped_(offset)) {
    LOGGER(ERROR, "Page %ld is already mapped.", page_id);
    return false;
  }

  auto vaddr = reinterpret_cast<generic_ptr_t>(
      reinterpret_cast<uintptr_t>(vaddr_) + offset);
  if (dev_.is_cuda()) {
    throw_on_gpu_error(gpu_vmm::mem_unmap(vaddr, page_size_),
                       "zero page unmap");
  }

  bool physical_page_mapped = false;
  try {
    auto page = make_unique_page(dev_, page_id, page_size_);
    if (!page->map(vaddr)) {
      throw std::runtime_error("physical page map returned false");
    }
    physical_page_mapped = true;
    mapping_.emplace(page_id, std::move(page));
  } catch (const std::exception &error) {
    std::string original_error = error.what();
    if (physical_page_mapped && dev_.is_cuda()) {
      auto status = gpu_vmm::mem_unmap(vaddr, page_size_);
      if (!gpu_vmm::is_success(status)) {
        throw_rollback_error("physical page map", original_error,
                             gpu_vmm::error_string(status));
      }
    }
    try {
      if (!map_(zero_page_.get(), offset)) {
        throw std::runtime_error("zero page map returned false");
      }
    } catch (const std::exception &rollback_error) {
      throw_rollback_error("physical page map", original_error,
                           rollback_error.what());
    }
    throw std::runtime_error("physical page map failed: " + original_error);
  }
  return true;
}

bool FTensor::unmap(offset_t offset) {
  std::unique_ptr<Page> retained_page;
  return unmap_retain_(offset, retained_page);
}

bool FTensor::is_mapped_(offset_t offset) const {
  validate_offset_(offset);
  assert(offset % page_size_ == 0); // Ensure alignment.
  return mapping_.find(offset / page_size_) != mapping_.end();
}

bool FTensor::unmap_retain_(offset_t offset,
                            std::unique_ptr<Page> &retained_page) {
  validate_offset_(offset);
  assert(offset % page_size_ == 0); // Ensure alignment.
  retained_page.reset();

  page_id_t page_id = offset / page_size_;
  auto mapping = mapping_.find(page_id);
  if (mapping == mapping_.end()) {
    LOGGER(ERROR, "Page %ld is not mapped.", page_id);
    return false;
  }

  auto vaddr = reinterpret_cast<generic_ptr_t>(
      reinterpret_cast<uintptr_t>(vaddr_) + offset);
  if (dev_.is_cuda()) {
    throw_on_gpu_error(gpu_vmm::mem_unmap(vaddr, page_size_),
                       "physical page unmap");
  }

  try {
    if (!map_(zero_page_.get(), offset)) {
      throw std::runtime_error("zero page map returned false");
    }
  } catch (const std::exception &error) {
    std::string original_error = error.what();
    try {
      if (!mapping->second->map(vaddr)) {
        throw std::runtime_error("physical page restore returned false");
      }
    } catch (const std::exception &rollback_error) {
      throw_rollback_error("physical page unmap", original_error,
                           rollback_error.what());
    }
    throw std::runtime_error("physical page unmap failed: " + original_error);
  }

  retained_page = std::move(mapping->second);
  mapping_.erase(mapping);
  return true;
}

bool FTensor::restore_mapping_(offset_t offset,
                               std::unique_ptr<Page> &retained_page) {
  validate_offset_(offset);
  assert(offset % page_size_ == 0); // Ensure alignment.
  if (!retained_page) {
    return true;
  }

  page_id_t page_id = offset / page_size_;
  if (mapping_.find(page_id) != mapping_.end()) {
    throw std::runtime_error(
        "state_inconsistency: cannot restore an already-mapped page");
  }

  auto vaddr = reinterpret_cast<generic_ptr_t>(
      reinterpret_cast<uintptr_t>(vaddr_) + offset);
  if (dev_.is_cuda()) {
    throw_on_gpu_error(gpu_vmm::mem_unmap(vaddr, page_size_),
                       "rollback zero page unmap");
  }

  bool physical_page_mapped = false;
  try {
    if (!retained_page->map(vaddr)) {
      throw std::runtime_error("physical page restore returned false");
    }
    physical_page_mapped = true;
    mapping_.emplace(page_id, std::move(retained_page));
  } catch (const std::exception &error) {
    std::string original_error = error.what();
    if (physical_page_mapped && dev_.is_cuda()) {
      auto status = gpu_vmm::mem_unmap(vaddr, page_size_);
      if (!gpu_vmm::is_success(status)) {
        throw_rollback_error("physical page restore", original_error,
                             gpu_vmm::error_string(status));
      }
    }
    try {
      if (!map_(zero_page_.get(), offset)) {
        throw std::runtime_error("zero page restore returned false");
      }
    } catch (const std::exception &rollback_error) {
      throw_rollback_error("physical page restore", original_error,
                           rollback_error.what());
    }
    throw std::runtime_error("physical page restore failed: " + original_error);
  }
  return true;
}

bool FTensor::map_(Page *page, offset_t offset, bool set_access) {
  validate_offset_(offset);
  assert(offset % page_size_ == 0); // Ensure alignment.
  assert(page);
  auto vaddr =
      reinterpret_cast<void *>(reinterpret_cast<uintptr_t>(vaddr_) + offset);
  return page->map(vaddr, set_access);
}

void FTensor::validate_offset_(offset_t offset) const {
  if (offset < 0 || static_cast<size_t>(offset) >= size_ ||
      page_size_ > size_ - static_cast<size_t>(offset)) {
    throw std::runtime_error(
        "KV tensor page offset is outside the reserved virtual address range");
  }
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
