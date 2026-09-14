// SPDX-FileCopyrightText: Copyright contributors to the kvcached project
// SPDX-License-Identifier: Apache-2.0

#include <fcntl.h>
#include <sys/mman.h>

#include <ATen/ops/from_blob.h>
#include <c10/core/ScalarType.h>

#include "constants.hpp"
#include "device_utils.hpp"
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
  if (is_accelerator(dev)) {
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
  if (is_accelerator(dev)) {
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
  if (!zero_page_backed_ && is_accelerator(dev_)) {
    install_anchor_page_();
  }

  auto num_elems = static_cast<int64_t>(size / c10::elementSize(dtype_));
  auto options =
      at::TensorOptions().dtype(dtype_).device(dev_).requires_grad(false);
  tensor_ =
      at::from_blob(reinterpret_cast<void *>(vaddr_), {num_elems}, options);
}

FTensor::~FTensor() {
  if (vaddr_) {
    if (is_accelerator(dev_)) {
      // Tolerate stale VMM mappings during teardown: log, do not abort.
      auto unmap_range = [](generic_ptr_t addr, size_t len) {
        auto res = gpu_vmm::mem_unmap(addr, len);
        if (!gpu_vmm::is_success(res)) {
          LOGGER(ERROR, "mem_unmap during FTensor cleanup failed: %s",
                 gpu_vmm::error_string(res));
        }
      };

      if (zero_page_backed_) {
        // Every virtual page is backed, so one range covers the reservation.
        unmap_range(vaddr_, size_);
      } else {
        // Only the pages in mapping_ (plus the anchor) are backed, and a
        // reservation-wide unmap spanning unbacked virtual pages faults inside
        // the Level Zero driver. Release the live mappings one at a time.
        if (anchor_page_) {
          unmap_range(vaddr_, page_size_);
        }
        for (const auto &entry : mapping_) {
          unmap_range(reinterpret_cast<generic_ptr_t>(
                          reinterpret_cast<uintptr_t>(vaddr_) +
                          entry.first * page_size_),
                      page_size_);
        }
      }

      auto res = gpu_vmm::address_free(vaddr_, size_);
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
  assert(offset % page_size_ == 0); // Ensure alignment.

  page_id_t page_id = offset / page_size_;
  if (mapping_.find(page_id) != mapping_.end()) {
    LOGGER(ERROR, "Page %ld is already mapped.", page_id);
    return false;
  }

  auto vaddr = reinterpret_cast<generic_ptr_t>(
      reinterpret_cast<uintptr_t>(vaddr_) + offset);
  // Only evict the zero page if one is actually mapped here; without the safety
  // net the virtual page is already free and unmapping it is an error. The lone
  // exception is the anchor page, which does occupy virtual page 0.
  if (is_accelerator(dev_)) {
    if (zero_page_backed_) {
      CHECK_GPU(gpu_vmm::mem_unmap(vaddr, page_size_));
    } else if (offset == 0) {
      release_anchor_page_();
    }
  }

  mapping_[page_id] = make_unique_page(dev_, page_id, page_size_);
  mapping_[page_id]->map(vaddr);
  return true;
}

bool FTensor::unmap(offset_t offset) {
  assert(offset % page_size_ == 0); // Ensure alignment.

  page_id_t page_id = offset / page_size_;
  if (mapping_.find(page_id) == mapping_.end()) {
    LOGGER(ERROR, "Page %ld is not mapped.", page_id);
    return false;
  }

  auto vaddr = reinterpret_cast<generic_ptr_t>(
      reinterpret_cast<uintptr_t>(vaddr_) + offset);
  if (is_accelerator(dev_)) {
    CHECK_GPU(gpu_vmm::mem_unmap(vaddr, page_size_));
  }

  // Map the zero page instead to ensure memory integrity. Skipped where the
  // backend has no shared-page support: the virtual page simply goes back to
  // being unbacked.
  if (zero_page_backed_) {
    map_(zero_page_.get(), offset);
  } else if (is_accelerator(dev_) && offset == 0) {
    // Virtual page 0 must stay backed for at::from_blob()'s device lookup.
    install_anchor_page_();
  }

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
  if (!is_accelerator(dev_)) {
    return true;
  }
  auto access_desc =
      gpu_vmm::make_device_rw_access_desc(resolve_device_index(dev_));
  CHECK_GPU(gpu_vmm::set_access(addr, size, &access_desc, 1));
  return true;
}

// at::from_blob() resolves an XPU tensor's device by asking the SYCL runtime
// what vaddr_ points at, and a reservation with nothing mapped into it answers
// "unknown" -- from_blob then throws "ptr is not a device type pointer". Only
// the base address is inspected, so backing the first virtual page with a page
// of its own is enough to make the reservation recognizable. The page is handed
// straight over to the allocator if it ever asks for offset 0, so it costs
// nothing once the KV cache is in use.
bool FTensor::install_anchor_page_() {
  assert(!anchor_page_);
  anchor_page_ = make_unique_page(dev_, ANCHOR_PAGE_ID, page_size_);
  return anchor_page_->map(vaddr_);
}

bool FTensor::release_anchor_page_() {
  if (!anchor_page_) {
    return true;
  }
  CHECK_GPU(gpu_vmm::mem_unmap(vaddr_, page_size_));
  anchor_page_.reset();
  return true;
}

bool FTensor::init_with_zero_() {
  assert(reinterpret_cast<uintptr_t>(vaddr_) % page_size_ ==
         0);                       // Ensure alignment.
  assert(size_ % page_size_ == 0); // Ensure alignment.

  // Backends that cannot map one physical page into several virtual ranges get
  // no zero-page safety net: the mapping calls would succeed and then fault the
  // device on first access. Leave the reservation unbacked instead, so a stray
  // read of a never-allocated region is a page fault rather than silent
  // corruption or a lost context. zero_page_ is null here -- the allocator does
  // not create one it cannot use.
  if (!uses_zero_page(dev_)) {
    zero_page_backed_ = false;
    return true;
  }

  // Set before the loop, not from its result: map_() routes through CHECK_GPU,
  // which aborts on failure, so on CUDA and HIP this is always true and every
  // branch below keeps the behavior those backends had before the XPU arm
  // existed.
  zero_page_backed_ = true;

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
