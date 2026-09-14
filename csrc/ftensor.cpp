// SPDX-FileCopyrightText: Copyright contributors to the kvcached project
// SPDX-License-Identifier: Apache-2.0

#include <atomic>
#include <fcntl.h>
#include <sys/mman.h>
#include <vector>

#include <stdexcept>
#include <string>

#include <torch/csrc/stable/ops.h>

#include "constants.hpp"
#include "device_utils.hpp"
#include "ftensor.hpp"
#include "gpu_utils.hpp"
#include "page.hpp"
#include "torch_utils.hpp"
#include "transaction_error.hpp"

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
  throw StateConsistencyError(std::string("state_inconsistency: ") + operation +
                              " failed: " + original_error +
                              "; rollback failed: " + rollback_error);
}

} // namespace

static std::atomic<size_t> g_vaddr_allocated_offset = 0;

static inline int resolve_device_index(const torch::stable::Device &dev) {
  if (dev.index() >= 0) {
    return dev.index();
  }
  return gpu_vmm::current_device();
}

static inline generic_ptr_t alloc_virtual_mem(const torch::stable::Device &dev,
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

static inline std::unique_ptr<Page>
make_unique_page(const torch::stable::Device &dev, page_id_t page_id,
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

FTensor::FTensor(const std::string &name, size_t size,
                 torch::headeronly::ScalarType dtype, torch::stable::Device dev,
                 std::shared_ptr<Page> zero_page, size_t page_size)
    : name_(name), vaddr_(nullptr), size_(size),
      page_size_(page_size > 0 ? page_size : kPageSize), dtype_(dtype),
      dev_(dev), zero_page_(zero_page) {
  vaddr_ = alloc_virtual_mem(dev_, size_);
  init_with_zero_();
  if (!zero_page_backed_ && is_accelerator(dev_)) {
    install_anchor_page_();
  }

  auto num_elems = static_cast<int64_t>(size / element_size(dtype_));
  std::vector<int64_t> sizes = {num_elems};
  std::vector<int64_t> strides = {1};
  tensor_ = torch::stable::from_blob(reinterpret_cast<void *>(vaddr_), sizes,
                                     strides, dev_, dtype_);
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
        if (anchor_mapped_) {
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
  validate_offset_(offset);
  assert(offset % page_size_ == 0); // Ensure alignment.

  page_id_t page_id = offset / page_size_;
  if (is_mapped_(offset)) {
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
      throw_on_gpu_error(gpu_vmm::mem_unmap(vaddr, page_size_),
                         "zero page unmap");
    } else if (offset == 0) {
      release_anchor_page_();
    }
  }

  bool physical_page_mapped = false;
  std::unique_ptr<Page> page;
  try {
    page = make_unique_page(dev_, page_id, page_size_);
    if (!page->map(vaddr)) {
      throw std::runtime_error("physical page map returned false");
    }
    physical_page_mapped = true;
    mapping_.emplace(page_id, std::move(page));
  } catch (const std::exception &error) {
    std::string original_error = error.what();
    if (dynamic_cast<const StateConsistencyError *>(&error)) {
      if (page) {
        failed_pages_.push_back(std::move(page));
      }
      throw;
    }
    if (physical_page_mapped && is_accelerator(dev_)) {
      auto status = gpu_vmm::mem_unmap(vaddr, page_size_);
      if (!gpu_vmm::is_success(status)) {
        if (page) {
          failed_pages_.push_back(std::move(page));
        }
        throw_rollback_error("physical page map", original_error,
                             gpu_vmm::error_string(status));
      }
    }
    try {
      // Leave the virtual page as map() found it: under the zero page where
      // there is one, otherwise unbacked, with the anchor put back if this was
      // virtual page 0.
      if (zero_page_backed_) {
        if (!map_(zero_page_.get(), offset)) {
          throw std::runtime_error("zero page map returned false");
        }
      } else if (is_accelerator(dev_) && offset == 0) {
        install_anchor_page_();
      }
      if (page) {
        page->release();
      }
    } catch (const std::exception &rollback_error) {
      if (page) {
        failed_pages_.push_back(std::move(page));
      }
      throw_rollback_error("physical page map", original_error,
                           rollback_error.what());
    }
    throw std::runtime_error("physical page map failed: " + original_error);
  }
  return true;
}

bool FTensor::unmap(offset_t offset) {
  std::unique_ptr<Page> retained_page;
  if (!unmap_retain_(offset, retained_page)) {
    return false;
  }
  try {
    retained_page->release();
  } catch (...) {
    failed_pages_.push_back(std::move(retained_page));
    throw;
  }
  return true;
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
  if (is_accelerator(dev_)) {
    throw_on_gpu_error(gpu_vmm::mem_unmap(vaddr, page_size_),
                       "physical page unmap");
  }

  // Map the zero page instead to ensure memory integrity. Skipped where the
  // backend has no shared-page support: the virtual page simply goes back to
  // being unbacked, so there is no zero-page failure to roll back from.
  if (zero_page_backed_) {
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
  } else if (is_accelerator(dev_) && offset == 0) {
    // Virtual page 0 must stay backed for from_blob()'s device lookup.
    // Rolled back like the zero page above: leaving this function with the
    // range unmapped but still in mapping_ would make a later unmap retry a
    // second unmap of an unbacked range, which faults inside the driver.
    try {
      if (!install_anchor_page_()) {
        throw std::runtime_error("anchor page map returned false");
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
  if (is_accelerator(dev_)) {
    if (zero_page_backed_) {
      throw_on_gpu_error(gpu_vmm::mem_unmap(vaddr, page_size_),
                         "rollback zero page unmap");
    } else if (offset == 0) {
      release_anchor_page_();
    }
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
    if (physical_page_mapped && is_accelerator(dev_)) {
      auto status = gpu_vmm::mem_unmap(vaddr, page_size_);
      if (!gpu_vmm::is_success(status)) {
        throw_rollback_error("physical page restore", original_error,
                             gpu_vmm::error_string(status));
      }
    }
    try {
      if (zero_page_backed_) {
        if (!map_(zero_page_.get(), offset)) {
          throw std::runtime_error("zero page restore returned false");
        }
      } else if (is_accelerator(dev_) && offset == 0) {
        install_anchor_page_();
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
  if (!is_accelerator(dev_)) {
    return true;
  }
  auto access_desc =
      gpu_vmm::make_device_rw_access_desc(resolve_device_index(dev_));
  CHECK_GPU(gpu_vmm::set_access(addr, size, &access_desc, 1));
  return true;
}

// from_blob() resolves an XPU tensor's device by asking the SYCL runtime
// what vaddr_ points at, and a reservation with nothing mapped into it answers
// "unknown" -- from_blob then throws "ptr is not a device type pointer". Only
// the base address is inspected, so backing the first virtual page with a page
// of its own is enough to make the reservation recognizable. The page is handed
// straight over to the allocator if it ever asks for offset 0, so it costs
// nothing once the KV cache is in use.
// The physical page is allocated at most once per FTensor and then parked
// rather than freed, because unmap_retain_() has to put the anchor back while
// the page it is releasing is still owned: allocating there would mean freeing
// virtual page 0 needs one MORE physical page than are already in use, so a
// release would fail on a full device -- exactly when the caller is freeing
// memory to make room. Mapping a page that is already held cannot fail for lack
// of memory. The cost is one parked page per FTensor while the allocator owns
// virtual page 0; it is the page the anchor would hold anyway the moment that
// page is freed.
bool FTensor::install_anchor_page_() {
  assert(!anchor_mapped_);
  if (!anchor_page_) {
    anchor_page_ = make_unique_page(dev_, ANCHOR_PAGE_ID, page_size_);
  }
  // anchor_mapped_ is set only once the mapping is in place: a half-installed
  // anchor would make ~FTensor unmap virtual page 0 twice, once for the anchor
  // and once for whatever mapping_ holds there.
  if (!anchor_page_->map(vaddr_)) {
    return false;
  }
  anchor_mapped_ = true;
  return true;
}

bool FTensor::release_anchor_page_() {
  if (!anchor_mapped_) {
    return true;
  }
  throw_on_gpu_error(gpu_vmm::mem_unmap(vaddr_, page_size_),
                     "anchor page unmap");
  // anchor_page_ deliberately kept: see install_anchor_page_().
  anchor_mapped_ = false;
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
