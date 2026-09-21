// SPDX-FileCopyrightText: Copyright contributors to the kvcached project
// SPDX-License-Identifier: Apache-2.0

#include "page.hpp"

#include <stdexcept>
#include <string>

#include "constants.hpp"
#include "gpu_utils.hpp"
#include "transaction_error.hpp"

namespace kvcached {

namespace {

template <typename Status>
std::runtime_error gpu_error(Status status, const char *operation) {
  return std::runtime_error(std::string(operation) + " failed in " +
                            gpu_vmm::backend_name() + ": " +
                            gpu_vmm::error_string(status));
}

} // namespace

GPUPage::GPUPage(page_id_t page_id, int dev_idx, size_t page_size)
    : page_id_(page_id), dev_idx_(dev_idx),
      page_size_(page_size > 0 ? page_size : kPageSize), handle_() {
  auto prop = gpu_vmm::make_pinned_device_allocation_prop(dev_idx_);
  auto status = gpu_vmm::mem_create(&handle_, page_size_, &prop);
  if (!gpu_vmm::is_success(status)) {
    throw gpu_error(status, "physical page allocation");
  }
}

GPUPage::~GPUPage() {
  try {
    release();
  } catch (const std::exception &error) {
    LOGGER(ERROR, "GPU page teardown: %s", error.what());
  }
}

void GPUPage::release() {
  if (released_) {
    return;
  }
  auto status = gpu_vmm::mem_release(handle_);
  if (!gpu_vmm::is_success(status)) {
    throw StateConsistencyError(
        gpu_error(status, "physical page release").what());
  }
  released_ = true;
}

bool GPUPage::map(generic_ptr_t vaddr, bool set_access) {
  auto access_desc = gpu_vmm::make_device_rw_access_desc(dev_idx_);
  // `set_access` also reaches mem_map: where the access mode is an argument to
  // the mapping call, that call is the only place to defer access.
  auto map_status = gpu_vmm::mem_map(vaddr, page_size_, 0, handle_, set_access);
  if (!gpu_vmm::is_success(map_status)) {
    if (gpu_vmm::is_state_uncertain(map_status)) {
      // The map failed and the backend could not undo what it had already done,
      // so this page may still be partly mapped at vaddr. Say so with the type
      // the caller keys on: a plain failure means "no room, try again later"
      // and lets the page be released, which would hand mapped memory back to
      // the driver.
      throw StateConsistencyError(
          gpu_error(map_status, "physical page map").what());
    }
    throw gpu_error(map_status, "physical page map");
  }
  if (set_access) {
    auto access_status =
        gpu_vmm::set_access(vaddr, page_size_, &access_desc, 1);
    if (!gpu_vmm::is_success(access_status)) {
      auto unmap_status = gpu_vmm::mem_unmap(vaddr, page_size_);
      std::string message =
          std::string("physical page access setup failed in ") +
          gpu_vmm::backend_name() + ": " + gpu_vmm::error_string(access_status);
      if (!gpu_vmm::is_success(unmap_status)) {
        message =
            "state_inconsistency: " + message + "; rollback unmap failed: ";
        message += gpu_vmm::error_string(unmap_status);
        throw StateConsistencyError(message);
      }
      throw std::runtime_error(message);
    }
  }
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
