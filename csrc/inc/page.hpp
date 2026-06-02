// SPDX-FileCopyrightText: Copyright contributors to the kvcached project
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <memory>

#include "constants.hpp"
#include "gpu_vmm.hpp"

namespace kvcached {

class Page {
public:
  virtual ~Page() = default;
  virtual bool map(void *vaddr, bool set_access = true) = 0;
  // unmap() is left for the caller to implement.
};

class GPUPage : public Page {
public:
  GPUPage(page_id_t page_id, int dev_idx, size_t page_size = 0);
  ~GPUPage();

  bool map(void *vaddr, bool set_access = true);

private:
  page_id_t page_id_;
  int dev_idx_;
  size_t page_size_;
  gpu_vmm::allocation_handle_t handle_;
};

class CPUPage : public Page {
public:
  CPUPage(page_id_t page_id, size_t page_size = 0);
  ~CPUPage();

  bool map(void *vaddr, bool set_access = true);

private:
  page_id_t page_id_;
  size_t page_size_;
  generic_ptr_t mapped_addr_;
};

} // namespace kvcached
