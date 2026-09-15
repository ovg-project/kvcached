// SPDX-FileCopyrightText: Copyright contributors to the kvcached project
// SPDX-License-Identifier: Apache-2.0

// Test-only LD_PRELOAD shim. Successful calls always reach the real driver.
#include <atomic>
#include <cuda.h>
#include <dlfcn.h>

namespace {
std::atomic<int> create_count{0}, unmap_count{0}, map_count{0};
std::atomic<int> fail_create{0}, fail_unmap{0}, fail_map_start{0};
std::atomic<int> injected{0};
std::atomic<int> release_count{0}, fail_release{0};
void *driver_symbol(const char *name) {
  // PyTorch may load libcuda locally, outside LD_PRELOAD's RTLD_NEXT scope.
  static void *driver = dlopen("libcuda.so.1", RTLD_NOW | RTLD_LOCAL);
  return dlsym(driver, name);
}
bool fail_at(std::atomic<int> &counter, int target) {
  return ++counter == target;
}
} // namespace

extern "C" void kvcached_fault_arm(int create_at, int unmap_at, int map_start) {
  create_count = unmap_count = map_count = 0;
  injected = 0;
  fail_create = create_at;
  fail_unmap = unmap_at;
  fail_map_start = map_start;
  release_count = fail_release = 0;
}

extern "C" void kvcached_fault_release(int release_at) {
  release_count = 0;
  fail_release = release_at;
}

extern "C" int kvcached_fault_hits() { return injected.load(); }

extern "C" CUresult CUDAAPI cuMemRelease(CUmemGenericAllocationHandle handle) {
  if (fail_at(release_count, fail_release)) {
    ++injected;
    return CUDA_ERROR_INVALID_VALUE;
  }
  auto original =
      reinterpret_cast<decltype(&cuMemRelease)>(driver_symbol("cuMemRelease"));
  return original(handle);
}

extern "C" CUresult CUDAAPI cuMemCreate(CUmemGenericAllocationHandle *handle,
                                        size_t size,
                                        const CUmemAllocationProp *prop,
                                        unsigned long long flags) {
  if (fail_at(create_count, fail_create)) {
    ++injected;
    return CUDA_ERROR_OUT_OF_MEMORY;
  }
  auto original =
      reinterpret_cast<decltype(&cuMemCreate)>(driver_symbol("cuMemCreate"));
  return original(handle, size, prop, flags);
}

extern "C" CUresult CUDAAPI cuMemUnmap(CUdeviceptr ptr, size_t size) {
  if (fail_at(unmap_count, fail_unmap)) {
    ++injected;
    return CUDA_ERROR_INVALID_VALUE;
  }
  auto original =
      reinterpret_cast<decltype(&cuMemUnmap)>(driver_symbol("cuMemUnmap"));
  return original(ptr, size);
}

extern "C" CUresult CUDAAPI cuMemMap(CUdeviceptr ptr, size_t size,
                                     size_t offset,
                                     CUmemGenericAllocationHandle handle,
                                     unsigned long long flags) {
  int count = ++map_count;
  int start = fail_map_start;
  if (start > 0 && (count == start || count == start + 1)) {
    ++injected;
    return CUDA_ERROR_INVALID_VALUE;
  }
  auto original =
      reinterpret_cast<decltype(&cuMemMap)>(driver_symbol("cuMemMap"));
  return original(ptr, size, offset, handle, flags);
}
