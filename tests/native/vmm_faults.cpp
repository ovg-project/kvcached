// SPDX-FileCopyrightText: Copyright contributors to the kvcached project
// SPDX-License-Identifier: Apache-2.0

// Test-only LD_PRELOAD shim. Successful calls always reach the real driver.
#include <atomic>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <dlfcn.h>
#include <string>

namespace {
std::atomic<int> create_count{0}, unmap_count{0}, map_count{0};
std::atomic<int> fail_create{0}, fail_unmap{0}, fail_map_start{0};
std::atomic<int> injected{0};
std::atomic<int> release_count{0}, fail_release{0};
std::atomic<bool> zero_free_memory{false};
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
extern "C" int kvcached_fault_release_calls() { return release_count.load(); }
extern "C" int kvcached_create_count() { return create_count.load(); }
extern "C" int kvcached_release_count() { return release_count.load(); }

extern "C" void kvcached_fault_zero_free_memory(int enabled) {
  zero_free_memory = enabled != 0;
}

extern "C" cudaError_t CUDARTAPI cudaMemGetInfo(size_t *free_bytes,
                                                size_t *total_bytes) {
  using GetInfo = decltype(&cudaMemGetInfo);
  static GetInfo original = []() -> GetInfo {
    auto next = reinterpret_cast<GetInfo>(dlsym(RTLD_NEXT, "cudaMemGetInfo"));
    if (next != nullptr && next != &cudaMemGetInfo) {
      return next;
    }
    const std::string soname =
        "libcudart.so." + std::to_string(CUDART_VERSION / 1000);
    void *runtime = dlopen(soname.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (runtime == nullptr) {
      return nullptr;
    }
    next = reinterpret_cast<GetInfo>(dlsym(runtime, "cudaMemGetInfo"));
    return next == &cudaMemGetInfo ? nullptr : next;
  }();
  if (original == nullptr) {
    return cudaErrorUnknown;
  }
  const auto result = original(free_bytes, total_bytes);
  if (result == cudaSuccess && zero_free_memory) {
    *free_bytes = 0;
  }
  return result;
}

extern "C" CUresult CUDAAPI cuMemGetInfo_v2(size_t *free_bytes,
                                            size_t *total_bytes) {
  auto original = reinterpret_cast<decltype(&cuMemGetInfo_v2)>(
      driver_symbol("cuMemGetInfo_v2"));
  const auto result = original(free_bytes, total_bytes);
  if (result == CUDA_SUCCESS && zero_free_memory) {
    *free_bytes = 0;
  }
  return result;
}

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
