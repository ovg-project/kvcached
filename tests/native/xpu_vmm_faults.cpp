// SPDX-FileCopyrightText: Copyright contributors to the kvcached project
// SPDX-License-Identifier: Apache-2.0

// Test-only LD_PRELOAD shim, the XPU counterpart of vmm_faults.cpp. Successful
// calls always reach the real SYCL runtime.
//
// It interposes the two functions gpu_vmm.hpp's XPU arm calls --
// physical_mem::map() and unmap() -- addressed by their mangled names, so this
// file needs no SYCL headers and builds with a plain host compiler. Interposing
// Level Zero instead does not work: measured on this stack, a preloaded
// zeVirtualMemMap/zeVirtualMemUnmap is never called, because the Unified
// Runtime reaches them through function pointers it loads itself.
//
// Arming is relative to the arm call, not to process start: each FTensor maps
// an anchor page in its constructor, so a test cannot know its own absolute
// call index.

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <dlfcn.h>
#include <link.h>
#include <stdexcept>
#include <string>

// Each name is needed twice -- to say what this file replaces, and to fetch
// what it replaced -- so it lives in one place where the two cannot disagree.
// clang-format off
#define KVCACHED_SYCL_PHYSICAL_MEM_MAP "_ZNK4sycl3_V13ext6oneapi12experimental12physical_mem3mapEmmNS3_19address_access_modeEm"
#define KVCACHED_SYCL_UNMAP "_ZN4sycl3_V13ext6oneapi12experimental5unmapEPKvmRKNS0_7contextE"
// clang-format on

namespace {

std::atomic<int> map_calls{0}, map_hits{0}, cleanup_hits{0};
std::atomic<int> fail_map_at{0}, fail_cleanup{0};
// Set by an injected map failure and consumed by the next unmap, which is the
// cleanup inside mem_map(). Anything later -- a rollback unmap, a real free --
// runs for real, so one arm call injects exactly the double failure.
std::atomic<int> cleanup_armed{0};

// The extension is loaded with RTLD_LOCAL, so libsycl is outside RTLD_NEXT's
// scope and has to be reopened by path. Its soname is version-specific, hence
// asking the loader which object is already in the process rather than naming
// one here.
int match_sycl(struct dl_phdr_info *info, size_t, void *out) {
  if (info->dlpi_name && std::strstr(info->dlpi_name, "libsycl.so")) {
    *static_cast<std::string *>(out) = info->dlpi_name;
    return 1;
  }
  return 0;
}

void *sycl_symbol(const char *name) {
  static void *lib = [] {
    std::string path;
    dl_iterate_phdr(match_sycl, &path);
    return path.empty() ? nullptr : dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
  }();
  void *fn = lib ? dlsym(lib, name) : nullptr;
  if (fn == nullptr) {
    throw std::runtime_error(std::string("fault shim cannot resolve ") + name);
  }
  return fn;
}

} // namespace

extern "C" void kvcached_xpu_fault_arm(int map_at, int fail_cleanup_after) {
  map_calls = map_hits = cleanup_hits = 0;
  cleanup_armed = 0;
  fail_map_at = map_at;
  fail_cleanup = fail_cleanup_after;
}

extern "C" int kvcached_xpu_fault_map_hits() { return map_hits.load(); }

extern "C" int kvcached_xpu_fault_cleanup_hits() { return cleanup_hits.load(); }

extern "C" int kvcached_xpu_fault_map_calls() { return map_calls.load(); }

// void *sycl::_V1::ext::oneapi::experimental::physical_mem::map(
//     uintptr_t, size_t, address_access_mode, size_t) const
void *kvcached_fault_map(const void *self, uintptr_t ptr, size_t size, int mode,
                         size_t offset) __asm__(KVCACHED_SYCL_PHYSICAL_MEM_MAP);

void *kvcached_fault_map(const void *self, uintptr_t ptr, size_t size, int mode,
                         size_t offset) {
  const int seq = ++map_calls;
  if (fail_map_at != 0 && seq == fail_map_at) {
    ++map_hits;
    if (fail_cleanup != 0) {
      cleanup_armed = 1;
    }
    // A real out-of-memory map throws too, and part of the range may be mapped
    // when it does; that is the case this stands in for.
    throw std::runtime_error("injected map failure (test shim)");
  }
  using map_fn = void *(*)(const void *, uintptr_t, size_t, int, size_t);
  auto original =
      reinterpret_cast<map_fn>(sycl_symbol(KVCACHED_SYCL_PHYSICAL_MEM_MAP));
  return original(self, ptr, size, mode, offset);
}

// void sycl::_V1::ext::oneapi::experimental::unmap(
//     const void *, size_t, const context &)
void kvcached_fault_unmap(const void *ptr, size_t size,
                          const void *ctx) __asm__(KVCACHED_SYCL_UNMAP);

void kvcached_fault_unmap(const void *ptr, size_t size, const void *ctx) {
  if (cleanup_armed.exchange(0) != 0) {
    ++cleanup_hits;
    throw std::runtime_error("injected cleanup unmap failure (test shim)");
  }
  using unmap_fn = void (*)(const void *, size_t, const void *);
  auto original = reinterpret_cast<unmap_fn>(sycl_symbol(KVCACHED_SYCL_UNMAP));
  original(ptr, size, ctx);
}
