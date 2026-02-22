// SPDX-FileCopyrightText: Copyright contributors to the kvcached project
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

#include "gpu_vmm.hpp"
#include "gpu_utils.hpp"

namespace vmm = kvcached::gpu_vmm;

static constexpr int kNumThds = 1;
static constexpr size_t kPageSize = 2ul << 20; // MB
static constexpr size_t kNumPages = 4096;
static constexpr size_t kMemSize = kPageSize * kNumPages;

void print_header();
void print_stats(const std::string &op_name,
                 const std::vector<double> latencies[kNumThds]);

int init_gpu() {
  int supports_vmm = 0;

  CHECK_GPU(vmm::initialize_runtime());
  CHECK_GPU(vmm::set_device(0));
  int dev_idx = vmm::current_device();

  size_t free_mem = 0, total_mem = 0;
#if defined(KVCACHED_USE_HIP)
  CHECK_GPU(hipMemGetInfo(&free_mem, &total_mem));
#elif defined(KVCACHED_USE_CUDA)
  CHECK_GPU(cudaMemGetInfo(&free_mem, &total_mem));
#endif

  std::cout << "Backend: " << vmm::backend_name() << std::endl;
  std::cout << "Total Free Memory: " << (float)free_mem / std::giga::num << "GB"
            << std::endl;

  CHECK_DRV(vmm::get_vmm_support(&supports_vmm, dev_idx));
  if (supports_vmm) {
    std::cout << "====== VMM Benchmark ======" << std::endl;
  } else {
    std::cout << "VMM not supported" << std::endl;
  }

  return 0;
}

void *alloc_virtual(size_t size) {
  void *addr = nullptr;
  CHECK_DRV(vmm::address_reserve(&addr, size, kPageSize));
  return addr;
}

int bench_physical_alloc(std::vector<vmm::allocation_handle_t> &handles) {
  std::vector<std::thread> thds;
  std::vector<double> latencies[kNumThds];

  handles.resize(kNumPages);

  int dev_idx = vmm::current_device();
  auto prop = vmm::make_pinned_device_allocation_prop(dev_idx);

  for (int i = 0; i < kNumThds; i++) {
    thds.emplace_back([&, tid = i]() {
      auto stt_page = kNumPages / kNumThds * tid;
      auto end_page = kNumPages / kNumThds * (tid + 1);
      for (size_t page_idx = stt_page; page_idx < end_page; page_idx++) {
        auto stt = std::chrono::high_resolution_clock::now();
        CHECK_DRV(vmm::mem_create(&handles[page_idx], kPageSize, &prop));
        auto end = std::chrono::high_resolution_clock::now();
        latencies[tid].push_back(
            std::chrono::duration_cast<std::chrono::microseconds>(end - stt)
                .count());
      }
    });
  }

  for (auto &thd : thds) {
    thd.join();
  }

  print_stats("mem_create", latencies);

  return 0;
}

void get_lat_stats(std::vector<double> latencies, double &avg, double &max,
                   double &p50, double &p90, double &p99) {
  if (latencies.empty()) {
    avg = max = p50 = p90 = p99 = 0.0;
    return;
  }
  double sum = 0;
  max = 0;
  p50 = 0;
  p90 = 0;
  for (const auto &lat : latencies) {
    sum += lat;
    max = std::max(max, lat);
  }
  avg = sum / latencies.size();

  std::sort(latencies.begin(), latencies.end());
  p50 = latencies[latencies.size() / 2];
  p90 = latencies[latencies.size() * 9 / 10];
  p99 = latencies[latencies.size() * 99 / 100];
}

void print_header() {
  std::cout << "Benchmarking with " << kNumThds << " threads and " << kNumPages
            << " pages of size " << (kPageSize >> 20) << "MB." << std::endl;
  std::cout << std::string(75, '-') << std::endl;
  std::cout << std::left << std::setw(15) << "Operation" << std::setw(15)
            << "avg (us)" << std::setw(15) << "p50 (us)" << std::setw(15)
            << "p90 (us)" << std::setw(15) << "p99 (us)" << std::setw(15)
            << "max (us)" << std::endl;
  std::cout << std::string(75, '-') << std::endl;
}

void print_stats(const std::string &op_name,
                 const std::vector<double> latencies[kNumThds]) {
  std::vector<double> all_latencies;
  for (int i = 0; i < kNumThds; i++) {
    all_latencies.insert(all_latencies.end(), latencies[i].begin(),
                         latencies[i].end());
  }

  double avg, max, p50, p90, p99;
  get_lat_stats(all_latencies, avg, max, p50, p90, p99);

  std::cout << std::left << std::setw(15) << op_name << std::fixed
            << std::setprecision(2) << std::setw(15) << avg << std::setw(15)
            << p50 << std::setw(15) << p90 << std::setw(15) << p99
            << std::setw(15) << max << std::endl;
}

int bench_mmap(void *addr,
               std::vector<vmm::allocation_handle_t> &handles) {
  std::vector<std::thread> thds;
  std::vector<double> latencies[kNumThds];
  char *base = static_cast<char *>(addr);

  for (int i = 0; i < kNumThds; i++) {
    thds.emplace_back([&, tid = i]() {
      auto stt = kNumPages / kNumThds * tid;
      auto end = kNumPages / kNumThds * (tid + 1);
      for (size_t i = stt; i < end; i++) {
        auto stt = std::chrono::high_resolution_clock::now();
        CHECK_DRV(vmm::mem_map(base + i * kPageSize, kPageSize, 0, handles[i]));
        auto end = std::chrono::high_resolution_clock::now();
        latencies[tid].push_back(
            std::chrono::duration_cast<std::chrono::microseconds>(end - stt)
                .count());
      }
    });
  }

  for (auto &thd : thds) {
    thd.join();
  }

  print_stats("mem_map", latencies);

  return 0;
}

int bench_setaccess(void *addr) {
  std::vector<std::thread> thds;
  std::vector<double> latencies[kNumThds];
  char *base = static_cast<char *>(addr);

  int dev_idx = vmm::current_device();
  auto access_desc = vmm::make_device_rw_access_desc(dev_idx);

  for (int i = 0; i < kNumThds; i++) {
    thds.emplace_back([&, tid = i]() {
      auto stt = kNumPages / kNumThds * tid;
      auto end = kNumPages / kNumThds * (tid + 1);
      for (size_t i = stt; i < end; i++) {
        auto stt = std::chrono::high_resolution_clock::now();
        CHECK_DRV(
            vmm::set_access(base + i * kPageSize, kPageSize, &access_desc, 1));
        auto end = std::chrono::high_resolution_clock::now();
        latencies[tid].push_back(
            std::chrono::duration_cast<std::chrono::microseconds>(end - stt)
                .count());
      }
    });
  }

  for (auto &thd : thds) {
    thd.join();
  }

  print_stats("set_access", latencies);

  return 0;
}

int bench_munmap(void *addr) {
  std::vector<std::thread> thds;
  std::vector<double> latencies[kNumThds];
  char *base = static_cast<char *>(addr);

  for (int i = 0; i < kNumThds; i++) {
    thds.emplace_back([&, tid = i]() {
      auto stt = kNumPages / kNumThds * tid;
      auto end = kNumPages / kNumThds * (tid + 1);
      for (size_t i = stt; i < end; i++) {
        auto stt = std::chrono::high_resolution_clock::now();
        CHECK_DRV(vmm::mem_unmap(base + i * kPageSize, kPageSize));
        auto end = std::chrono::high_resolution_clock::now();
        latencies[tid].push_back(
            std::chrono::duration_cast<std::chrono::microseconds>(end - stt)
                .count());
      }
    });
  }

  for (auto &thd : thds) {
    thd.join();
  }

  print_stats("mem_unmap", latencies);

  return 0;
}

void free_physical(std::vector<vmm::allocation_handle_t> &handles) {
  for (const auto &handle : handles) {
    CHECK_DRV(vmm::mem_release(handle));
  }
}

void free_virtual(void *addr) {
  CHECK_DRV(vmm::address_free(addr, kMemSize));
}

int main() {
  init_gpu();

  auto stt = std::chrono::high_resolution_clock::now();
  void *addr = alloc_virtual(kMemSize);
  auto end = std::chrono::high_resolution_clock::now();
  auto lat =
      std::chrono::duration_cast<std::chrono::microseconds>(end - stt).count();
  std::cout << "\naddress_reserve (" << (kMemSize >> 30)
            << "GB) latency: " << lat << " us\n"
            << std::endl;

  std::vector<vmm::allocation_handle_t> handles;

  print_header();
  bench_physical_alloc(handles);
  bench_mmap(addr, handles);
  bench_setaccess(addr);
  bench_munmap(addr);

  free_physical(handles);
  free_virtual(addr);

  return 0;
}
