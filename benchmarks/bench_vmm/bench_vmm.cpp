#include <algorithm>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

#include <cuda.h>
#include <cuda_runtime.h>

#include "cuda_utils.hpp"

static constexpr int kNumThds = 1;
static constexpr size_t kPageSize = 2ul << 20; // MB
static constexpr size_t kNumPages = 4096;
static constexpr size_t kMemSize = kPageSize * kNumPages;

void print_header();
void print_stats(const std::string &op_name,
                 const std::vector<double> latencies[kNumThds]);

int init_cuda() {
  size_t free;
  typedef unsigned char ElemType;
  CUcontext ctx;
  CUdevice dev;
  int supportsVMM = 0;

  CHECK_RT(cudaFree(0));

  CHECK_DRV(cuInit(0));
  CHECK_DRV(cuDevicePrimaryCtxRetain(&ctx, 0));
  CHECK_DRV(cuCtxSetCurrent(ctx));
  CHECK_DRV(cuCtxGetDevice(&dev));

  CHECK_DRV(cuMemGetInfo(&free, NULL));
  std::cout << "Total Free Memory: " << (float)free / std::giga::num << "GB"
            << std::endl;

  CHECK_DRV(cuDeviceGetAttribute(
      &supportsVMM, CU_DEVICE_ATTRIBUTE_VIRTUAL_ADDRESS_MANAGEMENT_SUPPORTED,
      dev));
  if (supportsVMM) {
    std::cout << "====== cuMemMap ElemSz=" << sizeof(ElemType)
              << " ======" << std::endl;
  } else {
    std::cout << "VMM not supported" << std::endl;
  }

  return 0;
}

CUdeviceptr alloc_virtual(size_t size) {
  CUdeviceptr addr;
  CHECK_DRV(cuMemAddressReserve(&addr, size, kPageSize, 0, 0));
  return addr;
}

int bench_physical_alloc(std::vector<CUmemGenericAllocationHandle> &handles) {
  std::vector<std::thread> thds;
  std::vector<double> latencies[kNumThds];

  handles.resize(kNumPages);

  CUdevice dev;
  CHECK_DRV(cuCtxGetDevice(&dev));

  CUmemAllocationProp prop = {
      .type = CU_MEM_ALLOCATION_TYPE_PINNED,
      .location =
          {
              .type = CU_MEM_LOCATION_TYPE_DEVICE,
              .id = dev,
          },
  };

  for (int i = 0; i < kNumThds; i++) {
    thds.emplace_back([&, tid = i]() {
      auto stt_page = kNumPages / kNumThds * tid;
      auto end_page = kNumPages / kNumThds * (tid + 1);
      for (size_t page_idx = stt_page; page_idx < end_page; page_idx++) {
        auto stt = std::chrono::high_resolution_clock::now();
        CHECK_DRV(cuMemCreate(&handles[page_idx], kPageSize, &prop, 0));
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

  print_stats("cuMemCreate", latencies);

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

int bench_mmap(CUdeviceptr addr,
               std::vector<CUmemGenericAllocationHandle> &handles) {
  std::vector<std::thread> thds;
  std::vector<double> latencies[kNumThds];

  for (int i = 0; i < kNumThds; i++) {
    thds.emplace_back([&, tid = i]() {
      auto stt = kNumPages / kNumThds * tid;
      auto end = kNumPages / kNumThds * (tid + 1);
      for (size_t i = stt; i < end; i++) {
        auto stt = std::chrono::high_resolution_clock::now();
        CHECK_DRV(cuMemMap(addr + i * kPageSize, kPageSize, 0, handles[i], 0));
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

  print_stats("cuMemMap", latencies);

  return 0;
}

int bench_setaccess(CUdeviceptr addr) {
  std::vector<std::thread> thds;
  std::vector<double> latencies[kNumThds];
  CUdevice dev;

  CHECK_DRV(cuCtxGetDevice(&dev));
  CUmemAccessDesc accessDesc{
      .location =
          {
              .type = CU_MEM_LOCATION_TYPE_DEVICE,
              .id = dev,
          },
      .flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE,
  };

  for (int i = 0; i < kNumThds; i++) {
    thds.emplace_back([&, tid = i]() {
      auto stt = kNumPages / kNumThds * tid;
      auto end = kNumPages / kNumThds * (tid + 1);
      for (size_t i = stt; i < end; i++) {
        auto stt = std::chrono::high_resolution_clock::now();
        CHECK_DRV(
            cuMemSetAccess(addr + i * kPageSize, kPageSize, &accessDesc, 1));
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

  print_stats("cuMemSetAccess", latencies);

  return 0;
}

int bench_munmap(CUdeviceptr addr) {
  std::vector<std::thread> thds;
  std::vector<double> latencies[kNumThds];

  for (int i = 0; i < kNumThds; i++) {
    thds.emplace_back([&, tid = i]() {
      auto stt = kNumPages / kNumThds * tid;
      auto end = kNumPages / kNumThds * (tid + 1);
      for (size_t i = stt; i < end; i++) {
        auto stt = std::chrono::high_resolution_clock::now();
        CHECK_DRV(cuMemUnmap(addr + i * kPageSize, kPageSize));
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

  print_stats("cuMemUnmap", latencies);

  return 0;
}

void free_physical(std::vector<CUmemGenericAllocationHandle> &handles) {
  for (const auto &handle : handles) {
    CHECK_DRV(cuMemRelease(handle));
  }
}

void free_virtual(CUdeviceptr addr) {
  CHECK_DRV(cuMemAddressFree(addr, kMemSize));
}

int main() {
  init_cuda();

  auto stt = std::chrono::high_resolution_clock::now();
  CUdeviceptr addr = alloc_virtual(kMemSize);
  auto end = std::chrono::high_resolution_clock::now();
  auto lat =
      std::chrono::duration_cast<std::chrono::microseconds>(end - stt).count();
  std::cout << "\ncuMemAddressReserve (" << (kMemSize >> 30)
            << "GB) latency: " << lat << " us\n"
            << std::endl;

  std::vector<CUmemGenericAllocationHandle> handles;

  print_header();
  bench_physical_alloc(handles);
  bench_mmap(addr, handles);
  bench_setaccess(addr);
  bench_munmap(addr);

  free_physical(handles);
  free_virtual(addr);

  return 0;
}