# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

"""Compile the real capacity-check code against a CPU device-runtime stand-in."""

import os
import subprocess
import sys
from pathlib import Path


def section(text: str, start: str, end: str) -> str:
    offset = text.index(start)
    return text[offset:text.index(end, offset)]


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    allocator = (root / "csrc/allocator.cpp").read_text()
    header = (root / "csrc/inc/allocator.hpp").read_text()
    stats = section(header, "struct PhysicalGrowthOperationStats {", "\nclass KVCACHED_HIDDEN")
    utilization = section(allocator, "double gpu_utilization_limit()", "\nstd::string physical_growth_lock_dir")
    # Allow the identical regression to run against the pre-fix source.
    guard = (section(allocator, "class ScopedDeviceGuard {", "\nvoid validate_physical_capacity")
             if "class ScopedDeviceGuard {" in allocator else "")
    capacity = section(allocator, "void validate_physical_capacity(", "\nsize_t checked_transaction_bytes")
    mock_runtime = r'''
#include <cassert>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
namespace gpu_vmm {
int device = 0;
int query_device = -1;
int selections = 0;
bool fail_query = false;
bool fail_select = false;
bool fail_restore = false;
int current_device() { return device; }
bool is_success(int status) { return status == 0; }
const char *error_string(int) { return "injected runtime error"; }
int set_device(int next) {
  ++selections;
  if ((fail_select && next == 1) || (fail_restore && next == 0)) {
    return 1;
  }
  device = next;
  return 0;
}
int mem_get_info(size_t *free_bytes, size_t *total_bytes) {
  query_device = device;
  *free_bytes = device == 1 ? 64 : 8;
  *total_bytes = 100;
  return fail_query ? 1 : 0;
}
}
// Discrete-GPU path of device_mem_info (gpu_utils.hpp).
int device_mem_info(int, size_t *free_bytes, size_t *total_bytes) {
  return gpu_vmm::mem_get_info(free_bytes, total_bytes);
}
// Match the production CHECK_GPU contract: restoration failure cannot continue.
#define CHECK_GPU(call) do { if (!gpu_vmm::is_success(call)) std::_Exit(86); } while (false)
'''
    cases = r'''
int main(int argc, char **) {
  if (argc > 1) {
    gpu_vmm::fail_restore = true;
    validate_physical_capacity(1, 1, nullptr);
    return 1;
  }
  // Same device, different device, capacity rejection, query error, invalid
  // utilization and failed selection. Each call must leave its caller on 0.
  for (int scenario = 0; scenario < 6; ++scenario) {
    gpu_vmm::device = 0;
    gpu_vmm::query_device = -1;
    gpu_vmm::selections = 0;
    gpu_vmm::fail_query = scenario == 3;
    gpu_vmm::fail_select = scenario == 5;
    setenv("KVCACHED_GPU_UTILIZATION", scenario == 4 ? "invalid" : "1", 1);
    const int target = scenario == 0 ? 0 : 1;
    PhysicalGrowthOperationStats stats;
    bool threw = false;
    try {
      validate_physical_capacity(target, scenario == 2 ? 65 : 1, &stats);
    } catch (const std::runtime_error &) {
      threw = true;
    }
    assert(threw == (scenario >= 2));
    assert(gpu_vmm::device == 0);
    assert(gpu_vmm::query_device == (scenario == 5 ? -1 : target));
    assert(gpu_vmm::selections == (scenario == 0 ? 0 : scenario == 5 ? 1 : 2));
    if (scenario <= 2) {
      assert(stats.free_bytes == (target == 1 ? 64u : 8u));
      assert(stats.capacity_rejections == (scenario == 2 ? 1u : 0u));
    }
  }
  std::cout << "physical capacity device context contracts passed\n";
}
'''
    executable = Path(sys.argv[1]) / "test_physical_capacity_device"
    subprocess.run([
        os.environ.get("CXX", "c++"), "-std=c++17", "-Wall", "-Wextra", "-Werror",
        "-x", "c++", "-o", str(executable), "-",
    ], input=mock_runtime + stats + utilization + guard + capacity + cases,
        text=True, check=True)
    subprocess.run([str(executable)], check=True)
    failed_restore = subprocess.run([str(executable), "restore-failure"], check=False)
    assert failed_restore.returncode == 86, "device restoration failure must stop execution"


if __name__ == "__main__":
    main()
