# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

"""Compile production device/query methods with CPU-only runtime boundaries.

The complete native allocator/preallocation path needs CUDA headers; these
portable cases exercise the actual C++ method bodies without a GPU/toolkit.
CUDA state and FTensor materialization are controlled boundaries.
"""

import shutil
import subprocess
from pathlib import Path

import pytest


def _method(source, signature):
    start = source.index(signature)
    opening = source.index("{", start)
    depth = 1
    end = opening + 1
    while depth:
        depth += (source[end] == "{") - (source[end] == "}")
        end += 1
    return source[start:end]


@pytest.fixture(scope="module")
def device_probe(tmp_path_factory):
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("C++ compiler required")
    root = Path(__file__).resolve().parents[1]
    allocator = (root / "csrc/allocator.cpp").read_text()
    pages = (root / "csrc/page_allocator.cpp").read_text()
    methods = "\n".join([
        _method(allocator, "void FTensorAllocator::init("),
        _method(allocator, "FTensorAllocator *FTensorAllocator::global_allocator("),
        _method(pages, "int64_t PageAllocator::get_avail_physical_pages() const"),
    ])
    build = tmp_path_factory.mktemp("page-device")
    source = build / "probe.cpp"
    source.write_text(_RUNTIME + methods + _CASES)
    binary = build / "probe"
    subprocess.run([compiler, "-std=c++17", "-pthread", str(source), "-o", str(binary)],
                   check=True, capture_output=True, text=True)
    return binary


@pytest.mark.parametrize("owner,caller,free0,free1,expected", [
    (0, 0, 95, 2, 90),
    (1, 1, 2, 95, 90),
    (1, 0, 2, 95, 90),
    (1, 0, 95, 2, 0),
    (1, 0, 95, 95, 90),
    (0, 1, 95, 2, 90),
    (1, 0, 95, 5, 0),
])
def test_query_uses_pool_device_and_preserves_caller(
    device_probe, owner, caller, free0, free1, expected,
):
    result = subprocess.run([str(device_probe), "query", str(owner), str(caller),
                             str(free0), str(free1)], capture_output=True, text=True,
                            check=True, timeout=10)
    assert f"pages={expected} device={caller} queried={owner}" in result.stdout
    assert f"selections={0 if owner == caller else 2} groups=1" in result.stdout


def test_query_from_a_new_thread(device_probe):
    result = subprocess.run([str(device_probe), "thread", "1", "0", "2", "95"],
                            capture_output=True, text=True, check=True, timeout=10)
    assert "pages=90 device=0 queried=1 selections=2 groups=1" in result.stdout


def test_implicit_device_is_pinned_before_lazy_group_creation(device_probe):
    result = subprocess.run([str(device_probe), "implicit", "1", "0", "2", "95"],
                            capture_output=True, text=True, check=True, timeout=10)
    assert "configured=1 group_owner=1" in result.stdout


def test_standalone_allocator_without_a_tensor_backend(device_probe):
    result = subprocess.run([str(device_probe), "standalone", "0", "1", "2", "95"],
                            capture_output=True, text=True, check=True, timeout=10)
    assert "pages=90 device=1 queried=1 selections=0 groups=0" in result.stdout


@pytest.mark.parametrize("mode,device,queried", [
    ("query_error", 0, 1),
    ("select_error", 0, -1),
    ("restore_error", 1, 1),
])
def test_runtime_errors_remain_fatal(device_probe, mode, device, queried):
    result = subprocess.run([str(device_probe), mode, "1", "0", "2", "95"],
                            capture_output=True, text=True, timeout=10, check=False)
    assert result.returncode == 86
    assert f"ERROR device={device} queried={queried}" in result.stdout


_RUNTIME = r'''
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
namespace gpu_vmm {
thread_local int device = 0, queried = -1, selections = 0;
int fail_target = -1;
bool fail_query = false;
size_t free_mib[2];
int current_device() { return device; }
int set_device(int next) {
  ++selections;
  if (next < 0 || next > 1 || next == fail_target) return 1;
  device = next;
  return 0;
}
int mem_get_info(size_t* free, size_t* total) {
  queried = device;
  *free = free_mib[device] * 1024 * 1024;
  *total = 100 * 1024 * 1024;
  return fail_query ? 1 : 0;
}
void check(int status) {
  if (status) {
    std::cout << "ERROR device=" << device << " queried=" << queried << std::endl;
    std::_Exit(86); // Model CHECK_GPU's terminal runtime-error boundary.
  }
}
}
#define CHECK_GPU(call) gpu_vmm::check(call)
#define LOGGER(...) ((void)0)
namespace torch { namespace stable {
class Device {
  std::string name_;
public:
  explicit Device(const std::string& name) : name_(name) {}
  bool is_cuda() const { return name_.find("cuda") == 0; }
  int index() const {
    auto colon = name_.find(':');
    return colon == std::string::npos ? -1 : std::stoi(name_.substr(colon + 1));
  }
};
}}
size_t kPageSize = 2 * 1024 * 1024;
class FTensorAllocator {
public:
  static std::mutex g_allocator_mutex_;
  static std::unordered_map<int64_t, std::unique_ptr<FTensorAllocator>> g_allocators_;
  static torch::stable::Device g_device_;
  static bool g_contiguous_layout_;
  int owner;
  FTensorAllocator(const torch::stable::Device& device, bool) {
    owner = device.index() < 0 ? gpu_vmm::current_device() : device.index();
    CHECK_GPU(gpu_vmm::set_device(owner));
  }
  static void init(const std::string&, size_t, bool);
  static FTensorAllocator* global_allocator(int64_t);
  // FTensor metadata boundary; init/global_allocator bodies below are production.
  static int device_index() { return g_device_.index(); }
};
std::mutex FTensorAllocator::g_allocator_mutex_;
std::unordered_map<int64_t, std::unique_ptr<FTensorAllocator>> FTensorAllocator::g_allocators_;
torch::stable::Device FTensorAllocator::g_device_("cpu");
bool FTensorAllocator::g_contiguous_layout_ = false;
class PageAllocator {
public:
  double gpu_utilization_ = 0.95;
  int64_t page_size_ = 1024 * 1024, num_layers_ = 1, num_kv_buffers_ = 1;
  int64_t get_avail_physical_pages() const;
};
size_t saturating_subtract(size_t a, size_t b) { return a > b ? a - b : 0; }
'''

_CASES = r'''
int main(int argc, char** argv) {
  assert(argc == 6);
  const std::string mode = argv[1];
  const int owner = std::stoi(argv[2]), caller = std::stoi(argv[3]);
  gpu_vmm::free_mib[0] = std::stoul(argv[4]);
  gpu_vmm::free_mib[1] = std::stoul(argv[5]);
  CHECK_GPU(gpu_vmm::set_device(owner));
  if (mode != "standalone")
    FTensorAllocator::init(mode == "implicit" ? "cuda" : "cuda:" + std::to_string(owner), 0, false);
  CHECK_GPU(gpu_vmm::set_device(caller));
  gpu_vmm::selections = 0;
  gpu_vmm::fail_query = mode == "query_error";
  gpu_vmm::fail_target = mode == "select_error" ? owner : mode == "restore_error" ? caller : -1;
  if (mode == "implicit") {
    auto group = FTensorAllocator::global_allocator(7);
    std::cout << "configured=" << FTensorAllocator::g_device_.index()
              << " group_owner=" << group->owner << std::endl;
    return 0;
  }
  // The pool may be constructed on a thread whose current device differs.
  PageAllocator allocator;
  auto query = [&]() {
    auto pages = allocator.get_avail_physical_pages();
    std::cout << "pages=" << pages << " device=" << gpu_vmm::device
              << " queried=" << gpu_vmm::queried
              << " selections=" << gpu_vmm::selections
              << " groups=" << FTensorAllocator::g_allocators_.size() << std::endl;
  };
  if (mode == "thread") {
    std::thread worker(query);
    worker.join();
  } else {
    query();
  }
}
'''
