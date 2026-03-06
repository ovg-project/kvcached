// SPDX-FileCopyrightText: Copyright contributors to the kvcached project
// SPDX-License-Identifier: Apache-2.0

#include <memory>
#include <mutex>
#include <torch/extension.h>
#include <unordered_map>

#include "allocator.hpp"
#include "constants.hpp"
#include "cuda_utils.hpp"
#include "ftensor.hpp"
#include "page.hpp"
#include "torch_utils.hpp"

namespace kvcached {
// Global configurable page size
size_t kPageSize = 2 * 1024 * 1024; // Default 2MB

std::unique_ptr<FTensorAllocator> FTensorAllocator::g_allocator_;
std::mutex FTensorAllocator::g_allocator_mutex_;

static inline std::shared_ptr<Page> make_shared_page(const torch::Device &dev,
                                                     page_id_t page_id,
                                                     size_t page_size = 0) {
  if (dev.is_cuda()) {
    return std::make_shared<GPUPage>(page_id, dev.index(), page_size);
  } else if (dev.is_cpu()) {
    return std::make_shared<CPUPage>(page_id, page_size);
  }
  ASSERT(false, "Unsupported device type.");
  return nullptr;
}

static inline size_t get_v_base_offset(const torch::Tensor &tensor) {
  size_t num_eles = tensor.numel() * tensor.element_size();
  ASSERT(num_eles % (2 * kPageSize) == 0,
         "Invalid tensor size: %zu, must be a multiple of 2 * page size %zu",
         num_eles, 2 * kPageSize);
  return num_eles / 2;
}

FTensorAllocator::FTensorAllocator(const torch::Device &device,
                                   bool contiguous_layout)
    : dev_(device), contiguous_layout_(contiguous_layout) {
  if (dev_.is_cuda()) {
    init_cuda_();
  }
}

FTensorAllocator::~FTensorAllocator() { destroy(); }

void FTensorAllocator::destroy() {
  std::lock_guard<std::mutex> lock(mtx_);
  kv_groups_.clear();
}

void FTensorAllocator::init(const std::string &dev_str, size_t page_size,
                            bool contiguous_layout) {
  std::lock_guard<std::mutex> lock(g_allocator_mutex_);
  if (g_allocator_) {
    LOGE("FTensorAllocator has been initialized. Re-initializing...")
    g_allocator_.reset();
  }

  // Set global page size if provided (0 means use default)
  if (page_size > 0) {
    // Validate that page_size is a multiple of 2MB
    size_t base_size = 2 * 1024 * 1024; // 2MB
    if (page_size % base_size != 0) {
      LOGE("Invalid page size: %zu, must be a multiple of 2MB (2097152 bytes)",
           page_size);
      abort();
    }
    kPageSize = page_size;
  }

  auto device = torch::Device(dev_str);
  g_allocator_ = std::make_unique<FTensorAllocator>(device, contiguous_layout);
}

FTensorAllocator *FTensorAllocator::global_allocator() {
  std::lock_guard<std::mutex> lock(g_allocator_mutex_);
  assert(g_allocator_);
  return g_allocator_.get();
}

void FTensorAllocator::shutdown() {
  std::lock_guard<std::mutex> lock(g_allocator_mutex_);
  if (g_allocator_) {
    g_allocator_.reset();
  }
}

FTensorAllocator::KVGroup &
FTensorAllocator::get_or_create_group_(int64_t group_id) {
  // mtx_ must be held by the caller.
  return kv_groups_[group_id]; // default-constructs if absent
}

std::vector<torch::Tensor> FTensorAllocator::create_kv_tensors(
    size_t size, torch::Dtype dtype, const std::string &dev_str,
    int64_t num_layers, int64_t num_kv_buffers, int64_t group_id) {
  std::lock_guard<std::mutex> lock(mtx_);

  auto &group = get_or_create_group_(group_id);

  assert(group.num_layers == 0 || group.num_layers == num_layers);
  group.num_layers = num_layers;

  // Ensure size is aligned to page size.
  size_t aligned_size = size;
  if (size % kPageSize != 0) {
    aligned_size = ((size + kPageSize - 1) / kPageSize) * kPageSize;
    LOGW("Size %zu is not aligned to page size %zu, aligning to %zu", size,
         kPageSize, aligned_size);
  }
  group.kv_tensor_size_per_layer = aligned_size;

  // Build a group-specific prefix so FTensor names don't collide across groups.
  auto prefix = std::string(kv_prefix) + "g" + std::to_string(group_id) + "_";

  if (contiguous_layout_) {
    // For contiguous layout, we use compound page which groups all layers
    // together for a single page. num_kv_buffers is 2 for MHA (K+V) and
    // 1 for MLA (combined KV).
    size_t compound_page_size = kPageSize * num_layers * num_kv_buffers;
    group.zero_page = make_shared_page(dev_, ZERO_PAGE_ID, compound_page_size);
    return create_kv_tensors_contiguous_(group, aligned_size, dtype, dev_str,
                                         num_layers, compound_page_size);
  } else {
    group.zero_page = make_shared_page(dev_, ZERO_PAGE_ID);
    return create_kv_tensors_per_layer_(group, prefix, aligned_size, dtype,
                                        dev_str, num_layers);
  }
}

bool FTensorAllocator::kv_tensors_created(int64_t group_id) {
  std::lock_guard<std::mutex> lock(mtx_);
  auto it = kv_groups_.find(group_id);
  if (it == kv_groups_.end()) {
    return false;
  }
  return it->second.num_layers > 0;
}

bool FTensorAllocator::map_to_kv_tensors(const std::vector<offset_t> &offsets,
                                         int64_t group_id) {
  std::unique_lock<std::mutex> lock(mtx_);
  auto it = kv_groups_.find(group_id);
  if (it == kv_groups_.end() || it->second.num_layers == 0) {
    LOGE("try to map to KV tensors when KV tensors are not created "
         "(group_id=%ld)",
         group_id);
    return false;
  }
  auto &group = it->second;

  if (contiguous_layout_) {
    // In contiguous layout, use the single contiguous tensor for mapping
    // Each offset maps a block that contains all layers
    auto ftensor = group.contiguous_kv_tensor.get();
    auto tensor = ftensor->get_tensor();

    for (auto offset : offsets) {
      // Map K and V regions for this block (covers all layers)
      ftensor->map(offset);
    }
  } else {
    // Per-layer mapping within this group
    for (int64_t i = 0; i < group.num_layers; i++) {
      auto kv_name = std::string(kv_prefix) + "g" + std::to_string(group_id) +
                     "_" + std::to_string(i);
      auto ftensor = group.ftensors[kv_name].get();
      /**
       * NOTE: we assume the K tensor and the V tensor are stacked at the 1st
       * dim. This is used for calculating the offset of the V tensor.
       * FIXME: (YIFAN) we may support other KV cache layouts later.
       */
      auto tensor = ftensor->get_tensor();
      auto v_base_offset = get_v_base_offset(tensor);
      for (auto offset : offsets) {
        auto koffset = offset;
        auto voffset = offset + v_base_offset;
        ftensor->map(koffset);
        ftensor->map(voffset);
      }
    }
  }
  return true;
}

bool FTensorAllocator::unmap_from_kv_tensors(
    const std::vector<offset_t> &offsets, int64_t group_id) {
  std::unique_lock<std::mutex> lock(mtx_);
  auto it = kv_groups_.find(group_id);
  if (it == kv_groups_.end() || it->second.num_layers == 0) {
    LOGE("try to unmap from KV tensors when KV tensors are not created "
         "(group_id=%ld)",
         group_id);
    return false;
  }
  auto &group = it->second;

  if (contiguous_layout_) {
    // In contiguous layout, unmap using the single contiguous tensor
    auto ftensor = group.contiguous_kv_tensor.get();
    auto tensor = ftensor->get_tensor();

    for (auto offset : offsets) {
      // Unmap K and V regions for this block (covers all layers)
      ftensor->unmap(offset);
    }
  } else {
    // Per-layer unmapping within this group
    for (int64_t i = 0; i < group.num_layers; i++) {
      auto kv_name = std::string(kv_prefix) + "g" + std::to_string(group_id) +
                     "_" + std::to_string(i);
      auto ftensor = group.ftensors[kv_name].get();
      /**
       * NOTE: we assume the K tensor and the V tensor are stacked at the 1st
       * dim. This is used for calculating the offset of the V tensor.
       * FIXME: (YIFAN) we may support other KV cache layouts later.
       */
      auto tensor = ftensor->get_tensor();
      auto v_base_offset = get_v_base_offset(tensor);
      for (auto offset : offsets) {
        ftensor->unmap(offset);
        ftensor->unmap(offset + v_base_offset);
      }
    }
  }
  return true;
}

std::string FTensorAllocator::get_anon_tensor_name_() {
  static constexpr std::string_view prefix = "anon_tensor_";
  static std::atomic<int> counter(0);
  return std::string(prefix) + std::to_string(counter++);
}

std::vector<torch::Tensor> FTensorAllocator::create_kv_tensors_per_layer_(
    KVGroup &group, std::string_view prefix, size_t size, torch::Dtype dtype,
    const std::string &dev_str, int64_t num_layers) {
  std::vector<torch::Tensor> ftensors;
  for (int64_t i = 0; i < num_layers; i++) {
    auto name = std::string(prefix) + std::to_string(i);
    auto tensor = create_ftensor_(group, size, dtype, dev_str, name);
    ftensors.push_back(tensor);
  }
  return ftensors;
}

std::vector<torch::Tensor> FTensorAllocator::create_kv_tensors_contiguous_(
    KVGroup &group, size_t size, torch::Dtype dtype, const std::string &dev_str,
    int64_t num_layers, size_t compound_page_size) {
  // In contiguous layout, Python passes per-layer size, and we multiply by
  // num_layers to get total size
  size_t total_kv_size = size * num_layers;

  // Create the single contiguous KV tensor (contains K and V for all layers)
  auto contiguous_name = std::string(kv_prefix) + "contiguous";
  group.contiguous_kv_tensor =
      std::make_unique<FTensor>(contiguous_name, total_kv_size, dtype, dev_,
                                group.zero_page, compound_page_size);

  // Get the contiguous tensor
  auto contiguous_tensor = group.contiguous_kv_tensor->get_tensor();
  return {contiguous_tensor};
}

/** this function is not thread-safe */
torch::Tensor FTensorAllocator::create_ftensor_(KVGroup &group, size_t size,
                                                torch::Dtype dtype,
                                                const std::string &dev_str,
                                                std::string name) {
  if (name.empty())
    name = get_anon_tensor_name_();

  if (group.ftensors.find(name) != group.ftensors.end()) {
    auto tensor = group.ftensors[name].get()->get_tensor();
    assert(tensor.numel() * tensor.element_size() == size);
    assert(tensor.device() == torch::Device(dev_str));
    return tensor;
  }

  // Create a new FTensor
  group.ftensors[name] =
      std::make_unique<FTensor>(name, size, dtype, dev_, group.zero_page);
  return group.ftensors[name]->get_tensor();
}

/** this function is not thread-safe */
void FTensorAllocator::free_ftensor_(KVGroup &group, torch::Tensor &ftensor) {
  auto name = ftensor.name();
  if (group.ftensors.find(name) == group.ftensors.end()) {
    return;
  }
  group.ftensors.erase(name);
}

void FTensorAllocator::init_cuda_() {
  CHECK_RT(cudaFree(0));

  CUdevice dev;
  CHECK_DRV(cuCtxGetDevice(&dev));

  int supportsVMM = 0;
  CHECK_DRV(cuDeviceGetAttribute(
      &supportsVMM, CU_DEVICE_ATTRIBUTE_VIRTUAL_ADDRESS_MANAGEMENT_SUPPORTED,
      dev));
  // LOGE("Supports VMM: %d", supportsVMM);

  CUcontext context;
  CHECK_DRV(cuCtxGetCurrent(&context));

  CUmemAllocationProp prop{
      .type = CU_MEM_ALLOCATION_TYPE_PINNED,
      .location =
          {
              .type = CU_MEM_LOCATION_TYPE_DEVICE,
              .id = dev,
          },
  };

  size_t chunk_sz = 0;
  CHECK_DRV(cuMemGetAllocationGranularity(&chunk_sz, &prop,
                                          CU_MEM_ALLOC_GRANULARITY_MINIMUM));
  ASSERT(kPageSize % chunk_sz == 0,
         "Invalid page size: %lu must be a multiple of CUDA granularity %lu\n",
         kPageSize, chunk_sz);
}

} // namespace kvcached
