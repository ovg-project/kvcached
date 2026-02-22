// SPDX-FileCopyrightText: Copyright contributors to the kvcached project
// SPDX-License-Identifier: Apache-2.0

#include <memory>
#include <mutex>
#include <unordered_map>

#include "allocator.hpp"
#include "constants.hpp"
#include "ftensor.hpp"
#include "gpu_utils.hpp"
#include "page.hpp"

namespace kvcached {
// Global configurable page size
size_t kPageSize = 2 * 1024 * 1024; // Default 2MB

std::unique_ptr<FTensorAllocator> FTensorAllocator::g_allocator_;
std::mutex FTensorAllocator::g_allocator_mutex_;

static inline std::shared_ptr<Page> make_shared_page(const c10::Device &dev,
                                                     page_id_t page_id,
                                                     size_t page_size = 0) {
  auto resolve_device_index = [](const c10::Device &device) -> int {
    if (device.index() >= 0) {
      return device.index();
    }
    return gpu_vmm::current_device();
  };

  // is_cuda() returns true for both NVIDIA (CUDA) and AMD (HIP/ROCm) devices,
  // because PyTorch's ROCm build masquerades HIP devices as CUDA.
  if (dev.is_cuda()) {
    return std::make_shared<GPUPage>(page_id, resolve_device_index(dev),
                                     page_size);
  } else if (dev.is_cpu()) {
    return std::make_shared<CPUPage>(page_id, page_size);
  }
  ASSERT(false, "Unsupported device type.");
  return nullptr;
}

static inline size_t get_v_base_offset(const at::Tensor &tensor) {
  size_t num_eles = tensor.numel() * tensor.element_size();
  ASSERT(num_eles % (2 * kPageSize) == 0,
         "Invalid tensor size: %zu, must be a multiple of 2 * page size %zu",
         num_eles, 2 * kPageSize);
  return num_eles / 2;
}

FTensorAllocator::FTensorAllocator(const c10::Device &device,
                                   bool contiguous_layout)
    : dev_(device), num_layers_(0), contiguous_layout_(contiguous_layout),
      kv_tensor_size_per_layer_(0) {
  if (dev_.is_cuda()) {
    init_gpu_();
  }
}

FTensorAllocator::~FTensorAllocator() { destroy(); }

void FTensorAllocator::destroy() {
  std::lock_guard<std::mutex> lock(mtx_);
  ftensors_.clear();
  contiguous_kv_tensor_.reset();
  zero_page_.reset();
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

  auto device = c10::Device(dev_str);
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

std::vector<at::Tensor> FTensorAllocator::create_kv_tensors(
    size_t size, c10::ScalarType dtype, const std::string &dev_str,
    int64_t num_layers, int64_t num_kv_buffers) {
  std::lock_guard<std::mutex> lock(mtx_);

  assert(num_layers_ == 0 || num_layers_ == num_layers);
  num_layers_ = num_layers;
  // Ensure size is aligned to page size.
  size_t aligned_size = size;
  if (size % kPageSize != 0) {
    aligned_size = ((size + kPageSize - 1) / kPageSize) * kPageSize;
    LOGW("Size %zu is not aligned to page size %zu, aligning to %zu", size,
         kPageSize, aligned_size);
  }
  kv_tensor_size_per_layer_ = aligned_size;

  if (contiguous_layout_) {
    // For contiguous layout, we use compound page which groups all layers
    // together for a single page. num_kv_buffers is 2 for MHA (K+V) and
    // 1 for MLA (combined KV).
    size_t compound_page_size = kPageSize * num_layers * num_kv_buffers;
    zero_page_ = make_shared_page(dev_, ZERO_PAGE_ID, compound_page_size);
    // We can use the aligned size directly for contiguous layout too because
    // both compound_page_size and aligned_size are already/will be multiplied
    // by num_layers.
    return create_kv_tensors_contiguous_(aligned_size, dtype, dev_str,
                                         num_layers, compound_page_size);
  } else {
    zero_page_ = make_shared_page(dev_, ZERO_PAGE_ID);
    return create_kv_tensors_per_layer_(kv_prefix, aligned_size, dtype, dev_str,
                                        num_layers);
  }
}

bool FTensorAllocator::kv_tensors_created() {
  std::lock_guard<std::mutex> lock(mtx_);
  return num_layers_ > 0;
}

bool FTensorAllocator::map_to_kv_tensors(const std::vector<offset_t> &offsets) {
  std::unique_lock<std::mutex> lock(mtx_);
  if (num_layers_ == 0) {
    LOGE("try to map to KV tensors when KV tensors are not created");
    return false;
  }

  if (contiguous_layout_) {
    // In contiguous layout, use the single contiguous tensor for mapping
    // Each offset maps a block that contains all layers
    auto ftensor = contiguous_kv_tensor_.get();
    auto tensor = ftensor->get_tensor();

    for (auto offset : offsets) {
      // Map K and V regions for this block (covers all layers)
      ftensor->map(offset);
    }
  } else {
    // Original per-layer mapping
    for (int64_t i = 0; i < num_layers_; i++) {
      auto kv_name = std::string(kv_prefix) + std::to_string(i);
      auto ftensor = ftensors_[kv_name].get();
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
    const std::vector<offset_t> &offsets) {
  std::unique_lock<std::mutex> lock(mtx_);
  if (num_layers_ == 0) {
    LOGE("try to unmap from KV tensors when KV tensors are not created");
    return false;
  }

  if (contiguous_layout_) {
    // In contiguous layout, unmap using the single contiguous tensor
    auto ftensor = contiguous_kv_tensor_.get();
    auto tensor = ftensor->get_tensor();

    for (auto offset : offsets) {
      // Unmap K and V regions for this block (covers all layers)
      ftensor->unmap(offset);
    }
  } else {
    // Original per-layer unmapping
    for (int64_t i = 0; i < num_layers_; i++) {
      auto kv_name = std::string(kv_prefix) + std::to_string(i);
      auto ftensor = ftensors_[kv_name].get();
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

std::vector<at::Tensor> FTensorAllocator::create_kv_tensors_per_layer_(
    std::string_view prefix, size_t size, c10::ScalarType dtype,
    const std::string &dev_str, int64_t num_layers) {
  std::vector<at::Tensor> ftensors;
  for (int64_t i = 0; i < num_layers; i++) {
    auto name = std::string(prefix) + std::to_string(i);
    auto tensor = create_ftensor_(size, dtype, dev_str, name);
    ftensors.push_back(tensor);
  }
  return ftensors;
}

std::vector<at::Tensor> FTensorAllocator::create_kv_tensors_contiguous_(
    size_t size, c10::ScalarType dtype, const std::string &dev_str,
    int64_t num_layers, size_t compound_page_size) {
  // In contiguous layout, Python passes per-layer size, and we multiply by
  // num_layers to get total size
  size_t total_kv_size = size * num_layers;

  // Create the single contiguous KV tensor (contains K and V for all layers)
  auto contiguous_name = std::string(kv_prefix) + "contiguous";
  contiguous_kv_tensor_ =
      std::make_unique<FTensor>(contiguous_name, total_kv_size, dtype, dev_,
                                zero_page_, compound_page_size);

  // Get the contiguous tensor
  auto contiguous_tensor = contiguous_kv_tensor_->get_tensor();
  return {contiguous_tensor};
}

/** this function is not thread-safe */
at::Tensor FTensorAllocator::create_ftensor_(size_t size, c10::ScalarType dtype,
                                             const std::string &dev_str,
                                             std::string name) {
  if (name.empty())
    name = get_anon_tensor_name_();

  if (ftensors_.find(name) != ftensors_.end()) {
    auto tensor = ftensors_[name].get()->get_tensor();
    assert(tensor.numel() * tensor.element_size() == size);
    assert(tensor.device() == c10::Device(dev_str));
    return tensor;
  }

  // Create a new FTensor
  ftensors_[name] =
      std::make_unique<FTensor>(name, size, dtype, dev_, zero_page_);
  return ftensors_[name]->get_tensor();
}

/** this function is not thread-safe */
void FTensorAllocator::free_ftensor_(at::Tensor &ftensor) {
  auto name = ftensor.name();
  if (ftensors_.find(name) == ftensors_.end()) {
    return;
  }
  ftensors_.erase(name);
}

void FTensorAllocator::init_gpu_() {
  CHECK_GPU(gpu_vmm::initialize_runtime());

  int dev_idx = dev_.index() >= 0 ? dev_.index() : gpu_vmm::current_device();
  CHECK_GPU(gpu_vmm::set_device(dev_idx));

  int supports_vmm = 0;
  CHECK_GPU(gpu_vmm::get_vmm_support(&supports_vmm, dev_idx));
  ASSERT(supports_vmm != 0,
         "VMM is not supported on %s device %d. kvcached requires GPU VMM "
         "support.",
         gpu_vmm::backend_name(), dev_idx);

  auto prop = gpu_vmm::make_pinned_device_allocation_prop(dev_idx);
  size_t chunk_sz = 0;
  CHECK_GPU(gpu_vmm::get_allocation_granularity(&chunk_sz, &prop));
  ASSERT(kPageSize % chunk_sz == 0,
         "Invalid page size: %lu must be a multiple of %s granularity %lu\n",
         kPageSize, gpu_vmm::backend_name(), chunk_sz);
}

} // namespace kvcached
