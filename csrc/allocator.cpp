// SPDX-FileCopyrightText: Copyright contributors to the kvcached project
// SPDX-License-Identifier: Apache-2.0

#include <memory>
#include <mutex>
#include <unordered_map>
#include <utility>
#include <vector>

#include "allocator.hpp"
#include "constants.hpp"
#include "ftensor.hpp"
#include "gpu_utils.hpp"
#include "page.hpp"

namespace kvcached {
// Global configurable page size
size_t kPageSize = 2 * 1024 * 1024; // Default 2MB

std::unordered_map<int64_t, std::unique_ptr<FTensorAllocator>>
    FTensorAllocator::g_allocators_;
std::mutex FTensorAllocator::g_allocator_mutex_;
c10::Device FTensorAllocator::g_device_(c10::kCPU);
bool FTensorAllocator::g_contiguous_layout_ = false;

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
      unified_pool_(false), kv_tensor_size_per_layer_(0) {
  if (dev_.is_cuda()) {
    init_gpu_();
  }
}

FTensorAllocator::~FTensorAllocator() { destroy(); }

void FTensorAllocator::destroy() {
  std::lock_guard<std::mutex> lock(mtx_);
  pending_unmap_.reset();
  finalized_unmap_transactions_.clear();
  finalized_unmap_order_.clear();
  ftensors_.clear();
  contiguous_kv_tensor_.reset();
  zero_page_.reset();
}

void FTensorAllocator::init(const std::string &dev_str, size_t page_size,
                            bool contiguous_layout) {
  std::lock_guard<std::mutex> lock(g_allocator_mutex_);
  if (!g_allocators_.empty()) {
    LOGGER(ERROR, "FTensorAllocator has been initialized. Re-initializing...");
    g_allocators_.clear();
  }

  // Set global page size if provided (0 means use default)
  if (page_size > 0) {
    // Validate that page_size is a multiple of 2MB
    size_t base_size = 2 * 1024 * 1024; // 2MB
    if (page_size % base_size != 0) {
      LOGGER(
          ERROR,
          "Invalid page size: %zu, must be a multiple of 2MB (2097152 bytes)",
          page_size);
      abort();
    }
    kPageSize = page_size;
  }

  auto device = c10::Device(dev_str);
  g_device_ = device;
  g_contiguous_layout_ = contiguous_layout;
  g_allocators_[0] =
      std::make_unique<FTensorAllocator>(device, contiguous_layout);
}

FTensorAllocator *FTensorAllocator::global_allocator(int64_t group_id) {
  std::lock_guard<std::mutex> lock(g_allocator_mutex_);
  auto it = g_allocators_.find(group_id);
  if (it == g_allocators_.end()) {
    // Lazily create a new allocator for this group,
    // using the device/layout from init().
    assert(!g_allocators_.empty() &&
           "FTensorAllocator::init() must be called first");
    g_allocators_[group_id] =
        std::make_unique<FTensorAllocator>(g_device_, g_contiguous_layout_);
    return g_allocators_[group_id].get();
  }
  return it->second.get();
}

void FTensorAllocator::shutdown() {
  std::lock_guard<std::mutex> lock(g_allocator_mutex_);
  g_allocators_.clear();
}

std::vector<at::Tensor> FTensorAllocator::create_kv_tensors(
    size_t size, c10::ScalarType dtype, const std::string &dev_str,
    int64_t num_layers, int64_t num_kv_buffers, bool unified_pool) {
  std::lock_guard<std::mutex> lock(mtx_);

  assert(num_layers_ == 0 || num_layers_ == num_layers);
  num_layers_ = num_layers;
  unified_pool_ = unified_pool;
  // Ensure size is aligned to page size.
  size_t aligned_size = size;
  if (size % kPageSize != 0) {
    aligned_size = ((size + kPageSize - 1) / kPageSize) * kPageSize;
    LOGGER(WARNING, "Size %zu is not aligned to page size %zu, aligning to %zu",
           size, kPageSize, aligned_size);
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
  return map_to_kv_tensors_with_result(offsets).first;
}

std::pair<bool, std::vector<offset_t>>
FTensorAllocator::map_to_kv_tensors_with_result(
    const std::vector<offset_t> &offsets) {
  std::unique_lock<std::mutex> lock(mtx_);
  if (num_layers_ == 0) {
    LOGGER(ERROR, "try to map to KV tensors when KV tensors are not created");
    return {false, {}};
  }
  reject_if_unmap_pending_locked_("map KV tensors");

  using MappingTarget = std::pair<FTensor *, offset_t>;
  struct MappingGroup {
    offset_t logical_offset;
    std::vector<MappingTarget> targets;
  };

  std::vector<MappingGroup> groups;
  groups.reserve(offsets.size());
  for (auto offset : offsets) {
    MappingGroup group{offset, {}};
    if (contiguous_layout_) {
      group.targets.emplace_back(contiguous_kv_tensor_.get(), offset);
    } else {
      for (int64_t i = 0; i < num_layers_; i++) {
        auto kv_name = std::string(kv_prefix) + std::to_string(i);
        auto ftensor = ftensors_[kv_name].get();
        group.targets.emplace_back(ftensor, offset);
        if (!unified_pool_) {
          auto v_base_offset = get_v_base_offset(ftensor->get_tensor());
          group.targets.emplace_back(ftensor, offset + v_base_offset);
        }
      }
    }
    groups.push_back(std::move(group));
  }

  std::vector<MappingTarget> mapped;
  std::vector<offset_t> newly_mapped_offsets;
  try {
    for (const auto &group : groups) {
      size_t existing = 0;
      for (const auto &[ftensor, offset] : group.targets) {
        existing += ftensor->is_mapped_(offset) ? 1 : 0;
      }
      if (existing == group.targets.size()) {
        continue;
      }
      if (existing != 0) {
        throw std::runtime_error("state_inconsistency: logical KV offset is "
                                 "only partially mapped: " +
                                 std::to_string(group.logical_offset));
      }

      for (const auto &[ftensor, offset] : group.targets) {
        if (!ftensor->map(offset)) {
          throw std::runtime_error("physical page map returned false");
        }
        mapped.emplace_back(ftensor, offset);
      }
      newly_mapped_offsets.push_back(group.logical_offset);
    }
  } catch (const std::exception &error) {
    std::vector<std::string> rollback_errors;
    for (auto it = mapped.rbegin(); it != mapped.rend(); ++it) {
      try {
        if (!it->first->unmap(it->second)) {
          rollback_errors.emplace_back("offset " + std::to_string(it->second) +
                                       " returned false");
        }
      } catch (const std::exception &rollback_error) {
        rollback_errors.emplace_back("offset " + std::to_string(it->second) +
                                     ": " + rollback_error.what());
      }
    }
    if (!rollback_errors.empty()) {
      std::string message =
          std::string("state_inconsistency: KV map failed: ") + error.what() +
          "; rollback failed: ";
      for (size_t i = 0; i < rollback_errors.size(); ++i) {
        if (i != 0) {
          message += "; ";
        }
        message += rollback_errors[i];
      }
      throw std::runtime_error(message);
    }
    throw;
  }
  return {true, std::move(newly_mapped_offsets)};
}

bool FTensorAllocator::unmap_from_kv_tensors(
    const std::vector<offset_t> &offsets) {
  std::unique_lock<std::mutex> lock(mtx_);
  if (num_layers_ == 0) {
    LOGGER(ERROR,
           "try to unmap from KV tensors when KV tensors are not created");
    return false;
  }
  reject_if_unmap_pending_locked_("unmap KV tensors");
  unmap_retain_locked_(offsets);
  return true;
}

bool FTensorAllocator::prepare_unmap_from_kv_tensors(
    const std::vector<offset_t> &offsets, const std::string &transaction_id) {
  std::unique_lock<std::mutex> lock(mtx_);
  if (transaction_id.empty()) {
    throw std::invalid_argument("unmap transaction id must not be empty");
  }
  if (num_layers_ == 0) {
    LOGGER(ERROR, "try to prepare unmap when KV tensors are not created");
    return false;
  }
  if (pending_unmap_) {
    if (pending_unmap_->id == transaction_id) {
      return true;
    }
    throw std::runtime_error(
        "state_inconsistency: another unmap transaction is pending: " +
        pending_unmap_->id);
  }
  auto finalized = finalized_unmap_transactions_.find(transaction_id);
  if (finalized != finalized_unmap_transactions_.end()) {
    if (finalized->second == UnmapTransactionOutcome::COMMITTED) {
      return true;
    }
    throw std::runtime_error("unmap transaction was already aborted: " +
                             transaction_id);
  }

  pending_unmap_.emplace(
      PendingUnmapTransaction{transaction_id, unmap_retain_locked_(offsets)});
  return true;
}

bool FTensorAllocator::commit_unmap_from_kv_tensors(
    const std::string &transaction_id) {
  std::unique_lock<std::mutex> lock(mtx_);
  if (transaction_id.empty()) {
    throw std::invalid_argument("unmap transaction id must not be empty");
  }
  if (pending_unmap_ && pending_unmap_->id == transaction_id) {
    remember_unmap_outcome_locked_(transaction_id,
                                   UnmapTransactionOutcome::COMMITTED);
    pending_unmap_.reset();
    return true;
  }
  if (pending_unmap_) {
    throw std::runtime_error("state_inconsistency: commit does not match "
                             "pending unmap transaction " +
                             pending_unmap_->id);
  }
  auto finalized = finalized_unmap_transactions_.find(transaction_id);
  if (finalized != finalized_unmap_transactions_.end() &&
      finalized->second == UnmapTransactionOutcome::COMMITTED) {
    return true;
  }
  if (finalized != finalized_unmap_transactions_.end()) {
    throw std::runtime_error("cannot commit an aborted unmap transaction: " +
                             transaction_id);
  }
  throw std::runtime_error("unknown unmap transaction: " + transaction_id);
}

bool FTensorAllocator::abort_unmap_from_kv_tensors(
    const std::string &transaction_id) {
  std::unique_lock<std::mutex> lock(mtx_);
  if (transaction_id.empty()) {
    throw std::invalid_argument("unmap transaction id must not be empty");
  }
  if (pending_unmap_ && pending_unmap_->id == transaction_id) {
    restore_retained_locked_(pending_unmap_->retained,
                             "distributed unmap aborted");
    pending_unmap_.reset();
    remember_unmap_outcome_locked_(transaction_id,
                                   UnmapTransactionOutcome::ABORTED);
    return true;
  }
  if (pending_unmap_) {
    throw std::runtime_error(
        "state_inconsistency: abort does not match pending unmap transaction " +
        pending_unmap_->id);
  }
  auto finalized = finalized_unmap_transactions_.find(transaction_id);
  if (finalized != finalized_unmap_transactions_.end()) {
    if (finalized->second == UnmapTransactionOutcome::ABORTED) {
      return true;
    }
    throw std::runtime_error("cannot abort a committed unmap transaction: " +
                             transaction_id);
  }

  // Prepare may not have reached this worker. Recording the abort prevents a
  // delayed prepare from reopening the transaction.
  remember_unmap_outcome_locked_(transaction_id,
                                 UnmapTransactionOutcome::ABORTED);
  return true;
}

std::vector<FTensorAllocator::RetainedMapping>
FTensorAllocator::unmap_retain_locked_(const std::vector<offset_t> &offsets) {

  using MappingTarget = std::pair<FTensor *, offset_t>;
  struct MappingGroup {
    offset_t logical_offset;
    std::vector<MappingTarget> targets;
  };
  std::vector<MappingGroup> groups;
  groups.reserve(offsets.size());
  for (auto offset : offsets) {
    MappingGroup group{offset, {}};
    if (contiguous_layout_) {
      group.targets.emplace_back(contiguous_kv_tensor_.get(), offset);
    } else {
      for (int64_t i = 0; i < num_layers_; i++) {
        auto kv_name = std::string(kv_prefix) + std::to_string(i);
        auto ftensor = ftensors_[kv_name].get();
        group.targets.emplace_back(ftensor, offset);
        if (!unified_pool_) {
          auto v_base_offset = get_v_base_offset(ftensor->get_tensor());
          group.targets.emplace_back(ftensor, offset + v_base_offset);
        }
      }
    }
    groups.push_back(std::move(group));
  }

  std::vector<RetainedMapping> retained;
  try {
    for (const auto &group : groups) {
      size_t existing = 0;
      for (const auto &[ftensor, offset] : group.targets) {
        existing += ftensor->is_mapped_(offset) ? 1 : 0;
      }
      if (existing == 0) {
        continue;
      }
      if (existing != group.targets.size()) {
        throw std::runtime_error("state_inconsistency: logical KV offset is "
                                 "only partially mapped: " +
                                 std::to_string(group.logical_offset));
      }

      for (const auto &[ftensor, offset] : group.targets) {
        std::unique_ptr<Page> page;
        if (!ftensor->unmap_retain_(offset, page) || !page) {
          throw std::runtime_error("physical page unmap returned no page");
        }
        retained.push_back({ftensor, offset, std::move(page)});
      }
    }
  } catch (const std::exception &error) {
    restore_retained_locked_(retained, error.what());
    throw;
  }
  return retained;
}

void FTensorAllocator::restore_retained_locked_(
    std::vector<RetainedMapping> &retained, const std::string &original_error) {
  std::vector<std::string> rollback_errors;
  for (auto it = retained.rbegin(); it != retained.rend(); ++it) {
    if (!it->page) {
      continue;
    }
    try {
      if (!it->ftensor->restore_mapping_(it->offset, it->page)) {
        rollback_errors.emplace_back("offset " + std::to_string(it->offset) +
                                     " returned false");
      }
    } catch (const std::exception &rollback_error) {
      rollback_errors.emplace_back("offset " + std::to_string(it->offset) +
                                   ": " + rollback_error.what());
    }
  }
  if (!rollback_errors.empty()) {
    std::string message =
        "state_inconsistency: KV unmap failed: " + original_error +
        "; rollback failed: ";
    for (size_t i = 0; i < rollback_errors.size(); ++i) {
      if (i != 0) {
        message += "; ";
      }
      message += rollback_errors[i];
    }
    throw std::runtime_error(message);
  }
}

void FTensorAllocator::remember_unmap_outcome_locked_(
    const std::string &transaction_id, UnmapTransactionOutcome outcome) {
  constexpr size_t max_finalized_transactions = 64;
  auto [it, inserted] =
      finalized_unmap_transactions_.insert_or_assign(transaction_id, outcome);
  (void)it;
  if (inserted) {
    finalized_unmap_order_.push_back(transaction_id);
  }
  while (finalized_unmap_order_.size() > max_finalized_transactions) {
    finalized_unmap_transactions_.erase(finalized_unmap_order_.front());
    finalized_unmap_order_.pop_front();
  }
}

void FTensorAllocator::reject_if_unmap_pending_locked_(
    const char *operation) const {
  if (pending_unmap_) {
    throw std::runtime_error(
        std::string("cannot ") + operation +
        " while unmap transaction is pending: " + pending_unmap_->id);
  }
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
