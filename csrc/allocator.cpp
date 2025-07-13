#include <memory>
#include <fcntl.h>
#include <sys/mman.h>
#include <torch/extension.h>

#include "allocator.hpp"
#include "constants.hpp"
#include "cuda_utils.hpp"
#include "ftensor.hpp"
#include "page.hpp"
#include "torch_utils.hpp"

namespace kvcached {
std::unique_ptr<FTensorAllocator> FTensorAllocator::g_allocator_;

static inline std::shared_ptr<Page> make_shared_page(const torch::Device &dev,
                                                     page_id_t page_id) {
  if (dev.is_cuda()) {
    return std::make_shared<GPUPage>(page_id, dev.index());
  } else if (dev.is_cpu()) {
    return std::make_shared<CPUPage>(page_id);
  }
  ASSERT(false, "Unsupported device type.");
  return nullptr;
}

static inline void* alloc_virtual_mem(const torch::Device &dev, size_t size) {
  ASSERT(size % kPageSize == 0, "alloc size not aligned."); // Ensure alignment.

  void* vaddr;
  if (dev.is_cuda()) {
    CHECK_DRV(cuMemAddressReserve(reinterpret_cast<CUdeviceptr *>(&vaddr), size,
                                  kPageSize, 0ULL, 0ULL));
  } else {
    vaddr = mmap(nullptr, size, PROT_READ | PROT_WRITE,
                 MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    ASSERT(vaddr != MAP_FAILED, "mmap failed.");
  }
  return vaddr;
}

static inline bool map_page(Page *page, offset_t offset, bool set_access, void* vaddr_base) {
  assert(offset % kPageSize == 0); // Ensure alignment.
  assert(page);
  auto vaddr =
      reinterpret_cast<void *>(reinterpret_cast<uintptr_t>(vaddr_base) + offset);
  return page->map(vaddr, set_access);
}

static inline bool init_with_zero(void* vaddr, size_t size, 
                                  std::shared_ptr<Page> zero_page) {
  assert(reinterpret_cast<uintptr_t>(vaddr) % kPageSize == 0); // Ensure alignment.
  assert(size % kPageSize == 0);  // Ensure alignment.

  bool succ = true;
  for (size_t offset = 0; offset < size; offset += kPageSize) {
    if (!map_page(zero_page.get(), offset, /* set_access = */ true, vaddr)) {
      succ = false;
      break;
    }
  }
  return succ;
}

FTensorAllocator::FTensorAllocator(size_t total_mem_size, const torch::Device &device, bool single_vaddr)
    : total_mem_size_(total_mem_size), dev_(device), num_layers_(0), single_vaddr_(single_vaddr), vaddr_(nullptr) {

  if (dev_.is_cuda()) {
    init_cuda_();
  }
  zero_page_ = make_shared_page(dev_, ZERO_PAGE_ID);
  
  if (single_vaddr_) {
    vaddr_ = alloc_virtual_mem(dev_, total_mem_size_);
    init_with_zero(vaddr_, total_mem_size_, zero_page_);
  }
}

FTensorAllocator::~FTensorAllocator() { destroy(); }

void FTensorAllocator::destroy() {
  ftensors_.clear();
  zero_page_.reset();
  if (vaddr_ && single_vaddr_) {
    CHECK_DRV(cuMemUnmap(reinterpret_cast<CUdeviceptr>(vaddr_), total_mem_size_));
    CHECK_DRV(cuMemAddressFree(reinterpret_cast<CUdeviceptr>(vaddr_), total_mem_size_));
  }
}

/* FIXME (YIFAN): this is not thread safe. */
void FTensorAllocator::init(size_t total_mem_size, const std::string &dev_str, bool single_vaddr) {
  if (g_allocator_) {
    LOGE("FTensorAllocator has been initialized. Re-initializing...")
    g_allocator_.reset();
  }

  auto device = torch::Device(dev_str);
  g_allocator_ = std::make_unique<FTensorAllocator>(total_mem_size, device, single_vaddr);
}

FTensorAllocator *FTensorAllocator::global_allocator() {
  assert(g_allocator_);
  return g_allocator_.get();
}

void FTensorAllocator::shutdown() {
  if (g_allocator_) {
    g_allocator_.reset();
  }
}

torch::Tensor FTensorAllocator::create_ftensor(size_t size, torch::Dtype dtype,
                                               const std::string &dev_str,
                                               std::string name,
                                               void* vaddr) {
  if (name.empty())
    name = get_anon_tensor_name_();

  if (ftensors_.find(name) != ftensors_.end()) {
    auto tensor = ftensors_[name].get()->get_tensor();
    assert(tensor.numel() * tensor.element_size() == size);
    assert(tensor.device() == torch::Device(dev_str));
    return tensor;
  }
  // Create a new FTensor.
  if (vaddr != nullptr) {
    ftensors_[name] =
        std::make_unique<FTensor>(name, size, dtype, dev_, zero_page_, vaddr);
  } else {
    ftensors_[name] =
        std::make_unique<FTensor>(name, size, dtype, dev_, zero_page_);
  }
  return ftensors_[name]->get_tensor();
}

void FTensorAllocator::free_ftensor(torch::Tensor &ftensor) {
  auto name = ftensor.name();
  if (ftensors_.find(name) == ftensors_.end()) {
    return;
  }
  ftensors_.erase(name);
}

std::vector<torch::Tensor>
FTensorAllocator::create_kv_tensors(size_t size, torch::Dtype dtype,
                                    const std::string &dev_str,
                                    int64_t num_layers) {
  assert(num_layers_ == 0 || num_layers_ == num_layers);
  num_layers_ = num_layers;
  if (single_vaddr_) {
    return create_kv_tensors_impl_(kv_prefix, size, dtype, dev_str, num_layers, vaddr_);
  } else {
    return create_kv_tensors_impl_(kv_prefix, size, dtype, dev_str, num_layers);
  }
}

void FTensorAllocator::free_kv_tensors() {
  ftensors_.clear();
}

bool FTensorAllocator::map_to_kv_tensors(const std::vector<offset_t> &offsets) {
  for (int64_t i = 0; i < num_layers_; i++) {
    auto kv_name = std::string(kv_prefix) + std::to_string(i);
    auto ftensor = ftensors_[kv_name].get();
    /**
     * NOTE: we assume the K tensor and the V tensor are stacked at the 1st
     * dim. This is used for calculating the offset of the V tensor.
     * FIXME: (YIFAN) we may support other KV cache layouts later.
     */
    auto tensor = ftensor->get_tensor();
    auto v_base_offset = (tensor.numel() * tensor.element_size()) / 2;
    for (auto offset : offsets) {
      auto koffset = offset;
      auto voffset = offset + v_base_offset;
      ftensor->map(koffset);
      ftensor->map(voffset);
    }
  }
  return true;
}

bool FTensorAllocator::unmap_from_kv_tensors(
    const std::vector<offset_t> &offsets) {
  for (int64_t i = 0; i < num_layers_; i++) {
    auto kv_name = std::string(kv_prefix) + std::to_string(i);
    auto ftensor = ftensors_[kv_name].get();
    /**
     * NOTE: we assume the K tensor and the V tensor are stacked at the 1st
     * dim. This is used for calculating the offset of the V tensor.
     * FIXME: (YIFAN) we may support other KV cache layouts later.
     */
    auto tensor = ftensor->get_tensor();
    auto v_base_offset = (tensor.numel() * tensor.element_size()) / 2;
    for (auto offset : offsets) {
      ftensor->unmap(offset);
      ftensor->unmap(offset + v_base_offset);
    }
  }
  return true;
}

std::string FTensorAllocator::get_anon_tensor_name_() {
  static constexpr std::string_view prefix = "anon_tensor_";
  static std::atomic<int> counter(0);
  return std::string(prefix) + std::to_string(counter++);
}

std::vector<torch::Tensor> FTensorAllocator::create_kv_tensors_impl_(
    std::string_view prefix, size_t size, torch::Dtype dtype,
    const std::string &dev_str, int64_t num_layers) {
  std::vector<torch::Tensor> ftensors;
  for (int64_t i = 0; i < num_layers; i++) {
    auto name = std::string(prefix) + std::to_string(i);
    auto tensor = create_ftensor(size, dtype, dev_str, name);
    ftensors.push_back(tensor);
  }
  return ftensors;
}

std::vector<torch::Tensor> FTensorAllocator::create_kv_tensors_impl_(
    std::string_view prefix, size_t size, torch::Dtype dtype,
    const std::string &dev_str, int64_t num_layers, generic_ptr_t vaddr) {
  std::vector<torch::Tensor> ftensors;
  size_t vaddr_offset = 0;
  for (int64_t i = 0; i < num_layers; i++) {
    auto name = std::string(prefix) + std::to_string(i);
    auto offset_vaddr = reinterpret_cast<generic_ptr_t>(reinterpret_cast<uintptr_t>(vaddr) + vaddr_offset);
    auto tensor = create_ftensor(size, dtype, dev_str, name, offset_vaddr);
    ftensors.push_back(tensor);
    vaddr_offset += (size + kPageSize - 1) / kPageSize * kPageSize; // Round up to the nearest page.
  }
  return ftensors;
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
  ASSERT(chunk_sz == kPageSize, "Invalid page size, %lu should be %lu %lu %d\n",
         kPageSize, chunk_sz, chunk_sz - kPageSize, chunk_sz == kPageSize);
}

} // namespace kvcached