// SPDX-FileCopyrightText: Copyright contributors to the kvcached project
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <cctype>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <limits>
#include <memory>
#include <mutex>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#include <dirent.h>
#include <fcntl.h>
#include <sys/file.h>
#include <unistd.h>

#include "allocator.hpp"
#include "constants.hpp"
#include "ftensor.hpp"
#include "gpu_utils.hpp"
#include "page.hpp"

namespace kvcached {
namespace {

int resolve_device_index(const c10::Device &device) {
  return device.index() >= 0 ? device.index() : gpu_vmm::current_device();
}

double gpu_utilization_limit() {
  constexpr double kDefaultUtilization = 0.95;
  const char *value = std::getenv("KVCACHED_GPU_UTILIZATION");
  if (value == nullptr || *value == '\0') {
    return kDefaultUtilization;
  }

  char *end = nullptr;
  errno = 0;
  const double parsed = std::strtod(value, &end);
  if (errno != 0 || end == value || *end != '\0' || !std::isfinite(parsed) ||
      parsed <= 0.0 || parsed > 1.0) {
    throw std::runtime_error(
        "KVCACHED_GPU_UTILIZATION must be in the range (0, 1]");
  }
  return parsed;
}

std::string physical_growth_lock_dir() {
  const char *configured_dir = std::getenv("KVCACHED_PHYSICAL_GROWTH_LOCK_DIR");
  std::string lock_dir = configured_dir != nullptr && *configured_dir != '\0'
                             ? configured_dir
                             : "/tmp";
  while (lock_dir.size() > 1 && lock_dir.back() == '/') {
    lock_dir.pop_back();
  }
  return lock_dir;
}

std::string sanitize_physical_device(std::string key) {
  if (key.empty()) {
    throw std::invalid_argument("physical GPU identifier must not be empty");
  }
  for (char &value : key) {
    if (!std::isalnum(static_cast<unsigned char>(value))) {
      value = '_';
    }
  }
  return key;
}

std::string physical_gpu_lock_path(const std::string &physical_device) {
  return physical_growth_lock_dir() + "/kvcached-physical-growth-" +
         sanitize_physical_device(physical_device) + ".lock";
}

std::string physical_gpu_identifier(int dev_idx) {
  char pci_bus_id[64] = {};
  const auto status = gpu_vmm::device_get_pci_bus_id(
      pci_bus_id, static_cast<int>(sizeof(pci_bus_id)), dev_idx);
  if (!gpu_vmm::is_success(status)) {
    throw std::runtime_error(std::string("failed to resolve physical GPU: ") +
                             gpu_vmm::error_string(status));
  }
  return std::string(pci_bus_id);
}

std::string physical_gpu_lock_path(int dev_idx) {
  return physical_gpu_lock_path(physical_gpu_identifier(dev_idx));
}

void validate_physical_capacity(int dev_idx, size_t required_bytes,
                                PhysicalGrowthOperationStats *stats) {
  auto status = gpu_vmm::set_device(dev_idx);
  if (!gpu_vmm::is_success(status)) {
    throw std::runtime_error(std::string("failed to select physical GPU: ") +
                             gpu_vmm::error_string(status));
  }
  size_t free_bytes = 0;
  size_t total_bytes = 0;
  status = gpu_vmm::mem_get_info(&free_bytes, &total_bytes);
  if (!gpu_vmm::is_success(status)) {
    throw std::runtime_error(std::string("GPU memory query failed: ") +
                             gpu_vmm::error_string(status));
  }
  const size_t headroom = static_cast<size_t>(static_cast<double>(total_bytes) *
                                              (1.0 - gpu_utilization_limit()));
  const size_t usable_bytes = free_bytes > headroom ? free_bytes - headroom : 0;
  if (stats != nullptr) {
    stats->capacity_checks = 1;
    stats->required_bytes = static_cast<uint64_t>(required_bytes);
    stats->free_bytes = static_cast<uint64_t>(free_bytes);
    stats->total_bytes = static_cast<uint64_t>(total_bytes);
    stats->headroom_bytes = static_cast<uint64_t>(headroom);
    stats->usable_bytes = static_cast<uint64_t>(usable_bytes);
  }
  if (required_bytes > usable_bytes) {
    const size_t shortfall = required_bytes - usable_bytes;
    if (stats != nullptr) {
      stats->capacity_rejections = 1;
      stats->shortfall_bytes = static_cast<uint64_t>(shortfall);
    }
    throw std::runtime_error(
        "capacity_exhausted: physical GPU headroom would be crossed");
  }
}

class FairFileTicket {
public:
  explicit FairFileTicket(const std::string &resource_path)
      : queue_path_(resource_path + ".queue"), ticket_fd_(-1),
        ticket_number_(0), owner_token_(process_owner_token()),
        turn_acquired_(false), wait_us_(0) {
    const auto wait_started = std::chrono::steady_clock::now();
    const int queue_fd = open(queue_path_.c_str(),
                              O_CREAT | O_CLOEXEC | O_NOFOLLOW | O_RDWR, 0660);
    if (queue_fd < 0) {
      throw std::runtime_error("failed to open physical growth ticket queue " +
                               queue_path_ + ": " + std::strerror(errno));
    }

    try {
      issue_ticket(queue_fd, resource_path);
    } catch (...) {
      cleanup_ticket();
      (void)close(queue_fd);
      throw;
    }

    (void)close(queue_fd);
    try {
      wait_until_first(resource_path);
      turn_acquired_ = true;
      wait_us_ = elapsed_us(wait_started);
    } catch (...) {
      cleanup_ticket();
      throw;
    }
  }

  FairFileTicket(const FairFileTicket &) = delete;
  FairFileTicket &operator=(const FairFileTicket &) = delete;

  ~FairFileTicket() {
    if (turn_acquired_) {
      record_last_owner();
    }
    cleanup_ticket();
  }

  uint64_t wait_us() const { return wait_us_; }

private:
  struct QueueState {
    uint64_t next_ticket = 0;
    uint64_t last_owner = 0;
  };

  static uint64_t process_owner_token() {
    static const uint64_t token = [] {
      std::random_device random;
      const uint64_t high = static_cast<uint64_t>(random()) << 32;
      const uint64_t low = static_cast<uint64_t>(random());
      return (high | low) ^ static_cast<uint64_t>(getpid());
    }();
    return token;
  }

  static uint64_t
  elapsed_us(const std::chrono::steady_clock::time_point &started) {
    return static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now() - started)
            .count());
  }

  QueueState read_queue_state(int queue_fd) const {
    QueueState state;
    const ssize_t read_bytes = pread(queue_fd, &state, sizeof(state), 0);
    if (read_bytes < 0 ||
        (read_bytes != 0 &&
         read_bytes != static_cast<ssize_t>(sizeof(uint64_t)) &&
         read_bytes != static_cast<ssize_t>(sizeof(state)))) {
      throw std::runtime_error("failed to read physical growth ticket queue " +
                               queue_path_);
    }
    // The previous format contained only next_ticket. Queue files survive
    // process upgrades, so a legacy 8-byte record keeps last_owner unset.
    if (read_bytes == static_cast<ssize_t>(sizeof(uint64_t))) {
      state.last_owner = 0;
    }
    return state;
  }

  void write_queue_state(int queue_fd, const QueueState &state) const {
    if (pwrite(queue_fd, &state, sizeof(state), 0) !=
            static_cast<ssize_t>(sizeof(state)) ||
        ftruncate(queue_fd, sizeof(state)) != 0) {
      throw std::runtime_error(
          "failed to update physical growth ticket queue " + queue_path_);
    }
  }

  void lock_queue(int queue_fd) const {
    if (flock(queue_fd, LOCK_EX) != 0) {
      throw std::runtime_error("failed to lock physical growth ticket queue " +
                               queue_path_ + ": " + std::strerror(errno));
    }
  }

  void issue_ticket(int queue_fd, const std::string &resource_path) {
    bool yielded_to_peer = false;
    while (true) {
      lock_queue(queue_fd);
      QueueState state;
      try {
        state = read_queue_state(queue_fd);
      } catch (...) {
        (void)flock(queue_fd, LOCK_UN);
        throw;
      }

      if (!yielded_to_peer && state.last_owner == owner_token_) {
        (void)flock(queue_fd, LOCK_UN);
        std::this_thread::sleep_for(std::chrono::microseconds(200));
        yielded_to_peer = true;
        continue;
      }
      if (state.next_ticket == std::numeric_limits<uint64_t>::max()) {
        (void)flock(queue_fd, LOCK_UN);
        throw std::overflow_error("physical growth ticket counter overflow");
      }

      ticket_number_ = state.next_ticket++;
      try {
        write_queue_state(queue_fd, state);
        ticket_path_ = ticket_path(resource_path, ticket_number_);
        ticket_fd_ =
            open(ticket_path_.c_str(),
                 O_CREAT | O_EXCL | O_CLOEXEC | O_NOFOLLOW | O_RDWR, 0660);
        if (ticket_fd_ < 0) {
          throw std::runtime_error("failed to create physical growth ticket " +
                                   ticket_path_ + ": " + std::strerror(errno));
        }
        if (flock(ticket_fd_, LOCK_EX) != 0) {
          throw std::runtime_error("failed to lock physical growth ticket " +
                                   ticket_path_ + ": " + std::strerror(errno));
        }
      } catch (...) {
        (void)flock(queue_fd, LOCK_UN);
        throw;
      }
      (void)flock(queue_fd, LOCK_UN);
      return;
    }
  }

  static std::string ticket_path(const std::string &resource_path,
                                 uint64_t number) {
    std::ostringstream path;
    path << resource_path << ".ticket." << std::setw(20) << std::setfill('0')
         << number;
    return path.str();
  }

  void wait_until_first(const std::string &resource_path) {
    const auto slash = resource_path.find_last_of('/');
    const std::string directory =
        slash == std::string::npos ? "." : resource_path.substr(0, slash);
    const std::string prefix =
        (slash == std::string::npos ? resource_path
                                    : resource_path.substr(slash + 1)) +
        ".ticket.";

    while (true) {
      uint64_t first_active = ticket_number_;
      std::string scan_error;
      DIR *dir = opendir(directory.c_str());
      if (dir == nullptr) {
        throw std::runtime_error("failed to scan physical growth tickets in " +
                                 directory + ": " + std::strerror(errno));
      }

      while (dirent *entry = readdir(dir)) {
        const std::string name(entry->d_name);
        if (name.compare(0, prefix.size(), prefix) != 0) {
          continue;
        }
        const std::string suffix = name.substr(prefix.size());
        if (suffix.empty() ||
            !std::all_of(suffix.begin(), suffix.end(), [](unsigned char value) {
              return std::isdigit(value);
            })) {
          continue;
        }

        uint64_t number = 0;
        try {
          number = std::stoull(suffix);
        } catch (const std::exception &) {
          continue;
        }
        if (number >= first_active || number == ticket_number_) {
          continue;
        }

        const std::string path = directory + "/" + name;
        const int fd = open(path.c_str(), O_CLOEXEC | O_NOFOLLOW | O_RDWR);
        if (fd < 0) {
          if (errno == ENOENT) {
            continue;
          }
          scan_error = "failed to open physical growth ticket " + path + ": " +
                       std::strerror(errno);
          break;
        }
        if (flock(fd, LOCK_EX | LOCK_NB) == 0) {
          (void)unlink(path.c_str());
          (void)flock(fd, LOCK_UN);
        } else if (errno == EWOULDBLOCK || errno == EAGAIN) {
          first_active = std::min(first_active, number);
        } else {
          scan_error = "failed to inspect physical growth ticket " + path +
                       ": " + std::strerror(errno);
        }
        (void)close(fd);
        if (!scan_error.empty()) {
          break;
        }
      }
      (void)closedir(dir);

      if (!scan_error.empty()) {
        throw std::runtime_error(scan_error);
      }

      if (first_active == ticket_number_) {
        return;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
  }

  void cleanup_ticket() noexcept {
    if (ticket_fd_ >= 0) {
      (void)flock(ticket_fd_, LOCK_UN);
      (void)close(ticket_fd_);
      ticket_fd_ = -1;
    }
    if (!ticket_path_.empty()) {
      (void)unlink(ticket_path_.c_str());
      ticket_path_.clear();
    }
  }

  void record_last_owner() noexcept {
    const int queue_fd =
        open(queue_path_.c_str(), O_CLOEXEC | O_NOFOLLOW | O_RDWR);
    if (queue_fd < 0 || flock(queue_fd, LOCK_EX) != 0) {
      if (queue_fd >= 0) {
        (void)close(queue_fd);
      }
      return;
    }
    try {
      QueueState state = read_queue_state(queue_fd);
      state.last_owner = owner_token_;
      write_queue_state(queue_fd, state);
    } catch (const std::exception &) {
      // Fairness metadata is advisory; the physical growth lock still
      // preserves correctness if this best-effort update fails.
    }
    (void)flock(queue_fd, LOCK_UN);
    (void)close(queue_fd);
  }

  std::string queue_path_;
  int ticket_fd_;
  uint64_t ticket_number_;
  uint64_t owner_token_;
  bool turn_acquired_;
  uint64_t wait_us_;
  std::string ticket_path_;
};

class PhysicalGrowthGuard {
public:
  PhysicalGrowthGuard(int dev_idx, size_t required_bytes,
                      PhysicalGrowthOperationStats *stats)
      : path_(physical_gpu_lock_path(dev_idx)), ticket_(path_), fd_(-1) {
    if (stats != nullptr) {
      stats->ticket_wait_us = ticket_.wait_us();
    }
    const std::string &path = path_;
    fd_ = open(path.c_str(), O_CREAT | O_CLOEXEC | O_NOFOLLOW | O_RDWR, 0660);
    if (fd_ < 0) {
      throw std::runtime_error("failed to open physical GPU growth lock " +
                               path + ": " + std::strerror(errno));
    }
    if (flock(fd_, LOCK_EX) != 0) {
      const int lock_errno = errno;
      close(fd_);
      fd_ = -1;
      throw std::runtime_error("failed to acquire physical GPU growth lock " +
                               path + ": " + std::strerror(lock_errno));
    }

    try {
      validate_physical_capacity(dev_idx, required_bytes, stats);
    } catch (...) {
      (void)flock(fd_, LOCK_UN);
      (void)close(fd_);
      fd_ = -1;
      throw;
    }
  }

  PhysicalGrowthGuard(const PhysicalGrowthGuard &) = delete;
  PhysicalGrowthGuard &operator=(const PhysicalGrowthGuard &) = delete;

  ~PhysicalGrowthGuard() {
    if (fd_ >= 0) {
      (void)flock(fd_, LOCK_UN);
      (void)close(fd_);
    }
  }

private:
  std::string path_;
  FairFileTicket ticket_;
  int fd_;
};

size_t checked_transaction_bytes(size_t bytes_per_offset, size_t offset_count) {
  if (offset_count != 0 &&
      bytes_per_offset > std::numeric_limits<size_t>::max() / offset_count) {
    throw std::overflow_error("KV mapping transaction size overflow");
  }
  return bytes_per_offset * offset_count;
}

} // namespace

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
      unified_pool_(false), kv_tensor_size_per_layer_(0),
      physical_bytes_per_offset_(0) {
  if (dev_.is_cuda()) {
    init_gpu_();
  }
}

FTensorAllocator::~FTensorAllocator() { destroy(); }

void FTensorAllocator::destroy() {
  std::lock_guard<std::mutex> lock(mtx_);
  prepared_map_transactions_.clear();
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
  if (num_layers <= 0 || num_kv_buffers <= 0) {
    throw std::invalid_argument(
        "num_layers and num_kv_buffers must both be positive");
  }
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
  physical_bytes_per_offset_ = checked_transaction_bytes(
      checked_transaction_bytes(kPageSize, static_cast<size_t>(num_layers)),
      static_cast<size_t>(num_kv_buffers));

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

std::vector<std::pair<FTensor *, offset_t>>
FTensorAllocator::mapping_targets_(const std::vector<offset_t> &offsets) {
  std::vector<std::pair<FTensor *, offset_t>> targets;

  if (contiguous_layout_) {
    // In contiguous layout, use the single contiguous tensor for mapping
    // Each offset maps a block that contains all layers
    auto ftensor = contiguous_kv_tensor_.get();
    auto tensor = ftensor->get_tensor();

    for (auto offset : offsets) {
      // Map K and V regions for this block (covers all layers)
      targets.emplace_back(ftensor, offset);
    }
  } else if (unified_pool_) {
    // Unified pool: K and V share a single block-interleaved FTensor per
    // layer. Each page id maps exactly one VMM page at pid * page_size.
    for (int64_t i = 0; i < num_layers_; i++) {
      auto kv_name = std::string(kv_prefix) + std::to_string(i);
      auto ftensor = ftensors_[kv_name].get();
      for (auto offset : offsets) {
        targets.emplace_back(ftensor, offset);
      }
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
        targets.emplace_back(ftensor, koffset);
        targets.emplace_back(ftensor, voffset);
      }
    }
  }
  return targets;
}

std::vector<FTensorAllocator::ReservedMapping>
FTensorAllocator::reserve_targets_(
    const std::vector<std::pair<FTensor *, offset_t>> &targets,
    const std::vector<offset_t> &offsets, PhysicalGrowthOperationStats *stats,
    bool adopt_existing_mappings) {
  std::vector<ReservedMapping> reserved;
  reserved.reserve(targets.size());
  if (stats != nullptr) {
    stats->targets_count = static_cast<uint64_t>(targets.size());
  }

  std::chrono::steady_clock::time_point reserve_started;
  bool reserve_started_set = false;
  const auto admission_started = std::chrono::steady_clock::now();
  bool admission_finished = false;
  try {
    // Reserve the physical CUDA handles while holding the process-shared fair
    // turn. Mapping the already-resident handles into virtual addresses is
    // local bookkeeping and intentionally happens after this critical section.
    std::unique_ptr<PhysicalGrowthGuard> growth_guard;
    size_t required_bytes = 0;
    for (const auto &[ftensor, offset] : targets) {
      if (adopt_existing_mappings && ftensor->is_mapped(offset)) {
        continue;
      }
      if (required_bytes >
          std::numeric_limits<size_t>::max() - ftensor->page_size_) {
        throw std::overflow_error("KV mapping transaction size overflow");
      }
      required_bytes += ftensor->page_size_;
    }
    if (dev_.is_cuda() && required_bytes != 0) {
      growth_guard = std::make_unique<PhysicalGrowthGuard>(
          resolve_device_index(dev_), required_bytes, stats);
    }
    if (stats != nullptr) {
      stats->admission_us = static_cast<uint64_t>(
          std::chrono::duration_cast<std::chrono::microseconds>(
              std::chrono::steady_clock::now() - admission_started)
              .count());
    }
    admission_finished = true;
    reserve_started = std::chrono::steady_clock::now();
    reserve_started_set = true;
    for (const auto &[ftensor, offset] : targets) {
      if (adopt_existing_mappings && ftensor->is_mapped(offset)) {
        reserved.push_back(ReservedMapping{ftensor, offset, nullptr});
      } else {
        reserved.push_back(
            ReservedMapping{ftensor, offset, ftensor->reserve_page(offset)});
      }
    }
  } catch (const std::exception &error) {
    if (stats != nullptr && !admission_finished) {
      stats->admission_us = static_cast<uint64_t>(
          std::chrono::duration_cast<std::chrono::microseconds>(
              std::chrono::steady_clock::now() - admission_started)
              .count());
    }
    if (stats != nullptr && reserve_started_set) {
      stats->reserve_us = static_cast<uint64_t>(
          std::chrono::duration_cast<std::chrono::microseconds>(
              std::chrono::steady_clock::now() - reserve_started)
              .count());
    }
    throw;
  }
  if (stats != nullptr && reserve_started_set) {
    stats->reserve_us = static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now() - reserve_started)
            .count());
  }

  return reserved;
}

bool FTensorAllocator::map_reserved_targets_(
    std::vector<ReservedMapping> reserved, PhysicalGrowthOperationStats *stats,
    bool rollback_on_failure) {
  std::vector<std::pair<FTensor *, offset_t>> mapped_offsets;
  mapped_offsets.reserve(reserved.size());
  const auto map_started = std::chrono::steady_clock::now();
  try {
    for (auto &mapping : reserved) {
      if (mapping.page == nullptr) {
        if (!mapping.ftensor->is_mapped(mapping.offset)) {
          throw std::runtime_error("Adopted KV page is no longer mapped");
        }
        continue;
      }
      if (!mapping.ftensor->map_reserved(mapping.offset,
                                         std::move(mapping.page))) {
        throw std::runtime_error("Reserved KV page map returned false");
      }
      mapped_offsets.emplace_back(mapping.ftensor, mapping.offset);
    }
  } catch (const std::exception &error) {
    if (stats != nullptr) {
      stats->map_us = static_cast<uint64_t>(
          std::chrono::duration_cast<std::chrono::microseconds>(
              std::chrono::steady_clock::now() - map_started)
              .count());
    }
    if (!rollback_on_failure) {
      // Distributed commits retain mapped pages for reconciliation/adoption.
      throw;
    }
    std::string failure = std::string("KV map failed: ") + error.what();
    for (auto it = mapped_offsets.rbegin(); it != mapped_offsets.rend(); ++it) {
      try {
        if (!it->first->unmap(it->second)) {
          failure += "; rollback returned false at offset " +
                     std::to_string(it->second);
        }
      } catch (const std::exception &rollback_error) {
        failure += "; rollback failed at offset " + std::to_string(it->second) +
                   ": " + rollback_error.what();
      }
    }
    // The failing target may itself have failed to restore its zero page.
    // Preserve this as terminal across pybind, even if earlier targets rolled
    // back.
    throw KVMappingStateError(failure);
  }
  if (stats != nullptr) {
    stats->map_us = static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now() - map_started)
            .count());
  }
  return true;
}

std::pair<bool, PhysicalGrowthOperationStats>
FTensorAllocator::prepare_map_to_kv_tensors(
    const std::string &transaction_id, const std::vector<offset_t> &offsets) {
  std::unique_lock<std::mutex> lock(mtx_);
  PhysicalGrowthOperationStats stats;
  stats.offsets_count = static_cast<uint64_t>(offsets.size());
  if (transaction_id.empty() || num_layers_ == 0 || offsets.empty()) {
    return {false, stats};
  }
  reject_if_unmap_pending_locked_("prepare KV map");
  const auto existing = prepared_map_transactions_.find(transaction_id);
  if (existing != prepared_map_transactions_.end()) {
    return {existing->second.offsets == offsets, stats};
  }

  try {
    auto targets = mapping_targets_(offsets);
    auto reserved = reserve_targets_(targets, offsets, &stats, true);
    prepared_map_transactions_.emplace(
        transaction_id, PreparedMapTransaction{offsets, std::move(reserved)});
    return {true, stats};
  } catch (const std::exception &error) {
    if (stats.capacity_rejections == 0) {
      LOGGER(ERROR, "Failed to prepare physical KV pages: %s", error.what());
    }
    return {false, stats};
  }
}

std::pair<bool, PhysicalGrowthOperationStats>
FTensorAllocator::commit_prepared_map(const std::string &transaction_id) {
  std::unique_lock<std::mutex> lock(mtx_);
  PhysicalGrowthOperationStats stats;
  const auto found = prepared_map_transactions_.find(transaction_id);
  if (found == prepared_map_transactions_.end()) {
    return {false, stats};
  }
  stats.offsets_count = static_cast<uint64_t>(found->second.offsets.size());
  stats.targets_count = static_cast<uint64_t>(found->second.mappings.size());
  auto reserved = std::move(found->second.mappings);
  prepared_map_transactions_.erase(found);
  const bool success =
      map_reserved_targets_(std::move(reserved), &stats, false);
  return {success, stats};
}

bool FTensorAllocator::abort_prepared_map(const std::string &transaction_id) {
  std::unique_lock<std::mutex> lock(mtx_);
  return prepared_map_transactions_.erase(transaction_id) > 0;
}

bool FTensorAllocator::has_prepared_map(
    const std::string &transaction_id) const {
  std::unique_lock<std::mutex> lock(mtx_);
  return prepared_map_transactions_.find(transaction_id) !=
         prepared_map_transactions_.end();
}

bool FTensorAllocator::unmap_from_kv_tensors(
    const std::vector<offset_t> &offsets) {
  std::unique_lock<std::mutex> lock(mtx_);
  if (num_layers_ == 0) {
    LOGGER(ERROR,
           "try to unmap from KV tensors when KV tensors are not created");
    return false;
  }
  reject_if_map_overlaps_locked_(offsets);

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
  reject_if_map_overlaps_locked_(offsets);
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

  remember_unmap_outcome_locked_(transaction_id,
                                 UnmapTransactionOutcome::ABORTED);
  return true;
}

std::vector<FTensorAllocator::RetainedMapping>
FTensorAllocator::unmap_retain_locked_(const std::vector<offset_t> &offsets) {
  std::vector<RetainedMapping> retained;
  try {
    for (auto logical_offset : offsets) {
      const auto targets = mapping_targets_({logical_offset});
      size_t existing = 0;
      for (const auto &[ftensor, offset] : targets) {
        existing += ftensor->is_mapped_(offset) ? 1 : 0;
      }
      if (existing == 0) {
        continue;
      }
      if (existing != targets.size()) {
        throw std::runtime_error("state_inconsistency: logical KV offset is "
                                 "only partially mapped: " +
                                 std::to_string(logical_offset));
      }
      for (const auto &[ftensor, offset] : targets) {
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

void FTensorAllocator::reject_if_map_overlaps_locked_(
    const std::vector<offset_t> &offsets) const {
  for (const auto &[id, transaction] : prepared_map_transactions_) {
    for (auto offset : offsets) {
      if (std::find(transaction.offsets.begin(), transaction.offsets.end(),
                    offset) != transaction.offsets.end()) {
        throw std::runtime_error(
            "cannot unmap a KV offset with a prepared map: " + id);
      }
    }
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
