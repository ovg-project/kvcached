// SPDX-FileCopyrightText: Copyright contributors to the kvcached project
// SPDX-License-Identifier: Apache-2.0
#include "gpu_budget.hpp"

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstring>
#include <fcntl.h>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <sys/file.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace kvcached {
namespace {
constexpr uint64_t kMagic = 0x4b56425544473034ULL;
constexpr size_t kInstances = 128;
constexpr size_t kPools = 1024;
constexpr size_t kName = 192;
constexpr off_t kPoolLeaseStart = 3;
static_assert(__atomic_always_lock_free(sizeof(uint64_t), nullptr),
              "shared budget needs lock-free aligned 64-bit publication");
template <typename T, typename V> void publish(T &target, V value) {
  __atomic_store_n(&target, static_cast<T>(value), __ATOMIC_SEQ_CST);
}
// Sequential ordering is used only for publication/observation. In particular,
// identity checks bracket atomic payload reads across slot reuse, without
// promising consistency between updates within the same identity.
template <typename T> T observe(const T &source) {
  return __atomic_load_n(&source, __ATOMIC_SEQ_CST);
}
uint64_t add(uint64_t a, uint64_t b) {
  if (b > UINT64_MAX - a)
    throw std::overflow_error("KV budget byte overflow");
  return a + b;
}
uint64_t sampled_capacity(uint64_t charged, int64_t free_after_margin) {
  if (free_after_margin >= 0)
    return add(charged, static_cast<uint64_t>(free_after_margin));
  // Preserve the deficit until it is combined with resident charges. Handle
  // INT64_MIN without signed negation overflow.
  const auto deficit = static_cast<uint64_t>(-(free_after_margin + 1)) + 1;
  return charged > deficit ? charged - deficit : 0;
}
std::runtime_error system_error(const char *operation) {
  return std::runtime_error(std::string(operation) + ": " +
                            std::strerror(errno));
}
int lease(int fd, short type, bool wait, off_t start = 0) {
#ifdef F_OFD_SETLK
  struct flock lock{};
  lock.l_type = type;
  lock.l_whence = SEEK_SET;
  lock.l_start = start;
  lock.l_len = 1;
  int result;
  do {
    result = fcntl(fd, wait ? F_OFD_SETLKW : F_OFD_SETLK, &lock);
  } while (result < 0 && errno == EINTR);
  return result;
#else
  (void)fd;
  (void)type;
  (void)wait;
  (void)start;
  errno = ENOTSUP;
  return -1;
#endif
}
// Serialize only samplers from sample start through publication. Grants,
// refunds and owner retirement never inspect or wait on this lease. Process
// death releases it, so a crashed sampler cannot leave a persistent gate.
struct CalibrationLease {
  int fd;
  explicit CalibrationLease(const std::string &path)
      : fd(open(path.c_str(), O_RDWR | O_CLOEXEC | O_NOFOLLOW)) {
    if (fd < 0)
      throw system_error("open calibration lease");
  }
  ~CalibrationLease() { close(fd); }
};
uint64_t monotonic_ns() {
  return static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::steady_clock::now().time_since_epoch())
          .count());
}
} // namespace

struct GpuBudget::Shared {
  uint64_t magic;
  uint64_t budget;
  uint64_t last_sample_ns;
  uint64_t next_generation; // writer-only, under the ledger lock
  struct Instance {
    char name[kName];
    uint64_t limit;
    uint64_t revision;
    uint64_t has_revision;
    uint64_t valid;
  } instances[kInstances];
  struct Pool {
    uint64_t generation;
    uint64_t owner;
    uint64_t instance;
    int64_t group;
    alignas(8) uint64_t charged;
  } pools[kPools];
};

BudgetObservation observe_gpu_budget(const std::string &path) {
  using Shared = GpuBudget::Shared;
  struct Mapping {
    int fd = -1;
    const Shared *data = nullptr;
    ~Mapping() {
      if (data)
        munmap(const_cast<Shared *>(data), sizeof(Shared));
      if (fd >= 0)
        close(fd);
    }
  } mapping;
  mapping.fd = open(path.c_str(), O_RDONLY | O_CLOEXEC | O_NOFOLLOW);
  if (mapping.fd < 0)
    throw system_error("open KV budget observation");
  // These nonblocking lifetime leases never acquire the accounting flock.
  // Pin an already live epoch before mapping: initialization/reset/truncation
  // cannot race the read, and a reader cannot resurrect an abandoned epoch.
  if (lease(mapping.fd, F_RDLCK, false, 2) < 0)
    throw system_error("KV budget observation initializing");
  struct flock usage{};
  usage.l_type = F_WRLCK;
  usage.l_whence = SEEK_SET;
  usage.l_len = 1;
  if (fcntl(mapping.fd, F_OFD_GETLK, &usage) < 0)
    throw system_error("inspect KV budget observation epoch");
  if (usage.l_type != F_RDLCK)
    throw std::runtime_error("KV budget observation has no live epoch");
  if (lease(mapping.fd, F_RDLCK, false) < 0)
    throw system_error("retain KV budget observation epoch");
  struct stat held{}, current{};
  auto validate = [&] {
    if (fstat(mapping.fd, &held) < 0 || lstat(path.c_str(), &current) < 0 ||
        !S_ISREG(held.st_mode) || !S_ISREG(current.st_mode) ||
        held.st_dev != current.st_dev || held.st_ino != current.st_ino ||
        held.st_size != static_cast<off_t>(sizeof(Shared)))
      throw std::runtime_error(
          "unavailable or incompatible KV budget observation");
  };
  validate();
  void *address =
      mmap(nullptr, sizeof(Shared), PROT_READ, MAP_SHARED, mapping.fd, 0);
  if (address == MAP_FAILED)
    throw system_error("map KV budget observation");
  mapping.data = static_cast<const Shared *>(address);
  const auto &shared = *mapping.data;
  if (observe(shared.magic) != kMagic)
    throw std::runtime_error("incompatible KV budget observation protocol");
  BudgetObservation result;
  result.budget_bytes = observe(shared.budget);
  result.last_sample_ns = observe(shared.last_sample_ns);
  for (const auto &pool : shared.pools) {
    const auto generation = observe(pool.generation);
    if (!generation) {
      // Interrupted retirement may leave charged resources in an invalid slot.
      if (observe(pool.owner) || observe(pool.charged))
        result.incomplete = true;
      continue;
    }
    const auto index = observe(pool.instance);
    if (index >= kInstances)
      throw std::runtime_error("invalid KV budget observation instance");
    const auto &instance = shared.instances[index];
    const auto identity = observe(instance.valid);
    BudgetPoolObservation entry;
    entry.generation = generation;
    for (const auto &byte : instance.name) {
      const auto value = observe(byte);
      if (!value)
        break;
      entry.instance += value;
    }
    entry.group = observe(pool.group);
    entry.charged_bytes = observe(pool.charged);
    entry.limit_bytes = observe(instance.limit);
    entry.revision = observe(instance.revision);
    entry.has_revision = observe(instance.has_revision) != 0;
    if (!identity || identity != observe(instance.valid) ||
        generation != observe(pool.generation)) {
      result.incomplete = true;
      continue;
    }
    result.pools.push_back(std::move(entry));
  }
  validate();
  return result;
}

struct GpuBudget::Locked {
  GpuBudget &budget;
  std::unique_lock<std::mutex> local;
  explicit Locked(GpuBudget &value) : budget(value), local(value.mutex_) {
    value.check_owner();
    const auto started = monotonic_ns();
    ++value.sync_acquisitions_;
    int result;
    do {
      result = flock(value.fd_, LOCK_EX | LOCK_NB);
    } while (result < 0 && errno == EINTR);
    if (result < 0 && (errno == EWOULDBLOCK || errno == EAGAIN)) {
      ++value.sync_contentions_;
      const auto waiting = monotonic_ns();
      do {
        result = flock(value.fd_, LOCK_EX);
      } while (result < 0 && errno == EINTR);
      value.sync_wait_ns_ += monotonic_ns() - waiting;
    }
    value.sync_acquire_ns_ += monotonic_ns() - started;
    if (result < 0)
      throw system_error("lock KV budget");
    struct stat held{}, current{};
    if (fstat(value.fd_, &held) < 0 ||
        lstat(value.path_.c_str(), &current) < 0 ||
        held.st_dev != current.st_dev || held.st_ino != current.st_ino ||
        !S_ISREG(current.st_mode)) {
      flock(value.fd_, LOCK_UN);
      throw std::runtime_error("KV budget segment was replaced");
    }
  }
  ~Locked() { flock(budget.fd_, LOCK_UN); }
};

std::shared_ptr<GpuBudget> GpuBudget::connect(const std::string &path,
                                              const Sampler &sampler) {
  return std::shared_ptr<GpuBudget>(new GpuBudget(path, sampler));
}

GpuBudget::GpuBudget(const std::string &path, const Sampler &sampler)
    : path_(path), pid_(getpid()) {
  static std::atomic<uint64_t> next_owner{1};
  owner_ = (static_cast<uint64_t>(pid_) << 32) ^
           static_cast<uint64_t>(
               std::chrono::steady_clock::now().time_since_epoch().count()) ^
           next_owner.fetch_add(1);
  if (owner_ == 0)
    owner_ = 1;
  fd_ = open(path.c_str(), O_RDWR | O_CREAT | O_CLOEXEC | O_NOFOLLOW, 0660);
  if (fd_ < 0)
    throw system_error("open KV budget");
  try {
    // Serialize connection setup separately from accounting. If an initializer
    // dies, its successor must retry exclusive epoch ownership before taking a
    // read lease; queued readers must not accept the previous epoch's header.
    if (lease(fd_, F_WRLCK, true, 2) < 0)
      throw system_error("lock KV budget initialization");
    // The first live connection starts a new accounting epoch. Sampling free
    // memory includes any old untracked allocations, so they are not credited.
    // Keep the usage lease exclusive during this one-time initialization, but
    // release the data flock before asking the GPU driver for a sample.
    bool first = lease(fd_, F_WRLCK, false) == 0;
    if (!first) {
      if (errno != EAGAIN && errno != EACCES)
        throw system_error("retain KV budget");
      if (lease(fd_, F_RDLCK, true) < 0)
        throw system_error("retain KV budget");
    }
    {
      Locked lock(*this);
      struct stat info{};
      if (fstat(fd_, &info) < 0)
        throw system_error("stat KV budget");
      if (first) {
        if (ftruncate(fd_, sizeof(Shared)) < 0)
          throw system_error("size KV budget");
      } else if (info.st_size != static_cast<off_t>(sizeof(Shared))) {
        throw std::runtime_error("incompatible live KV budget layout");
      }
      void *address = mmap(nullptr, sizeof(Shared), PROT_READ | PROT_WRITE,
                           MAP_SHARED, fd_, 0);
      if (address == MAP_FAILED)
        throw system_error("map KV budget");
      shared_ = static_cast<Shared *>(address);
      if (first)
        publish(shared_->magic, 0);
    }
    if (first) {
      const auto initial_budget = sampled_capacity(0, sampler());
      Locked lock(*this);
      std::memset(shared_, 0, sizeof(Shared));
      shared_->budget = initial_budget;
      shared_->last_sample_ns = monotonic_ns();
      publish(shared_->magic, kMagic);
      if (lease(fd_, F_RDLCK, false) < 0)
        throw system_error("retain initialized KV budget");
    } else {
      Locked lock(*this);
      if (shared_->magic != kMagic)
        throw std::runtime_error("incompatible KV budget protocol");
    }
    if (lease(fd_, F_UNLCK, false, 2) < 0)
      throw system_error("unlock KV budget initialization");
  } catch (...) {
    if (shared_)
      munmap(shared_, sizeof(Shared));
    close(fd_);
    throw;
  }
}

GpuBudget::~GpuBudget() {
  if (getpid() == pid_ && shared_) {
    try {
      Locked lock(*this);
      for (auto slot : slots_) {
        auto &pool = shared_->pools[slot];
        // Unknown resources remain charged while any other participant lives.
        if (pool.owner == owner_ && pool.charged == 0) {
          publish(pool.generation, 0);
          publish(pool.owner, 0);
        }
        if (pool.owner == 0 &&
            lease(fd_, F_UNLCK, false, kPoolLeaseStart + slot) < 0)
          throw system_error("release KV budget pool owner");
      }
      reclaim_instances_locked();
    } catch (const std::exception &error) {
      std::cerr << "KV budget teardown: " << error.what() << '\n';
    }
  }
  if (shared_)
    munmap(shared_, sizeof(Shared));
  if (fd_ >= 0)
    close(fd_);
}

void GpuBudget::check_owner() const {
  if (getpid() != pid_)
    throw std::runtime_error("reconnect KV budget after fork");
}

void GpuBudget::reclaim_instances_locked() {
  bool referenced[kInstances]{};
  for (const auto &pool : shared_->pools) {
    // Retain limits for live participants and for resources whose release has
    // not been confirmed, including resources left behind by a dead owner.
    if (pool.owner != 0 || pool.charged != 0) {
      if (pool.instance >= kInstances)
        throw std::runtime_error("invalid KV budget instance reference");
      referenced[pool.instance] = true;
    }
  }
  for (size_t i = 0; i < kInstances; ++i) {
    if (!referenced[i])
      publish(shared_->instances[i].valid, 0);
  }
}

// Retiring an owner is not a CUDA refund. Reduce the budget by the same
// charge before removing accounting, so spendable GPU credit cannot increase.
// A later sample can recover capacity actually observed free. Every shared
// change stays under the same short accounting lock.
void GpuBudget::retire_dead_pools() {
  Locked lock(*this);
  for (size_t i = 0; i < kPools; ++i) {
    auto &pool = shared_->pools[i];
    if (!pool.owner || pool.owner == owner_)
      continue;
    struct flock owner_lock{};
    owner_lock.l_type = F_WRLCK;
    owner_lock.l_whence = SEEK_SET;
    owner_lock.l_start = kPoolLeaseStart + i;
    owner_lock.l_len = 1;
    if (fcntl(fd_, F_OFD_GETLK, &owner_lock) < 0)
      throw system_error("inspect KV budget pool owner");
    if (owner_lock.l_type != F_UNLCK)
      continue;
    publish(pool.generation, 0);
    const auto charged = pool.charged;
    const auto budget = shared_->budget;
    publish(shared_->budget, budget > charged ? budget - charged : 0);
    accounting_test_hook(60);
    publish(pool.charged, 0);
    accounting_test_hook(61);
    publish(pool.owner, 0);
    accounting_test_hook(62);
    ++sync_dead_owner_recoveries_;
  }
  reclaim_instances_locked();
}

size_t GpuBudget::register_pool(const std::string &instance, int64_t group) {
  if (instance.empty() || instance.size() >= kName ||
      instance.find('\0') != std::string::npos)
    throw std::invalid_argument(
        "KV budget instance name must contain 1..191 bytes");
  retire_dead_pools();
  Locked lock(*this);
  // Also recover an instance published before pool registration was
  // interrupted.
  reclaim_instances_locked();
  size_t instance_index = kInstances;
  for (size_t i = 0; i < kInstances; ++i) {
    if (shared_->instances[i].valid && instance == shared_->instances[i].name) {
      instance_index = i;
      break;
    }
  }
  if (instance_index == kInstances) {
    for (size_t i = 0; i < kInstances; ++i) {
      auto &entry = shared_->instances[i];
      if (!entry.valid) {
        const auto generation = add(shared_->next_generation, 1);
        shared_->next_generation = generation;
        for (size_t n = 0; n < kName; ++n)
          publish(entry.name[n], n < instance.size() ? instance[n] : '\0');
        publish(entry.limit, UINT64_MAX);
        publish(entry.revision, 0);
        publish(entry.has_revision, 0);
        publish(entry.valid, generation);
        instance_index = i;
        break;
      }
    }
  }
  if (instance_index == kInstances)
    throw std::runtime_error("KV budget instance table is full");
  for (auto slot : slots_) {
    const auto &entry = shared_->pools[slot];
    if (entry.instance == instance_index && entry.group == group)
      return slot;
  }
  for (size_t i = 0; i < kPools; ++i) {
    auto &entry = shared_->pools[i];
    if (entry.owner == 0 && entry.charged == 0) {
      slots_.push_back(i); // allocate before publishing ownership
      if (lease(fd_, F_WRLCK, false, kPoolLeaseStart + i) < 0) {
        slots_.pop_back();
        throw system_error("retain KV budget pool owner");
      }
      const auto generation = add(shared_->next_generation, 1);
      shared_->next_generation = generation;
      publish(entry.instance, instance_index);
      publish(entry.group, group);
      publish(entry.owner, owner_);
      publish(entry.generation, generation);
      return i;
    }
  }
  throw std::runtime_error("KV budget pool table is full");
}

BudgetSnapshot GpuBudget::snapshot_locked(size_t slot) const {
  if (slot >= kPools || shared_->pools[slot].owner != owner_)
    throw std::runtime_error("invalid KV budget owner");
  BudgetSnapshot result;
  const auto instance = shared_->pools[slot].instance;
  result.budget_bytes = shared_->budget;
  result.instance_limit_bytes = shared_->instances[instance].limit;
  result.sync_acquisitions = sync_acquisitions_;
  result.sync_contentions = sync_contentions_;
  result.sync_acquire_ns = sync_acquire_ns_;
  result.sync_wait_ns = sync_wait_ns_;
  result.sync_dead_owner_recoveries = sync_dead_owner_recoveries_;
  for (const auto &entry : shared_->pools) {
    result.charged_bytes = add(result.charged_bytes, entry.charged);
    if (entry.instance == instance && entry.owner != 0)
      result.instance_charged_bytes =
          add(result.instance_charged_bytes, entry.charged);
  }
  return result;
}

BudgetSnapshot GpuBudget::snapshot(size_t slot, const Sampler &sampler) {
  bool sample_due = false;
  if (sampler) {
    Locked lock(*this);
    constexpr uint64_t interval_ns = 1000000000ULL;
    sample_due = monotonic_ns() - shared_->last_sample_ns >= interval_ns;
  }
  // Admission queries also refresh an exhausted budget: waiting for another
  // grant would otherwise prevent a scheduler from observing recovered memory.
  if (sample_due)
    recalibrate(sampler);
  Locked lock(*this);
  return snapshot_locked(slot);
}

void GpuBudget::accounting_test_hook(int stage) {
#ifdef KVCACHED_BUDGET_TESTING
  if (accounting_test_hook_)
    accounting_test_hook_(stage);
#else
  (void)stage;
#endif
}

std::unique_ptr<BudgetGrant> GpuBudget::grant(size_t slot, uint64_t bytes,
                                              BudgetSnapshot *observed,
                                              const Sampler &sampler) {
  auto result = std::make_unique<BudgetGrant>(shared_from_this(), slot, 0);
  bool attempted_sample = false;
  for (;;) {
    {
      Locked lock(*this);
      constexpr uint64_t interval_ns = 1000000000ULL;
      const bool sample_due =
          sampler && !attempted_sample &&
          monotonic_ns() - shared_->last_sample_ns >= interval_ns;
      if (!sample_due) {
        const auto state = snapshot_locked(slot);
        if (observed)
          *observed = state;
        if (state.charged_bytes > state.budget_bytes ||
            bytes > state.budget_bytes - state.charged_bytes ||
            state.instance_charged_bytes > state.instance_limit_bytes ||
            bytes > state.instance_limit_bytes - state.instance_charged_bytes)
          throw std::runtime_error(
              "capacity_exhausted: shared KV budget would be exceeded"
              " (requested=" +
              std::to_string(bytes) +
              ", budget=" + std::to_string(state.budget_bytes) +
              ", charged=" + std::to_string(state.charged_bytes) + ")");
        publish(shared_->pools[slot].charged,
                add(shared_->pools[slot].charged, bytes));
#ifdef KVCACHED_BUDGET_TESTING
        if (grant_test_hook_)
          grant_test_hook_();
#endif
        result->unused_ = bytes;
        result->active_ = true;
        return result;
      }
    }
    attempted_sample = true;
    // A concurrent sampler may win. Admission still proceeds against the last
    // published budget instead of waiting for that sampler.
    recalibrate(sampler);
  }
}

std::string GpuBudget::set_instance_limit(size_t slot, uint64_t bytes,
                                          uint64_t revision) {
  Locked lock(*this);
  (void)snapshot_locked(slot);
  auto &entry = shared_->instances[shared_->pools[slot].instance];
  if (entry.has_revision && revision < entry.revision)
    return "stale";
  if (entry.has_revision && revision == entry.revision)
    return entry.limit == bytes ? "applied" : "conflict";
  publish(entry.limit, bytes);
  publish(entry.revision, revision);
  publish(entry.has_revision, 1);
  return "applied";
}

bool GpuBudget::recalibrate(const Sampler &sampler) {
  CalibrationLease calibration(path_);
  {
    Locked lock(*this);
    struct stat held{}, selected{};
    if (fstat(fd_, &held) < 0 || fstat(calibration.fd, &selected) < 0 ||
        held.st_dev != selected.st_dev || held.st_ino != selected.st_ino)
      throw std::runtime_error("KV budget calibration segment was replaced");
    if (lease(calibration.fd, F_WRLCK, false, 1) < 0) {
      if (errno == EAGAIN || errno == EACCES)
        return false;
      throw system_error("lock calibration lease");
    }
  }
  const int64_t free_budget = sampler();
  Locked lock(*this);
  uint64_t charged = 0;
  for (const auto &pool : shared_->pools)
    charged = add(charged, pool.charged);
  // Sum at publication, including any concurrent grant/refund/retirement.
  // This is an estimate: unmaterialized grants, released-but-unrefunded bytes
  // and changes since the driver sample can temporarily overstate capacity.
  // Negative headroom remains signed until combined with current charges.
  publish(shared_->budget, sampled_capacity(charged, free_budget));
  publish(shared_->last_sample_ns, monotonic_ns());
  return true;
}

void GpuBudget::begin_mutation() {
  check_owner();
  std::lock_guard<std::mutex> mutation_lock(mutation_mutex_);
  auto &local = mutations_[std::this_thread::get_id()];
  if (local.depth == 0)
    flush_refunds(local); // retry a previous scope's failed completion
  ++local.depth;
}

void GpuBudget::return_locked(size_t slot, uint64_t bytes) {
  if (slot >= kPools || shared_->pools[slot].owner != owner_ ||
      bytes > shared_->pools[slot].charged)
    throw std::runtime_error("state_inconsistency: invalid KV budget refund");
  publish(shared_->pools[slot].charged, shared_->pools[slot].charged - bytes);
}

void GpuBudget::flush_refunds(LocalMutation &local) {
  if (!local.bytes)
    return;
  Locked lock(*this);
  accounting_test_hook(50);
  // Validate everything before publishing. A failed lock or validation keeps
  // the queued receipts owned here and retryable, with no partial refund.
  for (const auto &[slot, bytes] : local.refunds) {
    if (slot >= kPools || shared_->pools[slot].owner != owner_ ||
        bytes > shared_->pools[slot].charged)
      throw std::runtime_error("state_inconsistency: invalid KV budget refund");
  }
  for (const auto &[slot, bytes] : local.refunds)
    return_locked(slot, bytes);
  local.refunds.clear();
  local.bytes = 0;
}

void GpuBudget::refund(size_t slot, uint64_t bytes) {
  if (!bytes)
    return;
  check_owner();
  {
    std::lock_guard<std::mutex> mutation_lock(mutation_mutex_);
    const auto found = mutations_.find(std::this_thread::get_id());
    if (found != mutations_.end()) {
      auto &local = found->second;
      if (!local.depth || bytes >= kBudgetRefundBatchBytes - local.bytes)
        flush_refunds(local);
      if (local.depth && bytes < kBudgetRefundBatchBytes) {
        auto &total = local.refunds[slot];
        total = add(total, bytes);
        local.bytes += bytes;
        return;
      }
      // Larger handles refund immediately. If flushing previous receipts
      // failed, the incoming receipt was not enqueued and still owns its bytes.
      if (!local.depth)
        mutations_.erase(found);
    }
  }
  Locked lock(*this);
  return_locked(slot, bytes);
}

void GpuBudget::end_mutation() {
  check_owner();
  std::lock_guard<std::mutex> mutation_lock(mutation_mutex_);
  auto found = mutations_.find(std::this_thread::get_id());
  if (found == mutations_.end() || !found->second.depth)
    throw std::runtime_error("unbalanced KV budget mutation");
  if (--found->second.depth != 0)
    return;
  // Close the scope even when flushing fails. Keep its pending bytes for the
  // next scope/refund rather than leaving a permanently nested outer scope.
  flush_refunds(found->second);
  mutations_.erase(found);
}

BudgetReceipt::BudgetReceipt(std::shared_ptr<GpuBudget> budget, size_t slot,
                             uint64_t bytes)
    : budget_(std::move(budget)), slot_(slot), bytes_(bytes) {}
void BudgetReceipt::release() {
  if (bytes_) {
    budget_->refund(slot_, bytes_);
    bytes_ = 0;
  }
}
BudgetGrant::BudgetGrant(std::shared_ptr<GpuBudget> budget, size_t slot,
                         uint64_t bytes)
    : budget_(std::move(budget)), slot_(slot), unused_(bytes),
      active_(bytes != 0) {}
BudgetGrant::~BudgetGrant() {
  try {
    finish();
  } catch (const std::exception &error) {
    std::cerr << "KV grant cleanup remains charged: " << error.what() << '\n';
  }
}
std::unique_ptr<BudgetReceipt> BudgetGrant::take(uint64_t bytes) {
  if (!active_ || bytes > unused_)
    throw std::runtime_error("invalid KV grant consumption");
  auto result = std::make_unique<BudgetReceipt>(budget_, slot_, bytes);
  unused_ -= bytes;
  return result;
}
void BudgetGrant::finish() {
  if (!active_)
    return;
  if (unused_)
    budget_->refund(slot_, unused_);
  unused_ = 0;
  active_ = false;
}
BudgetMutation::BudgetMutation(std::shared_ptr<GpuBudget> budget)
    : budget_(std::move(budget)) {
  if (budget_)
    budget_->begin_mutation();
}
BudgetMutation::~BudgetMutation() {
  try {
    finish();
  } catch (const std::exception &error) {
    std::cerr << "KV mutation cleanup remains charged: " << error.what()
              << '\n';
  }
}
void BudgetMutation::finish() {
  if (!active_)
    return;
  active_ = false;
  if (budget_)
    budget_->end_mutation();
}
} // namespace kvcached
