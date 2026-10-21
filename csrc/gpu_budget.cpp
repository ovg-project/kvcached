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
constexpr uint64_t kMagic = 0x4b56425544473035ULL;
constexpr size_t kInstances = 128;
constexpr size_t kPools = 1024;
constexpr size_t kName = 192;
constexpr off_t kPoolLeaseStart = 3;
static_assert(__atomic_always_lock_free(sizeof(uint64_t), nullptr),
              "shared budget needs lock-free aligned 64-bit publication");
void publish(uint64_t &target, uint64_t value) {
  __atomic_store_n(&target, value, __ATOMIC_RELEASE);
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
  uint64_t next_generation;
  struct Instance {
    char name[kName];
    uint64_t limit;
    uint64_t revision;
    uint64_t has_revision;
    uint64_t valid;
    uint64_t participants;
    uint64_t generation;
    uint64_t expected_participants;
  } instances[kInstances];
  struct Pool {
    uint64_t owner;
    uint64_t instance;
    int64_t group;
    alignas(8) uint64_t charged;
    uint64_t capacity;
    uint64_t page_bundle;
    uint64_t ready;
    uint64_t can_shrink;
    uint64_t target;
    uint64_t acknowledged_revision;
    uint64_t acknowledged;
    uint64_t failed_revision;
    uint64_t failed;
  } pools[kPools];
};

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

std::shared_ptr<GpuBudget> GpuBudget::open_control(const std::string &path) {
  return std::shared_ptr<GpuBudget>(new GpuBudget(path));
}

GpuBudget::GpuBudget(const std::string &path) : path_(path), pid_(getpid()) {
  fd_ = open(path.c_str(), O_RDWR | O_CLOEXEC | O_NOFOLLOW);
  if (fd_ < 0)
    throw system_error("open live KV control");
  try {
    // Serialize with epoch initialization, then require an existing user.
    if (lease(fd_, F_WRLCK, true, 2) < 0)
      throw system_error("lock KV control attachment");
    if (lease(fd_, F_WRLCK, false) == 0)
      throw std::runtime_error("KV control instance is not running");
    if (errno != EAGAIN && errno != EACCES)
      throw system_error("inspect live KV control");
    if (lease(fd_, F_RDLCK, true) < 0)
      throw system_error("retain live KV control");
    {
      Locked lock(*this);
      struct stat info{};
      if (fstat(fd_, &info) < 0 ||
          info.st_size != static_cast<off_t>(sizeof(Shared)))
        throw std::runtime_error("incompatible live KV control layout");
      void *address = mmap(nullptr, sizeof(Shared), PROT_READ | PROT_WRITE,
                           MAP_SHARED, fd_, 0);
      if (address == MAP_FAILED)
        throw system_error("map live KV control");
      shared_ = static_cast<Shared *>(address);
      if (shared_->magic != kMagic)
        throw std::runtime_error("incompatible live KV control protocol");
    }
    if (lease(fd_, F_UNLCK, false, 2) < 0)
      throw system_error("unlock KV control attachment");
  } catch (...) {
    if (shared_)
      munmap(shared_, sizeof(Shared));
    close(fd_);
    throw;
  }
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
      shared_->next_generation = shared_->last_sample_ns;
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
        if (pool.owner == owner_ && pool.charged == 0)
          publish(pool.owner, 0);
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
        std::memset(&entry, 0, sizeof(entry));
        std::memcpy(entry.name, instance.c_str(), instance.size() + 1);
        entry.limit = UINT64_MAX;
        entry.generation = add(shared_->next_generation, 1);
        shared_->next_generation = entry.generation;
        publish(entry.valid, 1);
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
  if (shared_->instances[instance_index].participants)
    throw std::runtime_error(
        "dynamic KV pool registration after control is unsupported");
  for (size_t i = 0; i < kPools; ++i) {
    auto &entry = shared_->pools[i];
    if (entry.owner == 0 && entry.charged == 0) {
      slots_.push_back(i); // allocate before publishing ownership
      if (lease(fd_, F_WRLCK, false, kPoolLeaseStart + i) < 0) {
        slots_.pop_back();
        throw system_error("retain KV budget pool owner");
      }
      std::memset(&entry, 0, sizeof(entry));
      entry.instance = instance_index;
      entry.group = group;
      publish(entry.owner, owner_);
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
  // The legacy grant-only entry point remains usable until engine registration
  // supplies geometry. Once unified control is active it cannot bypass target
  // publication or acknowledgement invalidation.
  bool configured = entry.participants != 0;
  for (const auto &pool : shared_->pools)
    configured |= pool.owner &&
                  pool.instance == shared_->pools[slot].instance &&
                  pool.page_bundle;
  if (configured)
    return publish_control_locked(shared_->pools[slot].instance, bytes,
                                  revision)
        .status;
  if (entry.has_revision && revision < entry.revision)
    return "stale";
  if (entry.has_revision && revision == entry.revision)
    return entry.limit == bytes ? "applied" : "conflict";
  publish(entry.limit, bytes);
  entry.revision = revision;
  entry.has_revision = 1;
  return "applied";
}

void GpuBudget::configure_control_pool(size_t slot, uint64_t capacity,
                                       uint64_t page_bundle, bool can_shrink,
                                       uint64_t expected_participants) {
  if (!page_bundle || capacity % page_bundle)
    throw std::invalid_argument(
        "KV control capacity must contain whole page bundles");
  Locked lock(*this);
  (void)snapshot_locked(slot);
  auto &pool = shared_->pools[slot];
  if (pool.ready || shared_->instances[pool.instance].participants)
    throw std::runtime_error("KV control geometry is already frozen");
  auto &instance = shared_->instances[pool.instance];
  if (instance.expected_participants &&
      instance.expected_participants != expected_participants)
    throw std::runtime_error("inconsistent KV control participant topology");
  instance.expected_participants = expected_participants;
  pool.capacity = capacity;
  pool.page_bundle = page_bundle;
  pool.can_shrink = can_shrink;
}

void GpuBudget::ready_control_pool(size_t slot) {
  Locked lock(*this);
  (void)snapshot_locked(slot);
  auto &pool = shared_->pools[slot];
  if (!pool.page_bundle)
    throw std::runtime_error("KV control pool has no geometry");
  pool.ready = 1;
}

InstanceControl GpuBudget::control_state_locked(size_t instance,
                                                size_t slot) const {
  const auto &entry = shared_->instances[instance];
  InstanceControl state;
  state.published = entry.has_revision;
  state.generation = entry.generation;
  state.revision = entry.revision;
  state.requested_bytes = entry.limit;
  state.participants = entry.participants;
  state.expected_participants = entry.expected_participants;
  state.status = entry.has_revision ? "pending" : "unpublished";
  bool unsupported = false;
  uint64_t live = 0;
  for (size_t i = 0; i < kPools; ++i) {
    const auto &pool = shared_->pools[i];
    if (!pool.owner || pool.instance != instance)
      continue;
    state.charged_bytes = add(state.charged_bytes, pool.charged);
    state.effective_bytes =
        add(state.effective_bytes,
            entry.has_revision ? pool.target : pool.capacity);
    // Check liveness only: a control query never refunds or repairs a worker.
    struct flock owner_lock{};
    owner_lock.l_type = F_WRLCK;
    owner_lock.l_whence = SEEK_SET;
    owner_lock.l_start = kPoolLeaseStart + i;
    owner_lock.l_len = 1;
    if (pool.owner != owner_) {
      if (fcntl(fd_, F_OFD_GETLK, &owner_lock) < 0)
        throw system_error("inspect KV control participant");
      if (owner_lock.l_type == F_UNLCK)
        continue;
    }
    ++live;
    if (i == slot) {
      state.pool_slot = i;
      state.pool_owner = pool.owner;
      state.pool_target_bytes = pool.target;
      state.pool_capacity_bytes = pool.capacity;
      state.page_bundle_bytes = pool.page_bundle;
      state.can_shrink = pool.can_shrink;
      state.ready = pool.ready;
    }
    if (pool.failed && pool.failed_revision == entry.revision) {
      ++state.failed;
      state.failure_reason = "quarantined_pages_prevent_resize";
    }
    unsupported |= !pool.can_shrink && pool.target < pool.capacity;
    if (pool.acknowledged && pool.acknowledged_revision == entry.revision &&
        pool.charged <= pool.target)
      ++state.completed;
  }
  if (entry.has_revision) {
    if (live != entry.participants)
      state.status = "unavailable";
    else if (state.failed)
      state.status = "failed";
    else if (unsupported)
      state.status = "unsupported";
    else if (state.completed == entry.participants)
      state.status = "converged";
  }
  state.registered_participants = live;
  state.convergence = state.status;
  return state;
}

InstanceControl
GpuBudget::publish_control_locked(size_t instance, uint64_t bytes,
                                  std::optional<uint64_t> revision) {
  auto &entry = shared_->instances[instance];
  auto rejected = [&](const char *status) {
    auto state = control_state_locked(instance);
    state.status = status;
    return state;
  };
  if (revision && entry.has_revision) {
    if (*revision < entry.revision)
      return rejected("stale");
    if (*revision == entry.revision) {
      if (entry.limit != bytes)
        return rejected("conflict");
      if (entry.participants)
        return rejected("idempotent");
    }
  }
  if (!revision && entry.has_revision && entry.revision == UINT64_MAX)
    return rejected("revision_exhausted");
  uint64_t capacity = 0, participants = 0;
  for (const auto &pool : shared_->pools) {
    if (!pool.owner || pool.instance != instance)
      continue;
    if (!pool.ready || !pool.page_bundle)
      return rejected("not_ready");
    capacity = add(capacity, pool.capacity);
    ++participants;
  }
  if (!participants || !capacity ||
      (entry.participants && participants != entry.participants))
    return rejected("unavailable");
  // Proportional startup-capacity shares, independently rounded DOWN to each
  // pool's physical page bundle. The remainder stays unassigned. No pool can
  // independently consume the whole instance cap and rounding cannot raise it.
  const auto bounded = std::min(bytes, capacity);
  for (auto &pool : shared_->pools) {
    if (!pool.owner || pool.instance != instance)
      continue;
    const auto share = static_cast<uint64_t>(
        static_cast<unsigned __int128>(bounded) * pool.capacity / capacity);
    pool.target = share / pool.page_bundle * pool.page_bundle;
    pool.acknowledged = 0;
    pool.failed = 0;
  }
  publish(entry.limit, bytes);
  entry.revision =
      revision.value_or(entry.has_revision ? entry.revision + 1 : 1);
  entry.has_revision = 1;
  entry.participants = participants;
  auto state = control_state_locked(instance);
  state.status = "accepted";
  return state;
}

InstanceControl GpuBudget::publish_control(const std::string &instance,
                                           uint64_t bytes, uint64_t generation,
                                           std::optional<uint64_t> revision) {
  retire_dead_pools();
  Locked lock(*this);
  for (size_t i = 0; i < kInstances; ++i) {
    if (shared_->instances[i].valid && instance == shared_->instances[i].name) {
      if (shared_->instances[i].generation != generation) {
        auto state = control_state_locked(i);
        state.status = "instance_changed";
        return state;
      }
      return publish_control_locked(i, bytes, revision);
    }
  }
  InstanceControl state;
  state.status = "unavailable";
  return state;
}

InstanceControl GpuBudget::query_control(const std::string &instance,
                                         size_t slot) {
  Locked lock(*this);
  for (size_t i = 0; i < kInstances; ++i) {
    if (shared_->instances[i].valid && instance == shared_->instances[i].name)
      return control_state_locked(i, slot);
  }
  InstanceControl state;
  state.status = "unavailable";
  return state;
}

InstanceControl GpuBudget::control_state(size_t slot) {
  Locked lock(*this);
  (void)snapshot_locked(slot);
  return control_state_locked(shared_->pools[slot].instance, slot);
}

bool GpuBudget::acknowledge_control(size_t slot, uint64_t revision) {
  Locked lock(*this);
  (void)snapshot_locked(slot);
  auto &pool = shared_->pools[slot];
  const auto &entry = shared_->instances[pool.instance];
  if (!entry.has_revision || revision != entry.revision)
    return false;
  pool.acknowledged_revision = revision;
  pool.acknowledged = 1;
  return true;
}

bool GpuBudget::fail_control(const std::string &instance, size_t slot,
                             uint64_t owner, uint64_t generation,
                             uint64_t revision) {
  Locked lock(*this);
  if (slot >= kPools || !owner)
    return false;
  auto &pool = shared_->pools[slot];
  if (pool.owner != owner)
    return false;
  const auto &entry = shared_->instances[pool.instance];
  if (!entry.valid || instance != entry.name ||
      entry.generation != generation || !entry.has_revision ||
      entry.revision != revision)
    return false;
  pool.failed_revision = revision;
  pool.failed = 1;
  pool.acknowledged = 0;
  return true;
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
  shared_->last_sample_ns = monotonic_ns();
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
