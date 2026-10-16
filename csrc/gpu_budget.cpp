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
constexpr uint64_t kMagic = 0x4b56425544473031ULL;
constexpr size_t kInstances = 128;
constexpr size_t kPools = 1024;
constexpr size_t kName = 192;
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
// A separate open-file description owns the calibration lease. Closing it,
// including on process death, releases admission without a persistent flag.
struct CalibrationLease {
  int fd;
  explicit CalibrationLease(const std::string &path)
      : fd(open(path.c_str(), O_RDWR | O_CLOEXEC | O_NOFOLLOW)) {
    if (fd < 0)
      throw system_error("open calibration lease");
  }
  ~CalibrationLease() { close(fd); }
};
bool calibration_active(int fd) {
  struct flock lock{};
  lock.l_type = F_RDLCK;
  lock.l_whence = SEEK_SET;
  lock.l_start = 1;
  lock.l_len = 1;
  if (fcntl(fd, F_OFD_GETLK, &lock) < 0)
    throw system_error("inspect calibration lease");
  return lock.l_type != F_UNLCK;
}
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
  uint64_t active;
  uint64_t version;
  uint64_t last_sample_ns;
  struct Instance {
    char name[kName];
    uint64_t limit;
    uint64_t revision;
    uint64_t has_revision;
    uint64_t valid;
  } instances[kInstances];
  struct Pool {
    uint64_t owner;
    uint64_t instance;
    int64_t group;
    alignas(8) uint64_t charged;
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
        if (pool.owner == owner_ && pool.charged == 0)
          publish(pool.owner, 0);
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

size_t GpuBudget::register_pool(const std::string &instance, int64_t group) {
  if (instance.empty() || instance.size() >= kName ||
      instance.find('\0') != std::string::npos)
    throw std::invalid_argument(
        "KV budget instance name must contain 1..191 bytes");
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
  for (size_t i = 0; i < kPools; ++i) {
    auto &entry = shared_->pools[i];
    if (entry.owner == 0 && entry.charged == 0) {
      slots_.push_back(i); // allocate before publishing ownership
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
  result.active_mutations = shared_->active;
  result.sync_acquisitions = sync_acquisitions_;
  result.sync_contentions = sync_contentions_;
  result.sync_acquire_ns = sync_acquire_ns_;
  result.sync_wait_ns = sync_wait_ns_;
  result.sync_cas_retries = sync_cas_retries_;
  result.sync_dead_owner_recoveries = sync_dead_owner_recoveries_;
  for (const auto &entry : shared_->pools) {
    result.charged_bytes = add(result.charged_bytes, entry.charged);
    if (entry.instance == instance && entry.owner != 0)
      result.instance_charged_bytes =
          add(result.instance_charged_bytes, entry.charged);
  }
  return result;
}

BudgetSnapshot GpuBudget::snapshot(size_t slot) {
  Locked lock(*this);
  return snapshot_locked(slot);
}

std::unique_ptr<BudgetGrant> GpuBudget::grant(size_t slot, uint64_t bytes,
                                              BudgetSnapshot *observed,
                                              const Sampler &sampler) {
  auto result = std::make_unique<BudgetGrant>(shared_from_this(), slot, 0);
  bool attempted_sample = false;
  for (;;) {
    bool sample_due = false;
    {
      Locked lock(*this);
      // This is a physical growth batch, never a token/block hot-path query.
      // Check the clock under the same lock used for admission; ordinary
      // grants incur no additional data-lock round trip.
      constexpr uint64_t interval_ns = 1000000000ULL;
      sample_due = sampler && !attempted_sample && !shared_->active &&
                   monotonic_ns() - shared_->last_sample_ns >= interval_ns;
      if (!sample_due) {
        const auto state = snapshot_locked(slot);
        if (observed)
          *observed = state;
        if (calibration_active(fd_))
          throw std::runtime_error(
              "capacity_exhausted: KV budget calibration in progress");
        if (state.charged_bytes > state.budget_bytes ||
            bytes > state.budget_bytes - state.charged_bytes ||
            state.instance_charged_bytes > state.instance_limit_bytes ||
            bytes > state.instance_limit_bytes - state.instance_charged_bytes)
          throw std::runtime_error(
              "capacity_exhausted: shared KV budget would be exceeded");
        // Only this owner's authoritative charge changes. There is no
        // separately maintained total balance that could disagree after process
        // interruption.
        shared_->active = add(shared_->active, 1);
        ++shared_->version;
        publish(shared_->pools[slot].charged,
                add(shared_->pools[slot].charged, bytes));
#ifdef KVCACHED_BUDGET_TESTING
        if (grant_test_hook_)
          grant_test_hook_();
#endif
        // Construction below cannot allocate. Activate the already-allocated
        // grant.
        result->unused_ = bytes;
        result->active_ = true;
        return result;
      }
    }
    attempted_sample = true;
    // Another worker may win the lease or start physical work; then skip the
    // sample and retain the previous conservative budget for this admission.
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
  entry.revision = revision;
  entry.has_revision = 1;
  ++shared_->version;
  return "applied";
}

bool GpuBudget::recalibrate(const Sampler &sampler) {
  uint64_t charged = 0;
  uint64_t version = 0;
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
    if (shared_->active)
      return false;
    for (const auto &pool : shared_->pools)
      charged = add(charged, pool.charged);
    version = shared_->version;
  }
  const int64_t free_budget = sampler();
  Locked lock(*this);
  if (shared_->active || shared_->version != version)
    return false;
  // All grants have settled and releases completed. Pending promises must
  // never be added to a driver sample, hence the active-mutation check above.
  // The first implementation only lowers a live epoch's budget. Growing the
  // epoch after non-KV usage falls is policy work, not implicit extra credit.
  publish(shared_->budget,
          std::min(shared_->budget, sampled_capacity(charged, free_budget)));
  shared_->last_sample_ns = monotonic_ns();
  ++shared_->version;
  return true;
}

void GpuBudget::begin_mutation() {
  check_owner();
  std::lock_guard<std::mutex> mutation_lock(mutation_mutex_);
  const auto thread = std::this_thread::get_id();
  auto &local = mutations_[thread];
  if (local.depth++ != 0)
    return;
  try {
    Locked lock(*this);
    shared_->active = add(shared_->active, 1);
    ++shared_->version;
  } catch (...) {
    mutations_.erase(thread);
    throw;
  }
}

void GpuBudget::return_locked(size_t slot, uint64_t bytes) {
  if (slot >= kPools || shared_->pools[slot].owner != owner_ ||
      bytes > shared_->pools[slot].charged)
    throw std::runtime_error("state_inconsistency: invalid KV budget refund");
  publish(shared_->pools[slot].charged, shared_->pools[slot].charged - bytes);
}

void GpuBudget::refund(size_t slot, uint64_t bytes) {
  if (!bytes)
    return;
  check_owner();
  {
    std::lock_guard<std::mutex> mutation_lock(mutation_mutex_);
    const auto found = mutations_.find(std::this_thread::get_id());
    if (found != mutations_.end() && found->second.depth) {
      auto &total = found->second.refunds[slot];
      total = add(total, bytes);
      return;
    }
  }
  Locked lock(*this);
  return_locked(slot, bytes);
  ++shared_->version;
}

void GpuBudget::end_mutation() {
  check_owner();
  std::lock_guard<std::mutex> mutation_lock(mutation_mutex_);
  auto found = mutations_.find(std::this_thread::get_id());
  if (found == mutations_.end() || !found->second.depth)
    throw std::runtime_error("unbalanced KV budget mutation");
  if (found->second.depth > 1) {
    --found->second.depth;
    return;
  }
  Locked lock(*this);
  if (!shared_->active)
    throw std::runtime_error("unbalanced shared KV mutation");
  // Validate the entire refund before publishing any part. A failed lock or
  // validation leaves the outer scope retryable, without a partial refund.
  for (const auto &[slot, bytes] : found->second.refunds) {
    if (slot >= kPools || shared_->pools[slot].owner != owner_ ||
        bytes > shared_->pools[slot].charged)
      throw std::runtime_error("state_inconsistency: invalid KV budget refund");
  }
  for (const auto &[slot, bytes] : found->second.refunds)
    return_locked(slot, bytes);
  mutations_.erase(found);
  --shared_->active;
  ++shared_->version;
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
  GpuBudget::Locked lock(*budget_);
  budget_->return_locked(slot_, unused_);
  unused_ = 0;
  if (!budget_->shared_->active)
    throw std::runtime_error("unbalanced KV grant");
  --budget_->shared_->active;
  ++budget_->shared_->version;
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
  if (active_ && budget_)
    budget_->end_mutation();
  active_ = false;
}
} // namespace kvcached
