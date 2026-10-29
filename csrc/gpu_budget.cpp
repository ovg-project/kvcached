// SPDX-FileCopyrightText: Copyright contributors to the kvcached project
// SPDX-License-Identifier: Apache-2.0
#include "gpu_budget.hpp"

#include <algorithm>
#include <atomic>
#include <cctype>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdlib>
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
constexpr uint64_t kMagic = 0x4b56425544473039ULL;
constexpr size_t kInstances = 128;
constexpr size_t kPools = 1024;
constexpr size_t kName = 192;
constexpr off_t kPoolLeaseStart = 3;
static_assert(__atomic_always_lock_free(sizeof(uint64_t), nullptr),
              "shared budget needs lock-free aligned 64-bit publication");
uint64_t add(uint64_t a, uint64_t b) {
  if (b > UINT64_MAX - a)
    throw std::overflow_error("KV budget byte overflow");
  return a + b;
}
uint64_t saturating_add(uint64_t a, uint64_t b) {
  return b > UINT64_MAX - a ? UINT64_MAX : a + b;
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
uint64_t env_bytes(const char *name) {
  const char *value = std::getenv(name);
  if (value == nullptr || *value == '\0')
    return 0;
  char *end = nullptr;
  errno = 0;
  const double number = std::strtod(value, &end);
  std::string suffix(end);
  std::transform(suffix.begin(), suffix.end(), suffix.begin(),
                 [](unsigned char c) { return std::tolower(c); });
  double scale = 0;
  if (suffix.empty() || suffix == "b")
    scale = 1;
  else if (suffix == "k" || suffix == "kb")
    scale = 1024.0;
  else if (suffix == "m" || suffix == "mb")
    scale = 1024.0 * 1024;
  else if (suffix == "g" || suffix == "gb")
    scale = 1024.0 * 1024 * 1024;
  const double bytes = number * scale;
  // 2^64 is exact as a double, and nothing at or above it fits.
  if (errno != 0 || end == value || scale == 0 || !std::isfinite(bytes) ||
      bytes < 0 || bytes >= 18446744073709551616.0)
    throw std::invalid_argument(std::string(name) +
                                " must be a byte count such as 512M or 2G");
  return static_cast<uint64_t>(bytes);
}
} // namespace

InstancePolicy startup_policy_from_env() {
  InstancePolicy policy;
  policy.min_bytes = env_bytes("KVCACHED_MIN");
  policy.headroom_bytes = env_bytes("KVCACHED_HEADROOM");
  const char *priority = std::getenv("KVCACHED_PRIORITY");
  if (priority != nullptr && *priority != '\0') {
    char *end = nullptr;
    errno = 0;
    const long long parsed = std::strtoll(priority, &end, 10);
    if (errno != 0 || end == priority || *end != '\0')
      throw std::invalid_argument("KVCACHED_PRIORITY must be an integer");
    policy.priority = parsed;
  }
  return policy;
}

// Every observational load is atomic, including strings and reused slots.
// Writer serialization still provides authoritative admission/control ordering.
template <typename T> struct Observed {
  T value{};
  Observed() = default;
  Observed(const Observed &other) : value(static_cast<T>(other)) {}
  operator T() const { return __atomic_load_n(&value, __ATOMIC_SEQ_CST); }
  Observed &operator=(T next) {
    __atomic_store_n(&value, next, __ATOMIC_SEQ_CST);
    return *this;
  }
  Observed &operator=(const Observed &other) {
    return *this = static_cast<T>(other);
  }
};
void publish(Observed<uint64_t> &target, uint64_t value) { target = value; }

struct GpuBudget::Shared {
  Observed<uint64_t> magic;
  Observed<uint64_t> sequence;
  Observed<uint64_t> budget;
  Observed<uint64_t> last_sample_ns;
  Observed<uint64_t> next_generation;
  struct Instance {
    Observed<char> name[kName];
    std::string name_string() const {
      std::string result;
      for (const auto &c : name) {
        const char value = c;
        if (!value)
          break;
        result += value;
      }
      return result;
    }
    Observed<uint64_t> limit;
    Observed<uint64_t> revision;
    Observed<uint64_t> has_revision;
    Observed<uint64_t> limit_revision;
    Observed<uint64_t> has_limit_revision;
    Observed<uint64_t> valid;
    Observed<uint64_t> participants;
    Observed<uint64_t> generation;
    Observed<uint64_t> expected_participants;
    Observed<uint64_t> min;
    Observed<int64_t> priority;
    Observed<uint64_t> headroom;
    Observed<uint64_t> demand;
  } instances[kInstances];
  struct Pool {
    Observed<uint64_t> owner;
    Observed<uint64_t> instance;
    Observed<int64_t> group;
    alignas(8) Observed<uint64_t> charged;
    Observed<uint64_t> capacity;
    Observed<uint64_t> page_bundle;
    Observed<uint64_t> minimum;
    Observed<uint64_t> ready;
    Observed<uint64_t> can_shrink;
    Observed<uint64_t> target;
    Observed<uint64_t> acknowledged_revision;
    Observed<uint64_t> acknowledged;
    Observed<uint64_t> failed_revision;
    Observed<uint64_t> failed;
  } pools[kPools];
};

struct GpuBudget::Locked {
  GpuBudget &budget;
  std::unique_lock<std::mutex> local;
  bool publishing = false;
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
    if (value.shared_ && value.shared_->magic == kMagic) {
      // A subsequent writer also recovers an odd sequence left by a dead
      // writer.
      value.shared_->sequence = (value.shared_->sequence + 1) | 1;
      publishing = true;
    }
  }
  ~Locked() {
    if (publishing)
      budget.shared_->sequence = budget.shared_->sequence + 1;
    flock(budget.fd_, LOCK_UN);
  }
};

std::shared_ptr<GpuBudget> GpuBudget::connect(const std::string &path,
                                              const Sampler &sampler) {
  return std::shared_ptr<GpuBudget>(new GpuBudget(path, sampler));
}

std::shared_ptr<GpuBudget> GpuBudget::open_control(const std::string &path,
                                                   bool observe) {
  return std::shared_ptr<GpuBudget>(new GpuBudget(path, observe));
}

GpuBudget::GpuBudget(const std::string &path, bool observe)
    : path_(path), pid_(getpid()) {
  fd_ = open(path.c_str(), O_RDWR | O_CLOEXEC | O_NOFOLLOW);
  if (fd_ < 0)
    throw system_error("open live KV control");
  try {
    // Serialize with epoch initialization, then require an existing user.
    if (lease(fd_, F_WRLCK, !observe, 2) < 0)
      throw system_error("lock KV control attachment");
    if (lease(fd_, F_WRLCK, false) == 0)
      throw std::runtime_error("KV control instance is not running");
    if (errno != EAGAIN && errno != EACCES)
      throw system_error("inspect live KV control");
    if (lease(fd_, F_RDLCK, true) < 0)
      throw system_error("retain live KV control");
    {
      // The initialization lease stabilizes the layout. Observation must not
      // queue behind the accounting flock, even during attachment/teardown.
      std::unique_ptr<Locked> lock;
      if (!observe)
        lock = std::make_unique<Locked>(*this);
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
      *shared_ = Shared{};
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
  if (getpid() == pid_ && shared_ && !slots_.empty()) {
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

size_t GpuBudget::register_pool(const std::string &instance, int64_t group,
                                const InstancePolicy &startup) {
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
    if (shared_->instances[i].valid &&
        instance == shared_->instances[i].name_string()) {
      instance_index = i;
      break;
    }
  }
  if (instance_index == kInstances) {
    for (size_t i = 0; i < kInstances; ++i) {
      auto &entry = shared_->instances[i];
      if (!entry.valid) {
        entry = Shared::Instance{};
        for (size_t n = 0; n < instance.size(); ++n)
          entry.name[n] = instance[n];
        entry.limit = UINT64_MAX;
        entry.min = startup.min_bytes;
        entry.priority = startup.priority;
        entry.headroom = startup.headroom_bytes;
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
      entry = Shared::Pool{};
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
  result.watermark_bytes = watermark_locked(instance, shared_);
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

// Liveness only, as control queries check it: a crashed instance's pools
// keep their entries, and nothing here refunds or retires them. Cleanup is
// the orchestrator's.
bool GpuBudget::pool_alive(size_t slot, const Shared *data) const {
  const auto &pool = data->pools[slot];
  if (!pool.owner)
    return false;
  if (pool.owner == owner_)
    return true;
  struct flock owner_lock{};
  owner_lock.l_type = F_WRLCK;
  owner_lock.l_whence = SEEK_SET;
  owner_lock.l_start = kPoolLeaseStart + slot;
  owner_lock.l_len = 1;
  if (fcntl(fd_, F_OFD_GETLK, &owner_lock) < 0)
    throw system_error("inspect KV budget pool owner");
  return owner_lock.l_type != F_UNLCK;
}

// The part of an instance's demand that its own max still allows on this GPU.
// What it could not take anyway keeps nothing from other instances.
uint64_t GpuBudget::allowed_demand(size_t instance, uint64_t charged,
                                   const Shared *data) {
  const auto &entry = data->instances[instance];
  const uint64_t limit = entry.limit;
  return std::min<uint64_t>(entry.demand,
                            limit > charged ? limit - charged : 0);
}

// headroom + shortfall of #585, where the shortfall is the demand that the
// headroom does not cover. An instance none of whose pools is alive is not
// read; only instances that would count are checked.
uint64_t GpuBudget::watermark_locked(size_t instance,
                                     const Shared *data) const {
  const int64_t priority = data->instances[instance].priority;
  bool counted[kInstances]{};
  for (size_t i = 0; i < kInstances; ++i) {
    const auto &entry = data->instances[i];
    counted[i] = i != instance && entry.valid && entry.priority > priority &&
                 std::max<uint64_t>(entry.headroom, entry.demand) > 0;
  }
  bool live[kInstances]{};
  uint64_t charged[kInstances]{};
  for (size_t slot = 0; slot < kPools; ++slot) {
    const auto &pool = data->pools[slot];
    const uint64_t owner = pool.instance;
    if (owner >= kInstances || !counted[owner] || !pool.owner)
      continue;
    charged[owner] = add(charged[owner], pool.charged);
    if (!live[owner] && pool_alive(slot, data))
      live[owner] = true;
  }
  uint64_t total = 0;
  for (size_t i = 0; i < kInstances; ++i)
    if (live[i])
      total = saturating_add(
          total, std::max<uint64_t>(data->instances[i].headroom,
                                    allowed_demand(i, charged[i], data)));
  return total;
}

bool GpuBudget::post_demand(const std::string &instance, uint64_t generation,
                            uint64_t bytes) {
  Locked lock(*this);
  for (auto &entry : shared_->instances) {
    if (entry.valid && entry.generation == generation &&
        instance == entry.name_string()) {
      if (entry.demand != bytes)
        publish(entry.demand, bytes);
      return true;
    }
  }
  return false;
}

void GpuBudget::set_demand(size_t slot, uint64_t bytes) {
  Locked lock(*this);
  (void)snapshot_locked(slot);
  auto &entry = shared_->instances[shared_->pools[slot].instance];
  if (entry.demand != bytes)
    publish(entry.demand, bytes);
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
        // Free memory under the watermark is kept for higher priorities.
        const auto usable = state.budget_bytes > state.watermark_bytes
                                ? state.budget_bytes - state.watermark_bytes
                                : 0;
        if (state.charged_bytes > usable ||
            bytes > usable - state.charged_bytes ||
            state.instance_charged_bytes > state.instance_limit_bytes ||
            bytes > state.instance_limit_bytes - state.instance_charged_bytes)
          throw std::runtime_error(
              "capacity_exhausted: shared KV budget would be exceeded"
              " (requested=" +
              std::to_string(bytes) +
              ", budget=" + std::to_string(state.budget_bytes) +
              ", charged=" + std::to_string(state.charged_bytes) +
              ", watermark=" + std::to_string(state.watermark_bytes) + ")");
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
    return publish_control_locked(shared_->pools[slot].instance,
                                  ControlUpdate{bytes, {}, {}, {}}, revision)
        .status;
  if (entry.has_revision && revision < entry.revision)
    return "stale";
  if (entry.has_revision && revision == entry.revision)
    return entry.limit == bytes ? "applied" : "conflict";
  publish(entry.limit, bytes);
  entry.revision = revision;
  entry.has_revision = 1;
  entry.limit_revision = revision;
  entry.has_limit_revision = 1;
  return "applied";
}

void GpuBudget::configure_control_pool(size_t slot, uint64_t capacity,
                                       uint64_t page_bundle, bool can_shrink,
                                       uint64_t expected_participants,
                                       bool reserve_null) {
  if (!page_bundle || capacity % page_bundle ||
      (reserve_null && capacity < page_bundle))
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
  pool.minimum = reserve_null ? page_bundle : 0;
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

InstanceControl GpuBudget::control_state_locked(size_t instance, size_t slot,
                                                const Shared *snapshot) const {
  const auto *data = snapshot ? snapshot : shared_;
  const auto &entry = data->instances[instance];
  InstanceControl state;
  state.published = entry.has_limit_revision;
  state.generation = entry.generation;
  state.revision = entry.revision;
  state.limit_revision = entry.limit_revision;
  state.requested_bytes = entry.limit;
  state.participants = entry.participants;
  state.expected_participants = entry.expected_participants;
  state.min_bytes = entry.min;
  state.priority = entry.priority;
  state.headroom_bytes = entry.headroom;
  state.watermark_bytes = watermark_locked(instance, data);
  uint64_t total = 0;
  for (const auto &pool : data->pools)
    total = add(total, pool.charged);
  const uint64_t budget = data->budget;
  const auto usable =
      budget > state.watermark_bytes ? budget - state.watermark_bytes : 0;
  state.status = state.published ? "pending" : "unpublished";
  bool unsupported = false, cannot_shrink = false;
  uint64_t live = 0;
  for (size_t i = 0; i < kPools; ++i) {
    const auto &pool = data->pools[i];
    if (!pool.owner || pool.instance != instance)
      continue;
    state.charged_bytes = add(state.charged_bytes, pool.charged);
    state.effective_bytes = add(state.effective_bytes,
                                state.published ? pool.target : pool.capacity);
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
      state.minimum_bytes = pool.minimum;
      state.can_shrink = pool.can_shrink;
      state.ready = pool.ready;
    }
    if (pool.failed && pool.failed_revision == entry.limit_revision) {
      ++state.failed;
      state.failure_reason = "quarantined_pages_prevent_resize";
    }
    cannot_shrink |= !pool.can_shrink;
    unsupported |= !pool.can_shrink && pool.target < pool.capacity;
    if (pool.acknowledged &&
        pool.acknowledged_revision == entry.limit_revision &&
        pool.charged <= pool.target)
      ++state.completed;
  }
  if (state.published) {
    if (live != entry.participants)
      state.status = "unavailable";
    else if (state.failed)
      state.status = "failed";
    else if (unsupported ||
             (cannot_shrink && state.effective_bytes > entry.limit))
      state.status = "unsupported";
    else if (state.completed == entry.participants &&
             state.charged_bytes <= entry.limit)
      state.status = "converged";
  }
  const auto demand = allowed_demand(instance, state.charged_bytes, data);
  state.shortfall_bytes = demand > entry.headroom ? demand - entry.headroom : 0;
  state.available_bytes = std::min<uint64_t>(
      usable > total ? usable - total : 0,
      entry.limit > state.charged_bytes ? entry.limit - state.charged_bytes
                                        : 0);
  state.registered_participants = live;
  state.convergence = state.status;
  return state;
}

InstanceControl
GpuBudget::publish_control_locked(size_t instance, const ControlUpdate &update,
                                  std::optional<uint64_t> revision) {
  auto &entry = shared_->instances[instance];
  auto rejected = [&](const char *status) {
    auto state = control_state_locked(instance);
    state.status = status;
    return state;
  };
  const uint64_t bytes = update.limit_bytes.value_or(entry.limit);
  if (revision && entry.has_revision) {
    if (*revision < entry.revision)
      return rejected("stale");
    if (*revision == entry.revision) {
      // A retry of this revision must carry the values it published.
      if ((update.limit_bytes &&
           (!entry.has_limit_revision || entry.limit_revision != *revision)) ||
          entry.limit != bytes ||
          (update.min_bytes && *update.min_bytes != entry.min) ||
          (update.priority && *update.priority != entry.priority) ||
          (update.headroom_bytes && *update.headroom_bytes != entry.headroom))
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
  // Only max gives the engines something to execute. Policy writes preserve
  // the executing revision, including reports arriving after this write.
  const bool retarget = update.limit_bytes || !entry.participants;
  // Proportional startup-capacity shares, independently rounded DOWN to each
  // pool's physical page bundle. The remainder stays unassigned. No pool can
  // independently consume the whole instance cap. Null-page floors constrain
  // physical targets only; the exact instance cap still gates native grants.
  const auto bounded = std::min(bytes, capacity);
  for (auto &pool : shared_->pools) {
    if (!retarget || !pool.owner || pool.instance != instance)
      continue;
    const auto share = static_cast<uint64_t>(
        static_cast<unsigned __int128>(bounded) * pool.capacity / capacity);
    pool.target = std::max(uint64_t(pool.minimum),
                           share / pool.page_bundle * pool.page_bundle);
    pool.acknowledged = 0;
    pool.failed = 0;
  }
  publish(entry.limit, bytes);
  if (update.min_bytes)
    publish(entry.min, *update.min_bytes);
  if (update.priority)
    entry.priority = *update.priority;
  if (update.headroom_bytes)
    publish(entry.headroom, *update.headroom_bytes);
  entry.revision =
      revision.value_or(entry.has_revision ? entry.revision + 1 : 1);
  entry.has_revision = 1;
  entry.participants = participants;
  if (update.limit_bytes) {
    entry.limit_revision = entry.revision;
    entry.has_limit_revision = 1;
  }
  auto state = control_state_locked(instance);
  state.status = "accepted";
  return state;
}

InstanceControl GpuBudget::publish_control(const std::string &instance,
                                           uint64_t bytes, uint64_t generation,
                                           std::optional<uint64_t> revision) {
  return publish_control(instance, ControlUpdate{bytes, {}, {}, {}}, generation,
                         revision);
}

InstanceControl GpuBudget::publish_control(const std::string &instance,
                                           const ControlUpdate &update,
                                           uint64_t generation,
                                           std::optional<uint64_t> revision) {
  retire_dead_pools();
  Locked lock(*this);
  for (size_t i = 0; i < kInstances; ++i) {
    if (shared_->instances[i].valid &&
        instance == shared_->instances[i].name_string()) {
      if (shared_->instances[i].generation != generation) {
        auto state = control_state_locked(i);
        state.status = "instance_changed";
        return state;
      }
      return publish_control_locked(i, update, revision);
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
    if (shared_->instances[i].valid &&
        instance == shared_->instances[i].name_string())
      return control_state_locked(i, slot);
  }
  InstanceControl state;
  state.status = "unavailable";
  return state;
}

InstanceControl GpuBudget::observe_control(const std::string &instance) {
  check_owner();
  InstanceControl unavailable;
  unavailable.status = unavailable.convergence = "unavailable";
  const uint64_t before = shared_->sequence;
  if (before & 1)
    return unavailable;
  // Bounded, atomic copy: never wait for a writer. The sequence rejects slot
  // reuse or interleaved revisions; all copied words remain race-free.
  auto snapshot = std::make_unique<Shared>(*shared_);
  InstanceControl result = unavailable;
  for (size_t i = 0; i < kInstances; ++i) {
    if (snapshot->instances[i].valid &&
        instance == snapshot->instances[i].name_string()) {
      result = control_state_locked(i, SIZE_MAX, snapshot.get());
      break;
    }
  }
  struct stat held{}, current{};
  if (fstat(fd_, &held) < 0 || lstat(path_.c_str(), &current) < 0 ||
      held.st_dev != current.st_dev || held.st_ino != current.st_ino ||
      !S_ISREG(current.st_mode))
    return unavailable;
  if (shared_->sequence != before)
    return unavailable;
  return result;
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
  if (!entry.has_limit_revision || revision != entry.limit_revision)
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
  if (!entry.valid || instance != entry.name_string() ||
      entry.generation != generation || !entry.has_limit_revision ||
      entry.limit_revision != revision)
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
