// SPDX-FileCopyrightText: Copyright contributors to the kvcached project
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace kvcached {

struct BudgetSnapshot {
  uint64_t budget_bytes = 0;
  uint64_t charged_bytes = 0;
  uint64_t instance_charged_bytes = 0;
  uint64_t instance_limit_bytes = UINT64_MAX;
  uint64_t sync_acquisitions = 0;
  uint64_t sync_contentions = 0;
  uint64_t sync_acquire_ns = 0;
  uint64_t sync_wait_ns = 0;
  uint64_t sync_dead_owner_recoveries = 0;
};

// Observations are forecasts, never admission authority. Fields can come from
// different updates; charged includes reservations and unconfirmed releases.
struct BudgetPoolObservation {
  uint64_t generation = 0;
  std::string instance;
  int64_t group = 0;
  uint64_t charged_bytes = 0;
  uint64_t limit_bytes = 0;
  uint64_t revision = 0;
  bool has_revision = false;
};
struct BudgetObservation {
  uint64_t budget_bytes = 0;
  uint64_t last_sample_ns = 0;
  // A slot changed identity during this pass and was omitted. No retries.
  bool incomplete = false;
  std::vector<BudgetPoolObservation> pools;
};
// Internal path-based reader. Does not create files, sample, acquire the ledger
// flock, or retain a connection after returning. Throws for unavailable,
// initializing, replaced or incompatible ledgers. Names/layout stay private.
BudgetObservation observe_gpu_budget(const std::string &path);

class GpuBudget;

// A receipt never refunds itself: destroying an owner is not proof that a
// physical allocation has been released. Only checked release may return it.
class BudgetReceipt {
public:
  BudgetReceipt(std::shared_ptr<GpuBudget> budget, size_t slot, uint64_t bytes);
  BudgetReceipt(const BudgetReceipt &) = delete;
  BudgetReceipt &operator=(const BudgetReceipt &) = delete;
  void release();
  std::shared_ptr<GpuBudget> budget() const { return budget_; }

private:
  std::shared_ptr<GpuBudget> budget_;
  size_t slot_;
  uint64_t bytes_;
};

// One grant covers a complete native page batch. Unused bytes are refundable
// without CUDA work. Issued receipts stay charged until physical release.
class BudgetGrant {
public:
  BudgetGrant(std::shared_ptr<GpuBudget> budget, size_t slot, uint64_t bytes);
  ~BudgetGrant();
  BudgetGrant(const BudgetGrant &) = delete;
  BudgetGrant &operator=(const BudgetGrant &) = delete;
  std::unique_ptr<BudgetReceipt> take(uint64_t bytes);
  void finish();

private:
  friend class GpuBudget;
  std::shared_ptr<GpuBudget> budget_;
  size_t slot_;
  uint64_t unused_;
  bool active_ = true;
};

// Batch confirmed refunds on this thread, flushing before another receipt
// would reach this byte threshold. A larger indivisible handle refunds
// directly. This is a per-thread batching bound, not a bound on global sampling
// error.
inline constexpr uint64_t kBudgetRefundBatchBytes = 64 * 1024 * 1024;

// Nested scopes share a bounded refund batch. They never gate calibration or
// publish a shared in-flight flag. Physical release must precede refund().
class BudgetMutation {
public:
  explicit BudgetMutation(std::shared_ptr<GpuBudget> budget);
  ~BudgetMutation();
  BudgetMutation(const BudgetMutation &) = delete;
  BudgetMutation &operator=(const BudgetMutation &) = delete;
  void finish();

private:
  std::shared_ptr<GpuBudget> budget_;
  bool active_ = true;
};

// Linux process-shared accounting. No CUDA or IPC is invoked under its data
// lock. The sampler runs outside that lock at initialization/recalibration.
class GpuBudget : public std::enable_shared_from_this<GpuBudget> {
public:
  // Signed free bytes minus the safety margin, before clamping.
  using Sampler = std::function<int64_t()>;
  static std::shared_ptr<GpuBudget> connect(const std::string &path,
                                            const Sampler &sampler);
  ~GpuBudget();
  size_t register_pool(const std::string &instance, int64_t pool);
  std::unique_ptr<BudgetGrant> grant(size_t slot, uint64_t bytes,
                                     BudgetSnapshot *state = nullptr,
                                     const Sampler &sampler = {});
  BudgetSnapshot snapshot(size_t slot, const Sampler &sampler = {});
  // Returns applied, stale or conflict. Shrink may leave existing charges
  // above the target, in which case new grants are denied until release.
  std::string set_instance_limit(size_t slot, uint64_t bytes,
                                 uint64_t revision);
  bool recalibrate(const Sampler &sampler);
  void begin_mutation();
  void end_mutation();
  void refund(size_t slot, uint64_t bytes);
#ifdef KVCACHED_BUDGET_TESTING
  void set_grant_test_hook(void (*hook)()) { grant_test_hook_ = hook; }
  void set_accounting_test_hook(void (*hook)(int)) {
    accounting_test_hook_ = hook;
  }
#endif

private:
  friend class BudgetGrant;
  friend BudgetObservation observe_gpu_budget(const std::string &path);
  struct Shared;
  struct Locked;
  struct LocalMutation {
    size_t depth = 0;
    uint64_t bytes = 0;
    std::unordered_map<size_t, uint64_t> refunds;
  };
  explicit GpuBudget(const std::string &path, const Sampler &sampler);
  void check_owner() const;
  void reclaim_instances_locked();
  void retire_dead_pools();
  BudgetSnapshot snapshot_locked(size_t slot) const;
  void return_locked(size_t slot, uint64_t bytes);
  void flush_refunds(LocalMutation &local);
  void accounting_test_hook(int stage);
  int fd_ = -1;
  Shared *shared_ = nullptr;
  std::string path_;
  int64_t pid_ = 0;
  uint64_t owner_ = 0;
  std::mutex mutex_;
  // Owned by the budget, so allocator static teardown never accesses dead TLS.
  std::mutex mutation_mutex_;
  std::unordered_map<std::thread::id, LocalMutation> mutations_;
  std::vector<size_t> slots_;
  uint64_t sync_acquisitions_ = 0;
  uint64_t sync_contentions_ = 0;
  uint64_t sync_acquire_ns_ = 0;
  uint64_t sync_wait_ns_ = 0;
  uint64_t sync_dead_owner_recoveries_ = 0;
#ifdef KVCACHED_BUDGET_TESTING
  void (*grant_test_hook_)() = nullptr;
  void (*accounting_test_hook_)(int) = nullptr;
#endif
};

} // namespace kvcached
