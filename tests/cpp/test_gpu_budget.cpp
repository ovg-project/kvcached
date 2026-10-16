// SPDX-FileCopyrightText: Copyright contributors to the kvcached project
// SPDX-License-Identifier: Apache-2.0
#include "gpu_budget.hpp"

#include <cassert>
#include <chrono>
#include <csignal>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <sys/wait.h>
#include <thread>
#include <unistd.h>

using namespace kvcached;
constexpr uint64_t page = 2 * 1024 * 1024;

static void rejected(const std::function<void()> &action) {
  bool failed = false;
  try {
    action();
  } catch (const std::runtime_error &) {
    failed = true;
  }
  assert(failed);
}
static void child_ok(pid_t child) {
  int status = 0;
  assert(waitpid(child, &status, 0) == child);
  assert(WIFEXITED(status) && WEXITSTATUS(status) == 0);
}
static void signal_pipe(int fd) {
  char value = 1;
  assert(write(fd, &value, 1) == 1);
}
static void await_pipe(int fd) {
  char value;
  assert(read(fd, &value, 1) == 1);
}

// Exercise the same ordering as the static allocator registry: cleanup runs
// after main-thread TLS destruction, without an explicit shutdown call.
struct ExitBudgetCleanup {
  std::shared_ptr<GpuBudget> budget;
  std::unique_ptr<BudgetReceipt> receipt;
  size_t slot = 0;
  std::string path;
  std::string directory;
  ~ExitBudgetCleanup() {
    if (!budget)
      return;
    {
      BudgetMutation mutation(budget);
      receipt->release();
      mutation.finish();
      assert(budget->snapshot(slot).charged_bytes == 0);
    }
    receipt.reset();
    budget.reset();
    unlink(path.c_str());
    rmdir(directory.c_str());
  }
};
static ExitBudgetCleanup exit_cleanup;

static void check_interrupted_initialization(const std::string &path) {
  for (bool crash : {false, true}) {
    {
      auto previous = GpuBudget::connect(path, [] { return 8 * page; });
      auto grant =
          previous->grant(previous->register_pool("old-epoch", 0), 2 * page);
      auto abandoned = grant->take(2 * page);
      grant->finish();
    }
    int sampling[2], finish[2], ready[2];
    assert(pipe(sampling) == 0 && pipe(finish) == 0 && pipe(ready) == 0);
    const auto initializer = fork();
    assert(initializer >= 0);
    if (!initializer) {
      rejected([&] {
        GpuBudget::connect(path, [&]() -> uint64_t {
          signal_pipe(sampling[1]);
          await_pipe(finish[0]);
          if (crash)
            _exit(0);
          throw std::runtime_error("interrupted initial sample");
        });
      });
      _exit(0);
    }
    await_pipe(sampling[0]);
    pid_t waiters[2];
    for (int i = 0; i < 2; ++i) {
      waiters[i] = fork();
      assert(waiters[i] >= 0);
      if (!waiters[i]) {
        signal_pipe(ready[1]);
        auto fresh = GpuBudget::connect(path, [] { return page; });
        const auto state =
            fresh->snapshot(fresh->register_pool("new-epoch", 0));
        assert(state.budget_bytes == page && state.charged_bytes == 0);
        _exit(0);
      }
    }
    await_pipe(ready[0]);
    await_pipe(ready[0]);
    // Let both connectors enter the kernel wait while the sampler is held.
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    signal_pipe(finish[1]);
    child_ok(initializer);
    child_ok(waiters[0]);
    child_ok(waiters[1]);
    for (int fd :
         {sampling[0], sampling[1], finish[0], finish[1], ready[0], ready[1]})
      close(fd);
  }
  std::cout << "PASS epoch initialization: failed and crashed samplers cannot "
               "publish stale state\n";
}

#ifdef KVCACHED_BUDGET_CAS_ACCOUNTING
static int race_begin[2], race_entered[2], race_finish[2], race_refunded[2],
    race_resume[2];
static void calibration_race_hook(int stage) {
  if (stage == 20) {
    signal_pipe(race_begin[1]);
    await_pipe(race_entered[0]);
  }
  if (stage == 30) {
    signal_pipe(race_refunded[1]);
    await_pipe(race_resume[0]);
  }
}
static void check_cas_calibration_race(const std::string &path) {
  auto keeper = GpuBudget::connect(path, [] { return 3 * page; });
  const auto slot = keeper->register_pool("calibration-race", 0);
  int ready[2];
  assert(pipe(ready) == 0 && pipe(race_begin) == 0 && pipe(race_entered) == 0 &&
         pipe(race_finish) == 0 && pipe(race_refunded) == 0 &&
         pipe(race_resume) == 0);
  auto child = fork();
  assert(child >= 0);
  if (!child) {
    auto owner = GpuBudget::connect(path, [] { return 0; });
    const auto own = owner->register_pool("calibration-race", 1000);
    auto grant = owner->grant(own, 2 * page);
    auto receipt = grant->take(2 * page);
    grant->finish();
    signal_pipe(ready[1]);
    await_pipe(race_begin[0]);
    BudgetMutation mutation(owner);
    signal_pipe(race_entered[1]);
    await_pipe(race_finish[0]);
    owner->set_accounting_test_hook(calibration_race_hook);
    receipt->release();
    mutation.finish();
    _exit(0);
  }
  await_pipe(ready[0]);
  keeper->set_accounting_test_hook(calibration_race_hook);
  const bool applied = keeper->recalibrate([] {
    signal_pipe(race_finish[1]);
    await_pipe(race_refunded[0]);
    return 2 * page;
  });
  signal_pipe(race_resume[1]);
  child_ok(child);
  keeper->set_accounting_test_hook(nullptr);
  assert(!applied); // The driver sample crossed a real release.
  assert(keeper->snapshot(slot).charged_bytes == 0);
  assert(keeper->recalibrate([] { return 2 * page; }));
  assert(keeper->snapshot(slot).budget_bytes == 2 * page);
  for (auto pair : {ready, race_begin, race_entered, race_finish, race_refunded,
                    race_resume})
    for (int i = 0; i < 2; ++i)
      close(pair[i]);
  std::cout << "PASS CAS calibration: release completion cannot validate a "
               "torn sample\n";
}
static int crash_stage = 0;
static void exit_at_stage(int stage) {
  if (stage == crash_stage)
    _exit(0);
}
static void check_cas_interruption(const std::string &path) {
  for (int stage : {1, 2, 3, 4, 10, 11, 12, 13}) {
    auto keeper = GpuBudget::connect(path, [] { return 3 * page; });
    const auto slot = keeper->register_pool("cas-crash", 0);
    const auto child = fork();
    assert(child >= 0);
    if (!child) {
      auto owner = GpuBudget::connect(path, [] { return 0; });
      const auto own = owner->register_pool("cas-crash", 1000);
      crash_stage = stage;
      if (stage < 10)
        owner->set_accounting_test_hook(exit_at_stage);
      auto grant = owner->grant(own, 2 * page);
      auto receipt = grant->take(2 * page);
      grant->finish();
      owner->set_accounting_test_hook(exit_at_stage);
      receipt->release();
      _exit(3);
    }
    child_ok(child);
    const auto state = keeper->snapshot(slot);
    const uint64_t gpu = (stage <= 2 || stage == 13) ? 0 : 2 * page;
    const uint64_t instance = (stage == 1 || stage >= 12) ? 0 : 2 * page;
    assert(state.charged_bytes == gpu);
    assert(state.instance_charged_bytes == instance);
    assert(!keeper->recalibrate([] { return 3 * page; }));
    // Surviving workers can still use the unambiguous remaining credit.
    keeper->grant(slot, page)->finish();
    assert(keeper->snapshot(slot).charged_bytes == gpu);
  }
  {
    auto budget = GpuBudget::connect(path, [] { return 3 * page; });
    const auto a = budget->register_pool("first", 0);
    const auto b = budget->register_pool("second", 0);
    auto first = budget->grant(a, 2 * page);
    rejected([&] { budget->grant(b, 2 * page); });
    assert(budget->snapshot(b).instance_charged_bytes == 0);
    auto second = budget->grant(b, page);
    first->finish();
    second->finish();
    assert(budget->snapshot(a).charged_bytes == 0);
  }
  std::cout << "PASS CAS accounting: eight interrupted debit/refund stages and "
               "partial reservation rollback\n";
}
#endif

static int calibration_wait_entered[2];
static void calibration_wait_hook(int stage) {
  if (stage == 40)
    signal_pipe(calibration_wait_entered[1]);
}
static void check_calibration_wait(const std::string &path) {
  auto budget = GpuBudget::connect(path, [] { return 3 * page; });
  auto contender = GpuBudget::connect(path, [] { return 0; });
  const auto slot = contender->register_pool("calibration-contender", 0);
  assert(pipe(calibration_wait_entered) == 0);
  contender->set_accounting_test_hook(calibration_wait_hook);
  std::thread worker;
  assert(budget->recalibrate([&] {
    // A recursive sampler call is rejected instead of deadlocking itself.
    rejected([&] { contender->grant(slot, page); });
    worker = std::thread([&] { contender->grant(slot, page)->finish(); });
    await_pipe(calibration_wait_entered[0]);
    // A transient calibration must not become an engine-visible allocation
    // miss. The waiter holds neither a data lock nor a pending charge.
    const auto state = contender->snapshot(slot);
    assert(state.charged_bytes == 0 && state.active_mutations == 0);
    return 2 * page;
  }));
  worker.join();
  contender->set_accounting_test_hook(nullptr);
  assert(contender->snapshot(slot).charged_bytes == 0);
  for (int fd : calibration_wait_entered)
    close(fd);
  std::cout
      << "PASS calibration contention: wait without a false capacity miss\n";
}

static void check_capacity_recovery(const std::string &path) {
  {
    auto budget = GpuBudget::connect(path, [] { return 10 * page; });
    const auto slot = budget->register_pool("live-recovery", 0);
    auto grant = budget->grant(slot, 2 * page);
    auto resident = grant->take(2 * page);
    grant->finish();
    assert(budget->recalibrate([] { return page; }));
    assert(budget->snapshot(slot).budget_bytes == 3 * page);
    rejected([&] { budget->grant(slot, 2 * page); });
    assert(budget->recalibrate([] { return 6 * page; }));
    assert(budget->snapshot(slot).budget_bytes == 8 * page);
    auto recovered = budget->grant(slot, 6 * page);
    rejected([&] { budget->grant(slot, 1); });
    recovered->finish();
    resident->release();
    assert(budget->snapshot(slot).charged_bytes == 0);
    assert(budget->recalibrate([] { return 0; }));
    std::this_thread::sleep_for(std::chrono::milliseconds(1100));
    const auto recovered_query = budget->snapshot(slot, [] { return page; });
    assert(recovered_query.budget_bytes == page);
  }
  // An idle owner can die before or during the sample, after a complete grant.
  // Its stale charge is retained on BOTH sides of budget-used; it must not
  // restrict future use of genuinely free memory or create extra credit.
  for (bool die_during_sample : {false, true}) {
    auto keeper = GpuBudget::connect(path, [] { return 3 * page; });
    const auto slot = keeper->register_pool("survivor", 0);
    int ready[2], finish[2];
    assert(pipe(ready) == 0 && pipe(finish) == 0);
    const auto child = fork();
    assert(child >= 0);
    if (!child) {
      auto owner = GpuBudget::connect(path, [] { return 0; });
      auto grant = owner->grant(owner->register_pool("dead-idle", 0), 2 * page);
      auto abandoned = grant->take(2 * page);
      grant->finish();
      signal_pipe(ready[1]);
      await_pipe(finish[0]);
      _exit(0);
    }
    await_pipe(ready[0]);
    if (!die_during_sample) {
      signal_pipe(finish[1]);
      child_ok(child);
    }
    assert(keeper->recalibrate([&] {
      if (die_during_sample) {
        signal_pipe(finish[1]);
        child_ok(child);
      }
      // The driver observes all 3 physical pages free after owner exit.
      return 3 * page;
    }));
    const auto state = keeper->snapshot(slot);
    assert(state.charged_bytes == 2 * page && state.budget_bytes == 5 * page);
    auto full_free = keeper->grant(slot, 3 * page);
    rejected([&] { keeper->grant(slot, 1); });
    full_free->finish();
    assert(keeper->snapshot(slot).charged_bytes == 2 * page);
    // A later external allocation consumes one physical page. No stale-owner
    // credit may mask it, and pending promises cannot trigger recalibration.
    assert(keeper->recalibrate([] { return 2 * page; }));
    auto pending = keeper->grant(slot, 2 * page);
    assert(!keeper->recalibrate([] { return 3 * page; }));
    rejected([&] { keeper->grant(slot, 1); });
    pending->finish();
    for (int fd : {ready[0], ready[1], finish[0], finish[1]})
      close(fd);
  }
  std::cout << "PASS capacity recovery: temporary pressure, dead idle owners, "
               "death during sampling and exact remaining-credit bound\n";
}

int main() {
  const char *temp = std::getenv("TMPDIR");
  std::string dir =
      std::string(temp ? temp : "/tmp") + "/kvcached-budget-test-XXXXXX";
  assert(mkdtemp(dir.data()));
  const std::string path = dir + "/budget";
  {
    auto budget = GpuBudget::connect(path, [] { return 6 * page; });
    auto same_gpu = GpuBudget::connect(path, []() -> uint64_t {
      throw std::runtime_error("resampled live epoch");
    });
    const auto a = budget->register_pool("instance-a", 0);
    const auto b = same_gpu->register_pool("instance-a", 1000);
    const auto c = same_gpu->register_pool("instance-b", 0);
    assert(budget->set_instance_limit(a, 4 * page, 7) == "applied");
    assert(same_gpu->set_instance_limit(b, 5 * page, 6) == "stale");
    assert(same_gpu->set_instance_limit(b, 5 * page, 7) == "conflict");
    auto first = budget->grant(a, 3 * page);
    auto receipt = first->take(2 * page);
    first->finish(); // unused credit returned exactly once
    first->finish();
    auto second = same_gpu->grant(b, 2 * page);
    auto other = second->take(2 * page);
    second->finish();
    assert(budget->snapshot(a).instance_charged_bytes == 4 * page);
    rejected([&] { same_gpu->grant(b, page); });
    auto unrelated = same_gpu->grant(c, 2 * page);
    rejected([&] { budget->grant(a, page); });
    assert(!budget->recalibrate([] { return 0; }));
    unrelated->finish();
    assert(budget->set_instance_limit(a, page, 8) == "applied");
    rejected([&] { same_gpu->grant(b, page); });
    {
      BudgetMutation outer(budget);
      BudgetMutation nested(budget);
      receipt->release();
      receipt->release();
      nested.finish();
      assert(budget->snapshot(a).charged_bytes == 4 * page);
      outer.finish();
    }
    other->release();
    assert(budget->snapshot(a).charged_bytes == 0);
    assert(budget->snapshot(a).active_mutations == 0);
    assert(budget->set_instance_limit(a, 6 * page, 9) == "applied");
    assert(budget->recalibrate([&] {
      // Sampling does not hold the data lock, but grants cannot race it.
      assert(same_gpu->snapshot(c).charged_bytes == 0);
      rejected([&] { same_gpu->grant(c, page); });
      return 5 * page;
    }));
    assert(budget->snapshot(a).budget_bytes == 5 * page);
    rejected([&] {
      budget->recalibrate([]() -> uint64_t {
        throw std::runtime_error("driver sample failed");
      });
    });
    auto recovered = same_gpu->grant(c, page);
    recovered->finish();
    // A release racing a sample changes the generation; no stale budget update.
    assert(!budget->recalibrate([&] {
      BudgetMutation mutation(same_gpu);
      return 0;
    }));
    assert(budget->snapshot(a).budget_bytes == 5 * page);
    const auto child = fork();
    assert(child >= 0);
    if (!child) {
      rejected([&] { budget->snapshot(a); });
      auto own = GpuBudget::connect(path, [] { return 0; });
      own->recalibrate([]() -> uint64_t { _exit(0); });
      _exit(2);
    }
    child_ok(child);
    auto after_crash = budget->grant(a, page);
    after_crash->finish(); // dead calibrator's kernel lease was released
    unsigned samples = 0;
    auto sampler = [&] {
      ++samples;
      return 4 * page;
    };
    budget->grant(a, page, nullptr, sampler)->finish();
    assert(samples == 0); // no query in the recent-sample fast path
    std::this_thread::sleep_for(std::chrono::milliseconds(1100));
    budget->grant(a, page, nullptr, sampler)->finish();
    budget->grant(a, page, nullptr, sampler)->finish();
    assert(samples == 1 && budget->snapshot(a).budget_bytes == 4 * page);
    auto resident = budget->grant(a, 2 * page);
    auto physical = resident->take(2 * page);
    assert(!budget->recalibrate([] { return page; }));
    resident->finish();
    assert(budget->recalibrate([] { return page; }));
    assert(budget->snapshot(a).budget_bytes == 3 * page);
    rejected([&] { budget->grant(a, 2 * page); });
    physical->release();
    assert(budget->snapshot(a).charged_bytes == 0);
  }
  {
    auto keeper = GpuBudget::connect(path, [] { return 3 * page; });
    const auto slot = keeper->register_pool("keeper", 0);
    assert(keeper->set_instance_limit(slot, 2 * page, 17) == "applied");
    // Keep the epoch alive while more than one full instance table turns over.
    // A pool shared with the keeper must retain its cap and revision.
    for (unsigned i = 0; i < 300; ++i) {
      auto participant = GpuBudget::connect(path, [] { return 0; });
      const auto shared = participant->register_pool("keeper", 1000);
      const auto temporary =
          participant->register_pool("churn-" + std::to_string(i), 0);
      participant->grant(temporary, page)->finish();
      assert(participant->snapshot(shared).instance_limit_bytes == 2 * page);
      assert(participant->set_instance_limit(shared, page, 16) == "stale");
    }
    assert(keeper->snapshot(slot).charged_bytes == 0);
    assert(keeper->set_instance_limit(slot, 3 * page, 18) == "applied");
    int ready[2], start[2], held[2], finish[2];
    assert(pipe(ready) == 0 && pipe(start) == 0 && pipe(held) == 0 &&
           pipe(finish) == 0);
    pid_t children[2];
    for (int i = 0; i < 2; ++i) {
      children[i] = fork();
      assert(children[i] >= 0);
      if (!children[i]) {
        auto participant = GpuBudget::connect(path, [] { return 0; });
        const auto owner =
            participant->register_pool("worker-" + std::to_string(i), 0);
        signal_pipe(ready[1]);
        await_pipe(start[0]);
        std::unique_ptr<BudgetGrant> grant;
        try {
          grant = participant->grant(owner, 2 * page);
        } catch (const std::runtime_error &) {
        }
        signal_pipe(held[1]);
        await_pipe(finish[0]);
        if (grant)
          grant->finish();
        _exit(0);
      }
    }
    await_pipe(ready[0]);
    await_pipe(ready[0]);
    signal_pipe(start[1]);
    signal_pipe(start[1]);
    await_pipe(held[0]);
    await_pipe(held[0]);
    assert(keeper->snapshot(slot).charged_bytes == 2 * page);
    signal_pipe(finish[1]);
    signal_pipe(finish[1]);
    child_ok(children[0]);
    child_ok(children[1]);
    assert(keeper->snapshot(slot).charged_bytes == 0);
    for (int fd : {ready[0], ready[1], start[0], start[1], held[0], held[1],
                   finish[0], finish[1]})
      close(fd);
    auto child = fork();
    assert(child >= 0);
    if (!child) {
      auto owner = GpuBudget::connect(path, [] { return 0; });
#ifdef KVCACHED_BUDGET_TESTING
      // Die after publishing the charge, while the accounting lock is held.
      // Both synchronization implementations must recover exclusion without
      // guessing that the dead worker's physical resources have been released.
      owner->set_grant_test_hook([] { _exit(0); });
#endif
      auto grant = owner->grant(owner->register_pool("crashed", 0), 2 * page);
      (void)grant->take(
          page); // losing an owner is not a confirmed physical free
      _exit(0);
    }
    child_ok(child);
    assert(keeper->snapshot(slot).charged_bytes == 2 * page);
    keeper->grant(slot, page)->finish(); // recovered exclusion remains usable
    rejected([&] { keeper->grant(slot, 2 * page); });
    assert(!keeper->recalibrate([] { return 3 * page; }));
  }
  {
    // Only a new epoch, after all participant leases close, discards old
    // credit.
    auto fresh = GpuBudget::connect(path, [] { return page; });
    auto slot = fresh->register_pool("new-epoch", 0);
    assert(fresh->snapshot(slot).charged_bytes == 0);
    assert(fresh->snapshot(slot).budget_bytes == page);
    const auto renamed = path + ".old";
    assert(rename(path.c_str(), renamed.c_str()) == 0);
    auto replacement = GpuBudget::connect(path, [] { return 0; });
    rejected([&] { fresh->grant(slot, page); });
    unlink(renamed.c_str());
  }
  {
    auto budget = GpuBudget::connect(path, [] { return 4 * page; });
    auto slot = budget->register_pool("headroom-deficit", 0);
    auto grant = budget->grant(slot, 2 * page);
    auto first = grant->take(page);
    auto second = grant->take(page);
    grant->finish();
    assert(budget->recalibrate([] { return -static_cast<int64_t>(page); }));
    assert(budget->snapshot(slot).budget_bytes == page);
    first->release();
    rejected([&] { budget->grant(slot, page); });
    second->release();
    budget->grant(slot, page)->finish();
    assert(budget->recalibrate([] { return INT64_MIN; }));
    assert(budget->snapshot(slot).budget_bytes == 0);
  }
  {
    auto budget = GpuBudget::connect(path, [] { return -1; });
    auto slot = budget->register_pool("initial-deficit", 0);
    assert(budget->snapshot(slot).budget_bytes == 0);
  }
  std::cout
      << "PASS calibration: headroom deficit survives release and retry\n";
  check_interrupted_initialization(path);
  check_capacity_recovery(path);
  check_calibration_wait(path);

#ifdef KVCACHED_BUDGET_CAS_ACCOUNTING
  check_cas_interruption(path);
  check_cas_calibration_race(path);
#endif
  // A real static owner keeps its ledger alive through process teardown.
  exit_cleanup.path = dir + "/exit-budget";
  exit_cleanup.directory = dir;
  exit_cleanup.budget =
      GpuBudget::connect(exit_cleanup.path, [] { return page; });
  exit_cleanup.slot = exit_cleanup.budget->register_pool("static-owner", 0);
  auto exit_grant = exit_cleanup.budget->grant(exit_cleanup.slot, page);
  exit_cleanup.receipt = exit_grant->take(page);
  exit_grant->finish();
  unlink(path.c_str());
  std::cout << "PASS shared budget: limits, batches, processes, refunds, "
               "calibration, crash, epoch, inode\n";
}
