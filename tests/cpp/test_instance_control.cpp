// SPDX-FileCopyrightText: Copyright contributors to the kvcached project
// SPDX-License-Identifier: Apache-2.0
#include "gpu_budget.hpp"

#include <atomic>
#include <cassert>
#include <chrono>
#include <fcntl.h>
#include <set>
#include <stdexcept>
#include <sys/file.h>
#include <sys/wait.h>
#include <thread>
#include <unistd.h>

using namespace kvcached;

int main() {
  char directory[] = "/tmp/kvcached-control-XXXXXX";
  assert(mkdtemp(directory));
  const auto path = std::string(directory) + "/ledger";
  {
    auto budget = GpuBudget::connect(path, [] { return 1000; });
    auto a = budget->register_pool("model", 0);
    auto b = budget->register_pool("model", 1);
    assert(budget
               ->publish_control("model", 100,
                                 budget->query_control("model").generation)
               .status == "not_ready");
    budget->configure_control_pool(a, 120, 12, true);
    budget->configure_control_pool(b, 240, 24, true);
    budget->ready_control_pool(a);
    budget->ready_control_pool(b);
    auto writer = GpuBudget::open_control(path);
    assert(!writer->query_control("model").published);
    assert(writer->query_control("model").effective_bytes == 360);
    auto first = writer->publish_control(
        "model", 180, budget->query_control("model").generation);
    assert(first.status == "accepted" && first.revision == 1);
    assert(first.effective_bytes == 180 && first.completed == 0);
    assert(budget->control_state(a).pool_target_bytes == 60);
    assert(budget->control_state(b).pool_target_bytes == 120);
    assert(writer
               ->publish_control("model", 180,
                                 budget->query_control("model").generation, 1)
               .status == "idempotent");
    assert(writer
               ->publish_control("model", 181,
                                 budget->query_control("model").generation, 1)
               .status == "conflict");
    assert(writer
               ->publish_control("model", 180,
                                 budget->query_control("model").generation, 0)
               .status == "stale");
    assert(budget->acknowledge_control(a, 1));
    assert(budget->control_state(a).status == "pending");
    assert(budget->acknowledge_control(b, 1));
    assert(budget->control_state(a).status == "converged");
    auto second = writer->publish_control(
        "model", 180, budget->query_control("model").generation);
    assert(second.revision == 2 && second.completed == 0);
    assert(!budget->acknowledge_control(a, 1));
    assert(budget->control_state(a).status == "pending");
    auto rounded = writer->publish_control(
        "model", 179, budget->query_control("model").generation);
    assert(rounded.effective_bytes == 144);
    std::set<uint64_t> revisions;
    std::mutex mutex;
    std::vector<std::thread> writers;
    for (int i = 0; i != 8; ++i) {
      writers.emplace_back([&] {
        auto client = GpuBudget::open_control(path);
        for (int n = 0; n != 40; ++n) {
          auto result = client->publish_control(
              "model", 180, budget->query_control("model").generation);
          assert(result.status == "accepted");
          std::lock_guard<std::mutex> guard(mutex);
          assert(revisions.insert(result.revision).second);
        }
      });
    }
    for (auto &thread : writers)
      thread.join();
    assert(revisions.size() == 320);
    assert(*revisions.begin() == 4 && *revisions.rbegin() == 323);
  }
  {
    auto budget = GpuBudget::connect(path, [] { return 1000; });
    auto slot = budget->register_pool("model", 0);
    budget->configure_control_pool(slot, 120, 12, false);
    budget->ready_control_pool(slot);
    assert(!budget->control_state(slot).published);
    auto grant = budget->grant(slot, 48);
    auto receipt = grant->take(48);
    grant->finish();
    assert(budget
               ->publish_control("model", 0,
                                 budget->query_control("model").generation)
               .revision == 1);
    assert(budget->control_state(slot).status == "unsupported");
    assert(budget->acknowledge_control(slot, 1));
    assert(budget->control_state(slot).status == "unsupported");
    bool blocked = false;
    try {
      budget->grant(slot, 12);
    } catch (const std::runtime_error &) {
      blocked = true;
    }
    assert(blocked);
    receipt->release();
    assert(budget->control_state(slot).status == "unsupported");
    auto restored = budget->publish_control(
        "model", 120, budget->query_control("model").generation);
    assert(restored.convergence == "pending");
    assert(budget->acknowledge_control(slot, restored.revision));
    assert(budget->control_state(slot).status == "converged");
  }
  {
    auto budget = GpuBudget::connect(path, [] { return 1000; });
    auto slot = budget->register_pool("migration", 0);
    assert(budget->set_instance_limit(slot, 120, 7) == "applied");
    budget->configure_control_pool(slot, 240, 12, true);
    budget->ready_control_pool(slot);
    assert(budget
               ->publish_control("migration", 120,
                                 budget->query_control("migration").generation,
                                 7)
               .status == "accepted");
    assert(budget->control_state(slot).pool_target_bytes == 120);
    assert(budget->set_instance_limit(slot, 60, 8) == "accepted");
    assert(budget->control_state(slot).pool_target_bytes == 60);
    auto exhausted = budget->publish_control(
        "migration", 120, budget->query_control("migration").generation,
        UINT64_MAX);
    assert(exhausted.status == "accepted");
    assert(budget
               ->publish_control("migration", 120,
                                 budget->query_control("migration").generation)
               .status == "revision_exhausted");
  }
  {
    auto budget = GpuBudget::connect(path, [] { return 1000; });
    auto slot = budget->register_pool("membership", 0);
    budget->configure_control_pool(slot, 120, 12, true);
    budget->ready_control_pool(slot);
    auto sibling = GpuBudget::connect(path, [] { return 0; });
    auto peer = sibling->register_pool("membership", 1);
    sibling->configure_control_pool(peer, 120, 12, true);
    sibling->ready_control_pool(peer);
    assert(budget
               ->publish_control("membership", 120,
                                 budget->query_control("membership").generation)
               .status == "accepted");
    assert(budget->acknowledge_control(slot, 1));
    assert(sibling->acknowledge_control(peer, 1));
    assert(budget->control_state(slot).status == "converged");
    sibling.reset();
    assert(budget->control_state(slot).status == "unavailable");
    assert(budget
               ->publish_control("membership", 240,
                                 budget->query_control("membership").generation)
               .status == "unavailable");
    bool rejected = false;
    try {
      budget->register_pool("membership", 2);
    } catch (const std::runtime_error &) {
      rejected = true;
    }
    assert(rejected);
  }
  {
    auto budget = GpuBudget::connect(path, [] { return 1000; });
    auto slot = budget->register_pool("crash", 0);
    budget->configure_control_pool(slot, 120, 12, true);
    budget->ready_control_pool(slot);
    int ready[2], go[2];
    assert(pipe(ready) == 0 && pipe(go) == 0);
    auto child = fork();
    assert(child >= 0);
    if (child == 0) {
      close(ready[0]);
      close(go[1]);
      auto peer = GpuBudget::connect(path, [] { return 0; });
      auto other = peer->register_pool("crash", 1);
      peer->configure_control_pool(other, 120, 12, true);
      peer->ready_control_pool(other);
      auto grant = peer->grant(other, 48);
      auto receipt = grant->take(48);
      grant->finish();
      char signal = 'r';
      assert(write(ready[1], &signal, 1) == 1);
      assert(read(go[0], &signal, 1) == 1);
      assert(peer->acknowledge_control(other, 1));
      _exit(0); // No destructor: liveness must prevent false convergence.
    }
    close(ready[1]);
    close(go[0]);
    char signal;
    assert(read(ready[0], &signal, 1) == 1);
    auto revision = budget->publish_control(
        "crash", 120, budget->query_control("crash").generation);
    assert(revision.status == "accepted");
    assert(budget->acknowledge_control(slot, 1));
    assert(write(go[1], &signal, 1) == 1);
    int status;
    assert(waitpid(child, &status, 0) == child && WIFEXITED(status) &&
           WEXITSTATUS(status) == 0);
    auto state = budget->query_control("crash");
    assert(state.status == "unavailable" && state.completed == 1);
    assert(state.charged_bytes == 48); // Query must not refund dead ranks.
    close(ready[0]);
    close(go[1]);
  }
  {
    auto old = GpuBudget::connect(path, [] { return 1000; });
    auto slot = old->register_pool("restart", 0);
    old->configure_control_pool(slot, 120, 12, true);
    old->ready_control_pool(slot);
    auto writer = GpuBudget::open_control(path);
    auto before = writer->publish_control(
        "restart", 60, old->query_control("restart").generation);
    old.reset();
    auto replacement = GpuBudget::connect(path, [] { return 1000; });
    slot = replacement->register_pool("restart", 0);
    replacement->configure_control_pool(slot, 120, 12, true);
    replacement->ready_control_pool(slot);
    auto after = replacement->control_state(slot);
    assert(after.generation != before.generation && !after.published);
    assert(
        writer
            ->publish_control("restart", 60, before.generation, before.revision)
            .status == "instance_changed");
  }
  {
    auto budget = GpuBudget::connect(path, [] { return 1000; });
    auto slot = budget->register_pool("failure", 0);
    budget->configure_control_pool(slot, 120, 12, true);
    budget->ready_control_pool(slot);
    auto participant = budget->control_state(slot);
    auto writer = GpuBudget::open_control(path);
    auto first = writer->publish_control("failure", 60, participant.generation);
    assert(!writer->fail_control("failure", slot, participant.pool_owner + 1,
                                 first.generation, first.revision));
    assert(!writer->fail_control("failure", slot, participant.pool_owner,
                                 first.generation + 1, first.revision));
    assert(!writer->fail_control("other", slot, participant.pool_owner,
                                 first.generation, first.revision));
    assert(writer->fail_control("failure", slot, participant.pool_owner,
                                first.generation, first.revision));
    auto failed = writer->query_control("failure");
    assert(failed.status == "failed" && failed.failed == 1);
    assert(failed.failure_reason == "quarantined_pages_prevent_resize");
    assert(failed.requested_bytes == 60 && failed.effective_bytes == 60);
    assert(budget->snapshot(slot).instance_limit_bytes == 60);
    assert(budget->acknowledge_control(slot, first.revision));
    assert(writer->query_control("failure").status == "failed");
    assert(
        writer->publish_control("failure", 60, first.generation, first.revision)
            .convergence == "failed");
    auto next = writer->publish_control("failure", 120, first.generation);
    assert(next.convergence == "pending" && next.failed == 0);
    assert(next.failure_reason.empty());
    assert(!writer->fail_control("failure", slot, participant.pool_owner,
                                 first.generation, first.revision));
    assert(budget->acknowledge_control(slot, next.revision));
    assert(writer->query_control("failure").status == "converged");
  }
  {
    auto budget = GpuBudget::connect(path, [] { return 1000; });
    auto a = budget->register_pool("floors", 0);
    auto b = budget->register_pool("floors", 1);
    auto c = budget->register_pool("floors", 2);
    budget->configure_control_pool(a, 120, 12, true, 3, true);
    budget->configure_control_pool(b, 240, 24, true, 3, true);
    budget->configure_control_pool(c, 120, 12, true, 3, false);
    for (auto slot : {a, b, c})
      budget->ready_control_pool(slot);
    auto grant_a = budget->grant(a, 24);
    auto receipt_a = grant_a->take(24);
    grant_a->finish();
    auto grant_b = budget->grant(b, 24);
    auto receipt_b = grant_b->take(24);
    grant_b->finish();
    const auto generation = budget->query_control("floors").generation;
    for (auto limit : {0ULL, 1ULL, 36ULL}) {
      auto state = budget->publish_control("floors", limit, generation);
      assert(state.requested_bytes == limit);
      assert(budget->control_state(a).pool_target_bytes == 12);
      assert(budget->control_state(b).pool_target_bytes == 24);
      assert(budget->control_state(c).pool_target_bytes == 0);
      for (auto slot : {a, b, c})
        assert(budget->acknowledge_control(slot, state.revision));
      assert(budget->query_control("floors").status == "pending");
      bool blocked = false;
      try {
        budget->grant(c, 12);
      } catch (const std::runtime_error &) {
        blocked = true;
      }
      assert(blocked);
    }
    // Removing excess charge allows convergence at the physical floor.
    receipt_a->release();
    assert(budget->query_control("floors").status == "converged");
    auto zero = budget->publish_control("floors", 0, generation);
    for (auto slot : {a, b, c})
      budget->acknowledge_control(slot, zero.revision);
    assert(budget->query_control("floors").status == "pending");
    receipt_b->release();
    assert(budget->query_control("floors").status == "converged");
  }
  {
    auto budget = GpuBudget::connect(path, [] { return 1000; });
    auto slot = budget->register_pool("observe", 0);
    budget->configure_control_pool(slot, 120, 12, true, 1);
    budget->ready_control_pool(slot);
    auto published = budget->publish_control(
        "observe", 60, budget->query_control("observe").generation);
    int held = open(path.c_str(), O_RDWR);
    assert(held >= 0 && flock(held, LOCK_EX) == 0);
    // Attachment, query, and destruction must all finish while flock is held.
    alarm(10);
    {
      auto reader = GpuBudget::open_control(path, true);
      auto state = reader->observe_control("observe");
      assert(state.generation == published.generation &&
             state.revision == published.revision);
      assert(state.status == "pending");
    }
    std::atomic<bool> started{false}, finished{false};
    std::thread writer([&] {
      started = true;
      assert(budget->acknowledge_control(slot, published.revision));
      finished = true;
    });
    while (!started)
      std::this_thread::yield();
    std::this_thread::sleep_for(std::chrono::milliseconds(30));
    assert(!finished);
    flock(held, LOCK_UN);
    close(held);
    writer.join();
    assert(finished);
    auto observer = GpuBudget::open_control(path, true);
    assert(observer->observe_control("observe").status == "converged");
    budget.reset();
    auto replacement = GpuBudget::connect(path, [] { return 1000; });
    slot = replacement->register_pool("observe", 0);
    replacement->configure_control_pool(slot, 120, 12, true, 1);
    replacement->ready_control_pool(slot);
    auto state = observer->observe_control("observe");
    assert(state.generation != published.generation && !state.published);
    alarm(0);
  }
  {
    // The controls share max's revision: a retry of a revision must carry the
    // values it published, and an older revision is stale. A publication
    // without max sets the pool targets only as the instance's first one.
    auto budget = GpuBudget::connect(path, [] { return 1000; });
    auto a = budget->register_pool("policy", 0);
    auto b = budget->register_pool("policy", 1);
    budget->configure_control_pool(a, 120, 12, true);
    budget->configure_control_pool(b, 120, 12, true);
    budget->ready_control_pool(a);
    budget->ready_control_pool(b);
    auto writer = GpuBudget::open_control(path);
    const auto generation = writer->query_control("policy").generation;

    ControlUpdate priority;
    priority.priority = 5;
    auto first = writer->publish_control("policy", priority, generation);
    assert(first.status == "accepted" && first.revision == 1);
    assert(!first.published && first.convergence == "unpublished");
    assert(first.requested_bytes == UINT64_MAX && first.effective_bytes == 240);
    // A policy revision cannot stand in for a max that was never published.
    auto unpublished = writer->publish_control("policy", UINT64_MAX, generation,
                                               first.revision);
    assert(unpublished.status == "conflict" && !unpublished.published);

    auto limit = writer->publish_control("policy", 120, generation);
    assert(limit.revision == 2 && limit.priority == 5);
    assert(budget->control_state(a).pool_target_bytes == 60);
    assert(budget->acknowledge_control(a, 2) &&
           budget->acknowledge_control(b, 2));
    assert(writer->query_control("policy").status == "converged");

    ControlUpdate floor;
    floor.min_bytes = 480;
    floor.headroom_bytes = 0;
    auto third = writer->publish_control("policy", floor, generation);
    assert(third.status == "accepted" && third.revision == 3);
    assert(third.limit_revision == limit.revision);
    assert(third.min_bytes == 480 && third.requested_bytes == 120);
    assert(budget->control_state(a).pool_target_bytes == 60);
    assert(third.convergence == "converged" && third.completed == 2);
    assert(writer->publish_control("policy", floor, generation, 3).status ==
           "idempotent");
    ControlUpdate subset;
    subset.headroom_bytes = 0;
    assert(writer->publish_control("policy", subset, generation, 3).status ==
           "idempotent");
    ControlUpdate changed = floor;
    changed.min_bytes = 0;
    assert(writer->publish_control("policy", changed, generation, 3).status ==
           "conflict");
    assert(writer->publish_control("policy", 60, generation, 3).status ==
           "conflict");
    ControlUpdate stale;
    stale.priority = 1;
    assert(writer->publish_control("policy", stale, generation, 2).status ==
           "stale");
    assert(writer->query_control("policy").priority == 5);
  }
  {
    // A publication without max keeps a failed resize failed.
    auto budget = GpuBudget::connect(path, [] { return 1000; });
    auto slot = budget->register_pool("carry-failure", 0);
    budget->configure_control_pool(slot, 120, 12, true);
    budget->ready_control_pool(slot);
    auto writer = GpuBudget::open_control(path);
    const auto participant = budget->control_state(slot);
    const auto generation = participant.generation;
    auto shrink = writer->publish_control("carry-failure", 60, generation);
    assert(writer->fail_control("carry-failure", slot, participant.pool_owner,
                                generation, shrink.revision));
    ControlUpdate priority;
    priority.priority = 1;
    auto failed =
        writer->publish_control("carry-failure", priority, generation);
    assert(failed.revision == shrink.revision + 1);
    assert(failed.convergence == "failed" && failed.failed == 1);
    assert(failed.failure_reason == "quarantined_pages_prevent_resize");
    assert(budget->control_state(slot).pool_target_bytes == 60);
    // Publishing max again lets the engine retry.
    auto retried = writer->publish_control("carry-failure", 60, generation);
    assert(retried.convergence == "pending" && retried.failed == 0);
  }
  {
    // A deferred shrink stays pending until its release, and an engine that
    // acknowledges after a policy publication still reports the original max.
    auto budget = GpuBudget::connect(path, [] { return 1000; });
    auto slot = budget->register_pool("carry-shrink", 0);
    budget->configure_control_pool(slot, 120, 12, true);
    budget->ready_control_pool(slot);
    auto grant = budget->grant(slot, 96);
    auto receipt = grant->take(96);
    grant->finish();
    auto writer = GpuBudget::open_control(path);
    const auto generation = budget->control_state(slot).generation;
    auto shrink = writer->publish_control("carry-shrink", 60, generation);
    assert(budget->acknowledge_control(slot, shrink.revision));
    assert(writer->query_control("carry-shrink").status == "pending");
    ControlUpdate headroom;
    headroom.headroom_bytes = 24;
    auto later = writer->publish_control("carry-shrink", headroom, generation);
    assert(later.convergence == "pending" && later.completed == 0);
    receipt->release();
    assert(writer->query_control("carry-shrink").status == "converged");
    auto grow = writer->publish_control("carry-shrink", 120, generation);
    ControlUpdate priority;
    priority.priority = 2;
    auto next = writer->publish_control("carry-shrink", priority, generation);
    assert(next.revision == grow.revision + 1 && next.convergence == "pending");
    assert(!budget->acknowledge_control(slot, next.revision));
    assert(budget->acknowledge_control(slot, grow.revision));
    assert(writer->query_control("carry-shrink").status == "converged");
    assert(budget->control_state(slot).pool_target_bytes == 120);
  }
  {
    // Reports race with policy writes in either order. Only a newer max may
    // invalidate them, even when that max has the same numeric target.
    auto budget = GpuBudget::connect(path, [] { return 1000; });
    auto slot = budget->register_pool("report-race", 0);
    budget->configure_control_pool(slot, 120, 12, true);
    budget->ready_control_pool(slot);
    auto writer = GpuBudget::open_control(path);
    const auto source = budget->control_state(slot);
    ControlUpdate policy;
    policy.priority = 4;
    for (bool fail : {false, true}) {
      for (bool policy_first : {false, true}) {
        auto request =
            writer->publish_control("report-race", 60, source.generation);
        auto report = [&] {
          return fail ? writer->fail_control(
                            "report-race", slot, source.pool_owner,
                            source.generation, request.limit_revision)
                      : budget->acknowledge_control(slot,
                                                    request.limit_revision);
        };
        if (policy_first)
          writer->publish_control("report-race", policy, source.generation);
        assert(report());
        if (!policy_first)
          writer->publish_control("report-race", policy, source.generation);
        auto state = writer->observe_control("report-race");
        assert(state.limit_revision == request.revision);
        assert(state.revision == request.revision + 1);
        assert(state.convergence == (fail ? "failed" : "converged"));
        // Losing an explicit-revision race to a policy writer must not report
        // that this same-value max retry was already executed.
        auto conflict = writer->publish_control(
            "report-race", 60, source.generation, state.revision);
        assert(conflict.status == "conflict");
        assert(conflict.limit_revision == request.limit_revision);
        assert(conflict.convergence == state.convergence);
        auto retry =
            writer->publish_control("report-race", 60, source.generation);
        assert(retry.limit_revision == retry.revision &&
               retry.convergence == "pending");
        assert(!report());
        writer->publish_control("report-race", policy, source.generation);
        assert(!report());
        assert(writer->query_control("report-race").convergence == "pending");
      }
    }
  }
  {
    // Zero is a valid max revision, not an unpublished sentinel.
    auto budget = GpuBudget::connect(path, [] { return 1000; });
    auto slot = budget->register_pool("zero-revision", 0);
    budget->configure_control_pool(slot, 120, 12, true);
    budget->ready_control_pool(slot);
    auto state = budget->control_state(slot);
    auto first =
        budget->publish_control("zero-revision", 60, state.generation, 0);
    assert(first.published && first.limit_revision == 0);
    ControlUpdate policy;
    policy.min_bytes = 24;
    budget->publish_control("zero-revision", policy, state.generation);
    assert(budget->acknowledge_control(slot, 0));
    assert(budget->control_state(slot).convergence == "converged");
  }
  unlink(path.c_str());
  rmdir(directory);
}
