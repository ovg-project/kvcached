// SPDX-FileCopyrightText: Copyright contributors to the kvcached project
// SPDX-License-Identifier: Apache-2.0
#include "gpu_budget.hpp"

#include <cassert>
#include <set>
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
  unlink(path.c_str());
  rmdir(directory);
}
