# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

"""Engine-side consumption of frozen, post-initialization ledger controls."""

from __future__ import annotations

import time
from typing import Any

from kvcached import prefix_cache_retention
from kvcached.errors import StateConsistencyError


class RuntimeControl:
    def __init__(self, manager: Any, *, worker_ipc: bool, can_shrink: bool, expected_participants: int):
        from kvcached import _C  # type: ignore[attr-defined]
        from kvcached.tp_ipc_util import control_workers

        self.manager = manager
        self.worker_ipc = worker_ipc
        pins = getattr(manager, "permanent_block_ids", ())
        floor = {"minimum_pages": len(pins) + 1} if pins else {}
        if worker_ipc:
            sources = control_workers(manager.world_size, manager.pp_rank,
                                      manager.group_id, configure=can_shrink,
                                      expected_participants=expected_participants,
                                      startup_pages=manager.mem_size // manager.page_size,
                                      reserve_null=getattr(manager, "null_block", None) is not None, **floor)
        else:
            sources = [_C.local_instance_control(manager.group_id, can_shrink, expected_participants,
                                                  manager.mem_size // manager.page_size,
                                                  getattr(manager, "null_block", None) is not None, **floor)]
        if not sources or any(not source["ready"] for source in sources):
            raise RuntimeError("KV runtime control requires every worker pool ready")
        self.sources = tuple(sources)
        self.current: tuple[dict, ...] = ()
        self.last_poll = 0.0
        self.executing: tuple[dict, ...] = ()
        self.acknowledged: tuple = ()

    def exact_retry_headroom(self) -> int | None:
        """Sample worker GPUs and return the number of complete distributed pages they can fund."""
        from kvcached.tp_ipc_util import control_workers

        try:
            workers = control_workers(self.manager.world_size, self.manager.pp_rank,
                                      self.manager.group_id, refresh_capacity=True)
            identity = ("gpu", "instance", "pool_slot", "pool_owner", "generation")
            if len(workers) != len(self.sources) or any(
                    worker[key] != source[key] for worker, source in zip(workers, self.sources) for key in identity):
                raise StateConsistencyError("KV control participant changed during exact recovery")
        except StateConsistencyError as exc:
            self.manager._record_native_fatal("control poll", exc)
            raise

        sampled = {worker["gpu"] for worker in workers if worker.get("capacity_refreshed")}
        if any(source["gpu"] not in sampled for source in self.sources):
            # Another process may still hold this GPU's calibration lease.
            return None

        # Read after all samplers return. Workers sharing a GPU ledger spend
        # and refund their combined bundle for each distributed page ID.
        bundles: dict[tuple[str, str], int] = {}
        balances: dict[tuple[str, str], int] = {}
        for source, state in zip(self.sources, self.poll(force=True)):
            ledger = (source["gpu"], source["instance"])
            bundles[ledger] = bundles.get(ledger, 0) + state["page_bundle_bytes"]
            balance = state["signed_available_bytes"]
            balances[ledger] = min(balances.get(ledger, balance), balance)
        return min(balances[key] // size for key, size in bundles.items())

    def poll(self, *, force: bool = False) -> tuple[dict, ...]:
        from kvcached import _C  # type: ignore[attr-defined]

        now = time.monotonic()
        if not force and self.current and now - self.last_poll < 0.1:
            return self.current
        states = []
        try:
            for source in self.sources:
                state = _C.query_instance_control(source["gpu"], source["instance"],
                                                  source["pool_slot"])
                if (state["generation"] != source["generation"]
                        or state["pool_owner"] != source["pool_owner"]
                        or state["status"] == "unavailable"):
                    raise StateConsistencyError("KV control participant changed or disappeared")
                states.append(state)
        except StateConsistencyError as exc:
            self.manager._record_native_fatal("control poll", exc)
            raise
        self.current = tuple(states)
        self.last_poll = now
        return self.current

    def apply(self) -> None:
        manager = self.manager
        states = self.poll()
        if not any(state["published"] for state in states):
            return
        # Bytes remain per GPU. A shared logical pool must fit its most
        # constrained rank, including PP stages with different layer counts.
        pages = min((state["pool_target_bytes"] if state["published"]
                     else state["pool_capacity_bytes"]) // state["page_bundle_bytes"]
                    for state in states)
        if any(not state["can_shrink"] and state["published"]
               and state["pool_target_bytes"] < state["pool_capacity_bytes"]
               for state in states):
            return  # Native cap already blocks growth; no unsafe engine shrink.
        signature = tuple((state["generation"], state["limit_revision"], state["published"])
                          for state in states)
        previous = tuple((state["generation"], state["limit_revision"], state["published"])
                         for state in self.executing)
        if signature != previous:
            from kvcached.errors import (
                QuarantinedResizeError,
                RetainedQuotaResizeError,
                RetainedResizeError,
            )

            try:
                manager.resize(pages * manager.page_size)
            except RetainedResizeError:
                # The same revision must retry after the retained batch is
                # adopted, including a revision already executing a shrink.
                manager._resize_rejected = True
                self.executing = ()
                return
            except (QuarantinedResizeError, RetainedQuotaResizeError) as error:
                manager._resize_rejected = True
                self.executing = states
                self.fail(retained_quota=isinstance(error, RetainedQuotaResizeError))
                return
            self.executing = states
        elif manager.in_shrink:
            # Continue the existing request without recording a new resize.
            manager._maybe_finish_shrink()
            if manager.in_shrink:
                # Reservations made since the initial resize must not pin the
                # request indefinitely. free() also tries to finish the shrink.
                manager.free_reserved()
        if not manager.in_shrink and not getattr(manager, "_resize_rejected", False):
            self.acknowledge()

    def fail(self, *, retained_quota: bool = False) -> None:
        """Report the executing max, even if newer controls were published."""
        from kvcached import _C  # type: ignore[attr-defined]

        for source, state in zip(self.sources, self.executing):
            if state["published"]:
                _C.fail_instance_control(
                    source["gpu"], source["instance"], source["pool_slot"],
                    source["pool_owner"], state["generation"], state["limit_revision"],
                    retained_quota=retained_quota)

    def acknowledge(self) -> None:
        from kvcached import _C  # type: ignore[attr-defined]
        from kvcached.tp_ipc_util import control_workers

        signature = tuple((s["generation"], s["limit_revision"], s["published"])
                          for s in self.executing)
        if not signature or signature == getattr(self, "acknowledged", ()):
            return
        accepted = False
        if self.worker_ipc:
            acknowledgements = {
                (source["pp_rank"], source["rank"]):
                (state["generation"], state["limit_revision"], state["published"])
                for source, state in zip(self.sources, self.executing)
            }
            accepted = control_workers(self.manager.world_size, self.manager.pp_rank,
                            self.manager.group_id, acknowledgements=acknowledgements)
        else:
            state = self.executing[0]
            if state["published"]:
                current = _C.local_instance_control(self.manager.group_id)
                if current["generation"] == state["generation"]:
                    accepted = _C.acknowledge_instance_control(self.manager.group_id, state["limit_revision"])
            else:
                accepted = True
        if accepted:
            self.acknowledged = signature


def initialize_runtime_controls(integration: str, *, can_shrink=None, ranks: int = 1) -> tuple[Any, ...]:
    from kvcached.pool_registry import get_registered_kv_cache_pools

    managers = tuple(manager for manager, _ in
                     get_registered_kv_cache_pools(integration=integration))
    for manager in managers:
        manager.wait_ready()
    worker_ipc = False
    if integration == "vllm":
        from kvcached.integration.vllm.interfaces import should_use_worker_ipc
        worker_ipc = should_use_worker_ipc()
    if worker_ipc and managers:
        from kvcached.tp_ipc_util import _target_pp_ranks
        ranks = managers[0].world_size * len(_target_pp_ranks(managers[0].pp_rank))
    for manager in managers:
        supported = True if can_shrink is None else bool(can_shrink(manager))
        manager._runtime_control = RuntimeControl(
            manager, worker_ipc=worker_ipc, can_shrink=supported,
            expected_participants=len(managers) * ranks)
    if managers:
        prefix_cache_retention._policy.instance_name = managers[0]._control_instance_name()
    return managers
