# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0
"""Keep kvcached pages mapped while vLLM's OffloadingConnector copies them.

OffloadingConnector frees a request's blocks as soon as the request finishes
and copies them to host memory later: the scheduler creates a store job in the
first ``build_connector_meta`` after the free at the latest, and the worker
submits it one step after that. vLLM only fences such a job when the block is
reallocated, so an unmap by kvcached in between lets the copy read memory that
no longer holds the request's KV.

The connector's public methods carry everything needed to close that window:
the jobs and their GPU block ids leave the scheduler in the metadata that
``build_connector_meta`` returns, and the workers report each job once its CUDA
event has completed. Each job pins its pages in the kvcached manager until all
workers have reported it, and a page retired by a free waits for the next build
(``KVCacheManager.enable_copy_fence``).

Sliding-window groups free blocks while their request still runs, and the
connector may store such a block any number of builds later, as long as vLLM
has not reallocated it. Their pages stay pinned from allocation until the
request's last build. The pins only keep pages mapped: vLLM still reuses the
blocks.
"""

from __future__ import annotations

from typing import Any, Iterator

from kvcached.integration.version_utils import VersionRange
from kvcached.utils import KVCachedConfigError, get_kvcached_logger

logger = get_kvcached_logger()

OFFLOADING_CONNECTOR = "OffloadingConnector"
SUPPORTED_RANGE = ">=0.29.0,<0.31.0"


def _connector_names(kv_transfer_config: Any) -> list[str]:
    name = getattr(kv_transfer_config, "kv_connector", None)
    if name is None:
        return []
    names = [name]
    extra = getattr(kv_transfer_config, "kv_connector_extra_config", None) or {}
    for nested in extra.get("connectors", ()) if isinstance(extra, dict) else ():
        if isinstance(nested, dict) and nested.get("kv_connector"):
            names.append(nested["kv_connector"])
    return names


def check_offloading_config(vllm_config: Any, version: str | None) -> None:
    """Reject CPU offloading configurations whose copies are not pinned."""
    kv_transfer_config = getattr(vllm_config, "kv_transfer_config", None)
    names = _connector_names(kv_transfer_config)
    if OFFLOADING_CONNECTOR not in names:
        return
    parallel_config = vllm_config.parallel_config
    if (
        names[0] != OFFLOADING_CONNECTOR
        or not (version and VersionRange(SUPPORTED_RANGE).contains(version))
        or not getattr(vllm_config, "use_v2_model_runner", False)
        or int(parallel_config.tensor_parallel_size) != 1
        or int(parallel_config.pipeline_parallel_size) != 1
    ):
        raise KVCachedConfigError(
            "kvcached supports vLLM CPU KV offloading (OffloadingConnector, "
            "--kv-offloading-size) only as the sole connector on vLLM 0.29-0.30 "
            "with Model Runner V2, TP=1 and PP=1; remove --kv-offloading-size "
            "or disable kvcached"
        )


def windowed_groups(group_configs: Any) -> tuple[int, ...]:
    """KV cache groups whose freed blocks the connector may still store.

    A full-attention block leaves its request only when the request finishes or
    is preempted, and the connector stores it in the next build at the latest.
    Mamba "align" groups store only the states the scheduler hands off in the
    same pass. Sliding-window (and other Mamba) groups free blocks while the
    request runs, and the connector keeps their ids until it stores them,
    which can be any number of builds later, so their pins last as long as the
    request.
    """
    return tuple(
        config.group_idx for config in group_configs
        if config.sliding_window_size_in_chunks is not None
        and not config.requires_cow_source)


def _gpu_block_ids(job: Any, gpu_spec_type: Any) -> Iterator[int]:
    for spec in (job.src_spec, job.dst_spec):
        if isinstance(spec, gpu_spec_type):
            yield from spec.block_ids.tolist()


class OffloadCopyPins:
    """Pin the pages of each transfer job until every worker completed it.

    For windowed groups, also pin the pages of every block a request was given
    until the connector can no longer store it: the request's last build, or
    its resumption after preemption, when the connector drops the old ids.
    """

    def __init__(self, manager: Any, num_workers: int, gpu_spec_type: Any,
                 windowed: tuple[int, ...] = (), yield_req_data: Any = None):
        self.manager = manager
        self.num_workers = num_workers
        self.gpu_spec_type = gpu_spec_type
        self.reports: dict[int, int] = {}
        self.windowed = windowed
        self.yield_req_data = yield_req_data
        self.held: dict[str, list[Any]] = {}
        self._next_hold = 0

    def after_build(self, scheduler_output: Any, meta: Any) -> None:
        # Before the metadata reaches a worker, and before releasing pages.
        for jobs in (meta.load_jobs, meta.store_jobs):
            for job_id, job in jobs.items():
                self.manager.pin_copy_job(
                    job_id, _gpu_block_ids(job, self.gpu_spec_type))
        if self.windowed:
            self._hold_windowed_blocks(scheduler_output)
        self.manager.advance_copy_fence()
        if getattr(self.manager, "physical_release_barrier", None) is None:
            # Without a batch queue no later step drains retired pages; the
            # GPU runs no batch while the scheduler builds.
            self.manager.release_retired_pages_through(
                self.manager.capture_physical_release_marker())

    def completed(self, completed_jobs: dict[int, int]) -> None:
        # Count jobs that a reset made stale for the connector too: their
        # copies still run on the workers.
        for job_id, count in completed_jobs.items():
            reports = self.reports.get(job_id, 0) + count
            if reports >= self.num_workers:
                self.reports.pop(job_id, None)
                self.manager.unpin_copy_job(job_id)
            else:
                self.reports[job_id] = reports

    def _hold_windowed_blocks(self, scheduler_output: Any) -> None:
        # The same request data the connector extends its block lists from.
        for req_id, new_block_ids, resumed in self.yield_req_data(scheduler_output):
            if resumed:
                self._drop(req_id)
            if not new_block_ids:
                continue
            block_ids = [block_id for group in self.windowed
                         for block_id in new_block_ids[group] if block_id]
            if block_ids:
                key = ("held", req_id, self._next_hold)
                self._next_hold += 1
                self.manager.pin_copy_job(key, block_ids)
                self.held.setdefault(req_id, []).append(key)
        # Their last stores are in this build's metadata, pinned above.
        for req_id in scheduler_output.finished_req_ids or ():
            self._drop(req_id)

    def _drop(self, req_id: str) -> None:
        for key in self.held.pop(req_id, ()):
            self.manager.unpin_copy_job(key)


def install_offload_copy_pins(engine_core: Any, manager: Any) -> bool:
    """Wrap the scheduler's OffloadingConnector, if any. Returns True if wrapped."""
    scheduler: Any = getattr(engine_core, "scheduler", None)
    connector = getattr(scheduler, "connector", None)
    if connector is None or type(connector).__name__ != OFFLOADING_CONNECTOR:
        return False
    try:
        from vllm.distributed.kv_transfer.kv_connector.utils import yield_req_data
        from vllm.distributed.kv_transfer.kv_connector.v1.offloading.common import (
            OffloadingConnectorMetadata,
            OffloadingWorkerMetadata,
        )
        from vllm.v1.kv_offload.base import GPULoadStoreSpec
    except ImportError as e:
        raise KVCachedConfigError(
            f"kvcached cannot track this OffloadingConnector's copies: {e}") from e
    for cls, fields in ((OffloadingConnectorMetadata, ("load_jobs", "store_jobs")),
                        (OffloadingWorkerMetadata, ("completed_jobs",))):
        missing = [f for f in fields if f not in getattr(cls, "__dataclass_fields__", {})]
        if missing:
            raise KVCachedConfigError(
                f"kvcached cannot track OffloadingConnector copies: "
                f"{cls.__name__} has no {', '.join(missing)}")
    if manager is None:
        raise KVCachedConfigError("kvcached found no KV cache manager for offloading")

    offload_config = connector.connector_scheduler.config
    windowed = windowed_groups(offload_config.kv_group_configs)
    num_workers = int(offload_config.num_workers)
    pins = OffloadCopyPins(manager, num_workers, GPULoadStoreSpec, windowed, yield_req_data)
    manager.enable_copy_fence()

    build_connector_meta = connector.build_connector_meta
    update_connector_output = connector.update_connector_output
    has_pending_push_work = connector.has_pending_push_work

    def build_with_pins(scheduler_output: Any) -> Any:
        meta = build_connector_meta(scheduler_output)
        pins.after_build(scheduler_output, meta)
        return meta

    def update_with_pins(connector_output: Any) -> None:
        meta = getattr(connector_output, "kv_connector_worker_meta", None)
        completed = dict(getattr(meta, "completed_jobs", None) or {})
        update_connector_output(connector_output)
        pins.completed(completed)

    def has_pending_work_with_pins() -> bool:
        # Keep stepping until pinned or fenced pages are released.
        return has_pending_push_work() or manager.has_pending_copy_release()

    connector.build_connector_meta = build_with_pins
    connector.update_connector_output = update_with_pins
    connector.has_pending_push_work = has_pending_work_with_pins
    engine_core._kvcached_offload_copy_pins = pins
    logger.info("Pinning kvcached pages for OffloadingConnector copies (%d workers, "
                "windowed groups %s)", num_workers, list(windowed))
    return True
