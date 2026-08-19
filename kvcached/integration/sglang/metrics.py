# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

"""Prometheus export for kvcached state in SGLang scheduler processes."""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple, Type

from kvcached.utils import get_kvcached_logger

logger = get_kvcached_logger()

_POOL_GAUGE_FIELDS = (
    "num_layers",
    "num_kv_buffers",
    "page_size_bytes",
    "block_size_bytes",
    "total_blocks",
    "available_blocks",
    "allocated_blocks",
    "reserved_blocks",
    "available_bytes",
    "allocated_bytes",
    "reserved_bytes",
    "null_block_reserved",
    "virtual_per_layer_bytes",
    "virtual_total_bytes",
    "mapped_bytes",
    "total_pages",
    "free_pages",
    "inuse_pages",
    "reserved_pages",
    "available_physical_pages",
    "effective_free_pages",
    "in_shrink",
    "shrink_target_blocks",
    "resize_target_bytes",
)

_OPERATION_COUNTER_FIELDS = (
    "allocation_requests_total",
    "allocation_successes_total",
    "allocation_failures_total",
    "capacity_exhausted_total",
    "allocated_blocks_total",
    "free_requests_total",
    "free_successes_total",
    "free_failures_total",
    "freed_blocks_total",
    "physical_page_allocations_total",
    "physical_page_allocation_failures_total",
    "physical_page_frees_total",
    "resize_requests_total",
    "resize_successes_total",
    "resize_deferred_total",
    "resize_completions_total",
    "trim_requests_total",
    "trim_successes_total",
    "clear_requests_total",
    "clear_successes_total",
    "operation_errors_total",
    "post_init_errors_total",
    "allocation_errors_total",
    "free_errors_total",
    "resize_errors_total",
    "trim_errors_total",
    "clear_errors_total",
    "state_inconsistency_errors_total",
)

_RUNTIME_GAUGE_FIELDS = (
    "initialized",
    "world_size",
    "pp_rank",
    "async_sched",
    "contiguous_layout",
    "is_worker",
)

_WRAPPED_COLLECTOR_CLASSES: Dict[Type[Any], Type[Any]] = {}


def _runtime_snapshot_dict() -> Dict[str, Any]:
    from kvcached.integration.sglang.interfaces import observability_snapshot_dict

    return observability_snapshot_dict()


def _pool_snapshot_dicts() -> List[Dict[str, Any]]:
    from kvcached.integration.sglang.interfaces import kv_cache_pool_snapshot_dicts

    return kv_cache_pool_snapshot_dicts()


def _pool_operation_snapshot_dicts() -> List[Dict[str, Any]]:
    from kvcached.integration.sglang.interfaces import (
        kv_cache_pool_operation_snapshot_dicts,
    )

    return kv_cache_pool_operation_snapshot_dicts()


def _metric_help(field: str, category: str) -> str:
    return f"kvcached {category} value for {field.replace('_', ' ')}."


class SGLangPrometheusMetricsExporter:
    """Update SGLang's metric backend from exporter-neutral snapshots."""

    def __init__(
        self,
        labels: Dict[str, Any],
        *,
        counter_cls: Optional[Type[Any]] = None,
        gauge_cls: Optional[Type[Any]] = None,
    ) -> None:
        if counter_cls is None or gauge_cls is None:
            from prometheus_client import Counter, Gauge

            counter_cls = counter_cls or Counter
            gauge_cls = gauge_cls or Gauge

        self._base_labels = dict(labels)
        self._base_label_names = tuple(self._base_labels.keys())
        self._pool_label_names = self._base_label_names + ("pool_name", "group_id")

        self._runtime_gauges = {
            field: gauge_cls(
                name=f"kvcached_runtime_{field}",
                documentation=_metric_help(field, "runtime"),
                labelnames=self._base_label_names,
                multiprocess_mode="mostrecent",
            )
            for field in _RUNTIME_GAUGE_FIELDS
        }
        self._pool_gauges = {
            field: gauge_cls(
                name=f"kvcached_kv_cache_pool_{field}",
                documentation=_metric_help(field, "KV cache pool"),
                labelnames=self._pool_label_names,
                multiprocess_mode="mostrecent",
            )
            for field in _POOL_GAUGE_FIELDS
        }
        self._operation_counters = {
            field: counter_cls(
                name=f"kvcached_kv_cache_pool_operation_{field}",
                documentation=_metric_help(field, "KV cache pool operation"),
                labelnames=self._pool_label_names,
            )
            for field in _OPERATION_COUNTER_FIELDS
        }
        self._last_error_timestamp = gauge_cls(
            name="kvcached_kv_cache_pool_last_error_timestamp_seconds",
            documentation="Unix timestamp of the last recorded KV cache pool error.",
            labelnames=self._pool_label_names,
            multiprocess_mode="mostrecent",
        )
        self._export_errors = counter_cls(
            name="kvcached_sglang_metrics_export_errors_total",
            documentation="Number of failures while exporting kvcached SGLang metrics.",
            labelnames=self._base_label_names,
        )
        self._last_success_timestamp = gauge_cls(
            name="kvcached_sglang_metrics_export_last_success_timestamp_seconds",
            documentation="Unix timestamp of the last successful kvcached metrics update.",
            labelnames=self._base_label_names,
            multiprocess_mode="mostrecent",
        )
        self._counter_values: Dict[Tuple[str, str, str], int] = {}
        self._seen_pool_labels: Dict[Tuple[str, str], Dict[str, Any]] = {}
        self._last_error_log_time = 0.0

    def _pool_labels(self, snapshot: Dict[str, Any]) -> Dict[str, Any]:
        labels = dict(self._base_labels)
        labels["pool_name"] = str(snapshot.get("pool_name") or "unknown")
        labels["group_id"] = str(snapshot.get("group_id", 0))
        return labels

    def _set_runtime_gauges(self, snapshot: Dict[str, Any]) -> None:
        for field, gauge in self._runtime_gauges.items():
            value = snapshot.get(field)
            if value is not None:
                gauge.labels(**self._base_labels).set(float(value))

    def _set_pool_gauges(self, snapshots: List[Dict[str, Any]]) -> None:
        current_keys = set()
        for snapshot in snapshots:
            labels = self._pool_labels(snapshot)
            key = (labels["pool_name"], labels["group_id"])
            current_keys.add(key)
            self._seen_pool_labels[key] = labels
            for field, gauge in self._pool_gauges.items():
                value = snapshot.get(field)
                if value is not None:
                    gauge.labels(**labels).set(float(value))

        for key in self._seen_pool_labels.keys() - current_keys:
            labels = self._seen_pool_labels[key]
            for gauge in self._pool_gauges.values():
                gauge.labels(**labels).set(0)
            self._last_error_timestamp.labels(**labels).set(0)

    def _increment_operation_counters(self, snapshots: List[Dict[str, Any]]) -> None:
        for snapshot in snapshots:
            labels = self._pool_labels(snapshot)
            pool_name = labels["pool_name"]
            group_id = labels["group_id"]
            for field, counter in self._operation_counters.items():
                value = int(snapshot.get(field, 0))
                key = (pool_name, group_id, field)
                previous = self._counter_values.get(key, 0)
                delta = value - previous if value >= previous else value
                if delta:
                    counter.labels(**labels).inc(delta)
                self._counter_values[key] = value

            timestamp_ns = snapshot.get("last_error_timestamp_ns")
            if timestamp_ns is not None:
                self._last_error_timestamp.labels(**labels).set(float(timestamp_ns) / 1_000_000_000)

    def update(self) -> None:
        self._set_runtime_gauges(_runtime_snapshot_dict())
        self._set_pool_gauges(_pool_snapshot_dicts())
        self._increment_operation_counters(_pool_operation_snapshot_dicts())
        self._last_success_timestamp.labels(**self._base_labels).set(time.time())

    def record_update_error(self) -> None:
        self._export_errors.labels(**self._base_labels).inc()
        now = time.monotonic()
        if now - self._last_error_log_time >= 60:
            logger.warning(
                "Failed to update kvcached metrics for SGLang",
                exc_info=True,
            )
            self._last_error_log_time = now


def wrap_scheduler_metrics_collector(base_cls: Type[Any]) -> Type[Any]:
    """Compose kvcached metrics with the collector selected by SGLang."""

    if getattr(base_cls, "__kvcached_metrics_collector__", False):
        return base_cls
    cached = _WRAPPED_COLLECTOR_CLASSES.get(base_cls)
    if cached is not None:
        return cached

    class KVCachedSchedulerMetricsCollector(base_cls):  # type: ignore[misc, valid-type]
        __kvcached_metrics_collector__ = True

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            self._kvcached_metrics_exporter: Optional[SGLangPrometheusMetricsExporter] = None
            try:
                exporter = SGLangPrometheusMetricsExporter(
                    labels=dict(self.labels),
                    counter_cls=getattr(self, "_counter_cls", None),
                    gauge_cls=getattr(self, "_gauge_cls", None),
                )
            except Exception:
                logger.warning(
                    "Failed to initialize kvcached metrics for SGLang",
                    exc_info=True,
                )
            else:
                self._kvcached_metrics_exporter = exporter
                try:
                    exporter.update()
                except Exception:
                    exporter.record_update_error()

        def log_stats(self, stats: Any) -> Any:
            result = super().log_stats(stats)
            exporter = self._kvcached_metrics_exporter
            if exporter is None:
                return result
            try:
                exporter.update()
            except Exception:
                exporter.record_update_error()
            return result

    KVCachedSchedulerMetricsCollector.__name__ = f"KVCached{base_cls.__name__}"
    KVCachedSchedulerMetricsCollector.__qualname__ = KVCachedSchedulerMetricsCollector.__name__
    _WRAPPED_COLLECTOR_CLASSES[base_cls] = KVCachedSchedulerMetricsCollector
    return KVCachedSchedulerMetricsCollector
