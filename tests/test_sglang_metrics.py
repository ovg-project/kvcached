# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

from types import ModuleType, SimpleNamespace
from typing import Any, Dict, Tuple

import pytest

from kvcached.integration.sglang import metrics
from kvcached.integration.sglang.patches import SGLangMetricsPatch


class FakeMetric:
    instances: Dict[str, "FakeMetric"] = {}

    def __init__(
        self,
        name: str,
        documentation: str,
        labelnames: Tuple[str, ...],
        **kwargs: Any,
    ) -> None:
        del documentation, kwargs
        assert name not in self.instances, f"duplicate metric registration: {name}"
        self.name = name
        self.labelnames = tuple(labelnames)
        self.values: Dict[Tuple[Tuple[str, Any], ...], float] = {}
        self._active_labels: Tuple[Tuple[str, Any], ...] = ()
        self.instances[name] = self

    def labels(self, **labels: Any) -> "FakeMetric":
        assert set(labels) == set(self.labelnames)
        child = object.__new__(FakeMetric)
        child.__dict__ = self.__dict__.copy()
        child._active_labels = tuple(sorted(labels.items()))
        return child

    def set(self, value: float) -> None:
        self.values[self._active_labels] = value

    def inc(self, value: float = 1) -> None:
        self.values[self._active_labels] = self.values.get(self._active_labels, 0) + value


@pytest.fixture(autouse=True)
def clear_fake_metrics():
    FakeMetric.instances.clear()
    metrics._WRAPPED_COLLECTOR_CLASSES.clear()


def _labels(**extra: Any) -> Tuple[Tuple[str, Any], ...]:
    return tuple(
        sorted(
            {
                "model_name": "test-model",
                "pool_name": "mha",
                "group_id": "2",
                **extra,
            }.items()
        )
    )


def test_exporter_updates_snapshots_and_counter_deltas(monkeypatch):
    operation_snapshot = {
        "pool_name": "mha",
        "group_id": 2,
        "allocation_requests_total": 3,
        "allocation_successes_total": 2,
        "last_error_timestamp_ns": 2_500_000_000,
    }
    monkeypatch.setattr(
        metrics,
        "_runtime_snapshot_dict",
        lambda: {
            "initialized": True,
            "world_size": 4,
            "pp_rank": 1,
            "async_sched": False,
            "contiguous_layout": True,
            "is_worker": False,
        },
    )
    monkeypatch.setattr(
        metrics,
        "_pool_snapshot_dicts",
        lambda: [
            {
                "pool_name": "mha",
                "group_id": 2,
                "total_blocks": 100,
                "available_blocks": 70,
                "allocated_blocks": 30,
                "in_shrink": False,
            }
        ],
    )
    monkeypatch.setattr(
        metrics,
        "_pool_operation_snapshot_dicts",
        lambda: [dict(operation_snapshot)],
    )

    exporter = metrics.SGLangPrometheusMetricsExporter(
        {"model_name": "test-model"},
        counter_cls=FakeMetric,
        gauge_cls=FakeMetric,
    )
    exporter.update()

    assert (
        FakeMetric.instances["kvcached_runtime_world_size"].values[
            tuple(sorted({"model_name": "test-model"}.items()))
        ]
        == 4
    )
    assert (
        FakeMetric.instances["kvcached_runtime_pp_rank"].values[
            tuple(sorted({"model_name": "test-model"}.items()))
        ]
        == 1
    )
    assert FakeMetric.instances["kvcached_kv_cache_pool_available_blocks"].values[_labels()] == 70
    requests = FakeMetric.instances[
        "kvcached_kv_cache_pool_operation_allocation_requests_total"
    ]
    assert requests.values[_labels()] == 3
    assert (
        FakeMetric.instances["kvcached_kv_cache_pool_last_error_timestamp_seconds"].values[
            _labels()
        ]
        == 2.5
    )

    operation_snapshot["allocation_requests_total"] = 5
    exporter.update()
    assert requests.values[_labels()] == 5

    operation_snapshot["allocation_requests_total"] = 1
    exporter.update()
    assert requests.values[_labels()] == 6


def test_exporter_zeros_gauges_for_removed_pool(monkeypatch):
    pool_snapshots = [{"pool_name": "mha", "group_id": 2, "available_blocks": 70}]
    monkeypatch.setattr(metrics, "_runtime_snapshot_dict", lambda: {})
    monkeypatch.setattr(metrics, "_pool_snapshot_dicts", lambda: list(pool_snapshots))
    monkeypatch.setattr(metrics, "_pool_operation_snapshot_dicts", lambda: [])

    exporter = metrics.SGLangPrometheusMetricsExporter(
        {"model_name": "test-model"},
        counter_cls=FakeMetric,
        gauge_cls=FakeMetric,
    )
    exporter.update()
    pool_snapshots.clear()
    exporter.update()

    assert FakeMetric.instances["kvcached_kv_cache_pool_available_blocks"].values[_labels()] == 0


def test_collector_wrapper_composes_and_isolates_export_errors(monkeypatch):
    monkeypatch.setattr(metrics, "_runtime_snapshot_dict", lambda: {})
    monkeypatch.setattr(metrics, "_pool_snapshot_dicts", lambda: [])
    monkeypatch.setattr(metrics, "_pool_operation_snapshot_dicts", lambda: [])

    class ExistingCollector:
        _counter_cls = FakeMetric
        _gauge_cls = FakeMetric

        def __init__(self, labels: Dict[str, Any], **kwargs: Any) -> None:
            del kwargs
            self.labels = labels
            self.logged: list[Any] = []

        def log_stats(self, stats: Any) -> str:
            self.logged.append(stats)
            return "base-result"

    wrapped_cls = metrics.wrap_scheduler_metrics_collector(ExistingCollector)
    assert metrics.wrap_scheduler_metrics_collector(ExistingCollector) is wrapped_cls
    collector = wrapped_cls(labels={"model_name": "test-model"})
    assert collector.log_stats("stats") == "base-result"
    assert collector.logged == ["stats"]

    def fail_snapshot():
        raise RuntimeError("snapshot failed")

    monkeypatch.setattr(metrics, "_runtime_snapshot_dict", fail_snapshot)
    assert collector.log_stats("more-stats") == "base-result"
    assert (
        FakeMetric.instances["kvcached_sglang_metrics_export_errors_total"].values[
            tuple(sorted({"model_name": "test-model"}.items()))
        ]
        == 1
    )


def test_collector_wrapper_isolates_exporter_initialization_errors(monkeypatch):
    class ExistingCollector:
        def __init__(self, labels: Dict[str, Any]) -> None:
            self.labels = labels

        def log_stats(self, stats: Any) -> str:
            return f"base:{stats}"

    def fail_exporter(*args: Any, **kwargs: Any) -> None:
        del args, kwargs
        raise RuntimeError("metric backend rejected registration")

    monkeypatch.setattr(metrics, "SGLangPrometheusMetricsExporter", fail_exporter)

    wrapped_cls = metrics.wrap_scheduler_metrics_collector(ExistingCollector)
    collector = wrapped_cls(labels={"model_name": "test-model"})

    assert collector.log_stats("stats") == "base:stats"
    assert collector._kvcached_metrics_exporter is None


def test_collector_wrapper_retries_after_initial_snapshot_error(monkeypatch):
    snapshots = iter([RuntimeError("snapshot temporarily unavailable"), {}])

    def runtime_snapshot():
        snapshot = next(snapshots)
        if isinstance(snapshot, Exception):
            raise snapshot
        return snapshot

    monkeypatch.setattr(metrics, "_runtime_snapshot_dict", runtime_snapshot)
    monkeypatch.setattr(metrics, "_pool_snapshot_dicts", lambda: [])
    monkeypatch.setattr(metrics, "_pool_operation_snapshot_dicts", lambda: [])

    class ExistingCollector:
        _counter_cls = FakeMetric
        _gauge_cls = FakeMetric

        def __init__(self, labels: Dict[str, Any]) -> None:
            self.labels = labels

        def log_stats(self, stats: Any) -> str:
            return f"base:{stats}"

    wrapped_cls = metrics.wrap_scheduler_metrics_collector(ExistingCollector)
    collector = wrapped_cls(labels={"model_name": "test-model"})

    assert collector._kvcached_metrics_exporter is not None
    assert collector.log_stats("stats") == "base:stats"
    assert FakeMetric.instances[
        "kvcached_sglang_metrics_export_last_success_timestamp_seconds"
    ].values


def test_metrics_patch_wraps_selected_scheduler_collector(monkeypatch):
    class ExistingCollector:
        pass

    module = ModuleType("fake_sglang_metrics")
    setattr(module, "STAT_LOGGER_ROLE_SCHEDULER", "scheduler")
    setattr(
        module,
        "resolve_collector_class",
        lambda server_args, role, default_cls: server_args.stat_loggers.get(role, default_cls)
    )
    monkeypatch.setenv("ENABLE_KVCACHED", "true")
    assert SGLangMetricsPatch().apply(module)

    server_args = SimpleNamespace(stat_loggers={"scheduler": ExistingCollector})
    resolver = getattr(module, "resolve_collector_class")
    selected = resolver(server_args, "scheduler", object)
    assert issubclass(selected, ExistingCollector)
    assert selected is not ExistingCollector
    assert resolver(server_args, "tokenizer", object) is object

    monkeypatch.setenv("ENABLE_KVCACHED", "false")
    assert resolver(server_args, "scheduler", object) is ExistingCollector
