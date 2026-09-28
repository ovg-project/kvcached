# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

import math
from dataclasses import fields
from types import ModuleType, SimpleNamespace
from typing import Any, Dict, Tuple

import pytest

from kvcached.integration.sglang import metrics
from kvcached.integration.sglang.patches import SGLangMetricsPatch
from kvcached.observability import KVCachePoolOperationSnapshot


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
        child.values.setdefault(child._active_labels, 0)
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


@pytest.mark.parametrize("field", ["shrink_target_blocks", "resize_target_bytes"])
@pytest.mark.parametrize("missing", [False, True])
def test_exporter_clears_unavailable_gauge(monkeypatch, field, missing):
    snapshot = {"pool_name": "mha", "group_id": 2, field: 100}
    monkeypatch.setattr(metrics, "_runtime_snapshot_dict", lambda: {})
    monkeypatch.setattr(metrics, "_pool_snapshot_dicts", lambda: [snapshot])
    monkeypatch.setattr(metrics, "_pool_operation_snapshot_dicts", lambda: [])
    exporter = metrics.SGLangPrometheusMetricsExporter(
        {"model_name": "test-model"}, counter_cls=FakeMetric, gauge_cls=FakeMetric,
    )
    exporter.update()
    gauge = FakeMetric.instances[f"kvcached_kv_cache_pool_{field}"]
    assert gauge.values[_labels()] == 100
    if missing:
        del snapshot[field]
    else:
        snapshot[field] = None
    exporter.update()
    assert math.isnan(gauge.values[_labels()])
    snapshot[field] = 0
    exporter.update()
    assert gauge.values[_labels()] == 0


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


def test_exporter_operation_fields_match_snapshot_contract():
    expected = {
        field.name for field in fields(KVCachePoolOperationSnapshot)
        if field.name.endswith("_total")
    }
    assert set(metrics._OPERATION_COUNTER_FIELDS) == expected


def test_exporter_exposes_zero_counters_before_first_event(monkeypatch):
    snapshot = {"pool_name": "mha", "group_id": 2}
    monkeypatch.setattr(metrics, "_runtime_snapshot_dict", lambda: {})
    monkeypatch.setattr(metrics, "_pool_snapshot_dicts", lambda: [])
    monkeypatch.setattr(metrics, "_pool_operation_snapshot_dicts", lambda: [snapshot])
    exporter = metrics.SGLangPrometheusMetricsExporter(
        {"model_name": "test-model"}, counter_cls=FakeMetric, gauge_cls=FakeMetric,
    )
    exporter.update()
    for field in metrics._OPERATION_COUNTER_FIELDS:
        counter = FakeMetric.instances[f"kvcached_kv_cache_pool_operation_{field}"]
        assert counter.values[_labels()] == 0
    snapshot["allocation_failures_total"] = 1
    exporter.update()
    counter = FakeMetric.instances["kvcached_kv_cache_pool_operation_allocation_failures_total"]
    assert counter.values[_labels()] == 1


@pytest.mark.parametrize("during_init", [False, True])
@pytest.mark.parametrize("failure_site", ["labels", "inc", "logging"])
def test_secondary_export_error_preserves_collector_and_recovery(
    monkeypatch, during_init, failure_site
):
    monkeypatch.setattr(metrics, "_runtime_snapshot_dict", lambda: {"initialized": True})
    monkeypatch.setattr(metrics, "_pool_snapshot_dicts", lambda: [])
    monkeypatch.setattr(metrics, "_pool_operation_snapshot_dicts", lambda: [])
    monkeypatch.setattr(metrics.time, "monotonic", lambda: 120.0)

    class ExistingCollector:
        _counter_cls = FakeMetric
        _gauge_cls = FakeMetric

        def __init__(self, labels):
            self.labels = labels
            self.logged = []

        def log_stats(self, stats):
            self.logged.append(stats)
            return "base-result"

    def fail(*args, **kwargs):
        raise OSError("injected exporter backend failure")

    wrapped = metrics.wrap_scheduler_metrics_collector(ExistingCollector)
    collector = None if during_init else wrapped({"model_name": "test-model"})
    with monkeypatch.context() as broken:
        broken.setattr(FakeMetric, "set", fail)
        if failure_site == "logging":
            broken.setattr(metrics.logger, "warning", fail)
        else:
            broken.setattr(FakeMetric, failure_site, fail)
        if during_init:
            collector = wrapped({"model_name": "test-model"})
        assert collector is not None
        assert collector.log_stats("during failure") == "base-result"
        assert collector.logged == ["during failure"]

    assert collector.log_stats("after recovery") == "base-result"
    assert collector.logged == ["during failure", "after recovery"]
    assert FakeMetric.instances[
        "kvcached_sglang_metrics_export_last_success_timestamp_seconds"
    ].values


def test_collector_wrapper_preserves_native_collector_errors(monkeypatch):
    monkeypatch.setattr(metrics, "_runtime_snapshot_dict", lambda: {})
    monkeypatch.setattr(metrics, "_pool_snapshot_dicts", lambda: [])
    monkeypatch.setattr(metrics, "_pool_operation_snapshot_dicts", lambda: [])
    failure = RuntimeError("native collector failed")

    class ExistingCollector:
        _counter_cls = FakeMetric
        _gauge_cls = FakeMetric

        def __init__(self):
            self.labels = {"model_name": "test-model"}

        def log_stats(self, stats):
            raise failure

    collector = metrics.wrap_scheduler_metrics_collector(ExistingCollector)()
    with pytest.raises(RuntimeError) as caught:
        collector.log_stats(None)
    assert caught.value is failure


@pytest.mark.parametrize("logging_fails", [False, True])
def test_collector_wrapper_isolates_exporter_initialization_errors(monkeypatch, logging_fails):
    class ExistingCollector:
        def __init__(self, labels: Dict[str, Any]) -> None:
            self.labels = labels

        def log_stats(self, stats: Any) -> str:
            return f"base:{stats}"

    def fail_exporter(*args: Any, **kwargs: Any) -> None:
        del args, kwargs
        raise RuntimeError("metric backend rejected registration")

    monkeypatch.setattr(metrics, "SGLangPrometheusMetricsExporter", fail_exporter)
    if logging_fails:
        monkeypatch.setattr(metrics.logger, "warning", fail_exporter)

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
    assert collector.log_stats("first") == "base:first"
    assert collector.log_stats("stats") == "base:stats"
    assert FakeMetric.instances[
        "kvcached_sglang_metrics_export_last_success_timestamp_seconds"
    ].values


def test_non_reporting_collector_does_not_publish_startup_samples(monkeypatch):
    calls = []

    def runtime_snapshot():
        calls.append("snapshot")
        return {"initialized": True}

    monkeypatch.setattr(metrics, "_runtime_snapshot_dict", runtime_snapshot)
    monkeypatch.setattr(metrics, "_pool_snapshot_dicts", lambda: [])
    monkeypatch.setattr(metrics, "_pool_operation_snapshot_dicts", lambda: [])

    class ExistingCollector:
        _counter_cls = FakeMetric
        _gauge_cls = FakeMetric

        def __init__(self):
            self.labels = {"model_name": "test-model", "tp_rank": "1"}

        def log_stats(self, stats):
            return stats

    collector = metrics.wrap_scheduler_metrics_collector(ExistingCollector)()
    assert calls == []
    assert all(not metric.values for metric in FakeMetric.instances.values())
    assert collector.log_stats("enabled by engine") == "enabled by engine"
    assert calls == ["snapshot"]
    assert FakeMetric.instances["kvcached_runtime_initialized"].values


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


@pytest.mark.parametrize("with_server_args", [False, True])
@pytest.mark.parametrize("keyword_call", [False, True])
def test_metrics_patch_preserves_both_resolver_signatures(
    monkeypatch, with_server_args, keyword_call
):
    class ExistingCollector:
        pass

    module = ModuleType("fake_sglang_metrics")
    def with_args(server_args, role, default_cls):
        return server_args.stat_loggers.get(role, default_cls)

    def without_args(role, default_cls):
        return ExistingCollector if role == "scheduler" else default_cls

    setattr(module, "resolve_collector_class", with_args if with_server_args else without_args)
    monkeypatch.setenv("ENABLE_KVCACHED", "true")
    assert SGLangMetricsPatch().apply(module)
    resolver = getattr(module, "resolve_collector_class")
    server_args = SimpleNamespace(stat_loggers={"scheduler": ExistingCollector})

    def call(role):
        if keyword_call:
            kwargs = {"role": role, "default_cls": object}
            if with_server_args:
                kwargs["server_args"] = server_args
            return resolver(**kwargs)
        if with_server_args:
            return resolver(server_args, role, object)
        return resolver(role, object)

    selected = call("scheduler")
    assert selected is not ExistingCollector
    assert issubclass(selected, ExistingCollector)
    assert call("tokenizer") is object
    monkeypatch.setenv("ENABLE_KVCACHED", "false")
    assert call("scheduler") is ExistingCollector
