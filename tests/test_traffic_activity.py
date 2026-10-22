# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

"""Request completion order must not move the model's activity backwards."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "controller"))

from controller import traffic_monitor
from controller.sleep_manager import SleepConfig, SleepManager


@pytest.mark.parametrize(
    "start_times, completion_order",
    [
        ((100.0, 400.0), (0, 1)),
        ((100.0, 400.0), (1, 0)),
        ((400.0, 400.0), (0, 1)),
        ((400.0, 400.0), (1, 0)),
    ],
    ids=["in-order", "out-of-order", "equal-starts", "equal-starts-reversed"],
)
@pytest.mark.parametrize("first_success", [True, False])
def test_completion_order_preserves_recent_activity(
    monkeypatch, start_times, completion_order, first_success
):
    now = 0.0
    monkeypatch.setattr(traffic_monitor.time, "time", lambda: now)
    monitor = traffic_monitor.TrafficMonitor(idle_threshold_seconds=300)
    sleep_manager = SleepManager(
        SleepConfig(idle_threshold_seconds=300), traffic_monitor=monitor
    )

    requests = []
    for now in start_times:
        requests.append(monitor.record_request_start("model", "/v1/completions"))

    successes = (first_success, True)
    successful_durations = []
    for position, request_index in enumerate(completion_order):
        now = 401.0 + position
        monitor.record_request_end(
            requests[request_index], success=successes[request_index]
        )
        if successes[request_index]:
            successful_durations.append(now - start_times[request_index])

    stats = monitor.get_model_stats("model")
    assert stats is not None
    assert stats.last_activity_time == 400.0
    assert stats.total_requests == 2
    assert stats.successful_requests == sum(successes)
    assert stats.failed_requests == 2 - sum(successes)
    assert stats.avg_response_time == pytest.approx(
        sum(successful_durations) / len(successful_durations)
    )
    assert list(stats.request_timestamps) == [
        start_times[index] for index in completion_order
    ]
    assert monitor.get_idle_models() == []
    assert monitor.get_active_models() == ["model"]
    assert sleep_manager.get_sleep_candidates() == []
    summary = monitor.get_traffic_summary()["model"]
    assert summary["last_activity_time"] == 400.0
    assert summary["idle_time_seconds"] == 2.0
    assert not summary["is_idle"]

    # Retain the existing request-start timestamp and strict idle threshold.
    now = 700.0
    assert sleep_manager.get_sleep_candidates() == []
    now = 701.0
    assert monitor.get_idle_models() == ["model"]
    assert monitor.get_active_models() == []
    assert sleep_manager.get_sleep_candidates() == ["model"]
