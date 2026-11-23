# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

"""What counts as model activity, and when a model may be put to sleep.

Two rules are pinned here:

* Request completion order must not move the model's activity backwards.
* A model serving a request is busy, however long that request has been
  running. ``SleepManager`` nominates models for a sleep call, and vLLM's sleep
  in abort mode kills running requests, so a long generation must never look
  like an idle model.
"""

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
    # Activity is the model's most recent work: the last completion, which is
    # later than every start, in every completion order.
    assert stats.last_activity_time == 402.0
    assert stats.inflight_requests == 0
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
    assert summary["last_activity_time"] == 402.0
    assert summary["idle_time_seconds"] == 0.0
    assert summary["inflight_requests"] == 0
    assert not summary["is_idle"]

    # Idle time is counted from the last completion, with a strict threshold.
    now = 702.0  # one second under the threshold
    assert sleep_manager.get_sleep_candidates() == []
    now = 703.0  # one second over it
    assert monitor.get_idle_models() == ["model"]
    assert monitor.get_active_models() == []
    assert sleep_manager.get_sleep_candidates() == ["model"]


def _monitor_and_manager(monkeypatch, idle_threshold_seconds=300):
    clock = {"now": 0.0}
    monkeypatch.setattr(traffic_monitor.time, "time", lambda: clock["now"])
    monitor = traffic_monitor.TrafficMonitor(
        idle_threshold_seconds=idle_threshold_seconds
    )
    sleep_manager = SleepManager(
        SleepConfig(idle_threshold_seconds=idle_threshold_seconds),
        traffic_monitor=monitor,
    )
    return clock, monitor, sleep_manager


def test_a_model_serving_a_long_request_is_never_a_sleep_candidate(monkeypatch):
    """A generation that runs for longer than the idle threshold is not idle.

    This is the shape of the regression: the activity clock used to be stamped
    with the request's *arrival*, so a model that had been generating for five
    minutes looked five minutes idle, and the controller answered by calling
    vLLM's sleep endpoint -- which aborts the running request.
    """
    clock, monitor, sleep_manager = _monitor_and_manager(monkeypatch)

    clock["now"] = 100.0
    in_flight = monitor.record_request_start("model", "/v1/chat/completions")

    # The request is still generating far past the idle threshold.
    for elapsed in (150.0, 299.0, 600.0):
        clock["now"] = 100.0 + elapsed
        stats = monitor.get_model_stats("model")
        assert stats is not None, "a model with an in-flight request must be tracked"
        assert stats.inflight_requests == 1
        assert not stats.is_idle(300)
        assert monitor.get_idle_models() == []
        assert monitor.get_active_models() == ["model"]
        assert sleep_manager.get_sleep_candidates() == []

    # Only once it finishes does the idle clock start.
    clock["now"] = 700.0
    monitor.record_request_end(in_flight, success=True)
    stats = monitor.get_model_stats("model")
    assert stats.inflight_requests == 0
    assert stats.last_activity_time == 700.0
    assert not stats.is_idle(300)
    assert sleep_manager.get_sleep_candidates() == []

    clock["now"] = 1001.0
    assert sleep_manager.get_sleep_candidates() == ["model"]


def test_inflight_counting_survives_a_duplicate_completion(monkeypatch):
    """A request closed twice must not drive the count negative or double count."""
    clock, monitor, _ = _monitor_and_manager(monkeypatch)

    clock["now"] = 10.0
    first = monitor.record_request_start("model", "/v1/completions")
    clock["now"] = 11.0
    second = monitor.record_request_start("model", "/v1/completions")
    assert monitor.get_model_stats("model").inflight_requests == 2

    clock["now"] = 20.0
    monitor.record_request_end(first, success=True)
    monitor.record_request_end(first, success=False, error_message="duplicate")
    stats = monitor.get_model_stats("model")
    assert stats.inflight_requests == 1
    assert stats.total_requests == 1
    assert stats.successful_requests == 1
    assert stats.failed_requests == 0

    monitor.record_request_end(second, success=False, error_message="boom")
    stats = monitor.get_model_stats("model")
    assert stats.inflight_requests == 0
    assert stats.total_requests == 2
    assert stats.failed_requests == 1


def test_sleep_candidates_exclude_a_busy_model_even_if_the_clock_says_idle(
    monkeypatch,
):
    """The candidate list is the last line of defence before a sleep call.

    If the activity clock is ever wrong again, a model that is still serving
    must not be nominated, because nomination is what aborts its requests.
    """
    clock, monitor, sleep_manager = _monitor_and_manager(monkeypatch)

    clock["now"] = 10.0
    monitor.record_request_start("model", "/v1/completions")

    # Pretend an activity-tracking defect reports the model as idle.
    monkeypatch.setattr(monitor, "get_idle_models", lambda *a, **kw: ["model"])

    assert monitor.get_model_stats("model").inflight_requests == 1
    assert sleep_manager.get_sleep_candidates() == []


def test_sleep_candidates_still_exclude_models_that_are_already_asleep(
    monkeypatch,
):
    clock, monitor, sleep_manager = _monitor_and_manager(monkeypatch)

    clock["now"] = 10.0
    request = monitor.record_request_start("model", "/v1/completions")
    clock["now"] = 20.0
    monitor.record_request_end(request, success=True)
    sleep_manager.sleeping_models["model"] = clock["now"]

    clock["now"] = 1000.0
    assert monitor.get_idle_models() == ["model"]
    assert sleep_manager.get_sleep_candidates() == []


def test_explicit_out_of_order_completions_do_not_move_activity_backwards(
    monkeypatch,
):
    """The activity clock only ever moves forward."""
    clock, monitor, _ = _monitor_and_manager(monkeypatch)

    clock["now"] = 100.0
    first = monitor.record_request_start("model", "/v1/completions")
    clock["now"] = 200.0
    second = monitor.record_request_start("model", "/v1/completions")

    # *Later* requests complete with an earlier end time.
    clock["now"] = 500.0
    monitor.record_request_end(second, success=True)
    assert monitor.get_model_stats("model").last_activity_time == 500.0

    clock["now"] = 510.0
    monitor.record_request_end(first, success=True, response_time=150.0)
    assert monitor.get_model_stats("model").last_activity_time == 500.0


def test_unknown_model_state_is_not_a_sleep_candidate(monkeypatch):
    """Unknown state fails closed: nomination is what issues the sleep call."""
    clock, monitor, sleep_manager = _monitor_and_manager(monkeypatch)

    clock["now"] = 1000.0
    # Pretend the clock reports the model as idle while its stats are missing.
    monkeypatch.setattr(monitor, "get_idle_models", lambda *a, **kw: ["model"])
    monkeypatch.setattr(monitor, "get_model_stats", lambda *a, **kw: None)

    assert sleep_manager.get_sleep_candidates() == []


async def test_auto_sleep_rechecks_busy_right_before_sleeping(monkeypatch):
    """Auto-sleep re-checks the model after nomination, before the sleep call."""
    clock, monitor, sleep_manager = _monitor_and_manager(monkeypatch)
    sleep_manager.config.vllm_models_config["model"] = {
        "host": "127.0.0.1",
        "port": "1",
    }
    calls = []

    async def fake_sleep(host, port, level=1):
        calls.append(level)
        return True

    monkeypatch.setattr(sleep_manager, "_call_vllm_sleep_api", fake_sleep)

    clock["now"] = 10.0
    request = monitor.record_request_start("model", "/v1/completions")
    clock["now"] = 20.0
    monitor.record_request_end(request, success=True)

    # The candidate list is stale by the time it is acted on.
    clock["now"] = 1000.0
    assert sleep_manager.get_sleep_candidates() == ["model"]
    monitor.record_request_start("model", "/v1/completions")

    assert await sleep_manager.put_model_to_sleep("model", manual=False) is False
    assert calls == []
    assert sleep_manager.get_sleeping_models() == {}


async def test_auto_sleep_rechecks_the_idle_interval_not_just_busy(monkeypatch):
    """A model re-busied after nomination must idle out again before sleeping."""
    clock, monitor, sleep_manager = _monitor_and_manager(
        monkeypatch, idle_threshold_seconds=1)
    sleep_manager.config.vllm_models_config["model"] = {
        "host": "127.0.0.1",
        "port": "1",
    }
    calls = []

    async def fake_sleep(host, port, level=1):
        calls.append(level)
        return True

    monkeypatch.setattr(sleep_manager, "_call_vllm_sleep_api", fake_sleep)

    clock["now"] = 0.0
    first = monitor.record_request_start("model", "/v1/completions")
    clock["now"] = 1.0
    monitor.record_request_end(first, success=True)

    # Nomination happened here: idle past the 1 s threshold.
    clock["now"] = 3.0
    assert sleep_manager.get_sleep_candidates() == ["model"]

    # The model served another request and finished moments ago: not busy, but
    # nowhere near the idle threshold.
    second = monitor.record_request_start("model", "/v1/completions")
    clock["now"] = 3.003
    monitor.record_request_end(second, success=True)

    assert await sleep_manager.put_model_to_sleep("model", manual=False) is False
    assert calls == []
    assert sleep_manager.get_sleeping_models() == {}


async def test_manual_sleep_still_honours_the_operator_while_busy(monkeypatch):
    """Manual sleep proceeds, but it is the operator's call, not an accident."""
    clock, monitor, sleep_manager = _monitor_and_manager(monkeypatch)
    sleep_manager.config.vllm_models_config["model"] = {
        "host": "127.0.0.1",
        "port": "1",
    }
    calls = []

    async def fake_sleep(host, port, level=1):
        calls.append(level)
        return True

    monkeypatch.setattr(sleep_manager, "_call_vllm_sleep_api", fake_sleep)

    clock["now"] = 10.0
    monitor.record_request_start("model", "/v1/completions")

    assert await sleep_manager.put_model_to_sleep("model", manual=True) is True
    assert calls == [1]
