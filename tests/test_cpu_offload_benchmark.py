# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

import pytest

from tools.benchmark_cpu_offload import summarize_ms


def test_summarize_ms_reports_tail_latency():
    summary = summarize_ms([1.0, 2.0, 3.0, 4.0, 100.0])

    assert summary == {
        "max": 100.0,
        "mean": 22.0,
        "min": 1.0,
        "p50": 3.0,
        "p95": 100.0,
        "p99": 100.0,
    }


def test_summarize_ms_rejects_empty_samples():
    with pytest.raises(ValueError, match="must not be empty"):
        summarize_ms([])
