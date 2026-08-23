# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

MODULE_PATH = (
    Path(__file__).parents[1] / "tools" / "check_test_classification.py"
)
SPEC = importlib.util.spec_from_file_location("check_test_classification", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
classification = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = classification
SPEC.loader.exec_module(classification)


def manifests(**overrides):
    values: dict[str, list[str]] = {
        category: [] for category in classification.CATEGORIES
    }
    values.update(overrides)
    return values


def test_exact_partition_is_valid():
    discovered = ["tests/test_a.py", "tests/test_b.py"]

    classification.validate_classification(
        discovered,
        manifests(cpu=["tests/test_a.py"], integration=["tests/test_b.py"]),
    )


@pytest.mark.parametrize(
    ("classified", "message"),
    [
        (
            manifests(cpu=["tests/test_a.py"]),
            "unclassified tests: tests/test_b.py",
        ),
        (
            manifests(
                cpu=["tests/test_a.py"],
                gpu=["tests/test_a.py"],
                integration=["tests/test_b.py"],
            ),
            "classified more than once: tests/test_a.py",
        ),
        (
            manifests(
                cpu=["tests/test_a.py", "tests/test_missing.py"],
                integration=["tests/test_b.py"],
            ),
            "unknown tests: tests/test_missing.py",
        ),
    ],
)
def test_invalid_partition_fails(classified, message):
    with pytest.raises(classification.ClassificationError, match=message):
        classification.validate_classification(
            ["tests/test_a.py", "tests/test_b.py"], classified
        )


def test_manifest_entries_must_be_sorted():
    with pytest.raises(classification.ClassificationError, match="unsorted manifests"):
        classification.validate_classification(
            ["tests/test_a.py", "tests/test_b.py"],
            manifests(cpu=["tests/test_b.py", "tests/test_a.py"]),
        )
