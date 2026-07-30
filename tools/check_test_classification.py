# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

"""Validate the exhaustive test execution-boundary manifests."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List


ROOT = Path(__file__).resolve().parents[1]
TEST_ROOT = ROOT / "tests"
MANIFEST_ROOT = TEST_ROOT / "manifests"
CATEGORIES = ("cpu", "gpu", "integration")


class ClassificationError(ValueError):
    """Raised when the test manifests do not form an exact partition."""


def discover_tests(test_root: Path = TEST_ROOT) -> List[str]:
    return sorted(
        path.relative_to(ROOT).as_posix() for path in test_root.glob("test_*.py")
    )


def read_manifest(path: Path) -> List[str]:
    entries = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.partition("#")[0].strip()
        if line:
            entries.append(line)
    return entries


def load_manifests(manifest_root: Path = MANIFEST_ROOT) -> Dict[str, List[str]]:
    missing = [
        category
        for category in CATEGORIES
        if not (manifest_root / f"{category}.txt").is_file()
    ]
    if missing:
        raise ClassificationError(
            "missing test manifests: " + ", ".join(f"{name}.txt" for name in missing)
        )
    return {
        category: read_manifest(manifest_root / f"{category}.txt")
        for category in CATEGORIES
    }


def validate_classification(
    discovered: Iterable[str], manifests: Dict[str, List[str]]
) -> None:
    discovered_set = set(discovered)
    entries = [entry for category in CATEGORIES for entry in manifests[category]]
    counts = Counter(entries)

    invalid_paths = sorted(
        entry
        for entry in counts
        if not entry.startswith("tests/test_")
        or not entry.endswith(".py")
        or Path(entry).is_absolute()
        or ".." in Path(entry).parts
    )
    duplicates = sorted(entry for entry, count in counts.items() if count > 1)
    classified_set = set(entries)
    missing = sorted(discovered_set - classified_set)
    unknown = sorted(classified_set - discovered_set)
    unsorted = [
        category
        for category in CATEGORIES
        if manifests[category] != sorted(manifests[category])
    ]

    problems = []
    if invalid_paths:
        problems.append("invalid paths: " + ", ".join(invalid_paths))
    if duplicates:
        problems.append("classified more than once: " + ", ".join(duplicates))
    if missing:
        problems.append("unclassified tests: " + ", ".join(missing))
    if unknown:
        problems.append("unknown tests: " + ", ".join(unknown))
    if unsorted:
        problems.append("unsorted manifests: " + ", ".join(unsorted))
    if problems:
        raise ClassificationError("; ".join(problems))


def classified_tests(category: str) -> List[str]:
    manifests = load_manifests()
    validate_classification(discover_tests(), manifests)
    return manifests[category]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--list-category", choices=CATEGORIES)
    args = parser.parse_args()

    manifests = load_manifests()
    discovered = discover_tests()
    validate_classification(discovered, manifests)

    if args.list_category:
        print("\n".join(manifests[args.list_category]))
    else:
        counts = ", ".join(
            f"{category}={len(manifests[category])}" for category in CATEGORIES
        )
        print(f"test classification valid: total={len(discovered)}, {counts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
