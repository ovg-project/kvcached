#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

"""Verify that an engine initialized the kvcached allocator successfully."""

from __future__ import annotations

import argparse
from pathlib import Path


def activation_marker(engine: str) -> str:
    return f"KVCACHED_ENGINE_INTEGRATION_READY engine={engine} device="


def verify_activation(engine: str, log_text: str) -> None:
    marker = activation_marker(engine)
    if marker not in log_text:
        raise ValueError(
            f"missing successful kvcached allocator initialization marker: {marker}"
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", choices=("vllm", "sglang"), required=True)
    parser.add_argument("--log", type=Path, required=True)
    args = parser.parse_args()

    try:
        verify_activation(args.engine, args.log.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError) as exc:
        print(f"kvcached activation check failed: {exc}")
        return 1

    print(f"KVCACHED_ACTIVATION_VERIFIED engine={args.engine}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
