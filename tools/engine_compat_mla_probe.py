#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0
"""Tiny FP16 DeepSeek-V2 MLA; dense FFNs, no V4, TP/PP or model-quality coverage."""

from functools import partial

from engine_compat_family_probe import probe, worker

if __name__ == "__main__":
    raise SystemExit(probe.main(worker_fn=partial(worker, "mla"),
                               script=__file__, description=__doc__))
