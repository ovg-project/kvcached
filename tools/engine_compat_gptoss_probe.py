#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0
"""Tiny FP16 GPT-OSS: sliding/full attention and MoE, not MXFP4 qualification."""

from functools import partial

from engine_compat_family_probe import probe, worker

if __name__ == "__main__":
    raise SystemExit(probe.main(worker_fn=partial(worker, "gptoss"),
                               script=__file__, description=__doc__))
