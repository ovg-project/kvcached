#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0
"""SGLang runtime/CUDA negative control, not a model-serving qualification.

Task-specific installed-engine behavior is checked by the separately frozen GPU
contracts. This unconditional check prevents a mock-only contract from claiming
that CUDA availability and allocation recovery were exercised.
"""

import argparse
import importlib.metadata
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "output", "version", "candidate-sha", "runner", "layout", "mode"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    import torch

    assert importlib.metadata.version("sglang").split("+")[0] == args.version
    assert torch.cuda.is_available(), "GPU checks cannot run on a CPU-only host"
    total = torch.cuda.get_device_properties(0).total_memory
    try:
        torch.empty(total * 2, dtype=torch.uint8, device="cuda:0")
    except torch.OutOfMemoryError:
        pass
    else:
        raise AssertionError("Expected oversized physical CUDA allocation to fail")
    value = torch.arange(128, device="cuda:0")
    assert int((value + 1).sum()) == 8256
    torch.cuda.synchronize(0)
    del value
    torch.cuda.empty_cache()
    (output / "result.json").write_text(json.dumps(dict(
        status="passed", comparison="passed", candidate_sha=args.candidate_sha,
        expected_vllm_version=args.version, requested_runner=args.runner, layout=args.layout,
        qualification="sglang-runtime-and-cuda-recovery-only",
    )) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
