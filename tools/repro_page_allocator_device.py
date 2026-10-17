#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

"""Hold bounded GPU0 pressure for the normal SGLang device-query reproduction.

Run with CUDA_VISIBLE_DEVICES=0 and ENABLE_KVCACHED=false. Wait for the ready
JSON, then start SGLang in another terminal as described in
docs/PAGE_ALLOCATOR_DEVICE_REPRO.md. Ctrl-C releases this process's allocation.
"""

import argparse
import json
import time


def hold_pressure():
    import torch

    if torch.cuda.device_count() != 1:
        raise SystemExit("Expose only the pressure GPU: CUDA_VISIBLE_DEVICES=0")
    torch.cuda.set_device(0)
    free, total = torch.cuda.mem_get_info()
    amount = int(total * 0.35)
    if free - amount < total * 0.5:
        raise SystemExit("Pressure GPU is not idle enough; leave at least half its memory free")

    allocation = torch.empty(amount, dtype=torch.uint8, device="cuda:0")
    allocation.fill_(19)
    torch.cuda.synchronize()
    after, _ = torch.cuda.mem_get_info()
    if after < total * 0.5:
        raise SystemExit("Pressure GPU memory changed during setup; stop and check other users")
    print(json.dumps({"state": "ready", "allocated_bytes": amount,
                      "free_before": free, "free_after": after, "total_bytes": total}), flush=True)
    while True:
        time.sleep(1)


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    try:
        hold_pressure()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
