# SPDX-License-Identifier: Apache-2.0
"""Read kvcached's published memory usage from its shared-memory segment.

`PageAllocator::update_memory_usage()` publishes, per KV cache group:
  used_size     = num_inuse_pages       * num_layers * page_size * num_kv_buffers
  prealloc_size = len(reserved_page_list) * num_layers * page_size * num_kv_buffers

`used_size` is exactly the physical memory pinned by mapped pages -- the
quantity issue #359 is about. `prealloc_size` is the reserved pool that
free_page() parks pages in before unmapping them; those are also still mapped.
"""
import argparse
import json
import os
import sys
import time

from kvcached.cli.utils import SHM_DIR, MemInfoStruct, RwLockedShm

GB = 1024**3


def detect_segments():
    """Return every /dev/shm segment that looks like a kvcached MemInfoStruct."""
    found = []
    for fname in sorted(os.listdir(SHM_DIR)):
        path = os.path.join(SHM_DIR, fname)
        try:
            if os.stat(path).st_size != MemInfoStruct.SHM_SIZE:
                continue
            with RwLockedShm(fname, MemInfoStruct.SHM_SIZE,
                             RwLockedShm.RLOCK) as mm:
                if MemInfoStruct.from_buffer(mm).total_size <= 0:
                    continue
        except Exception:  # noqa: BLE001 -- unreadable segments are not ours
            continue
        found.append(fname)
    return found


def read_mem(name):
    with RwLockedShm(name, MemInfoStruct.SHM_SIZE, RwLockedShm.RLOCK) as mm:
        info = MemInfoStruct.from_buffer(mm)
    return {
        "seg": name,
        "total_gb": info.total_size / GB,
        "used_gb": info.used_size / GB,
        "prealloc_gb": info.prealloc_size / GB,
        "pinned_gb": (info.used_size + info.prealloc_size) / GB,
        "used_bytes": info.used_size,
        "prealloc_bytes": info.prealloc_size,
    }


def read_all(names=None):
    names = names or detect_segments()
    out = []
    for n in names:
        try:
            out.append(read_mem(n))
        except Exception as e:  # noqa: BLE001
            out.append({"seg": n, "error": f"{type(e).__name__}: {e}"})
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seg", default=None)
    ap.add_argument("--watch", type=float, default=0.0)
    ap.add_argument("--duration", type=float, default=0.0)
    ap.add_argument("--jsonl", default=None)
    args = ap.parse_args()

    names = [args.seg] if args.seg else None
    if args.watch <= 0:
        print(json.dumps(read_all(names), indent=1))
        sys.exit(0)

    out = open(args.jsonl, "w") if args.jsonl else None
    t0 = time.time()
    pinned_names = names
    while True:
        if not pinned_names:
            detected = detect_segments()
            if detected:
                pinned_names = detected
        recs = read_all(pinned_names) if pinned_names else []
        line = json.dumps({"t": time.time() - t0, "segs": recs})
        if out:
            out.write(line + "\n")
            out.flush()
        else:
            print(line, flush=True)
        if args.duration and time.time() - t0 >= args.duration:
            break
        time.sleep(args.watch)
    if out:
        out.close()
