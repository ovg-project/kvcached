# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

"""
Record kvcached shared-memory pool usage over time.

Reads bench-config-amd.yaml (or any kvcached config) to discover instance
names and ports, resolves each port → PID → PGID → IPC segment name, and
logs per-instance memory usage to CSV.

Run this concurrently with a benchmark to capture how the elastic KV cache
pool is redistributed across models under load.

Usage:
    python record_kvmem.py --config bench-config-amd.yaml --output results/kvmem.csv
    python record_kvmem.py --config bench-config-amd.yaml --interval 0.5 --duration 300
"""

import argparse
import csv
import os
import signal
import sys
import time
from typing import Dict, List, Optional, Tuple

import yaml

# ---------------------------------------------------------------------------
# Resolve kvcached on sys.path
# ---------------------------------------------------------------------------
_here = os.path.dirname(os.path.abspath(__file__))
_root = os.path.join(_here, "..", "..")
if _root not in sys.path:
    sys.path.insert(0, _root)

from kvcached.cli.utils import (
    SHM_DIR,
    MemInfoStruct,
    RwLockedShm,
    _format_size,
    get_ipc_name,
)


# ---------------------------------------------------------------------------
# Config parsing
# ---------------------------------------------------------------------------

def load_instances(config_path: str) -> List[Dict]:
    """
    Parse a kvcached YAML config and return a list of instance dicts, each with:
      name, model, engine, port
    """
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    instances = []
    for inst in cfg.get("instances", []):
        port = None
        for arg in inst.get("engine_args", []):
            if arg.startswith("--port="):
                port = int(arg.split("=", 1)[1])
            elif arg == "--port":
                pass  # next arg would be value; not used in our config
        instances.append({
            "name":   inst.get("name", "unknown"),
            "model":  inst.get("model", "unknown"),
            "engine": inst.get("engine", "vllm"),
            "port":   port,
        })
    return instances


# ---------------------------------------------------------------------------
# Port → PID → PGID → IPC name resolution
# ---------------------------------------------------------------------------

def port_to_pid(port: int) -> Optional[int]:
    """Find the PID of the process listening on a given TCP port via /proc/net/tcp."""
    # Build inode → pid map from /proc
    inode_to_pid: Dict[int, int] = {}
    try:
        for pid_str in os.listdir("/proc"):
            if not pid_str.isdigit():
                continue
            pid = int(pid_str)
            fd_dir = f"/proc/{pid}/fd"
            try:
                for fd in os.listdir(fd_dir):
                    link = os.readlink(f"{fd_dir}/{fd}")
                    if link.startswith("socket:["):
                        inode = int(link[8:-1])
                        inode_to_pid[inode] = pid
            except (PermissionError, FileNotFoundError):
                continue
    except Exception:
        return None

    # Search /proc/net/tcp and /proc/net/tcp6
    hex_port = f"{port:04X}"
    for tcp_file in ("/proc/net/tcp", "/proc/net/tcp6"):
        try:
            with open(tcp_file) as f:
                next(f)  # skip header line
                for line in f:
                    parts = line.split()
                    if len(parts) < 10:
                        continue
                    local_addr = parts[1]
                    state = parts[3]
                    inode = int(parts[9])
                    # state 0A = listening
                    if state != "0A":
                        continue
                    local_port = local_addr.split(":")[1]
                    if local_port.upper() == hex_port:
                        return inode_to_pid.get(inode)
        except FileNotFoundError:
            continue
    return None


def pid_to_pgid(pid: int) -> Optional[int]:
    try:
        return os.getpgid(pid)
    except Exception:
        return None


def pgid_to_ipc_name(pgid: int, engine: str = "vllm") -> str:
    """Reconstruct the IPC name kvcached uses: kvcached_<EngineTag>_<PGID>."""
    tag = "vLLM" if engine.lower() == "vllm" else "SGLang"
    return f"kvcached_{tag}_{pgid}"


def resolve_ipc_for_instance(inst: Dict) -> Optional[str]:
    """
    Given an instance dict (with port and engine), return the IPC segment name
    by following: port → PID → PGID → IPC name.
    Returns None if resolution fails.
    """
    port = inst.get("port")
    if not port:
        return None
    pid = port_to_pid(port)
    if pid is None:
        return None
    pgid = pid_to_pgid(pid)
    if pgid is None:
        return None
    return pgid_to_ipc_name(pgid, inst.get("engine", "vllm"))


# ---------------------------------------------------------------------------
# Fallback: auto-detect all active kvcached IPC segments
# ---------------------------------------------------------------------------

def detect_all_ipc_names() -> List[str]:
    candidates: List[str] = []
    try:
        for fname in os.listdir(SHM_DIR):
            path = os.path.join(SHM_DIR, fname)
            try:
                st = os.stat(path)
            except Exception:
                continue
            if st.st_size != MemInfoStruct.SHM_SIZE:
                continue
            try:
                with RwLockedShm(fname, MemInfoStruct.SHM_SIZE, RwLockedShm.RLOCK) as mm:
                    if MemInfoStruct.from_buffer(mm).total_size > 0:
                        candidates.append(fname)
            except Exception:
                continue
    except FileNotFoundError:
        pass
    return sorted(candidates)


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

def sample_ipc(ipc_name: str) -> Optional[Tuple[int, int, int]]:
    """Returns (total_bytes, used_bytes, prealloc_bytes) or None."""
    try:
        with RwLockedShm(get_ipc_name(ipc_name), MemInfoStruct.SHM_SIZE, RwLockedShm.RLOCK) as mm:
            info = MemInfoStruct.from_buffer(mm)
            return int(info.total_size), int(info.used_size), int(info.prealloc_size)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Record kvcached KV-cache pool usage to CSV, labelled by instance."
    )
    parser.add_argument(
        "--config",
        default="bench-config-amd.yaml",
        help="Path to kvcached YAML config (default: bench-config-amd.yaml).",
    )
    parser.add_argument(
        "--output",
        default="results/kvmem.csv",
        help="Output CSV file (default: results/kvmem.csv).",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=1.0,
        help="Sampling interval in seconds (default: 1.0).",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        help="Stop after this many seconds. Runs until Ctrl-C if omitted.",
    )
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else ".", exist_ok=True)

    # Load instances from config
    instances = load_instances(args.config)
    print(f"Loaded {len(instances)} instance(s) from {args.config}:")
    for inst in instances:
        print(f"  {inst['name']:12s}  model={inst['model']}  port={inst['port']}")
    print()

    # Graceful shutdown
    stop = {"flag": False}
    def _handler(sig, frame):
        stop["flag"] = True
    signal.signal(signal.SIGINT, _handler)
    signal.signal(signal.SIGTERM, _handler)

    print(f"Recording → {args.output}  (interval={args.interval}s, Ctrl-C to stop)\n")

    start_wall = time.time()
    row_count = 0

    # Cache IPC name resolutions (re-resolve if None, as servers may start later)
    ipc_cache: Dict[str, Optional[str]] = {inst["name"]: None for inst in instances}

    with open(args.output, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "timestamp",
            "elapsed_s",
            "instance_name",
            "model",
            "port",
            "ipc_name",
            "total_gb",
            "used_gb",
            "prealloc_gb",
            "free_gb",
            "used_pct",
        ])

        while not stop["flag"]:
            now = time.time()
            elapsed = now - start_wall
            any_data = False

            for inst in instances:
                name = inst["name"]

                # Resolve IPC name if not yet cached
                if ipc_cache[name] is None:
                    ipc_cache[name] = resolve_ipc_for_instance(inst)

                ipc_name = ipc_cache[name]
                if ipc_name is None:
                    print(f"[{elapsed:6.1f}s] {name:12s}  waiting for server on port {inst['port']}...")
                    continue

                result = sample_ipc(ipc_name)
                if result is None:
                    # Segment gone — clear cache so we re-resolve next tick
                    ipc_cache[name] = None
                    continue

                total, used, prealloc = result
                free = max(total - used - prealloc, 0)
                used_pct = (used + prealloc) / total * 100 if total else 0

                writer.writerow([
                    f"{now:.3f}",
                    f"{elapsed:.2f}",
                    name,
                    inst["model"],
                    inst["port"],
                    ipc_name,
                    f"{total / 1e9:.3f}",
                    f"{used / 1e9:.3f}",
                    f"{prealloc / 1e9:.3f}",
                    f"{free / 1e9:.3f}",
                    f"{used_pct:.1f}",
                ])
                row_count += 1
                any_data = True

                print(
                    f"[{elapsed:6.1f}s] {name:12s}  "
                    f"used={_format_size(used):>10s}  "
                    f"prealloc={_format_size(prealloc):>10s}  "
                    f"free={_format_size(free):>10s}  "
                    f"({used_pct:.1f}%)"
                )

            if any_data:
                f.flush()

            if args.duration and elapsed >= args.duration:
                break

            time.sleep(args.interval)

    print(f"\nDone. Wrote {row_count} rows to {args.output}")


if __name__ == "__main__":
    main()
