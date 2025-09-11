# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

import argparse
import curses
import os
import time
from typing import List, Optional, Union

from kvcached.cli.utils import SHM_DIR, MemInfoStruct, RwLockedShm, _format_size, get_ipc_name


def _detect_kvcache_ipc_names() -> List[str]:
    """Detect all shared memory segments that look like KVCacheManager info."""
    candidates: List[str] = []
    try:
        for fname in os.listdir(SHM_DIR):
            path = os.path.join(SHM_DIR, fname)
            try:
                st = os.stat(path)
            except Exception:
                continue
            # Heuristic: MemInfoStruct segments have fixed size
            if st.st_size != MemInfoStruct.SHM_SIZE:
                continue
            try:
                with RwLockedShm(fname, MemInfoStruct.SHM_SIZE,
                                 RwLockedShm.RLOCK) as mm:
                    total_size = MemInfoStruct.from_buffer(mm).total_size
                    if total_size <= 0:
                        continue
                candidates.append(fname)
            except Exception:
                # Skip files we cannot map / lock
                continue
    except FileNotFoundError:
        pass
    return sorted(candidates)


def _draw_kvtop(stdscr: "curses._CursesWindow", ipc_names: Optional[List[str]],
                refresh_rate: float):
    """Curses UI loop that mimics a minimal `htop` for KV cache memory."""
    curses.curs_set(0)  # Hide cursor
    stdscr.nodelay(True)  # Non-blocking getch

    # ------------------------------------------------------------------
    # Color initialisation (if supported)
    # ------------------------------------------------------------------
    use_colors = False
    if curses.has_colors():
        try:
            curses.start_color()
            curses.use_default_colors()
            curses.init_pair(1, curses.COLOR_GREEN, -1)  # Low usage
            curses.init_pair(2, curses.COLOR_YELLOW, -1)  # Medium usage
            curses.init_pair(3, curses.COLOR_RED, -1)  # High usage
            curses.init_pair(4, curses.COLOR_CYAN, -1)  # Section headers
            use_colors = True
        except Exception:
            # Fallback gracefully if terminal has broken color support
            use_colors = False

    # ------------------------------------------------------------------
    # Show initializing message & pre-load heavy libs (torch)
    # ------------------------------------------------------------------
    stdscr.erase()
    stdscr.addstr(0, 0, "Initializing... please wait", curses.A_DIM)
    stdscr.refresh()

    try:
        import torch  # Heavy import occurs once here
        torch_available = torch.cuda.is_available()
    except Exception:
        torch = None  # type: ignore
        torch_available = False

    while True:
        # Determine which IPC names to show this frame
        names_to_show = ipc_names if ipc_names else _detect_kvcache_ipc_names()

        # ------------------------------------------------------------------
        # GPU physical memory usage (if CUDA available) – compute once
        # ------------------------------------------------------------------
        if torch_available:
            try:
                avail_gpu, total_gpu = torch.cuda.mem_get_info()
                gpu_used = total_gpu - avail_gpu
                gpu_percent = (gpu_used / total_gpu * 100) if total_gpu else 0
            except Exception:
                total_gpu = gpu_used = gpu_percent = None
        else:
            total_gpu = gpu_used = gpu_percent = None

        # Clear screen first
        stdscr.erase()
        height, width = stdscr.getmaxyx()
        header_attr = curses.color_pair(
            4) | curses.A_BOLD if use_colors else curses.A_BOLD
        stdscr.addstr(0, 0, "KVCache Memory Usage", header_attr)

        current_row = 2

        bar_width = max(10, width - 2)

        # Iterate over each detected/selected IPC name
        for idx, name in enumerate(names_to_show):
            if current_row >= height - 4:  # Reserve some rows for footer/GPU
                break

            # Fetch memory info for this IPC
            try:
                with RwLockedShm(get_ipc_name(name), MemInfoStruct.SHM_SIZE,
                                 RwLockedShm.RLOCK) as mm:
                    mem_info = MemInfoStruct.from_buffer(mm)
                    total = int(mem_info.total_size)
                    used = int(mem_info.used_size)
                    prealloc = int(mem_info.prealloc_size)
            except FileNotFoundError:
                total = used = prealloc = 0

            free = max(total - used - prealloc, 0)
            percent_total = ((used + prealloc) / total * 100) if total else 0

            # Determine colour based on combined utilisation (used + prealloc)
            if percent_total < 50:
                bar_color = curses.color_pair(1) if use_colors else 0
            elif percent_total < 80:
                bar_color = curses.color_pair(2) if use_colors else 0
            else:
                bar_color = curses.color_pair(3) if use_colors else 0

            used_width = int(bar_width * used / total) if total else 0
            prealloc_only_width = int(bar_width * prealloc /
                                      total) if total else 0
            bar_prealloc = "=" * prealloc_only_width
            bar_used = "#" * used_width
            bar_free = "-" * (bar_width - prealloc_only_width - used_width)

            # Section header per IPC
            stdscr.addstr(current_row, 0, f"IPC: {name}", header_attr)
            current_row += 1

            stdscr.addstr(current_row, 0, "[", header_attr)
            stdscr.addstr(bar_prealloc[:max(0, width - 2)],
                          curses.color_pair(2) if use_colors else 0)
            stdscr.addstr(bar_used[:max(0, width - 2 - len(bar_prealloc))],
                          bar_color)
            stdscr.addstr(
                bar_free[:max(0, width - 2 - len(bar_prealloc) -
                              len(bar_used))])
            if width - 1 > 0:
                stdscr.addstr(
                    current_row,
                    min(width - 1,
                        len("[" + bar_prealloc + bar_used + bar_free)), "]",
                    header_attr)

            current_row += 1
            stdscr.addstr(
                current_row,
                0,
                f"Prealloc: {_format_size(prealloc)} | Used: {_format_size(used)} / {_format_size(total)} ({percent_total:.1f}%) | Free: {_format_size(free)}"[:
                                                                                                                                                               width
                                                                                                                                                               -
                                                                                                                                                               1],
            )
            current_row += 2  # Blank line after each IPC section

        # ------------------------------------------------------------------
        # GPU section (if available) – placed after all IPCs
        # ------------------------------------------------------------------
        if total_gpu is not None and current_row < height - 4:
            gpu_bar_width = max(10, width - 2)
            gpu_used_width = int(gpu_bar_width * gpu_percent / 100)
            gpu_header_attr = header_attr
            stdscr.addstr(current_row, 0, "GPU Memory Usage", gpu_header_attr)

            if gpu_percent < 50:
                gpu_bar_color = curses.color_pair(1) if use_colors else 0
            elif gpu_percent < 80:
                gpu_bar_color = curses.color_pair(2) if use_colors else 0
            else:
                gpu_bar_color = curses.color_pair(3) if use_colors else 0

            stdscr.addstr(current_row + 1, 0, "[", gpu_header_attr)
            stdscr.addstr("#" * gpu_used_width, gpu_bar_color)
            stdscr.addstr("-" * (gpu_bar_width - gpu_used_width))
            if width - 1 > 0:
                stdscr.addstr(
                    current_row + 1,
                    min(
                        width - 1,
                        len("[" + "#" * gpu_used_width + "-" *
                            (gpu_bar_width - gpu_used_width))), "]",
                    gpu_header_attr)

            gpu_free = total_gpu - gpu_used if total_gpu is not None else 0
            stdscr.addstr(
                current_row + 2,
                0,
                f"Used: {_format_size(gpu_used)} / {_format_size(total_gpu)} ({gpu_percent:.1f}%) | Free: {_format_size(gpu_free)}"[:
                                                                                                                                    width
                                                                                                                                    -
                                                                                                                                    1],
            )
            current_row += 4
        elif total_gpu is None and current_row < height - 1:
            stdscr.addstr(current_row, 0, "GPU memory info unavailable",
                          curses.A_DIM)
            current_row += 2

        # Footer
        stdscr.addstr(current_row, 0, "Press 'q' to quit", curses.A_DIM)
        stdscr.refresh()

        # Handle input
        ch = stdscr.getch()
        if ch == ord("q") or ch == ord("Q"):
            break
        time.sleep(refresh_rate)


def kvtop(ipc_names: Union[str, List[str], None] = None,
          refresh_rate: float = 1.0) -> None:
    """Launch the curses UI similar to `htop` for KV cache memory.

    Args:
        ipc_names: A single IPC name, a list of names, or ``None`` to auto-detect.
        refresh_rate: Screen refresh interval in seconds.
    """
    if isinstance(ipc_names, str):
        ipc_names = [ipc_names]
    # Preserve backward compatibility with previous default behaviour
    if ipc_names == []:
        ipc_names = None

    curses.wrapper(_draw_kvtop, ipc_names, refresh_rate)


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------


def main() -> None:
    """Main entry point for kvtop CLI."""
    parser = argparse.ArgumentParser(
        description="Interactive TUI for monitoring KVCached memory usage.")
    parser.add_argument(
        "ipc_names",
        nargs="*",
        default=[],
        help=
        "Optional list of IPC names to monitor. If omitted, auto-detect all running KVCached managers."
    )
    parser.add_argument("--refresh",
                        type=float,
                        default=1.0,
                        help="Screen refresh interval in seconds.")
    args = parser.parse_args()

    kvtop(args.ipc_names if args.ipc_names else None, args.refresh)


if __name__ == "__main__":
    main()
