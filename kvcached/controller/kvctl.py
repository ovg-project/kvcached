import argparse
import atexit
import json
import os
import shlex
import subprocess
import sys
import time
from typing import Dict, List, Optional

from kvcached.controller.cli import get_kv_cache_limit, update_kv_cache_limit
from kvcached.controller.kvtop import _detect_kvcache_ipc_names
from kvcached.controller.kvtop import kvtop as kvtop_ui

try:
    import readline  # type: ignore
    READLINE_AVAILABLE = True
except ImportError:  # pragma: no cover – win / minimal envs
    READLINE_AVAILABLE = False

# ANSI colour handling -------------------------------------------------------

_ANSI_COLOR_CODES = {
    'reset': '\033[0m',
    'bold': '\033[1m',
    'red': '\033[31m',
    'green': '\033[32m',
    'yellow': '\033[33m',
    'blue': '\033[34m',
    'magenta': '\033[35m',
    'cyan': '\033[36m',
}


def _supports_color() -> bool:
    # Honour NO_COLOR spec and only colour TTYs
    if os.getenv('NO_COLOR') is not None:
        return False
    return sys.stdout.isatty()


COLOR_ENABLED = _supports_color()


def _clr(text: str, color: str | None = None, *, bold: bool = False) -> str:
    if not COLOR_ENABLED:
        return text
    seq = ''
    if bold:
        seq += _ANSI_COLOR_CODES['bold']
    if color and color in _ANSI_COLOR_CODES:
        seq += _ANSI_COLOR_CODES[color]
    if not seq:
        return text
    return f"{seq}{text}{_ANSI_COLOR_CODES['reset']}"


COMMANDS = [
    'list', 'limit', 'limit-percent', 'watch', 'top', 'help', 'exit', 'quit'
]

# Nicely formatted help text for the interactive shell.
HELP_TEXT = """\
Available commands:
  list [ipc ...]               List IPC segments and usage
  limit <ipc> <size>           Set absolute limit (e.g. 512M, 2G)
  limit-percent <ipc> <pct>    Set limit as percentage of total GPU RAM
  watch [-n sec] [ipc ...]     Continuously display usage table
  top [ipc ...] [--refresh r]  Launch curses kvtop UI (q to quit)
  !<shell cmd>                 Run command in system shell
  help                         Show this help message
  exit | quit                  Exit the shell
"""


def _setup_readline():
    """Configure readline for history and tab-completion if available."""
    if not READLINE_AVAILABLE:
        return

    hist_file = os.path.expanduser('~/.kvctl_history')
    try:
        readline.read_history_file(hist_file)
    except FileNotFoundError:
        pass

    def _save_history():
        try:
            readline.write_history_file(hist_file)
        except Exception:
            pass

    atexit.register(_save_history)

    def _complete(text: str, state: int):  # noqa: D401 – simple fn
        buffer = readline.get_line_buffer()
        begidx = readline.get_begidx()
        # Split safely until the completion point
        try:
            tokens = shlex.split(buffer[:begidx])
        except ValueError:
            tokens = buffer.split()

        if len(tokens) == 0:  # completing first word
            options = [cmd for cmd in COMMANDS if cmd.startswith(text)]
        elif len(tokens) == 1:  # still completing command name
            options = [cmd for cmd in COMMANDS if cmd.startswith(text)]
        else:
            cmd = tokens[0]
            if cmd in ('limit', 'limit-percent', 'list', 'watch'):
                options = [
                    n for n in _detect_kvcache_ipc_names()
                    if n.startswith(text)
                ]
            else:
                options = []
        if state < len(options):
            return options[state]
        return None

    readline.set_completer(_complete)
    readline.parse_and_bind('tab: complete')


SIZE_SUFFIXES = {
    'b': 1,
    'k': 1024,
    'kb': 1024,
    'm': 1024**2,
    'mb': 1024**2,
    'g': 1024**3,
    'gb': 1024**3,
}


def _parse_size(size_str: str) -> int:
    """Convert human-friendly size strings like '512M', '1g', '100_000' to bytes."""
    s = size_str.strip().lower().replace(',', '').replace('_', '')
    for suf, mul in SIZE_SUFFIXES.items():
        if s.endswith(suf):
            num = float(s[:-len(suf)])
            return int(num * mul)
    # No suffix – assume raw bytes
    return int(float(s))


def _format_size(num_bytes: int) -> str:
    # Reuse simple formatter (non-curses version of kvtop helper)
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if num_bytes < 1024 or unit == 'TB':
            return f"{num_bytes:.1f} {unit}"
        num_bytes /= 1024
    return f"{num_bytes} B"


# ---------------------------------------------------------------------------
# Core command implementations
# ---------------------------------------------------------------------------


def cmd_list(ipcs: Optional[List[str]] = None, json_out: bool = False):
    names = ipcs or _detect_kvcache_ipc_names()
    res: List[Dict[str, int]] = []
    for name in names:
        info = get_kv_cache_limit(name)
        if info is None:
            continue
        res.append({
            'ipc': name,
            'limit_bytes': info.total_size,
            'used_bytes': info.used_size,
        })

    if json_out:
        print(json.dumps(res, indent=2))
    else:
        if not res:
            print("No active KVCached segments found.")
            return
        print(
            _clr(f"{'IPC':24} {'Limit':>12} {'Used':>12} {'%':>6}",
                 'cyan',
                 bold=True))
        for entry in res:
            lim = entry['limit_bytes']
            used = entry['used_bytes']
            pct = used / lim * 100 if lim else 0
            # Choose colour based on utilisation
            if pct < 50:
                clr = 'green'
            elif pct < 80:
                clr = 'yellow'
            else:
                clr = 'red'

            line = f"{entry['ipc']:<24} {_format_size(lim):>12} {_format_size(used):>12} {pct:5.1f} %"
            print(_clr(line, clr))


def cmd_limit(ipc: str, size_str: str):
    size_bytes = _parse_size(size_str)
    update_kv_cache_limit(ipc, size_bytes)


def cmd_limit_percent(ipc: str, percent: float):
    from kvcached.controller.cli import get_total_gpu_memory

    total_mem = get_total_gpu_memory()
    if total_mem <= 0:
        print("CUDA unavailable; cannot compute size from percentage",
              file=sys.stderr)
        sys.exit(1)
    size_bytes = int(total_mem * percent / 100.0)
    update_kv_cache_limit(ipc, size_bytes)


def cmd_watch(interval: float = 1.0, ipcs: Optional[List[str]] = None):
    try:
        while True:
            subprocess.run(['clear'])
            cmd_list(ipcs, json_out=False)
            time.sleep(interval)
    except KeyboardInterrupt:
        pass


def cmd_top(ipcs: Optional[List[str]] = None, refresh: float = 1.0):
    """Launch the kvtop curses UI (blocks until user quits)."""
    kvtop_ui(ipcs, refresh)


# ---------------------------------------------------------------------------
# Interactive shell
# ---------------------------------------------------------------------------


def interactive_shell():
    _setup_readline()
    print("Entering kvcached shell. Type 'help' for commands, 'exit' to quit.")
    while True:
        try:
            line = input('kvcached> ')
        except KeyboardInterrupt:
            # Ignore Ctrl-C inside shell – just move to new prompt.
            print()  # Newline after ^C
            continue
        except EOFError:
            break
        line = line.strip()
        if not line:
            continue
        if line in ('exit', 'quit'):
            break
        if line == 'help':
            print(HELP_TEXT)
            continue
        # Allow shell commands prefixed with !
        if line.startswith('!'):
            os.system(line[1:])
            continue
        # Parse and dispatch
        try:
            tokens = shlex.split(line)
            cmd = tokens[0]
            if cmd == 'list':
                cmd_list(tokens[1:] if len(tokens) > 1 else None)
            elif cmd == 'limit' and len(tokens) == 3:
                cmd_limit(tokens[1], tokens[2])
            elif cmd == 'limit-percent' and len(tokens) == 3:
                cmd_limit_percent(tokens[1], float(tokens[2]))
            elif cmd == 'watch':
                interval = float(tokens[1]) if len(tokens) > 1 else 1.0
                cmd_watch(interval)
            elif cmd == 'top':
                # Syntax: top [refresh] [ipc...]
                # If first arg numeric → refresh, else ipc name.
                refresh = 1.0
                ipcs = []
                if len(tokens) >= 2:
                    try:
                        refresh = float(tokens[1])
                        ipcs = tokens[2:]
                    except ValueError:
                        ipcs = tokens[1:]
                cmd_top(ipcs if ipcs else None, refresh)
            else:
                # Fallback to system shell
                os.system(line)
        except Exception as e:
            print(f"Error: {e}")

        if READLINE_AVAILABLE and line:
            try:
                readline.add_history(line)
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Entry point when used as a CLI script
# ---------------------------------------------------------------------------


def _main():
    parser = argparse.ArgumentParser(description="KVCached control utility")
    sub = parser.add_subparsers(dest='command')

    # list
    p_list = sub.add_parser('list', help='List IPC segments and usage')
    p_list.add_argument('ipc', nargs='*', help='Specific IPC names (optional)')
    p_list.add_argument('--json', action='store_true', help='Output JSON')

    # limit
    p_limit = sub.add_parser('limit', help='Set absolute limit')
    p_limit.add_argument('ipc')
    p_limit.add_argument('size', help="Size, e.g. 512M, 2G")

    # limit-percent
    p_lp = sub.add_parser('limit-percent', help='Set limit as % of total GPU')
    p_lp.add_argument('ipc')
    p_lp.add_argument('percent', type=float)

    # watch
    p_watch = sub.add_parser('watch', help='Continuously list')
    p_watch.add_argument('-n', '--interval', type=float, default=1.0)
    p_watch.add_argument('ipc', nargs='*')

    # top
    p_top = sub.add_parser('top', help='Launch curses kvtop UI')
    p_top.add_argument('-r',
                       '--refresh',
                       type=float,
                       default=1.0,
                       help='Refresh interval')
    p_top.add_argument('ipc', nargs='*', help='IPC names (optional)')

    # shell
    sub.add_parser('shell', help='Start interactive shell')

    args = parser.parse_args()

    if args.command == 'list':
        cmd_list(args.ipc if args.ipc else None, json_out=args.json)
    elif args.command == 'limit':
        cmd_limit(args.ipc, args.size)
    elif args.command == 'limit-percent':
        cmd_limit_percent(args.ipc, args.percent)
    elif args.command == 'watch':
        cmd_watch(args.interval, args.ipc if args.ipc else None)
    elif args.command == 'top':
        cmd_top(args.ipc if args.ipc else None, args.refresh)
    elif args.command == 'shell' or args.command is None:
        interactive_shell()
    else:
        parser.print_help()


if __name__ == '__main__':
    _main()
