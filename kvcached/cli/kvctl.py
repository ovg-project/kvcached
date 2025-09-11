# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

import argparse
import atexit
import json
import os
import shlex
import subprocess
import sys
import time
from typing import List, Optional, TypedDict

from kvcached.cli.kvtop import _detect_kvcache_ipc_names, kvtop as kvtop_ui
from kvcached.cli.utils import _format_size, get_kv_cache_limit, update_kv_cache_limit

try:
    import readline  # type: ignore
    READLINE_AVAILABLE = True
except ImportError:  # pragma: no cover – win / minimal envs
    READLINE_AVAILABLE = False
    print("readline not available. Auto-completion will not work.",
          file=sys.stderr)

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


def _clr(text: str, color: Optional[str] = None, *, bold: bool = False) -> str:
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
    'list', 'limit', 'limit-percent', 'watch', 'kvtop', 'delete', 'help',
    'exit', 'quit'
]

# Nicely formatted help text for the interactive shell.
HELP_TEXT = """\
Available commands:
  list [ipc ...]               List IPC segments and usage
  limit <ipc> <size>           Set absolute limit (e.g. 512M, 2G)
  limit-percent <ipc> <pct>    Set limit as percentage of total GPU RAM
  watch [-n sec] [ipc ...]     Continuously display usage table
  kvtop [ipc ...] [--refresh r]  Launch curses kvtop UI (q to quit)
  !<shell cmd>                 Run command in system shell
  help                         Show this help message
  delete <ipc>                 Delete IPC segment and its limit entry
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
        elif len(tokens) == 1:
            # We have typed exactly one complete token (the command) and are
            # now completing the *first* argument.  For the set of commands
            # that accept an IPC name as the next token, offer those names;
            # otherwise continue completing the command itself.

            cmd = tokens[0]

            if cmd in ('limit', 'limit-percent', 'list', 'watch', 'delete'):
                ipc_names = _detect_kvcache_ipc_names()
                # Case-insensitive matching so "VLLM" also matches "vllm".
                options = [
                    n for n in ipc_names if n.lower().startswith(text.lower())
                ]
            else:
                options = [c for c in COMMANDS if c.startswith(text)]
        else:
            cmd = tokens[0]
            if cmd in ('limit', 'limit-percent', 'list', 'watch', 'delete'):
                options = [
                    n for n in _detect_kvcache_ipc_names()
                    if n.lower().startswith(text.lower())
                ]
            else:
                options = []
        if state < len(options):
            return options[state]
        return None

    readline.set_completer(_complete)
    # Use appropriate key binding depending on whether Python's readline
    # is linked against GNU readline or the libedit compatibility layer
    # (common on macOS and some minimal Linux builds).  The latter uses a
    # different command syntax.
    if getattr(readline, "__doc__", "").startswith("Importing this module enables command line editing using libedit") or \
            (readline.__doc__ and "libedit" in readline.__doc__):
        # libedit style
        readline.parse_and_bind('bind ^I rl_complete')
    else:
        # GNU readline style
        readline.parse_and_bind('tab: complete')

    # Ensure hyphens are treated as part of a word so commands like
    # "limit-percent" can be completed after typing "limit-per".
    try:
        delims = readline.get_completer_delims()
        if '-' in delims:
            readline.set_completer_delims(delims.replace('-', ''))
    except AttributeError:
        # Some readline/libedit shims may not expose these functions
        pass


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
    """
    Convert human-friendly size strings such as ``512M``, ``1g`` or
    ``100_000`` into a byte count.

    Because some suffixes overlap (e.g. ``b`` vs ``mb``), we sort the suffix
    table by *descending length* so that the longest suffix wins.  Invalid
    strings raise ``ValueError`` instead of crashing later.
    """
    s = size_str.strip().lower().replace(',', '').replace('_', '')

    # Try to match the longest suffix first ("mb" before "b", etc.)
    for suf, mul in sorted(SIZE_SUFFIXES.items(), key=lambda kv: -len(kv[0])):
        if s.endswith(suf):
            num_part = s[:-len(suf)] or "0"
            try:
                num = float(num_part)
            except ValueError as exc:
                raise ValueError(f"Invalid size string '{size_str}'") from exc
            return int(num * mul)

    # No recognised suffix – assume the string is raw bytes
    try:
        return int(float(s))
    except ValueError as exc:
        raise ValueError(f"Invalid size string '{size_str}'") from exc


# ---------------------------------------------------------------------------
# Core command implementations
# ---------------------------------------------------------------------------


class _IpcStats(TypedDict):
    ipc: str
    limit_bytes: int
    used_bytes: int


def cmd_list(ipcs: Optional[List[str]] = None, json_out: bool = False):
    names = ipcs or _detect_kvcache_ipc_names()
    res: List[_IpcStats] = []
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
    """Set an absolute limit for an existing IPC segment.

    We first validate that the supplied ``ipc`` name corresponds to a running
    segment; otherwise we refuse the operation to avoid accidentally creating
    a new (wrong-case) shared-memory file.
    """
    if get_kv_cache_limit(ipc) is None:
        print(_clr(f"Error: IPC '{ipc}' not found.", 'red', bold=True),
              file=sys.stderr)
        avail = _detect_kvcache_ipc_names()
        if avail:
            print("Active IPC names:", ", ".join(avail), file=sys.stderr)
        return

    size_bytes = _parse_size(size_str)
    update_kv_cache_limit(ipc, size_bytes)


def cmd_limit_percent(ipc: str, percent: float):
    """Set limit as percentage of total GPU RAM for an existing IPC."""
    if get_kv_cache_limit(ipc) is None:
        print(_clr(f"Error: IPC '{ipc}' not found.", 'red', bold=True),
              file=sys.stderr)
        return

    from kvcached.cli.utils import get_total_gpu_memory

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
# Delete IPC command
# ---------------------------------------------------------------------------


def cmd_delete(ipc: str):
    from kvcached.cli.utils import delete_kv_cache_segment

    if delete_kv_cache_segment(ipc):
        print(_clr(f"Deleted IPC '{ipc}'.", 'green'))
    else:
        print(_clr(f"IPC '{ipc}' not found.", 'red', bold=True),
              file=sys.stderr)


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
                # Syntax: watch [-n SEC] [ipc ...]  (matches CLI behaviour)
                interval: float = 1.0
                ipcs_watch: List[str] = []
                i = 1
                while i < len(tokens):
                    tok = tokens[i]
                    if tok in ('-n', '--interval'):
                        i += 1
                        if i >= len(tokens):
                            raise ValueError(
                                "Expected number after '-n/--interval'")
                        interval = float(tokens[i])
                    else:
                        # If token is a bare number and interval wasn't set via flag,
                        # treat it as the legacy positional interval argument.
                        if not ipcs_watch and tok.replace('.', '',
                                                          1).isdigit():
                            interval = float(tok)
                        else:
                            ipcs_watch.append(tok)
                    i += 1
                cmd_watch(interval, ipcs_watch if ipcs_watch else None)
            elif cmd == 'kvtop':
                # Syntax: kvtop [-r/--refresh SEC] [ipc ...]
                refresh: float = 1.0
                ipcs_top: List[str] = []
                i = 1
                while i < len(tokens):
                    tok = tokens[i]
                    if tok in ('-r', '--refresh'):
                        i += 1
                        if i >= len(tokens):
                            raise ValueError(
                                "Expected float after '-r/--refresh'")
                        refresh = float(tokens[i])
                    else:
                        ipcs_top.append(tok)
                    i += 1
                cmd_top(ipcs_top if ipcs_top else None, refresh)
            elif cmd == 'delete' and len(tokens) == 2:
                cmd_delete(tokens[1])
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


def main():
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

    # kvtop
    p_kvtop = sub.add_parser('kvtop', help='Launch curses kvtop UI')
    p_kvtop.add_argument('-r',
                         '--refresh',
                         type=float,
                         default=1.0,
                         help='Refresh interval')
    p_kvtop.add_argument('ipc', nargs='*', help='IPC names (optional)')

    # delete
    p_del = sub.add_parser('delete', help='Delete IPC segment')
    p_del.add_argument('ipc')

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
    elif args.command == 'kvtop':
        cmd_top(args.ipc if args.ipc else None, args.refresh)
    elif args.command == 'delete':
        cmd_delete(args.ipc)
    elif args.command == 'shell' or args.command is None:
        interactive_shell()
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
