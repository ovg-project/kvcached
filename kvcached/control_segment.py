# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0
"""Lifecycle-only control-file protocol shared by Python trackers and tools.

All users/writers/deleters of ONE segment must stop and upgrade together. The
marker identifies the protocol; it cannot exclude old code ignoring its locks.
The 24-byte memory-info payload is unchanged. Non-Linux files remain unmarked
and visible, and safe deletion is unavailable. No inference-path work is added.
"""

import ctypes
import errno
import fcntl
import os
import stat
import sys
from typing import Optional

MARKER_OFFSET = 24
MARKER = b"KVCOFD01"
SEGMENT_SIZE = MARKER_OFFSET + len(MARKER)


class _Flock(ctypes.Structure):
    # Native Linux LP64 layout, also used by Linux aarch64. We explicitly refuse
    # other ABIs rather than passing an incorrectly packed struct to the kernel.
    _fields_ = [
        ("type", ctypes.c_short),
        ("whence", ctypes.c_short),
        ("start", ctypes.c_long),
        ("length", ctypes.c_long),
        ("pid", ctypes.c_int),
    ]


def supported() -> bool:
    return (
        sys.platform.startswith("linux")
        and ctypes.sizeof(ctypes.c_long) == 8
        and hasattr(fcntl, "F_OFD_SETLK")
    )


def lock(fd: int, exclusive: bool = False) -> None:
    if not supported():
        raise OSError(errno.ENOTSUP, "Linux OFD control-segment locks are unavailable")
    request = _Flock(fcntl.F_WRLCK if exclusive else fcntl.F_RDLCK, os.SEEK_SET, 0, 1, 0)
    while True:
        try:
            fcntl.fcntl(fd, fcntl.F_OFD_SETLK, bytes(request))
            return
        except InterruptedError:
            continue


def is_current(fd: int, path: str) -> bool:
    try:
        current, opened = os.lstat(path), os.fstat(fd)
        return (
            stat.S_ISREG(current.st_mode)
            and stat.S_ISREG(opened.st_mode)
            and os.path.samestat(current, opened)
        )
    except FileNotFoundError:
        return False


def publish(fd: int) -> None:
    if sys.platform.startswith("linux"):
        if os.pwrite(fd, MARKER, MARKER_OFFSET) != len(MARKER):
            raise OSError(errno.EIO, "short control-segment marker write")
        os.ftruncate(fd, SEGMENT_SIZE)


def known(fd: int) -> bool:
    return (
        os.fstat(fd).st_size == SEGMENT_SIZE and os.pread(fd, len(MARKER), MARKER_OFFSET) == MARKER
    )


def active(path: str) -> Optional[bool]:
    """True if contended, False if known-unused/missing, None if uncertain.

    Only False may be hidden. Unknown legacy files stay visible. This is a
    point-in-time observation; deletion must independently acquire its lock.
    """
    if not supported():
        return None
    try:
        fd = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
    except FileNotFoundError:
        return False
    except OSError:
        return None
    try:
        if not known(fd) or not is_current(fd, path):
            return None
        request = _Flock(fcntl.F_WRLCK, os.SEEK_SET, 0, 1, 0)
        result = fcntl.fcntl(fd, fcntl.F_OFD_GETLK, bytes(request))
        return _Flock.from_buffer_copy(result).type != fcntl.F_UNLCK
    except OSError:
        return None
    finally:
        os.close(fd)


def delete(path: str) -> bool:
    """Delete one known idle control inode, never a socket directory.

    The independent open description is essential: dup of a holder's fd could
    upgrade its shared lock despite inherited users. Hold EX through unlink.
    Uncooperative external pathname replacement is outside this protocol.
    """
    try:
        fd = os.open(path, os.O_RDWR | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0))
    except FileNotFoundError:
        return False
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise ValueError("not a regular control segment")
        if not known(fd):
            raise ValueError(
                "legacy or unknown control segment; stop all its users before manual cleanup"
            )
        try:
            lock(fd, exclusive=True)
        except OSError as exc:
            if exc.errno in (errno.EAGAIN, errno.EACCES):
                raise OSError(errno.EBUSY, "control segment is in use", path) from exc
            raise
        if not is_current(fd, path):
            return False
        os.unlink(path)
        return True
    finally:
        os.close(fd)
