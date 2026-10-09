# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

"""Linux control-segment leases and an engine-independent cleanup reaper.

Kept outside the kvcached package so the daemon does not import PyTorch.
The protocol is opt-in; all users of an IPC name must use the same mode.
"""

import array
import contextlib
import ctypes
import fcntl
import logging
import os
import signal
import socket
import stat
import struct
import sys
import time
from typing import Optional

logger = logging.getLogger(__name__)
_SIZE = 24
_VERSION = struct.Struct("!I")
_ACK = struct.Struct("!BQQ")


def enabled() -> bool:
    return os.getenv("KVCACHED_IPC_CLEANUP") == "reaper"


def directory() -> str:
    root = os.getenv("KVCACHED_REAPER_DIR") or f"/dev/shm/.kvcached-lifecycle-{os.geteuid()}"
    if not os.path.isabs(root):
        raise ValueError("Reaper directory must be absolute")
    return os.path.normpath(root)


def control_path(path: str) -> str:
    if not os.path.isabs(path):
        path = os.path.join("/dev/shm", path)
    if (path != os.path.join("/dev/shm", os.path.basename(path)) or
            os.path.basename(path) in ("", ".", "..") or "\0" in path):
        raise ValueError("Reaper mode requires a control file directly in /dev/shm")
    return path


def identity(fd: int) -> tuple[int, int]:
    info = os.fstat(fd)
    return info.st_dev, info.st_ino


def reopen(fd: int) -> int:
    new = os.open(f"/proc/self/fd/{fd}", os.O_RDWR | os.O_CLOEXEC)
    try:
        if identity(new) != identity(fd):
            raise ValueError("Reopened file identity changed")
    except BaseException:
        os.close(new)
        raise
    return new


def prepare(root: str) -> None:
    if sys.platform != "linux" or not hasattr(fcntl, "F_OFD_SETLK"):
        raise RuntimeError("Reaper mode requires Linux OFD locks")
    os.makedirs(root, mode=0o700, exist_ok=True)
    info = os.lstat(root)
    if (not stat.S_ISDIR(info.st_mode) or info.st_uid != os.geteuid()
            or info.st_mode & 0o077):
        raise PermissionError("Reaper directory must be private and owned by the current UID")


def slot(path: str) -> int:
    # Match ipc_cleanup.hpp; the slot files must stay stable across instances.
    value = 14695981039346656037
    for byte in os.fsencode(path):
        value = ((value ^ byte) * 1099511628211) & ((1 << 64) - 1)
    return value % 64


@contextlib.contextmanager
def name_lock(root: str, path: str, blocking: bool = True):
    fd = os.open(os.path.join(root, f"gate-{slot(path)}"),
                 os.O_RDWR | os.O_CREAT | os.O_CLOEXEC | os.O_NOFOLLOW, 0o600)
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or info.st_uid != os.geteuid():
            raise PermissionError("Unverified name lock")
        fcntl.flock(fd, fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB))
        yield
    finally:
        os.close(fd)


class _Flock(ctypes.Structure):
    _fields_ = [("type", ctypes.c_short), ("whence", ctypes.c_short),
                ("start", ctypes.c_longlong), ("length", ctypes.c_longlong),
                ("pid", ctypes.c_int)]


def usage_lock(fd: int, kind: int) -> None:
    request = _Flock(kind, os.SEEK_SET, 0, 1, 0)
    fcntl.fcntl(fd, fcntl.F_OFD_SETLK, bytes(request))


def register(root: str, path: str, fd: int) -> None:
    proof = reopen(fd)
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET) as connection:
            connection.settimeout(1.0)
            connection.connect(os.path.join(root, "reaper.sock"))
            _, uid, _ = struct.unpack("=iii", connection.getsockopt(
                socket.SOL_SOCKET, socket.SO_PEERCRED, 12))
            if uid != os.geteuid():
                raise PermissionError("Reaper UID mismatch")
            connection.sendmsg([_VERSION.pack(1) + os.fsencode(path)],
                               [(socket.SOL_SOCKET, socket.SCM_RIGHTS,
                                 array.array("i", [proof]))])
            reply = connection.recv(_ACK.size)
            if len(reply) != _ACK.size:
                raise OSError("Reaper did not acknowledge registration")
            ok, device, inode = _ACK.unpack(reply)
            if not ok or (device, inode) != identity(fd):
                raise OSError("Reaper rejected the control file")
    finally:
        os.close(proof)


class Lease:
    """Hold a shared OFD lock until all local users have stopped.

    Closing an inherited FD must not explicitly unlock the description: a
    fork child can still be using that same lease.
    """

    def __init__(self, path: str, *, create: bool = True, register_owner: bool = True):
        self.path = control_path(path)
        self.root = directory()
        self.fd: Optional[int] = None
        self.registered = False
        prepare(self.root)
        with name_lock(self.root, self.path):
            fd = os.open(self.path, os.O_RDWR | os.O_CLOEXEC | os.O_NOFOLLOW |
                         (os.O_CREAT if create else 0), 0o666)
            try:
                info = os.fstat(fd)
                if not stat.S_ISREG(info.st_mode) or info.st_uid != os.geteuid():
                    raise PermissionError("Unverified control file")
                if info.st_size == 0 and create:
                    os.ftruncate(fd, _SIZE)
                elif info.st_size != _SIZE:
                    raise ValueError("Unexpected control-file layout")
                usage_lock(fd, fcntl.F_RDLCK)
                self.fd = fd
                if register_owner:
                    self.register()
            except BaseException:
                self.fd = None
                os.close(fd)
                raise

    def register(self) -> bool:
        if self.fd is None:
            raise RuntimeError("Cannot register a closed lease")
        try:
            register(self.root, self.path, self.fd)
        except (OSError, ValueError) as error:
            self.registered = False
            logger.warning("Automatic IPC cleanup unavailable for %s: %s", self.path, error)
        else:
            self.registered = True
        return self.registered

    def close(self) -> None:
        fd, self.fd = self.fd, None
        if fd is not None:
            os.close(fd)


class Reaper:
    def __init__(self, root: str):
        self.root = os.path.abspath(root)
        self.records: dict[tuple[int, int], tuple[str, int]] = {}

    def add(self, path: str, fd: int) -> tuple[int, int]:
        path = control_path(path)
        info = os.fstat(fd)
        if (not stat.S_ISREG(info.st_mode) or info.st_uid != os.geteuid()
                or info.st_size != _SIZE):
            raise ValueError("Unverified control file")
        # SCM_RIGHTS alone shares the sender's open description. Normalize it
        # before ACK, even if a client mistakenly sent its usage-lock FD.
        retained = reopen(fd)
        key = info.st_dev, info.st_ino
        if key in self.records:
            os.close(retained)
        else:
            self.records[key] = path, retained
        return key

    def tick(self) -> None:
        for key, (path, fd) in list(self.records.items()):
            try:
                with name_lock(self.root, path, blocking=False):
                    usage_lock(fd, fcntl.F_WRLCK)
                    done = False
                    try:
                        try:
                            current = os.stat(path, follow_symlinks=False)
                        except FileNotFoundError:
                            done = True
                        else:
                            if (current.st_dev, current.st_ino) == identity(fd):
                                os.unlink(path)
                            done = True
                    finally:
                        # A failed unlink retains the identity, not the lock.
                        # Unlock before letting a new user take the name lock.
                        usage_lock(fd, fcntl.F_UNLCK)
                        if done:
                            os.close(fd)
                            del self.records[key]
            except BlockingIOError:
                pass
            except OSError:
                logger.debug("Keeping control segment %s for a cleanup retry", path,
                             exc_info=True)

    def receive(self, connection: socket.socket) -> None:
        received: list[int] = []
        try:
            _, uid, _ = struct.unpack("=iii", connection.getsockopt(
                socket.SOL_SOCKET, socket.SO_PEERCRED, 12))
            if uid != os.geteuid():
                raise PermissionError("Registration UID mismatch")
            payload, ancillary, flags, _ = connection.recvmsg(
                8192, socket.CMSG_SPACE(8 * array.array("i").itemsize),
                socket.MSG_CMSG_CLOEXEC)
            for level, kind, data in ancillary:
                if level == socket.SOL_SOCKET and kind == socket.SCM_RIGHTS:
                    values = array.array("i")
                    values.frombytes(data)
                    received.extend(values)
            if (flags & (socket.MSG_TRUNC | socket.MSG_CTRUNC) or len(received) != 1
                    or len(payload) <= _VERSION.size
                    or _VERSION.unpack(payload[:_VERSION.size])[0] != 1):
                raise ValueError("Expected protocol v1 and one FD")
            key = self.add(os.fsdecode(payload[_VERSION.size:]), received[0])
            os.close(received.pop())
            connection.sendall(_ACK.pack(1, *key))
        except (OSError, ValueError):
            logger.debug("Rejected control-segment registration", exc_info=True)
            try:
                connection.sendall(_ACK.pack(0, 0, 0))
            except OSError:
                pass
        finally:
            for fd in received:
                os.close(fd)

    def close(self) -> None:
        for _, fd in self.records.values():
            os.close(fd)
        self.records.clear()


def remove(path: str) -> bool:
    """Explicit tool deletion still respects live leases and the name lock."""
    path = control_path(path)
    root = directory()
    prepare(root)
    with name_lock(root, path):
        try:
            fd = os.open(path, os.O_RDWR | os.O_CLOEXEC | os.O_NOFOLLOW)
        except FileNotFoundError:
            return False
        try:
            usage_lock(fd, fcntl.F_WRLCK)
            current = os.stat(path, follow_symlinks=False)
            if (current.st_dev, current.st_ino) != identity(fd):
                return False
            os.unlink(path)
            return True
        finally:
            os.close(fd)


def serve(root: str, interval: float = 0.2) -> None:
    root = os.path.abspath(root)
    prepare(root)
    state = Reaper(root)
    endpoint = os.path.join(root, "reaper.sock")
    guard = os.open(os.path.join(root, "reaper.lock"),
                    os.O_RDWR | os.O_CREAT | os.O_CLOEXEC | os.O_NOFOLLOW, 0o600)
    endpoint_identity = None
    running = True

    def stop(signum, frame):
        nonlocal running
        running = False

    try:
        fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if os.path.lexists(endpoint):
            info = os.lstat(endpoint)
            if not stat.S_ISSOCK(info.st_mode) or info.st_uid != os.geteuid():
                raise PermissionError("Unverified reaper endpoint")
            os.unlink(endpoint)
        with socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET) as server:
            server.bind(endpoint)
            os.chmod(endpoint, 0o600)
            info = os.lstat(endpoint)
            endpoint_identity = info.st_dev, info.st_ino
            server.listen()
            server.settimeout(interval)
            for signum in (signal.SIGINT, signal.SIGTERM):
                signal.signal(signum, stop)
            print(f"kvcached reaper ready: {endpoint}", flush=True)
            next_tick = time.monotonic()
            while running:
                # Continuous registrations must not starve cleanup.
                if time.monotonic() >= next_tick:
                    state.tick()
                    next_tick = time.monotonic() + interval
                try:
                    connection, _ = server.accept()
                except socket.timeout:
                    continue
                with connection:
                    connection.settimeout(1.0)
                    state.receive(connection)
    finally:
        state.close()
        if endpoint_identity is not None:
            try:
                info = os.lstat(endpoint)
                if (info.st_dev, info.st_ino) == endpoint_identity:
                    os.unlink(endpoint)
            except FileNotFoundError:
                pass
        os.close(guard)


def main() -> None:
    # Console scripts normally execute site hooks first. Re-exec without them
    # so the long-lived daemon also stays independent of engine import hooks.
    if not sys.flags.no_site:
        os.execv(sys.executable, [sys.executable, "-S", os.path.abspath(__file__),
                                 *sys.argv[1:]])
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", default=directory())
    parser.add_argument("--interval", type=float, default=0.2)
    args = parser.parse_args()
    if args.interval < 0.01:
        parser.error("--interval must be at least 0.01 seconds")
    logging.basicConfig(level=logging.INFO)
    serve(args.directory, args.interval)


if __name__ == "__main__":
    main()
