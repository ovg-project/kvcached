# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

"""Real Linux IPC, processes and locks; no engine, PyTorch or CUDA imports."""

import array
import errno
import fcntl
import os
import select
import shutil
import signal
import socket
import struct
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import pytest

import kvcached_reaper as ipc

pytestmark = pytest.mark.skipif(sys.platform != "linux", reason="Linux OFD locks required")
ROOT = Path(__file__).resolve().parents[1]


def event(process):
    assert select.select([process.stdout], [], [], 5)[0], "Process did not reach its barrier"
    line = process.stdout.readline().decode().strip()
    assert line, process.stderr.read().decode()
    return line


def command(process, value):
    process.stdin.write((value + "\n").encode())
    process.stdin.flush()


def stop(process):
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)
    for stream in (process.stdin, process.stdout, process.stderr):
        if stream is not None:
            stream.close()


def wait_removed(path):
    deadline = time.monotonic() + 5
    while path.exists() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert not path.exists()


@pytest.fixture
def namespace(monkeypatch):
    with tempfile.TemporaryDirectory(prefix="kvcached-reaper-test-", dir="/dev/shm") as name:
        root = Path(name)
        segment = root.with_name(root.name + "-segment")
        monkeypatch.setenv("KVCACHED_IPC_CLEANUP", "reaper")
        monkeypatch.setenv("KVCACHED_REAPER_DIR", str(root))
        ipc.prepare(str(root))
        yield root, segment
        segment.unlink(missing_ok=True)


@pytest.fixture
def daemon(namespace):
    root, _ = namespace
    process = subprocess.Popen([
        sys.executable, "-u", str(ROOT / "kvcached_reaper.py"),
        "--directory", str(root), "--interval", "0.01",
    ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True)
    try:
        assert event(process).startswith("kvcached reaper ready:")
        yield process
    finally:
        stop(process)


@pytest.fixture(scope="module")
def native(tmp_path_factory):
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("C++ compiler required for the native registration client")
    assert compiler is not None
    output = tmp_path_factory.mktemp("ipc-native") / "lease"
    subprocess.run([compiler, "-std=c++17", "-Wall", "-Wextra", "-Werror",
                    "-I" + str(ROOT / "csrc/inc"), str(ROOT / "tests/native/ipc_lease.cpp"),
                    "-o", str(output)], check=True, timeout=60)
    return output


def worker(namespace, native):
    root, path = namespace
    env = dict(os.environ, KVCACHED_REAPER_DIR=str(root))
    process = subprocess.Popen([str(native), str(path)], env=env,
                               stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, start_new_session=True)
    assert event(process) == f"ready 1 {ipc.slot(str(path))}"
    return process


@pytest.mark.parametrize("method", ["exit", "crash", "sigkill", "group-sigkill"])
def test_native_last_owner_cleanup(namespace, daemon, native, method):
    _, path = namespace
    first = worker(namespace, native)
    second = worker(namespace, native)
    retained = os.open(path, os.O_RDWR)
    try:
        with pytest.raises(BlockingIOError):
            ipc.usage_lock(retained, fcntl.F_WRLCK)
        for process in (first, second):
            if method == "sigkill":
                process.kill()
            elif method == "group-sigkill":
                os.killpg(process.pid, signal.SIGKILL)
            else:
                command(process, method)
            process.wait(timeout=5)
            if process is first:
                assert path.exists()
                with pytest.raises(BlockingIOError):
                    ipc.usage_lock(retained, fcntl.F_WRLCK)
        assert daemon.poll() is None
        wait_removed(path)
    finally:
        os.close(retained)
        stop(first)
        stop(second)


def state_with_owner(namespace):
    root, path = namespace
    owner = ipc.Lease(str(path), register_owner=False)
    state = ipc.Reaper(str(root))
    assert owner.fd is not None
    state.add(str(path), owner.fd)
    return state, owner


def test_two_local_owners_and_data_flock_are_independent(namespace):
    state, first = state_with_owner(namespace)
    second = ipc.Lease(first.path, register_owner=False)
    try:
        for value in range(1000):
            with open(first.path, "r+b") as data:
                fcntl.flock(data, fcntl.LOCK_EX)
                data.write(struct.pack("=qqq", 4096, value, value * 2))
        state.tick()
        assert Path(first.path).stat().st_size == 24
        assert struct.unpack("=qqq", Path(first.path).read_bytes()) == (4096, 999, 1998)
        first.close()
        state.tick()
        assert Path(first.path).exists()
        second.close()
        state.tick()
        assert not Path(first.path).exists()
    finally:
        first.close()
        second.close()
        state.close()


def test_data_flock_cannot_be_reused_as_a_lifetime_lock(namespace):
    _, path = namespace
    owner = ipc.Lease(str(path), register_owner=False)
    try:
        with open(path, "r+b") as reader, open(path, "r+b") as writer:
            fcntl.flock(reader, fcntl.LOCK_SH)
            with pytest.raises(BlockingIOError):
                fcntl.flock(writer, fcntl.LOCK_EX | fcntl.LOCK_NB)
    finally:
        owner.close()


def test_receiver_normalizes_a_mistaken_usage_fd(namespace, daemon):
    root, path = namespace
    owner = ipc.Lease(str(path), register_owner=False)
    assert owner.fd is not None
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET) as connection:
            connection.settimeout(5)
            connection.connect(str(root / "reaper.sock"))
            connection.sendmsg([ipc._VERSION.pack(1) + os.fsencode(path)],
                               [(socket.SOL_SOCKET, socket.SCM_RIGHTS,
                                 array.array("i", [owner.fd]))])
            assert ipc._ACK.unpack(connection.recv(17)) == (1, *ipc.identity(owner.fd))
        probe = ipc.reopen(owner.fd)
        try:
            with pytest.raises(BlockingIOError):
                ipc.usage_lock(probe, fcntl.F_WRLCK)
        finally:
            os.close(probe)
        owner.close()
        wait_removed(path)
    finally:
        owner.close()


def test_newcomer_open_before_lease_is_protected_by_name_lock(namespace):
    root, path = namespace
    state, old = state_with_owner(namespace)
    old.close()
    program = (
        "import os,sys; sys.path.insert(0,sys.argv[1]); import kvcached_reaper as i; "
        "\nwith i.name_lock(sys.argv[2],sys.argv[3]):\n"
        " fd=os.open(sys.argv[3],os.O_RDWR); print('opened',flush=True); "
        "sys.stdin.readline(); i.usage_lock(fd,i.fcntl.F_RDLCK)\n"
        "print('leased',flush=True); sys.stdin.readline(); os.close(fd)"
    )
    newcomer = subprocess.Popen([sys.executable, "-S", "-u", "-c", program,
                                str(ROOT), str(root), str(path)],
                               stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE)
    try:
        assert event(newcomer) == "opened"
        state.tick()
        assert path.exists()
        command(newcomer, "continue")
        assert event(newcomer) == "leased"
        state.tick()
        assert path.exists()
        command(newcomer, "exit")
        assert newcomer.wait(timeout=5) == 0
        state.tick()
        assert not path.exists()
    finally:
        stop(newcomer)
        state.close()


def test_cleanup_first_recreates_a_new_inode(namespace):
    state, old = state_with_owner(namespace)
    original = ipc.identity(old.fd)
    pin = ipc.reopen(old.fd)
    try:
        old.close()
        state.tick()
        newcomer = ipc.Lease(old.path, register_owner=False)
        try:
            assert newcomer.fd is not None
            assert ipc.identity(newcomer.fd) != original
            state.tick()
            assert Path(old.path).exists()
        finally:
            newcomer.close()
    finally:
        os.close(pin)
        state.close()


@pytest.mark.parametrize("timing", ["before-registration", "after-registration"])
def test_same_name_replacement_is_never_adopted(namespace, timing):
    root, path = namespace
    owner = ipc.Lease(str(path), register_owner=False)
    assert owner.fd is not None
    proof = ipc.reopen(owner.fd)
    state = ipc.Reaper(str(root))
    try:
        if timing == "after-registration":
            state.add(str(path), proof)
        owner.close()
        path.unlink()
        path.write_bytes(b"new-instance".ljust(24, b"!"))
        if timing == "before-registration":
            state.add(str(path), proof)
        state.tick()
        assert path.read_bytes() == b"new-instance".ljust(24, b"!")
        assert not state.records
    finally:
        owner.close()
        os.close(proof)
        state.close()


@pytest.mark.parametrize("fault", ["stat", "fstat", "unlink"])
@pytest.mark.parametrize("replace", [False, True])
def test_failure_retries_only_original_identity(namespace, monkeypatch, fault, replace):
    _, path = namespace
    state, owner = state_with_owner(namespace)
    key = ipc.identity(owner.fd)
    retained = state.records[key][1]
    owner.close()
    original = getattr(ipc.os, fault)
    hits: list[bool] = []

    def fail_once(target, *args, **kwargs):
        if not hits and ((fault == "fstat" and target == retained) or
                         (fault != "fstat" and os.fspath(target) == str(path))):
            hits.append(True)
            raise PermissionError(errno.EACCES, "injected failure")
        return original(target, *args, **kwargs)

    try:
        with monkeypatch.context() as patch:
            patch.setattr(ipc.os, fault, fail_once)
            state.tick()
        assert hits and path.exists() and key in state.records
        if replace:
            path.unlink()
            path.write_bytes(b"replacement".ljust(24, b"!"))
            state.tick()
            assert path.read_bytes() == b"replacement".ljust(24, b"!")
        else:
            newcomer = ipc.Lease(str(path), register_owner=False)
            state.tick()
            assert path.exists()
            newcomer.close()
            state.tick()
            assert not path.exists()
        assert not state.records
    finally:
        state.close()


@pytest.mark.parametrize("failed_query", [1, 2])
def test_reopen_query_failure_closes_new_fd(namespace, monkeypatch, failed_query):
    _, path = namespace
    owner = ipc.Lease(str(path), register_owner=False)
    assert owner.fd is not None
    before = len(list(Path("/proc/self/fd").iterdir()))
    original = ipc.identity
    queries = []

    def fail(fd):
        queries.append(fd)
        if len(queries) == failed_query:
            raise OSError(errno.EIO, "injected identity failure")
        return original(fd)

    try:
        with monkeypatch.context() as patch:
            patch.setattr(ipc, "identity", fail)
            with pytest.raises(OSError):
                ipc.reopen(owner.fd)
        assert len(list(Path("/proc/self/fd").iterdir())) == before
    finally:
        owner.close()


def test_registration_failure_keeps_lease_and_can_retry(namespace, daemon, monkeypatch):
    _, path = namespace
    owner = ipc.Lease(str(path), register_owner=False)
    assert owner.fd is not None

    def fail(fd):
        raise OSError(errno.EMFILE, "injected registration FD failure")

    try:
        with monkeypatch.context() as patch:
            patch.setattr(ipc, "reopen", fail)
            assert not owner.register()
        probe = ipc.reopen(owner.fd)
        try:
            with pytest.raises(BlockingIOError):
                ipc.usage_lock(probe, fcntl.F_WRLCK)
        finally:
            os.close(probe)
        assert owner.register()
        owner.close()
        wait_removed(path)
    finally:
        owner.close()


def test_failed_usage_lock_aborts_without_fd_leak(namespace, monkeypatch):
    _, path = namespace
    before = len(list(Path("/proc/self/fd").iterdir()))

    def fail(fd, kind):
        raise OSError(errno.EIO, "injected usage lock failure")

    monkeypatch.setattr(ipc, "usage_lock", fail)
    with pytest.raises(OSError):
        ipc.Lease(str(path))
    assert len(list(Path("/proc/self/fd").iterdir())) == before


def test_missing_daemon_does_not_remove_unregistered_files(namespace):
    root, path = namespace
    owner = ipc.Lease(str(path))
    try:
        assert not owner.registered
        owner.close()
        state = ipc.Reaper(str(root))
        state.tick()
        assert path.exists() and not state.records
    finally:
        owner.close()


def test_native_missing_daemon_keeps_usage_lease(namespace, native):
    root, path = namespace
    process = subprocess.Popen([str(native), str(path)],
                               stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE)
    probe = None
    try:
        assert event(process) == f"ready 0 {ipc.slot(str(path))}"
        probe = os.open(path, os.O_RDWR)
        with pytest.raises(BlockingIOError):
            ipc.usage_lock(probe, fcntl.F_WRLCK)
        command(process, "exit")
        process.wait(timeout=5)
        assert path.exists()
        state = ipc.Reaper(str(root))
        state.tick()
        assert not state.records
    finally:
        if probe is not None:
            os.close(probe)
        stop(process)


def test_fork_child_keeps_lease_after_parent_close(namespace):
    state, owner = state_with_owner(namespace)
    reader, writer = os.pipe()
    child = os.fork()
    if child == 0:
        os.close(writer)
        os.read(reader, 1)
        os._exit(0)
    os.close(reader)
    waited = False
    writer_closed = False
    try:
        owner.close()
        state.tick()
        assert Path(owner.path).exists()
        os.close(writer)
        writer_closed = True
        assert os.waitpid(child, 0)[1] == 0
        waited = True
        state.tick()
        assert not Path(owner.path).exists()
    finally:
        if not writer_closed:
            os.close(writer)
        if not waited:
            os.waitpid(child, 0)
        owner.close()
        state.close()


def test_cloexec_does_not_pass_lease_to_unrelated_program(namespace):
    state, owner = state_with_owner(namespace)
    program = (
        "import os,sys; target=tuple(map(int,sys.argv[1:])); found=[]; "
        "\nfor entry in os.listdir('/proc/self/fd'):\n"
        " try:\n"
        "  s=os.fstat(int(entry))\n"
        "  if (s.st_dev,s.st_ino)==target: found.append(entry)\n"
        " except OSError: pass\n"
        "assert not found"
    )
    try:
        assert not os.get_inheritable(owner.fd)
        subprocess.run([sys.executable, "-S", "-c", program,
                        *map(str, ipc.identity(owner.fd))], close_fds=False,
                       check=True, timeout=5)
        owner.close()
        state.tick()
        assert not Path(owner.path).exists()
    finally:
        owner.close()
        state.close()


@pytest.mark.parametrize("bad", ["version", "two-fds", "too-many-fds", "oversize"])
def test_invalid_registration_closes_all_received_fds(namespace, bad):
    state, owner = state_with_owner(namespace)
    sender, receiver = socket.socketpair(socket.AF_UNIX, socket.SOCK_SEQPACKET)
    before = len(list(Path("/proc/self/fd").iterdir()))
    try:
        for _ in range(20):
            payload = ipc._VERSION.pack(99 if bad == "version" else 1) + os.fsencode(owner.path)
            fds = [owner.fd] * (2 if bad == "two-fds" else 9 if bad == "too-many-fds" else 1)
            if bad == "oversize":
                payload += b"x" * 8192
            sender.sendmsg([payload], [(socket.SOL_SOCKET, socket.SCM_RIGHTS,
                                        array.array("i", fds))])
            state.receive(receiver)
            assert ipc._ACK.unpack(sender.recv(17))[0] == 0
        assert len(list(Path("/proc/self/fd").iterdir())) == before
        assert len(state.records) == 1
    finally:
        sender.close()
        receiver.close()
        owner.close()
        state.close()


@pytest.mark.parametrize("reregister", [False, True])
def test_reaper_restart_preserves_unregistered_identity(namespace, daemon, reregister):
    root, path = namespace
    owner = ipc.Lease(str(path))
    assert owner.registered
    daemon.kill()
    daemon.wait(timeout=5)
    if not reregister:
        owner.close()
    replacement = subprocess.Popen([
        sys.executable, "-S", "-u", str(ROOT / "kvcached_reaper.py"),
        "--directory", str(root), "--interval", "0.01",
    ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    probe_path = path.with_name(path.name + "-probe")
    try:
        assert event(replacement).startswith("kvcached reaper ready:")
        probe = ipc.Lease(str(probe_path))
        assert probe.registered
        probe.close()
        wait_removed(probe_path)  # The new daemon has completed a cleanup tick.
        assert path.exists()
        if reregister:
            assert owner.register()
            owner.close()
            wait_removed(path)
    finally:
        owner.close()
        stop(replacement)
        probe_path.unlink(missing_ok=True)


def test_explicit_deletion_respects_live_owner(namespace):
    _, path = namespace
    owner = ipc.Lease(str(path), register_owner=False)
    try:
        with pytest.raises(BlockingIOError):
            ipc.remove(str(path))
        assert path.exists()
        owner.close()
        assert ipc.remove(str(path))
        assert not ipc.remove(str(path))
    finally:
        owner.close()


def test_repeated_registration_and_cleanup_does_not_leak_fds(namespace):
    root, path = namespace
    state = ipc.Reaper(str(root))
    before = len(list(Path("/proc/self/fd").iterdir()))
    try:
        for _ in range(40):
            owner = ipc.Lease(str(path), register_owner=False)
            assert owner.fd is not None
            for _ in range(3):
                state.add(str(path), owner.fd)
            assert len(state.records) == 1
            owner.close()
            state.tick()
            assert not path.exists() and not state.records
        assert len(list(Path("/proc/self/fd").iterdir())) == before
    finally:
        state.close()


def test_name_locks_have_a_bounded_number_of_slots(namespace):
    root, _ = namespace
    for index in range(200):
        with ipc.name_lock(str(root), f"/dev/shm/model-{index}"):
            pass
    assert 1 <= len(list(root.glob("gate-*"))) <= 64


def test_second_daemon_does_not_replace_live_endpoint(namespace, daemon):
    root, _ = namespace
    original = (root / "reaper.sock").stat()
    second = subprocess.Popen([sys.executable, "-S", str(ROOT / "kvcached_reaper.py"),
                               "--directory", str(root)], stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE)
    try:
        assert second.wait(timeout=5) != 0
        assert os.path.samestat(original, (root / "reaper.sock").stat())
    finally:
        stop(second)


def test_daemon_does_not_load_torch_or_cuda(namespace, daemon):
    maps = Path(f"/proc/{daemon.pid}/maps").read_text()
    assert not any(name in maps for name in ("libtorch", "libcuda", "libcudart"))


@pytest.mark.parametrize("invalid", ["/tmp/control", "/dev/shm/../control", "bad/name",
                                     "/dev/shm//control"])
def test_rejects_out_of_namespace_paths(namespace, invalid):
    with pytest.raises(ValueError):
        ipc.Lease(invalid)


def test_rejects_symlinks_and_wrong_layout(namespace):
    root, path = namespace
    target = root / "target"
    target.write_bytes(b"protected")
    path.symlink_to(target)
    with pytest.raises(OSError):
        ipc.Lease(str(path))
    assert target.read_bytes() == b"protected"
    path.unlink()
    path.write_bytes(b"not a control file")
    with pytest.raises(ValueError):
        ipc.Lease(str(path))
    assert path.read_bytes() == b"not a control file"


def test_native_and_python_reject_symlinked_directory(namespace, native, monkeypatch):
    root, path = namespace
    alias = root / "alias"
    target = root / "private"
    target.mkdir(mode=0o700)
    alias.symlink_to(target, target_is_directory=True)
    monkeypatch.setenv("KVCACHED_REAPER_DIR", str(alias) + "/")
    with pytest.raises(PermissionError):
        ipc.Lease(str(path))
    result = subprocess.run([str(native), str(path)], capture_output=True, timeout=5)
    assert result.returncode != 0
    assert b"private and owned" in result.stderr
    assert not path.exists()


def test_empty_directory_option_uses_default(monkeypatch):
    monkeypatch.setenv("KVCACHED_REAPER_DIR", "")
    assert ipc.directory() == f"/dev/shm/.kvcached-lifecycle-{os.geteuid()}"
