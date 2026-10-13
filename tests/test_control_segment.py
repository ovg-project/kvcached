# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0
"""Real control lifetime, upgrade boundary and deletion races (no GPU needed)."""

import errno
import json
import os
import select
import signal
import subprocess
import sys
import time

import pytest

from kvcached import control_segment as protocol, mem_info_tracker as tracker_module
from kvcached.cli import utils
from kvcached.cli.kvctl import cmd_delete, cmd_list
from kvcached.cli.kvtop import _detect_kvcache_ipc_names

pytestmark = pytest.mark.skipif(not protocol.supported(), reason="Linux OFD locks")


@pytest.fixture
def segment(tmp_path, monkeypatch):
    monkeypatch.setattr(utils, "SHM_DIR", str(tmp_path))
    monkeypatch.setattr(tracker_module, "DEFAULT_IPC_NAME", "test-control")
    from kvcached.cli import kvtop

    monkeypatch.setattr(kvtop, "SHM_DIR", str(tmp_path))
    return tmp_path / "test-control"


def tracker():
    return tracker_module.MemInfoTracker(4096)


def test_live_hidden_after_close_and_explicit_delete(segment, capsys):
    owner = tracker()
    owner.update_memory_usage(128, 64)
    assert protocol.active(str(segment)) is True
    assert _detect_kvcache_ipc_names() == [segment.name]
    assert cmd_delete(segment.name) is False
    assert "in use" in capsys.readouterr().err
    owner.close()
    assert segment.exists() and protocol.active(str(segment)) is False
    assert _detect_kvcache_ipc_names() == []
    assert _detect_kvcache_ipc_names(include_inactive=True) == [segment.name]
    cmd_list([segment.name], json_out=True)
    assert json.loads(capsys.readouterr().out) == []
    cmd_list([segment.name], json_out=True, include_inactive=True)
    assert json.loads(capsys.readouterr().out)[0]["used_bytes"] == 128
    assert cmd_delete(segment.name) is True
    assert not segment.exists()


def test_independent_trackers_and_duplicate_wait_for_last_reference(segment):
    one, two = tracker(), tracker()
    duplicate = os.dup(one._fd)
    try:
        one.close()
        two.close()
        with pytest.raises(OSError, match="in use"):
            protocol.delete(str(segment))
    finally:
        os.close(duplicate)
    assert protocol.delete(str(segment))


def test_fork_close_preserves_child_usage_until_sigkill(segment):
    owner = tracker()
    reader, writer = os.pipe()
    pid = os.fork()
    if pid == 0:
        os.close(reader)
        os.write(writer, b"x")
        signal.pause()
        os._exit(0)
    os.close(writer)
    try:
        assert os.read(reader, 1) == b"x"
        owner.close()
        assert protocol.active(str(segment)) is True
        with pytest.raises(OSError, match="in use"):
            protocol.delete(str(segment))
    finally:
        os.kill(pid, signal.SIGKILL)
        os.waitpid(pid, 0)
        os.close(reader)
        owner.close()
    assert segment.exists() and protocol.delete(str(segment))


def test_old_tracker_close_never_unlinks_replacement(segment):
    old = tracker()
    old_inode = segment.stat().st_ino
    segment.unlink()  # Deliberate external replacement outside the protocol.
    new = tracker()
    try:
        assert segment.stat().st_ino != old_inode
        old.close()
        assert segment.exists() and protocol.active(str(segment)) is True
    finally:
        old.close()
        new.close()
    assert protocol.delete(str(segment))


def test_legacy_visibility_refusal_and_coordinated_restart(segment):
    # Before migration, absence of locks says NOTHING about old users. All
    # writers/deleters of this one segment must stop before the new init below.
    segment.write_bytes((4096).to_bytes(8, sys.byteorder) + bytes(16))
    assert protocol.active(str(segment)) is None
    assert _detect_kvcache_ipc_names() == [segment.name]
    with pytest.raises(ValueError, match="legacy or unknown"):
        protocol.delete(str(segment))
    original = segment.stat()
    owner = tracker()
    try:
        assert os.path.samestat(original, segment.stat())
        assert utils.get_kv_cache_limit(segment.name) == utils.MemInfoStruct(4096, 0, 0)
        assert protocol.active(str(segment)) is True
    finally:
        owner.close()


@pytest.mark.parametrize("fault", [errno.EIO, errno.EINVAL, errno.ENOLCK])
def test_lock_failures_keep_files_and_refuse_unlocked_start(segment, monkeypatch, fault):
    owner = tracker()
    owner.close()

    def fail(*args, **kwargs):
        raise OSError(fault, "injected lock failure")

    monkeypatch.setattr(protocol, "lock", fail)
    with pytest.raises(OSError):
        tracker()
    with pytest.raises(OSError):
        protocol.delete(str(segment))
    assert segment.exists()


def test_unsupported_and_probe_error_are_uncertain(segment, monkeypatch):
    owner = tracker()
    owner.close()
    monkeypatch.setattr(protocol, "supported", lambda: False)
    assert protocol.active(str(segment)) is None
    assert _detect_kvcache_ipc_names() == [segment.name]
    with pytest.raises(OSError, match="unavailable"):
        protocol.delete(str(segment))
    assert segment.exists()


def test_deleted_between_open_and_usage_is_retried_before_write(segment, monkeypatch):
    owner = tracker()
    owner.close()
    original_lock = protocol.lock
    hits: list[bool] = []

    def replace(fd, exclusive=False):
        if not hits:
            hits.append(True)
            assert protocol.delete(str(segment))
            # The initializer must reject its unlinked fd and retry a new open.
        return original_lock(fd, exclusive)

    # Let the nested delete acquire its lock without re-entering the hook.
    def dispatch(fd, exclusive=False):
        return original_lock(fd, exclusive) if exclusive else replace(fd, exclusive)

    monkeypatch.setattr(protocol, "lock", dispatch)
    new = tracker()
    try:
        assert len(hits) == 1 and protocol.active(str(segment)) is True
        info = utils.get_kv_cache_limit(segment.name)
        assert info is not None and info.total_size == 4096
    finally:
        new.close()


def test_replacement_between_delete_open_and_lock_is_preserved(segment, monkeypatch):
    owner = tracker()
    owner.close()
    original_lock = protocol.lock

    def replace(fd, exclusive=False):
        segment.unlink()
        segment.write_bytes(b"replacement")
        return original_lock(fd, exclusive)

    monkeypatch.setattr(protocol, "lock", replace)
    assert protocol.delete(str(segment)) is False
    assert segment.read_bytes() == b"replacement"


def test_unlink_error_is_reported_and_retry_remains_safe(segment, monkeypatch):
    owner = tracker()
    owner.close()

    def fail(*args, **kwargs):
        raise PermissionError("injected unlink failure")

    with monkeypatch.context() as patch:
        patch.setattr(protocol.os, "unlink", fail)
        with pytest.raises(PermissionError):
            protocol.delete(str(segment))
    assert segment.exists() and protocol.delete(str(segment))


def test_symlink_delete_and_initializer_are_refused(segment):
    target = segment.with_name("target")
    target.write_bytes(bytes(24) + protocol.MARKER)
    segment.symlink_to(target)
    with pytest.raises(OSError):
        protocol.delete(str(segment))
    with pytest.raises(OSError):
        tracker()
    assert target.exists()


def test_short_data_flock_does_not_release_usage(segment):
    owner = tracker()
    try:
        for _ in range(10):
            owner.update_memory_usage(64, 32)
            utils.update_kv_cache_limit(segment.name, 8192)
            assert protocol.active(str(segment)) is True
        assert owner.check_and_get_resize_target(1024, 2) == 2048
    finally:
        owner.close()


@pytest.mark.parametrize("kind", ["python", "native"])
def test_real_worker_start_delete_exit_interleavings(segment, kind):
    native = os.getenv("KVCACHED_TEST_NATIVE_TRACKER", "")
    if kind == "native" and not native:
        pytest.skip("set KVCACHED_TEST_NATIVE_TRACKER to the real C++ tracker probe")
    script = r"""
import sys
from kvcached import mem_info_tracker as tracker
from kvcached.cli import utils
utils.SHM_DIR, tracker.DEFAULT_IPC_NAME = sys.argv[1:]
owner = tracker.MemInfoTracker(4096)
print('ready', flush=True)
sys.stdin.readline()
owner.close()
print('closed', flush=True)
sys.stdin.readline()
"""
    for _ in range(30):
        command = (
            [native, str(segment)]
            if kind == "native"
            else [sys.executable, "-c", script, str(segment.parent), segment.name]
        )
        process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        try:
            assert process.stdin is not None and process.stdout is not None
            assert _read_line(process) == "ready"
            with pytest.raises(OSError, match="in use"):
                protocol.delete(str(segment))
            process.stdin.write("close\n")
            process.stdin.flush()
            assert _read_line(process) == "closed"
            assert protocol.active(str(segment)) is False
            assert protocol.delete(str(segment))
            replacement = tracker()
            process.stdin.write("exit\n")
            process.stdin.flush()
            assert process.wait(timeout=5) == 0
            assert segment.exists() and protocol.active(str(segment)) is True
            replacement.close()
            assert protocol.delete(str(segment))
        finally:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=5)


def _read_line(process):
    assert select.select([process.stdout], [], [], 10)[0], "worker output timed out"
    return process.stdout.readline().strip()


@pytest.mark.parametrize("kind", ["python", "native"])
def test_real_startup_races_with_explicit_delete(segment, kind):
    native = os.getenv("KVCACHED_TEST_NATIVE_TRACKER", "")
    if kind == "native" and not native:
        pytest.skip("set KVCACHED_TEST_NATIVE_TRACKER to the real C++ tracker probe")
    script = r"""
import sys
from kvcached import mem_info_tracker as tracker
from kvcached.cli import utils
utils.SHM_DIR, tracker.DEFAULT_IPC_NAME = sys.argv[1:]
print('waiting', flush=True)
sys.stdin.readline()
try:
    owner = tracker.MemInfoTracker(4096)
except OSError:
    print('failed', flush=True)
    sys.exit(3)
print('ready', flush=True)
sys.stdin.readline()
owner.close()
print('closed', flush=True)
"""
    for _ in range(20):
        owner = tracker()
        owner.close()
        command = (
            [native, str(segment), "--wait-for-start"]
            if kind == "native"
            else [sys.executable, "-c", script, str(segment.parent), segment.name]
        )
        process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        try:
            assert process.stdin is not None and process.stdout is not None
            assert _read_line(process) == "waiting"
            process.stdin.write("start\n")
            process.stdin.flush()
            deadline = time.monotonic() + 10
            while not select.select([process.stdout], [], [], 0)[0]:
                assert time.monotonic() < deadline, "startup timed out"
                try:
                    protocol.delete(str(segment))
                except OSError as error:
                    assert error.errno == errno.EBUSY
                except ValueError:
                    # A first initializer may not yet have published the marker.
                    pass
            state = _read_line(process)
            if state == "failed":
                # Nonblocking acquisition may reject startup under contention.
                # Running without a valid lifetime lock is never acceptable.
                assert process.wait(timeout=5) == 3
                continue
            assert state == "ready"
            identity = segment.stat()
            assert protocol.active(str(segment)) is True
            for _ in range(5):
                with pytest.raises(OSError, match="in use"):
                    protocol.delete(str(segment))
            assert os.path.samestat(identity, segment.stat())
            process.stdin.write("close\n")
            process.stdin.flush()
            assert _read_line(process) == "closed"
            process.stdin.close()
            assert process.wait(timeout=5) == 0
            assert protocol.delete(str(segment))
        finally:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=5)
            segment.unlink(missing_ok=True)
