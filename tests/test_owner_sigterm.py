# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0
"""Real signal exits: unwind held locks, preserve policy and signal status."""
import os
import signal
import subprocess
import sys
import time

import pytest

_SCRIPT = r"""
import os, signal, sys, threading, time
from pathlib import Path
from kvcached.utils import register_owner_sigterm_cleanup
root, mode = Path(sys.argv[1]), sys.argv[2]
lock = threading.Lock()
class Owner:
    def __init__(self):
        self.fired = False
    def shutdown(self):
        with lock:
            if mode == 'reentrant' and not self.fired:
                self.fired = True
                os.kill(os.getpid(), signal.SIGTERM)
            with (root/'cleaned').open('a') as f:
                f.write('cleaned\n')
            if mode == 'second':
                (root/'cleaning').touch()
                time.sleep(30)
if mode == 'local-owner':
    import gc
    def run():
        local_owner = Owner()
        assert register_owner_sigterm_cleanup(local_owner.shutdown)
        try:
            (root/'ready').touch()
            while True: time.sleep(.05)
        finally:
            del local_owner
            gc.collect()
    run()
owner = Owner()
if mode == "sglang-noop-hook":
    threading._register_atexit = lambda *args, **kwargs: None
if mode in ("non-daemon", "sglang-noop-hook"):
    threading.Thread(target=threading.Event().wait).start()
if mode == 'custom':
    def custom(signum, frame):
        (root/'custom').touch()
    signal.signal(signal.SIGTERM, custom)
elif mode == 'ignored':
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
registered = register_owner_sigterm_cleanup(owner.shutdown)
assert registered == (mode not in ('custom', 'ignored'))
register_owner_sigterm_cleanup(owner.shutdown)  # no duplicate dispatch
if mode == 'fork':
    pid = os.fork()
    if pid == 0:
        (root/'ready').touch()
        while True: time.sleep(.05)
    (root/'child').write_text(str(pid))
    waited, status = os.waitpid(pid, 0)
    assert os.WIFSIGNALED(status) and os.WTERMSIG(status) == signal.SIGTERM
    assert not (root/'cleaned').exists()
    sys.exit(0)
if mode == 'reentrant':
    owner.shutdown()
else:
    with lock:
        (root/'ready').touch()
        while True: time.sleep(.05)
"""


def _wait(path, process, timeout=10):
    deadline = time.monotonic() + timeout
    while not path.exists() and process.poll() is None and time.monotonic() < deadline:
        time.sleep(.02)
    assert path.exists(), f"missing {path.name}; process status={process.poll()}"


@pytest.mark.skipif(os.name != "posix", reason="POSIX termination status")
@pytest.mark.parametrize("mode", ["held-lock", "reentrant", "custom", "ignored", "second", "fork", "non-daemon", "sglang-noop-hook", "local-owner"])
def test_owner_sigterm_real_process(tmp_path, mode):
    process = subprocess.Popen([sys.executable, "-c", _SCRIPT, str(tmp_path), mode])
    try:
        if mode == "reentrant":
            assert process.wait(timeout=10) == -signal.SIGTERM
        else:
            _wait(tmp_path / "ready", process)
            target = process.pid
            if mode == "fork":
                _wait(tmp_path / "child", process)
                target = int((tmp_path / "child").read_text())
            os.kill(target, signal.SIGTERM)
            if mode in ("custom", "ignored"):
                if mode == "custom":
                    _wait(tmp_path / "custom", process)
                else:
                    time.sleep(.1)
                assert process.poll() is None
                assert not (tmp_path / "cleaned").exists()
                return
            if mode == "second":
                _wait(tmp_path / "cleaning", process)
                os.kill(process.pid, signal.SIGTERM)
            assert process.wait(timeout=10) == (0 if mode == "fork" else -signal.SIGTERM)
        if mode != "fork":
            assert (tmp_path / "cleaned").read_text() == "cleaned\n"
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)


def test_owner_sigterm_registration_off_main_thread_is_inert():
    import threading

    from kvcached.utils import register_owner_sigterm_cleanup
    class Owner:
        def shutdown(self):
            raise AssertionError("must not run")
    owner = Owner()
    result = []
    thread = threading.Thread(target=lambda: result.append(register_owner_sigterm_cleanup(owner.shutdown)))
    thread.start()
    thread.join()
    assert result == [False]

_LOOP_SCRIPT = r"""
import asyncio, os, signal, sys, time
from pathlib import Path
from kvcached.utils import register_owner_sigterm_cleanup, register_owner_sigterm_loop
root, mode = Path(sys.argv[1]), sys.argv[2]
class Owner:
    def shutdown(self):
        (root/'cleaned').touch()
owner = Owner()
assert register_owner_sigterm_cleanup(owner.shutdown)
previous = signal.getsignal(signal.SIGTERM)
loop = asyncio.new_event_loop()
def native():
    (root/'native').touch()
    loop.stop()
loop.add_signal_handler(signal.SIGTERM, native)
assert register_owner_sigterm_loop(loop, previous)
bridge = signal.getsignal(signal.SIGTERM)
assert register_owner_sigterm_cleanup(owner.shutdown)
assert signal.getsignal(signal.SIGTERM) is bridge
if mode == 'running':
    loop.call_soon((root/'ready').touch)
    loop.run_forever()
    assert (root/'native').exists() and not (root/'cleaned').exists()
    loop.close()
else:
    loop.run_until_complete(asyncio.sleep(0))
    (root/'ready').touch()
    while True: time.sleep(.05)
"""


@pytest.mark.skipif(os.name != "posix", reason="POSIX asyncio signal handling")
@pytest.mark.parametrize("mode", ["running", "stopped"])
def test_owner_sigterm_running_and_stopped_asyncio_loop(tmp_path, mode):
    process = subprocess.Popen([sys.executable, "-c", _LOOP_SCRIPT, str(tmp_path), mode])
    try:
        _wait(tmp_path / "ready", process)
        os.kill(process.pid, signal.SIGTERM)
        assert process.wait(timeout=10) == (0 if mode == "running" else -signal.SIGTERM)
        assert (tmp_path / "native").exists() is (mode == "running")
        assert (tmp_path / "cleaned").exists() is (mode == "stopped")
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)


_FORK_LOOP_SCRIPT = r"""
import asyncio, multiprocessing as mp, os, signal, sys, threading, time
from pathlib import Path
root, parent_state, mode = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
fork_method = sys.argv[4]
parent_pid = os.getpid()
child_loop = None
custom_fds = None
injected = False
detached_fd = None
lock = threading.Lock()

def mark(name):
    with (root/name).open('a') as f:
        f.write(str(os.getpid()) + '\n')

def custom(signum, frame):
    mark('custom')

def native_child():
    mark('child-native')
    child_loop.stop()

def child_policy():
    global child_loop, custom_fds, detached_fd
    if mode in ('custom', 'early-custom'):
        signal.signal(signal.SIGTERM, custom)
    elif mode in ('ignored', 'early-ignored'):
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
    elif mode in ('early-loop', 'new-loop', 'new-stopped-loop'):
        child_loop = asyncio.new_event_loop()
        child_loop.add_signal_handler(signal.SIGTERM, native_child)
    elif mode in ('early-fd', 'early-signal', 'early-detached', 'reused-fd', 'closed-fd'):
        custom_fds = os.pipe()
        os.set_blocking(custom_fds[0], False)
        os.set_blocking(custom_fds[1], False)
        fd = custom_fds[1]
        if mode == 'reused-fd':
            # Replace only the child's descriptor, without closing the loop
            # or changing the selector shared with the parent.
            fd = loop._csock.fileno()
            os.dup2(custom_fds[1], fd)
        elif mode == 'closed-fd':
            os.close(loop._csock.fileno())
        elif mode == 'early-detached':
            detached_fd = loop._csock.detach()
        signal.set_wakeup_fd(fd, warn_on_full_buffer=False)
        signal.signal(signal.SIGTERM, custom)
        if mode == 'early-signal':
            # Deliver a real signal inside reset, without replacing the setter.
            # The old probe loses this byte; an owned-fd redirect preserves it.
            def during_reset(frame, event, function):
                global injected
                if ((event == 'c_return' and function is signal.set_wakeup_fd)
                        or (event == 'c_call' and function is os.dup2)):
                    sys.setprofile(None)
                    injected = True
                    os.kill(os.getpid(), signal.SIGTERM)
            sys.setprofile(during_reset)

# This hook deliberately runs BEFORE the utility's own child hook.
if mode.startswith('early-') or mode in ('reused-fd', 'closed-fd'):
    os.register_at_fork(after_in_child=child_policy)

from kvcached.utils import register_owner_sigterm_cleanup, register_owner_sigterm_loop

class Owner:
    def __init__(self, name):
        self.name = name
    def shutdown(self):
        with lock:
            assert (root/'unwound').exists()
            mark(self.name)

owner = Owner('parent-cleaned')
assert register_owner_sigterm_cleanup(owner.shutdown)
loop = asyncio.new_event_loop()
previous = signal.getsignal(signal.SIGTERM)
parent_hits = []
loop.add_signal_handler(signal.SIGTERM, lambda: parent_hits.append('native'))
assert register_owner_sigterm_loop(loop, previous)
parent_bridge = signal.getsignal(signal.SIGTERM)

def child():
    if mode == 'early-signal':
        assert injected, 'signal did not arrive during reset'
        assert os.read(custom_fds[0], 4096) == bytes([signal.SIGTERM])
        assert (root/'custom').read_text() == str(os.getpid()) + '\n'
        (root/'custom').unlink()
    if mode == 'early-detached':
        # The original socket object no longer owns this descriptor. It must
        # still refer to a socket, even though its kernel identity is unchanged.
        import socket
        with socket.socket(fileno=detached_fd):
            pass
    child_owner = Owner('child-cleaned')
    if mode in ('custom', 'ignored'):
        child_policy()
    if mode != 'no-owner':
        accepted = mode in ('owner', 'new-loop', 'new-stopped-loop')
        assert register_owner_sigterm_cleanup(child_owner.shutdown) == accepted
        assert register_owner_sigterm_cleanup(child_owner.shutdown) == accepted
    if mode in ('new-loop', 'new-stopped-loop'):
        previous = signal.getsignal(signal.SIGTERM)
        child_policy()
        assert register_owner_sigterm_loop(child_loop, previous)
        assert register_owner_sigterm_cleanup(child_owner.shutdown)
    if mode in ('early-loop', 'new-loop'):
        child_loop.call_soon(os.kill, os.getpid(), signal.SIGTERM)
        child_loop.call_later(2, child_loop.stop)
        child_loop.run_forever()
        assert (root/'child-native').read_text() == str(os.getpid()) + '\n'
        assert not (root/'child-cleaned').exists()
        os._exit(0)
    with lock:
        try:
            os.kill(os.getpid(), signal.SIGTERM)
            if mode in ('owner', 'no-owner', 'new-stopped-loop'):
                # Only the broken bridge reaches this point; the parent has
                # an independent deadline and will kill/reap this child.
                while True:
                    time.sleep(.02)
            if mode in ('custom', 'early-custom', 'early-fd', 'early-signal', 'early-detached', 'reused-fd', 'closed-fd'):
                assert (root/'custom').read_text() == str(os.getpid()) + '\n'
            else:
                assert not (root/'custom').exists()
            if custom_fds is not None:
                assert os.read(custom_fds[0], 1) == bytes([signal.SIGTERM])
                # Independent wakeup-only pipes keep their quiet full-buffer
                # policy whether the old loop socket is still open or gone.
                errors = []
                sys.unraisablehook = errors.append
                while True:
                    try:
                        os.write(custom_fds[1], b'x' * 4096)
                    except BlockingIOError:
                        break
                os.kill(os.getpid(), signal.SIGTERM)
                assert (root/'custom').read_text() == (str(os.getpid()) + '\n') * 2
                assert not errors, errors
            assert not (root/'child-cleaned').exists()
            os._exit(0)
        finally:
            mark('unwound')

pid = None
process = None
status = None
deadline = None

def fork():
    global pid, process, deadline
    if fork_method == 'multiprocessing':
        process = mp.get_context('fork').Process(target=child)
        process.start()
        pid = process.pid
    else:
        pid = os.fork()
        if pid == 0:
            child()
            os._exit(2)
    deadline = time.monotonic() + 3

def poll():
    global status
    if process is not None:
        process.join(0)
        code = process.exitcode
        waited = code is not None
        result = None if code is None else (-code if code < 0 else code << 8)
    else:
        waited, result = os.waitpid(pid, os.WNOHANG)
    if waited:
        status = result
        loop.stop()
    elif time.monotonic() >= deadline:
        os.kill(pid, signal.SIGKILL)
        os.waitpid(pid, 0)
        raise AssertionError('child ignored SIGTERM')
    else:
        loop.call_later(.01, poll)

try:
    if parent_state == 'running':
        loop.call_soon(fork)
    else:
        loop.run_until_complete(asyncio.sleep(0))
        fork()
    loop.call_soon(poll)
    loop.call_later(5, loop.stop)
    loop.run_forever()  # Keep pumping the parent's socket while child signals.
    assert status is not None, 'child was not reaped before timeout'
    assert not parent_hits, parent_hits
    assert not (root/'parent-cleaned').exists()
    assert signal.getsignal(signal.SIGTERM) is parent_bridge
    if mode in ('owner', 'no-owner', 'new-stopped-loop'):
        assert os.WIFSIGNALED(status) and os.WTERMSIG(status) == signal.SIGTERM, status
    else:
        assert os.WIFEXITED(status) and os.WEXITSTATUS(status) == 0, status
    if mode in ('owner', 'new-stopped-loop'):
        assert (root/'child-cleaned').read_text() == str(pid) + '\n'
    else:
        assert not (root/'child-cleaned').exists()
    # Verify the parent's signal fd and callback still work after child exit.
    loop.call_soon(os.kill, parent_pid, signal.SIGTERM)
    loop.call_later(.1, loop.stop)
    loop.run_forever()
    assert parent_hits == ['native'], parent_hits
finally:
    if os.getpid() == parent_pid:
        if pid is not None and status is None:
            try:
                os.kill(pid, signal.SIGKILL)
                os.waitpid(pid, 0)
            except ProcessLookupError:
                pass
        loop.close()
"""


@pytest.mark.skipif(not hasattr(os, "fork"), reason="POSIX fork and asyncio signals")
@pytest.mark.parametrize("fork_method", ["fork", "multiprocessing"])
@pytest.mark.parametrize("parent_state", ["running", "stopped"])
@pytest.mark.parametrize("mode", [
    "owner", "no-owner", "custom", "ignored", "new-loop", "new-stopped-loop",
    "early-loop", "early-custom", "early-ignored", "early-fd", "early-signal", "early-detached", "reused-fd", "closed-fd",
])
def test_owner_sigterm_forked_loop(tmp_path, parent_state, mode, fork_method):
    process = subprocess.Popen(
        [sys.executable, "-c", _FORK_LOOP_SCRIPT, str(tmp_path), parent_state, mode, fork_method],
        start_new_session=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    try:
        stdout, stderr = process.communicate(timeout=15)
        assert process.returncode == 0, (stdout.decode(), stderr.decode())
    finally:
        # The complete session belongs to this test. Reap descendants even if
        # a regression makes the owner/child unresponsive or setup fails.
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.communicate(timeout=5)
