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
