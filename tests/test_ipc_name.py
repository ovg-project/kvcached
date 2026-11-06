# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

import importlib
import os
import subprocess
import sys
import types
from pathlib import Path

import pytest

# Inode of Linux's initial PID namespace, PID_NS_INIT_INO in <linux/nsfs.h>.
HOST_PID_NAMESPACE = 0xEFFFFFFC


def _utils(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", types.ModuleType("torch"))
    return importlib.import_module("kvcached.utils")


def test_explicit_ipc_name_is_exact_without_probing(monkeypatch):
    utils = _utils(monkeypatch)

    monkeypatch.setenv("KVCACHED_IPC_NAME", "shared-instance-ipc")

    def fail_if_probed(name):
        raise AssertionError(f"explicit IPC name should not probe {name}")

    monkeypatch.setattr(utils, "_ipc_segment_exists", fail_if_probed)

    assert utils._obtain_default_ipc_name() == "shared-instance-ipc"


@pytest.mark.parametrize("pgid,pid_namespace,expected", [
    # runc and crun start each container's init with setsid(), so an engine
    # launched as the entrypoint has PGID 1 in every container.
    (1, 4026532461, "kvcached_vLLM_pidns4026532461_1"),
    (1, 4026532789, "kvcached_vLLM_pidns4026532789_1"),
    (4242, HOST_PID_NAMESPACE, "kvcached_vLLM_4242"),
    (4242, None, "kvcached_vLLM_4242"),  # no /proc/self/ns/pid
])
def test_default_name(monkeypatch, pgid, pid_namespace, expected):
    utils = _utils(monkeypatch)
    monkeypatch.delenv("KVCACHED_IPC_NAME", raising=False)
    monkeypatch.setattr(utils, "_detect_engine_tag", lambda: "vLLM")
    monkeypatch.setattr(utils, "_ipc_segment_exists", lambda name: False)
    monkeypatch.setattr(utils.os, "getpgid", lambda pid: pgid)
    real_stat = os.stat

    def stat(path, *args, **kwargs):
        if path != "/proc/self/ns/pid":
            return real_stat(path, *args, **kwargs)
        if pid_namespace is None:
            raise FileNotFoundError(path)
        return types.SimpleNamespace(st_ino=pid_namespace)

    monkeypatch.setattr(utils.os, "stat", stat)
    assert utils._obtain_default_ipc_name() == expected


@pytest.mark.skipif(sys.platform != "linux", reason="Linux example launch scripts")
@pytest.mark.parametrize("example,roles", [
    ("01_simple_two_models/start_two_models.sh", ("a", "b")),
    ("05_multi_agents/start_multi_agent_models.sh", ("research", "writing")),
])
@pytest.mark.parametrize("engine", ["vllm", "sglang"])
def test_example_launches_have_distinct_ipc_names(tmp_path, example, roles, engine):
    """Run the real launch scripts, replacing only the serving executables."""
    stub = tmp_path / "vllm"
    stub.write_text(
        f"#!{sys.executable}\n"
        "import os\n"
        "from pathlib import Path\n"
        "output = Path(os.environ['IPC_CAPTURE']) / str(os.getpid())\n"
        "output.write_text(os.environ['KVCACHED_IPC_NAME'])\n",
        encoding="utf-8",
    )
    stub.chmod(0o755)
    env = dict(os.environ, PATH=f"{tmp_path}{os.pathsep}{os.environ['PATH']}",
               PYTHON=str(stub), KVCACHED_IPC_NAME="inherited-name")
    script = Path(__file__).resolve().parents[1] / "examples" / example
    names = []
    for launch in range(2):
        output = tmp_path / f"launch-{launch}"
        output.mkdir()
        env["IPC_CAPTURE"] = str(output)
        options = []
        for index, role in enumerate(roles):
            engine_arg = f"--engine-{role}" if role in ("a", "b") else f"--{role}-engine"
            port_arg = f"--port-{role}" if role in ("a", "b") else f"--{role}-port"
            options.extend([engine_arg, engine, port_arg, str(18000 + 2 * launch + index)])
        subprocess.run(["bash", str(script), *options], env=env,
                       check=True, capture_output=True, text=True, timeout=20)
        launched = [path.read_text() for path in output.iterdir()]
        assert len(launched) == 2
        names.extend(launched)
    assert len(set(names)) == 4
    assert all(name != "inherited-name" and 0 < len(name) <= 64 for name in names)
