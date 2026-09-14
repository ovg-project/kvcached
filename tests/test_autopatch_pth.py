# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0
"""kvcached_autopatch.pth must reach site-packages, or say so (issue #470).

The vLLM/SGLang import hooks are registered by kvcached_autopatch.pth, which
the interpreter only executes from site-packages. PEP 660 editable installs
never ran the legacy develop command that used to copy it, so
ENABLE_KVCACHED / KVCACHED_AUTOPATCH silently did nothing. These tests pin
down the two halves of the fix: the .pth is appended to the editable wheel
(with a valid RECORD entry, so pip installs it), and importing kvcached with
the knobs set but the .pth absent warns.

CPU-only: setup.py is not imported (it needs a GPU toolchain); the wheel
helper is loaded from its source.
"""

import ast
import base64
import hashlib
import os
import subprocess
import sys
import warnings
import zipfile
from pathlib import Path

import pytest

import kvcached

ROOT = Path(__file__).resolve().parents[1]
PTH_FILE = "kvcached_autopatch.pth"


def _load_add_pth_to_wheel():
    """Load setup.py's add_pth_to_wheel without executing setup.py."""
    tree = ast.parse((ROOT / "setup.py").read_text())
    func = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "add_pth_to_wheel"
    )
    namespace = {
        "base64": base64,
        "hashlib": hashlib,
        "os": os,
        "zipfile": zipfile,
        "SCRIPT_PATH": str(ROOT),
        "PTH_FILE": PTH_FILE,
    }
    exec(compile(ast.Module(body=[func], type_ignores=[]), "setup.py", "exec"), namespace)
    return namespace["add_pth_to_wheel"]


def _record_entry(name: str, data: bytes) -> str:
    digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=")
    return f"{name},sha256={digest.decode()},{len(data)}"


def _write_minimal_wheel(path: Path) -> dict:
    """A tiny but valid wheel: one module, METADATA, WHEEL, RECORD."""
    files = {
        "kvctest.py": b"VALUE = 1\n",
        "kvctest-0.1.dist-info/METADATA": b"Metadata-Version: 2.1\nName: kvctest\nVersion: 0.1\n",
        "kvctest-0.1.dist-info/WHEEL": (
            b"Wheel-Version: 1.0\nGenerator: test\nRoot-Is-Purelib: true\nTag: py3-none-any\n"
        ),
    }
    record = "".join(f"{_record_entry(name, data)}\n" for name, data in files.items())
    record += "kvctest-0.1.dist-info/RECORD,,\n"
    files["kvctest-0.1.dist-info/RECORD"] = record.encode()
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as wheel:
        for name, data in files.items():
            wheel.writestr(name, data)
    return files


def test_add_pth_to_wheel_appends_pth_and_record_entry(tmp_path):
    wheel_path = tmp_path / "kvctest-0.1-py3-none-any.whl"
    original = _write_minimal_wheel(wheel_path)
    pth_data = (ROOT / PTH_FILE).read_bytes()

    _load_add_pth_to_wheel()(str(wheel_path))

    with zipfile.ZipFile(wheel_path) as wheel:
        assert wheel.testzip() is None
        names = wheel.namelist()
        assert names.count(PTH_FILE) == 1
        assert wheel.read(PTH_FILE) == pth_data
        for name, data in original.items():
            if not name.endswith("RECORD"):
                assert wheel.read(name) == data
        record = wheel.read("kvctest-0.1.dist-info/RECORD").decode().splitlines()
    assert _record_entry(PTH_FILE, pth_data) in record
    assert record[:-1] == original["kvctest-0.1.dist-info/RECORD"].decode().splitlines()
    assert not (tmp_path / "kvctest-0.1-py3-none-any.whl.tmp").exists()


def test_add_pth_to_wheel_rejects_a_wheel_without_record(tmp_path):
    wheel_path = tmp_path / "broken-0.1-py3-none-any.whl"
    with zipfile.ZipFile(wheel_path, "w") as wheel:
        wheel.writestr("broken.py", b"")

    with pytest.raises(RuntimeError, match="RECORD"):
        _load_add_pth_to_wheel()(str(wheel_path))


def test_pip_installs_the_appended_pth_into_the_target_root(tmp_path):
    """pip must accept the rewritten wheel and place the .pth at the install
    root (site-packages for a real install), where the interpreter runs it."""
    wheel_path = tmp_path / "kvctest-0.1-py3-none-any.whl"
    _write_minimal_wheel(wheel_path)
    _load_add_pth_to_wheel()(str(wheel_path))
    target = tmp_path / "target"

    subprocess.run(
        [sys.executable, "-m", "pip", "install", "--quiet", "--no-index",
         "--no-deps", "--target", str(target), str(wheel_path)],
        check=True,
    )

    assert (target / PTH_FILE).read_bytes() == (ROOT / PTH_FILE).read_bytes()
    assert (target / "kvctest.py").exists()


@pytest.fixture
def site_dirs(monkeypatch, tmp_path):
    system = tmp_path / "site-packages"
    user = tmp_path / "user-site"
    system.mkdir()
    user.mkdir()
    monkeypatch.setattr(kvcached.site, "getsitepackages", lambda: [str(system)])
    monkeypatch.setattr(kvcached.site, "getusersitepackages", lambda: str(user))
    monkeypatch.delenv("ENABLE_KVCACHED", raising=False)
    monkeypatch.delenv("KVCACHED_AUTOPATCH", raising=False)
    return system, user


def _assert_no_warning():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        kvcached._warn_if_autopatch_pth_missing()


@pytest.mark.parametrize("knob", ["ENABLE_KVCACHED", "KVCACHED_AUTOPATCH"])
def test_import_warns_when_autopatch_is_requested_but_pth_is_missing(site_dirs, monkeypatch, knob):
    monkeypatch.setenv(knob, "1")

    with pytest.warns(RuntimeWarning, match="kvcached_autopatch.pth is not installed"):
        kvcached._warn_if_autopatch_pth_missing()


def test_no_warning_when_pth_is_in_site_packages(site_dirs, monkeypatch):
    system, _ = site_dirs
    (system / PTH_FILE).write_text("")
    monkeypatch.setenv("ENABLE_KVCACHED", "true")

    _assert_no_warning()


def test_no_warning_when_pth_is_in_user_site(site_dirs, monkeypatch):
    _, user = site_dirs
    (user / PTH_FILE).write_text("")
    monkeypatch.setenv("KVCACHED_AUTOPATCH", "1")

    _assert_no_warning()


def test_no_warning_when_autopatch_is_not_requested(site_dirs, monkeypatch):
    monkeypatch.setenv("ENABLE_KVCACHED", "false")

    _assert_no_warning()


def test_warning_survives_site_lookup_failures(site_dirs, monkeypatch):
    def boom():
        raise AttributeError("no getsitepackages in this virtualenv")

    monkeypatch.setattr(kvcached.site, "getsitepackages", boom)
    monkeypatch.setenv("ENABLE_KVCACHED", "1")

    with pytest.warns(RuntimeWarning):
        kvcached._warn_if_autopatch_pth_missing()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
