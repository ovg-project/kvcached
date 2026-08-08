# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

import os
import subprocess
import sys
import textwrap
from pathlib import Path
from unittest import mock


# Mock heavy dependencies so this regression test can run without torch/CUDA.
# This must happen before importing kvcached.
_torch_mock = mock.MagicMock()
_torch_mock.__version__ = "2.6.0"
_torch_mock.version.hip = None
_torch_mock.version.cuda = None
sys.modules.setdefault("torch", _torch_mock)
sys.modules.setdefault("torch.cuda", _torch_mock.cuda)
sys.modules.setdefault("torch.utils", _torch_mock.utils)
sys.modules.setdefault("torch.utils.cpp_extension", _torch_mock.utils.cpp_extension)
sys.modules.setdefault("kvcached.vmm_ops", mock.MagicMock())


ROOT = Path(__file__).resolve().parents[1]


def _write(path: Path, text: str = "") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def test_pth_registers_vllm_hook_before_script_sets_enable_kvcached(tmp_path):
    site_dir = tmp_path / "site-packages"
    site_dir.mkdir()
    marker = tmp_path / "autopatch_marker.txt"

    _write(site_dir / "kvcached" / "__init__.py")
    _write(site_dir / "kvcached" / "integration" / "__init__.py")
    _write(site_dir / "kvcached" / "integration" / "vllm" / "__init__.py")
    _write(
        site_dir / "kvcached" / "integration" / "vllm" / "autopatch.py",
        "import os\n"
        "from pathlib import Path\n"
        "Path(os.environ['KVCACHED_TEST_MARKER']).write_text('imported')\n",
    )
    _write(site_dir / "vllm" / "__init__.py")

    (site_dir / "kvcached_autopatch.pth").write_text(
        (ROOT / "kvcached_autopatch.pth").read_text()
    )

    script = textwrap.dedent(
        """
        import os
        import site
        import sys
        from pathlib import Path

        site.addsitedir(sys.argv[1])
        sys.path.insert(0, sys.argv[1])

        os.environ["ENABLE_KVCACHED"] = "1"
        os.environ["KVCACHED_TEST_MARKER"] = sys.argv[2]

        import vllm  # noqa: F401

        assert Path(sys.argv[2]).read_text() == "imported"
        """
    )

    env = os.environ.copy()
    env.pop("ENABLE_KVCACHED", None)
    env.pop("KVCACHED_AUTOPATCH", None)
    env.pop("PYTHONPATH", None)

    subprocess.run(
        [sys.executable, "-c", script, str(site_dir), str(marker)],
        check=True,
        env=env,
        cwd=tmp_path,
    )


def test_vllm_env_enabled_accepts_enable_kvcached(monkeypatch):
    monkeypatch.delenv("KVCACHED_AUTOPATCH", raising=False)
    monkeypatch.setenv("ENABLE_KVCACHED", "1")

    from kvcached.integration.vllm.autopatch import _env_enabled

    assert _env_enabled()


def test_sglang_env_enabled_accepts_enable_kvcached(monkeypatch):
    monkeypatch.delenv("KVCACHED_AUTOPATCH", raising=False)
    monkeypatch.setenv("ENABLE_KVCACHED", "true")

    from kvcached.integration.sglang.autopatch import _env_enabled

    assert _env_enabled()


def test_autopatch_env_enabled_accepts_kvcached_autopatch(monkeypatch):
    monkeypatch.delenv("ENABLE_KVCACHED", raising=False)
    monkeypatch.setenv("KVCACHED_AUTOPATCH", "1")

    from kvcached.integration.vllm.autopatch import _env_enabled as vllm_env_enabled
    from kvcached.integration.sglang.autopatch import _env_enabled as sglang_env_enabled

    assert vllm_env_enabled()
    assert sglang_env_enabled()


def test_autopatch_env_disabled_by_default(monkeypatch):
    monkeypatch.delenv("ENABLE_KVCACHED", raising=False)
    monkeypatch.delenv("KVCACHED_AUTOPATCH", raising=False)

    from kvcached.integration.vllm.autopatch import _env_enabled as vllm_env_enabled
    from kvcached.integration.sglang.autopatch import _env_enabled as sglang_env_enabled

    assert not vllm_env_enabled()
    assert not sglang_env_enabled()
