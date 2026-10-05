# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import Mock

# Importing kvcached only requires torch to be loaded first; these tests exercise
# patch selection and do not need any torch APIs. Remove the local stub after the
# import so it cannot leak into other test modules in the same process.
try:
    import torch  # noqa: F401
except ImportError:
    sys.modules["torch"] = ModuleType("torch")
    _remove_torch_stub = True
else:
    _remove_torch_stub = False

try:
    from kvcached.integration import patch_base  # noqa: E402
    from kvcached.integration.patch_base import (  # noqa: E402
        PatchManager,
        is_integration_version_supported,
    )
finally:
    if _remove_torch_stub:
        sys.modules.pop("torch", None)


def test_unsupported_integrations_run_without_kvcached(monkeypatch):
    cases = [
        ("vllm", None, ">=0.8.4", "version could not be detected"),
        ("sglang", "0.0.0", ">=0.4.9", "unsupported version 0.0.0"),
    ]
    manager = PatchManager("vllm").version_manager
    warning = Mock()
    monkeypatch.setattr(patch_base.logger, "warning", warning)

    for library, detected_version, supported_range, reason in cases:
        monkeypatch.setattr(
            manager, "detect_version", lambda _, value=detected_version: value
        )
        warning.reset_mock()
        assert not is_integration_version_supported(library, supported_range)

        message, *args = warning.call_args.args
        assert message % tuple(args) == (
            f"{library} integration disabled: {reason}; running without kvcached"
        )


def test_patch_manager_rejects_unknown_version(monkeypatch):
    manager = PatchManager("vllm")
    monkeypatch.setattr(manager.version_manager, "detect_version", lambda _: None)

    assert not manager._is_patch_compatible(Mock(patch_name="test_patch"), ">=0.8.4")


def test_autopatches_guard_versions_before_constructing_manager():
    for integration, supported_range in (
        ("vllm", "VLLM_ALL_RANGE"),
        ("sglang", "SGLANG_ALL_RANGE"),
    ):
        path = (
            Path(__file__).resolve().parents[1]
            / "kvcached"
            / "integration"
            / integration
            / "autopatch.py"
        )
        source = path.read_text(encoding="utf-8")
        guard = (
            f'if not is_integration_version_supported("{integration}", '
            f"{supported_range}):"
        )
        assert source.index(guard) < source.index(
            f'patch_manager = PatchManager("{integration}")'
        )
