# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


MODULE_PATH = Path(__file__).parents[1] / "tools" / "check_engine_compat.py"
SPEC = importlib.util.spec_from_file_location("check_engine_compat", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
compat = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = compat
SPEC.loader.exec_module(compat)


def write_module(root: Path, relative: str, source: str) -> None:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source, encoding="utf-8")


def materialize_contract(repo: Path, engine: str) -> Path:
    package_root = repo / ("vllm" if engine == "vllm" else "python/sglang")
    for contract in compat.CONTRACTS[engine]:
        source = "\n".join(
            f"class {symbol}:\n    def marker(self, value=None):\n        pass\n"
            for symbol in contract.required
        )
        write_module(package_root, contract.path, source)
    return package_root


def test_vllm_contract_records_class_method_fingerprints(tmp_path):
    package_root = materialize_contract(tmp_path, "vllm")

    result = compat.check_repository(tmp_path, "vllm")

    assert result.status == "compatible"
    block_pool = next(
        module for module in result.modules
        if module.path == str(package_root / "v1/core/block_pool.py")
    )
    assert block_pool.symbols["KVCacheBlock"] == ["marker(self, value)"]


def test_sglang_contract_supports_python_package_layout(tmp_path):
    package_root = materialize_contract(tmp_path, "sglang")

    result = compat.check_repository(tmp_path, "sglang")

    assert result.status == "compatible"
    assert result.package_root == str(package_root)


def test_missing_required_symbol_fails_contract(tmp_path):
    package_root = materialize_contract(tmp_path, "vllm")
    write_module(package_root, "v1/engine/core.py", "class RenamedEngineCore:\n    pass\n")

    result = compat.check_repository(tmp_path, "vllm")

    assert result.status == "incompatible"
    engine_core = next(
        module for module in result.modules
        if module.path == str(package_root / "v1/engine/core.py")
    )
    assert engine_core.missing_required == ["EngineCore"]
