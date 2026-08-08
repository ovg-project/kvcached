# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
import shutil
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
        definitions = []
        for symbol in contract.required:
            methods = contract.required_methods.get(symbol, ("marker",))
            body = "\n".join(
                f"    def {method}(self, value=None):\n        pass"
                for method in methods
            )
            definitions.append(f"class {symbol}:\n{body}\n")
        source = "\n".join(definitions)
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


def test_sglang_contract_accepts_legacy_single_file_allocator(tmp_path):
    package_root = materialize_contract(tmp_path, "sglang")
    allocator_package = package_root / "srt/mem_cache/allocator"
    legacy_allocator = package_root / "srt/mem_cache/allocator.py"
    shutil.rmtree(allocator_package)
    legacy_allocator.write_text(
        "class BaseTokenToKVPoolAllocator:\n"
        "    def __init__(self): pass\n"
        "    def available_size(self): pass\n"
        "    def free_group_begin(self): pass\n"
        "    def free_group_end(self): pass\n"
        "    def clear(self): pass\n"
        "    def alloc(self): pass\n"
        "    def free(self): pass\n\n"
        "class PagedTokenToKVPoolAllocator:\n"
        "    def alloc_extend(self): pass\n"
        "    def alloc_decode(self): pass\n\n"
        "def alloc_decode_kernel(): pass\n\n"
        "def alloc_extend_kernel(): pass\n",
        encoding="utf-8",
    )
    result = compat.check_repository(tmp_path, "sglang")

    assert result.status == "compatible"
    allocator_modules = [
        module for module in result.modules if module.path == str(legacy_allocator)
    ]
    assert len(allocator_modules) == 2


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


def test_missing_required_method_fails_contract(tmp_path):
    package_root = materialize_contract(tmp_path, "sglang")
    write_module(
        package_root,
        "srt/mem_cache/allocator/paged.py",
        "class PagedTokenToKVPoolAllocator:\n"
        "    def alloc_decode(self): pass\n\n"
        "def alloc_decode_kernel(): pass\n\n"
        "def alloc_extend_kernel(): pass\n",
    )

    result = compat.check_repository(tmp_path, "sglang")

    assert result.status == "incompatible"
    paged = next(
        module for module in result.modules
        if module.path == str(package_root / "srt/mem_cache/allocator/paged.py")
    )
    assert paged.missing_required_methods == [
        "PagedTokenToKVPoolAllocator.alloc_extend"
    ]
