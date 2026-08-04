# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import ast
from pathlib import Path


PATCHES = Path(__file__).parents[1] / "kvcached/integration/sglang/patches.py"


def _elastic_paged_allocator() -> ast.ClassDef:
    tree = ast.parse(PATCHES.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "ElasticPagedTokenToKVPoolAllocator":
            return node
    raise AssertionError("ElasticPagedTokenToKVPoolAllocator was not found")


def test_paged_allocator_accepts_upstream_num_new_pages_hint():
    allocator = _elastic_paged_allocator()
    alloc_extend = next(
        node
        for node in allocator.body
        if isinstance(node, ast.FunctionDef) and node.name == "alloc_extend"
    )

    assert alloc_extend.args.args[-1].arg == "num_new_pages"
    assert isinstance(alloc_extend.args.defaults[-1], ast.Constant)
    assert alloc_extend.args.defaults[-1].value is None


def test_refactored_allocator_kernel_path_is_supported():
    tree = ast.parse(PATCHES.read_text(encoding="utf-8"))
    imports = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
    ]

    assert any(
        node.module == "sglang.srt.mem_cache.triton_ops"
        and any(alias.name == "allocator" for alias in node.names)
        for node in imports
    )


def test_allocator_kernel_launch_adapts_to_upstream_parameters():
    source = PATCHES.read_text(encoding="utf-8")

    assert "inspect.signature(alloc_extend_kernel_fn).parameters" in source
    assert 'if "ret_values" in alloc_extend_param_names:' in source
    assert 'if "max_num_extend_tokens" in alloc_extend_param_names:' in source
