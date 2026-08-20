# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

import ast
from pathlib import Path


def test_virtual_capacity_patch_runs_after_memory_pool_aliases():
    autopatch_path = (
        Path(__file__).parents[1]
        / "kvcached"
        / "integration"
        / "sglang"
        / "autopatch.py"
    )
    tree = ast.parse(autopatch_path.read_text(encoding="utf-8"))

    patch_sglang = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_patch_sglang"
    )
    registration_calls = [
        node
        for node in ast.walk(patch_sglang)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "register_patches_with_versions"
    ]
    assert len(registration_calls) == 1

    registration = registration_calls[0]
    assert len(registration.args) == 1
    patch_list = registration.args[0]
    assert isinstance(patch_list, ast.List)

    patch_names = []
    patch_versions = {}
    for entry in patch_list.elts:
        assert isinstance(entry, ast.Tuple) and len(entry.elts) == 2
        constructor = entry.elts[0]
        assert isinstance(constructor, ast.Call)
        assert isinstance(constructor.func, ast.Name)
        patch_name = constructor.func.id
        patch_names.append(patch_name)

        version = entry.elts[1]
        if isinstance(version, ast.Constant):
            patch_versions[patch_name] = version.value

    virtual_index = patch_names.index("SGLangVirtualKVCapacityPatch")
    for pool_patch in (
        "ElasticMemoryPoolPatch",
        "ElasticMLAMemoryPoolPatch",
        "ElasticMambaPoolPatch",
        "ElasticHybridLinearKVPoolPatch",
    ):
        assert patch_names.index(pool_patch) < virtual_index

    assert patch_versions["SGLangVirtualKVCapacityPatch"] == ">=0.5.11"
