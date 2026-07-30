#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

"""Statically verify upstream engine symbols used by kvcached patches."""

from __future__ import annotations

import argparse
import ast
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Set


@dataclass(frozen=True)
class ModuleContract:
    path: str
    required: Sequence[str]
    optional: Sequence[str] = ()


CONTRACTS: Mapping[str, Sequence[ModuleContract]] = {
    "vllm": (
        ModuleContract("v1/core/block_pool.py", ("KVCacheBlock",)),
        ModuleContract("v1/engine/core.py", ("EngineCore",)),
        ModuleContract(
            "v1/core/kv_cache_coordinator.py",
            ("KVCacheCoordinator",),
        ),
        ModuleContract("v1/core/kv_cache_manager.py", ("KVCacheManager",)),
        ModuleContract("v1/worker/gpu_model_runner.py", ("GPUModelRunner",)),
        ModuleContract("v1/worker/gpu_worker.py", ("Worker",)),
        ModuleContract(
            "v1/kv_cache_interface.py",
            ("FullAttentionSpec",),
            ("MLAAttentionSpec", "MambaSpec", "KVCacheTensor"),
        ),
    ),
    "sglang": (
        ModuleContract(
            "srt/mem_cache/allocator.py",
            (
                "BaseTokenToKVPoolAllocator",
                "alloc_decode_kernel",
                "alloc_extend_kernel",
            ),
        ),
        ModuleContract(
            "srt/mem_cache/memory_pool.py",
            ("KVCache", "MHATokenToKVPool", "MLATokenToKVPool"),
            ("HybridLinearKVPool", "MambaPool"),
        ),
        ModuleContract("srt/managers/scheduler.py", ("Scheduler",)),
        ModuleContract("srt/mem_cache/radix_cache.py", ("RadixCache",)),
    ),
}


@dataclass
class ModuleResult:
    path: str
    missing_required: List[str] = field(default_factory=list)
    optional_present: List[str] = field(default_factory=list)
    symbols: Dict[str, List[str]] = field(default_factory=dict)
    parse_error: str = ""


@dataclass
class CompatibilityResult:
    engine: str
    package_root: str
    status: str
    modules: List[ModuleResult]


def find_package_root(repository: Path, engine: str) -> Path:
    candidates = (
        (repository / "vllm",)
        if engine == "vllm"
        else (repository / "python" / "sglang", repository / "sglang")
    )
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    rendered = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"could not find {engine} package root; checked {rendered}")


def format_arguments(arguments: ast.arguments) -> str:
    positional = [*arguments.posonlyargs, *arguments.args]
    parts = [argument.arg for argument in positional]
    if arguments.vararg is not None:
        parts.append(f"*{arguments.vararg.arg}")
    elif arguments.kwonlyargs:
        parts.append("*")
    parts.extend(argument.arg for argument in arguments.kwonlyargs)
    if arguments.kwarg is not None:
        parts.append(f"**{arguments.kwarg.arg}")
    return f"({', '.join(parts)})"


def assigned_names(node: ast.AST) -> Iterable[str]:
    if isinstance(node, ast.Name):
        yield node.id
    elif isinstance(node, (ast.Tuple, ast.List)):
        for element in node.elts:
            yield from assigned_names(element)


def collect_symbols(tree: ast.Module) -> Dict[str, List[str]]:
    symbols: Dict[str, List[str]] = {}
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            methods = [
                f"{child.name}{format_arguments(child.args)}"
                for child in node.body
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
            ]
            symbols[node.name] = methods
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            symbols[node.name] = [format_arguments(node.args)]
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                symbols[alias.asname or alias.name.split(".")[-1]] = []
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                for name in assigned_names(target):
                    symbols[name] = []
        elif isinstance(node, ast.AnnAssign):
            for name in assigned_names(node.target):
                symbols[name] = []
    return symbols


def inspect_module(package_root: Path, contract: ModuleContract) -> ModuleResult:
    path = package_root / contract.path
    result = ModuleResult(path=str(path))
    if not path.is_file():
        result.missing_required = list(contract.required)
        result.parse_error = "module file not found"
        return result

    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (OSError, SyntaxError, UnicodeError) as exc:
        result.missing_required = list(contract.required)
        result.parse_error = str(exc)
        return result

    result.symbols = collect_symbols(tree)
    present: Set[str] = set(result.symbols)
    result.missing_required = sorted(set(contract.required) - present)
    result.optional_present = sorted(set(contract.optional) & present)
    return result


def check_repository(repository: Path, engine: str) -> CompatibilityResult:
    package_root = find_package_root(repository, engine)
    modules = [
        inspect_module(package_root, contract)
        for contract in CONTRACTS[engine]
    ]
    compatible = all(not module.missing_required for module in modules)
    return CompatibilityResult(
        engine=engine,
        package_root=str(package_root),
        status="compatible" if compatible else "incompatible",
        modules=modules,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", choices=tuple(CONTRACTS), required=True)
    parser.add_argument("--repo", type=Path, default=Path("."))
    parser.add_argument("--report", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        result = check_repository(args.repo.resolve(), args.engine)
        payload = asdict(result)
    except Exception as exc:
        payload = {
            "engine": args.engine,
            "package_root": "",
            "status": "error",
            "modules": [],
            "error": str(exc),
        }

    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    print(rendered, end="")
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(rendered, encoding="utf-8")
    return 0 if payload["status"] == "compatible" else 2


if __name__ == "__main__":
    raise SystemExit(main())
