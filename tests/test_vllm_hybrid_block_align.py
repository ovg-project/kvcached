# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0
"""kvcached grows a hybrid attention block so its KV unit tiles the page."""
# ruff: noqa: F811
import types
from unittest import mock

import pytest
from test_vllm_pool_exhaustion import vllm_patches  # noqa: F401

MIB = 1024 * 1024


def _cache_config(block_size=784, padded=784 * 4096, mode="align", user=False):
    return types.SimpleNamespace(
        block_size=block_size, mamba_page_size_padded=padded, mamba_cache_mode=mode,
        mamba_block_size=block_size if mode == "align" else None,
        user_specified_block_size=user, user_specified_mamba_block_size=False)


def _align(vllm_patches, monkeypatch, cfg, page_mb):
    import kvcached.utils
    monkeypatch.setattr(kvcached.utils, "PAGE_SIZE", page_mb * MIB)
    vllm_patches._align_block_size_to_kvcached_page(cfg, mock.Mock())
    return cfg


@pytest.mark.parametrize("mode", ["align", "all"])
def test_qwen38_27b_block_grows_to_tile_a_4mib_page(vllm_patches, monkeypatch, mode):
    cfg = _align(vllm_patches, monkeypatch, _cache_config(mode=mode), 4)
    assert cfg.block_size == 1024
    assert cfg.mamba_page_size_padded == 4 * MIB
    assert cfg.mamba_block_size == 1024


def test_mamba_block_size_untouched_without_prefix_caching(vllm_patches, monkeypatch):
    cfg = _align(vllm_patches, monkeypatch, _cache_config(mode="none"), 4)
    assert cfg.block_size == 1024 and cfg.mamba_block_size is None


@pytest.mark.parametrize("cfg_kwargs,page_mb", [
    ({"user": True}, 4),                            # user-chosen block size is kept
    ({"block_size": 1024, "padded": 4 * MIB}, 4),   # already tiles
    ({"padded": None}, 4),                          # not a hybrid model
    ({}, 2),                                        # no block fits a 2 MiB page
])
def test_left_unchanged(vllm_patches, monkeypatch, cfg_kwargs, page_mb):
    cfg = _cache_config(**cfg_kwargs)
    before = dict(vars(cfg))
    _align(vllm_patches, monkeypatch, cfg, page_mb)
    assert vars(cfg) == before


def test_patch_wraps_the_platform_classmethod(vllm_patches, monkeypatch):
    calls = []

    class Platform:
        @classmethod
        def _align_hybrid_block_size(cls, vllm_config, backend_cls):
            calls.append(backend_cls)
            vllm_config.cache_config.block_size = 784
            vllm_config.cache_config.mamba_page_size_padded = 784 * 4096

    class CudaPlatform(Platform):
        pass

    module = types.ModuleType("vllm.platforms.interface")
    module.Platform = Platform  # type: ignore[attr-defined]
    patch = vllm_patches.HybridBlockSizeAlignPatch()
    monkeypatch.setattr(patch.version_manager, "detect_version", lambda _name: "0.29.0")
    monkeypatch.setattr(vllm_patches, "enable_kvcached", lambda: True)
    import kvcached.utils
    monkeypatch.setattr(kvcached.utils, "PAGE_SIZE", 4 * MIB)

    assert patch.apply(module)
    assert patch.apply(module)  # idempotent
    vllm_config = types.SimpleNamespace(cache_config=_cache_config())
    CudaPlatform._align_hybrid_block_size(vllm_config, "backend")
    assert calls == ["backend"]
    assert vllm_config.cache_config.block_size == 1024


def test_disabled_kvcached_keeps_native_behavior(vllm_patches, monkeypatch):
    class Platform:
        @classmethod
        def _align_hybrid_block_size(cls, vllm_config, backend_cls):
            vllm_config.cache_config.block_size = 784

    module = types.ModuleType("vllm.platforms.interface")
    module.Platform = Platform  # type: ignore[attr-defined]
    patch = vllm_patches.HybridBlockSizeAlignPatch()
    monkeypatch.setattr(patch.version_manager, "detect_version", lambda _name: "0.29.0")
    monkeypatch.setattr(vllm_patches, "enable_kvcached", lambda: False)
    assert patch.apply(module)
    vllm_config = types.SimpleNamespace(cache_config=_cache_config())
    Platform._align_hybrid_block_size(vllm_config, None)
    assert vllm_config.cache_config.block_size == 784


def test_release_without_the_hook_is_a_noop(vllm_patches, monkeypatch):
    module = types.ModuleType("vllm.platforms.interface")
    module.Platform = type("Platform", (), {})  # type: ignore[attr-defined]
    patch = vllm_patches.HybridBlockSizeAlignPatch()
    monkeypatch.setattr(patch.version_manager, "detect_version", lambda _name: "0.25.1")
    assert patch.apply(module)
