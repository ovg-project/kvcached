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
])
def test_left_unchanged(vllm_patches, monkeypatch, cfg_kwargs, page_mb):
    cfg = _cache_config(**cfg_kwargs)
    before = dict(vars(cfg))
    _align(vllm_patches, monkeypatch, cfg, page_mb)
    assert vars(cfg) == before


@pytest.mark.parametrize("tokens,per_token,expected,page_mb", [
    (784, 4096, 1024, 4), (1056, 2048, 2048, 4), (528, 4096, 1024, 4),
    (1728, 1280, 1728, 6), (896, 2560, 896, 6),
    (2048, 1280, 2048, 6),  # safe non-divisible pages are allowed by #522
    (1088, 2048, 2048, 4),  # do not chase the original block's 34 MiB tiling page
    (512, 2048, 512, 2),    # default page already holds an exactly tiling block
    (768, 2048, 1024, 2),   # #522 can still align a block smaller than the default page
    (1024, 2048, 1024, 2),  # equal sizes do not trigger page enlargement
])
def test_default_page_and_block_are_selected_together(
        vllm_patches, monkeypatch, tokens, per_token, expected, page_mb):
    from kvcached import utils
    from kvcached.kv_geometry import check_page_geometry

    monkeypatch.delenv("KVCACHED_PAGE_SIZE_MB", raising=False)
    cfg = _cache_config(tokens, tokens * per_token)
    _align(vllm_patches, monkeypatch, cfg, 2)
    assert cfg.block_size == expected
    assert cfg.mamba_page_size_padded == expected * per_token
    assert cfg.mamba_block_size == expected
    actual_page = utils.get_page_size_for_block(cfg.mamba_page_size_padded, 2 * MIB)
    assert actual_page == page_mb * MIB
    assert check_page_geometry(cfg.mamba_page_size_padded, actual_page) is None
    before = dict(vars(cfg))
    _align(vllm_patches, monkeypatch, cfg, 2)
    assert vars(cfg) == before


@pytest.mark.parametrize("flag", ["user_specified_block_size", "user_specified_mamba_block_size"])
def test_default_page_grows_without_overriding_fixed_blocks(vllm_patches, monkeypatch, flag):
    from kvcached.utils import get_page_size_for_block

    monkeypatch.delenv("KVCACHED_PAGE_SIZE_MB", raising=False)
    cfg = _cache_config(2048, 5 * MIB // 2)
    setattr(cfg, flag, True)
    before = dict(vars(cfg))
    _align(vllm_patches, monkeypatch, cfg, 2)
    assert vars(cfg) == before
    assert get_page_size_for_block(cfg.mamba_page_size_padded, 2 * MIB) == 6 * MIB


def test_explicit_small_page_is_not_overridden(vllm_patches, monkeypatch):
    from kvcached.kv_geometry import check_page_geometry
    from kvcached.utils import get_page_size_for_block

    monkeypatch.setenv("KVCACHED_PAGE_SIZE_MB", "2")
    cfg = _cache_config()
    before = dict(vars(cfg))
    _align(vllm_patches, monkeypatch, cfg, 2)
    assert vars(cfg) == before
    page = get_page_size_for_block(cfg.mamba_page_size_padded, 2 * MIB)
    assert page == 2 * MIB
    assert check_page_geometry(cfg.mamba_page_size_padded, page) is not None


@pytest.mark.parametrize("user", [False, True])
@pytest.mark.parametrize("page_mb", [4, 6])
def test_initial_page_holding_block_keeps_upstream_validation(
        vllm_patches, monkeypatch, user, page_mb):
    from kvcached.kv_geometry import check_page_geometry
    from kvcached.utils import get_page_size_for_block

    monkeypatch.setenv("KVCACHED_PAGE_SIZE_MB", str(page_mb))
    cfg = _cache_config(2048, 5 * MIB // 2, user=user)
    before = dict(vars(cfg))
    _align(vllm_patches, monkeypatch, cfg, page_mb)
    assert vars(cfg) == before
    page = get_page_size_for_block(cfg.mamba_page_size_padded, page_mb * MIB)
    assert page == page_mb * MIB
    error = check_page_geometry(cfg.mamba_page_size_padded, page)
    assert (error is not None) == (page_mb == 4)


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
