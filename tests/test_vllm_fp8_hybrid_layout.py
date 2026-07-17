# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch


def test_native_hybrid_layout_update_supports_old_and_new_signatures():
    from kvcached.integration.vllm.patches import _call_hybrid_layout_update

    kv_caches = {"attn.0": object()}
    kernel_block_sizes = (16,)
    calls = []

    def legacy_update(caches):
        calls.append(("legacy", caches, None))

    def current_update(caches, block_sizes):
        calls.append(("current", caches, block_sizes))

    _call_hybrid_layout_update(legacy_update, kv_caches, kernel_block_sizes)
    _call_hybrid_layout_update(current_update, kv_caches, kernel_block_sizes)

    assert calls == [
        ("legacy", kv_caches, None),
        ("current", kv_caches, kernel_block_sizes),
    ]


def _assert_fp8_hybrid_flashinfer_layout():
    fp8_dtype = getattr(torch, "float8_e4m3fn", None)
    assert fp8_dtype is not None

    vmm_ops = ModuleType("kvcached.vmm_ops")
    setattr(vmm_ops, "PageAllocator", type("PageAllocator", (), {}))
    setattr(vmm_ops, "InternalPage", type("InternalPage", (), {}))
    for name in (
        "create_kv_tensors",
        "init_kvcached",
        "kv_tensors_created",
        "map_to_kv_tensors",
        "shutdown_kvcached",
        "unmap_from_kv_tensors",
    ):
        setattr(vmm_ops, name, lambda *args, **kwargs: None)
    sys.modules["kvcached.vmm_ops"] = vmm_ops

    from kvcached.integration.vllm import interfaces

    interfaces._contiguous_layout = False
    interfaces._kvcached_initialized = True
    interfaces.normalize_gpu_device = lambda device: device
    interfaces.torch.cuda.is_available = lambda: True
    interfaces.torch.cuda.get_device_properties = lambda device: SimpleNamespace(
        total_memory=4 * interfaces.PAGE_SIZE
    )

    def create_cpu_kv_tensors(
        ftensor_bytes_per_layer,
        dtype_itemsize,
        device,
        num_layers,
        **kwargs,
    ):
        assert kwargs["unified_pool"] is True
        return [
            torch.zeros(
                ftensor_bytes_per_layer // dtype_itemsize,
                dtype=torch.uint8,
            )
            for _ in range(num_layers)
        ]

    interfaces.create_kv_tensors = create_cpu_kv_tensors

    num_blocks = 3
    block_size = 2
    num_heads = 1
    head_size = 4
    hidden_size = block_size * num_heads * head_size
    elements_per_block = 2 * hidden_size
    views, raw_info = interfaces.alloc_kv_cache(
        (num_blocks, 2, block_size, num_heads, head_size),
        block_size,
        fp8_dtype,
        "cuda:0",
        num_layers=1,
        attention_type="HYBRID_LINEAR",
        kv_layout="NHD",
    )
    page_size_bytes = raw_info["page_size_bytes"]

    view = views[0]
    assert view.dtype == fp8_dtype
    assert view.shape[0] >= num_blocks
    assert view.shape[1:] == (2, block_size, num_heads, head_size)
    assert view.stride() == (elements_per_block, hidden_size, head_size, head_size, 1)
    assert page_size_bytes == elements_per_block * fp8_dtype.itemsize

    for block_idx in range(num_blocks):
        k_byte_offset = (
            view[block_idx, 0].storage_offset() * fp8_dtype.itemsize
        )
        v_byte_offset = (
            view[block_idx, 1].storage_offset() * fp8_dtype.itemsize
        )
        assert k_byte_offset == block_idx * page_size_bytes
        assert v_byte_offset == k_byte_offset + hidden_size * fp8_dtype.itemsize


def test_fp8_hybrid_flashinfer_view_interleaves_kv_per_block():
    """FP8 hybrid K/V must share each physical block, not split the pool."""
    if getattr(torch, "float8_e4m3fn", None) is None:
        pytest.skip("torch does not provide float8_e4m3fn")

    test_file = Path(__file__).resolve()
    repo_root = test_file.parent.parent
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        filter(None, (str(repo_root), env.get("PYTHONPATH")))
    )
    result = subprocess.run(
        [sys.executable, str(test_file)],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_vllm_022_hybrid_reshape_uses_shared_pools_and_native_layout(monkeypatch):
    from kvcached.integration.vllm import patches

    torch_utils = ModuleType("vllm.utils.torch_utils")
    setattr(torch_utils, "get_dtype_size", lambda dtype: 1)
    vllm_utils = ModuleType("vllm.utils")
    setattr(vllm_utils, "torch_utils", torch_utils)
    vllm = ModuleType("vllm")
    setattr(vllm, "utils", vllm_utils)
    monkeypatch.setitem(sys.modules, "vllm", vllm)
    monkeypatch.setitem(sys.modules, "vllm.utils", vllm_utils)
    monkeypatch.setitem(sys.modules, "vllm.utils.torch_utils", torch_utils)

    attention_spec = object()
    mamba_spec = object()
    monkeypatch.setattr(patches, "_is_mamba_spec", lambda spec: spec is mamba_spec)
    monkeypatch.setattr(
        patches,
        "_reshape_mamba_non_contiguous",
        lambda raw, spec, get_dtype_size: f"state:{raw}",
    )
    monkeypatch.setattr(patches, "enable_kvcached", lambda: True)

    kv_cache_config = SimpleNamespace(
        kv_cache_tensors=[
            SimpleNamespace(shared_by=("attn.1", "mamba.1")),
            SimpleNamespace(shared_by=("attn.0", "mamba.0")),
        ],
        kv_cache_groups=[
            SimpleNamespace(
                kv_cache_spec=attention_spec,
                layer_names=("attn.0", "attn.1"),
            ),
            SimpleNamespace(
                kv_cache_spec=mamba_spec,
                layer_names=("mamba.0", "mamba.1"),
            ),
        ],
    )

    class FakeGPUModelRunner:
        kv_cache_config = None
        _kvcached_mamba_raw_info = None

        def _reshape_kv_cache_tensors(self, *args, **kwargs):
            raise AssertionError("native reshape should be patched")

        def _update_hybrid_attention_mamba_layout(self, kv_caches, block_sizes):
            self.native_layout_call = (dict(kv_caches), block_sizes)

    patch = patches.GPUModelRunnerPatch()
    assert patch.add_reshape_methods(FakeGPUModelRunner) is True
    assert patch.patch_reshape_methods(FakeGPUModelRunner) is True

    runner = FakeGPUModelRunner()
    runner.kv_cache_config = kv_cache_config
    runner._kvcached_mamba_raw_info = {
        "buffers": ("pool-0", "pool-1"),
        "is_contiguous": False,
    }
    raw_attention_pools = ("attn-pool-0", "attn-pool-1")
    kernel_block_sizes = (16, 16)

    kv_caches = runner._reshape_kv_cache_tensors(
        raw_attention_pools,
        kernel_block_sizes,
    )

    assert kv_caches == {
        "attn.0": "attn-pool-1",
        "attn.1": "attn-pool-0",
        "mamba.0": "state:pool-1",
        "mamba.1": "state:pool-0",
    }
    assert runner.native_layout_call == (kv_caches, kernel_block_sizes)


def test_legacy_hybrid_reshape_keeps_config_signature(monkeypatch):
    from kvcached.integration.vllm import patches

    vllm_utils = ModuleType("vllm.utils")
    setattr(vllm_utils, "get_dtype_size", lambda dtype: 1)
    vllm = ModuleType("vllm")
    setattr(vllm, "utils", vllm_utils)
    monkeypatch.setitem(sys.modules, "vllm", vllm)
    monkeypatch.setitem(sys.modules, "vllm.utils", vllm_utils)

    attention_spec = object()
    monkeypatch.setattr(patches, "_is_mamba_spec", lambda spec: False)
    monkeypatch.setattr(patches, "enable_kvcached", lambda: True)
    kv_cache_config = SimpleNamespace(
        kv_cache_tensors=[SimpleNamespace(shared_by=("attn.0",))],
        kv_cache_groups=[
            SimpleNamespace(
                kv_cache_spec=attention_spec,
                layer_names=("attn.0",),
            )
        ],
    )

    class FakeGPUModelRunner:
        def _reshape_kv_cache_tensors(self, *args, **kwargs):
            raise AssertionError("native reshape should be patched")

    patch = patches.GPUModelRunnerPatch()
    assert patch.add_reshape_methods(FakeGPUModelRunner) is True
    assert patch.patch_reshape_methods(FakeGPUModelRunner) is True

    runner = FakeGPUModelRunner()
    assert runner._reshape_kv_cache_tensors(
        kv_cache_config=kv_cache_config,
        kv_cache_raw_tensors=("attn-pool-0",),
    ) == {"attn.0": "attn-pool-0"}


if __name__ == "__main__":
    _assert_fp8_hybrid_flashinfer_layout()
