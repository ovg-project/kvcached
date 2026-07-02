# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

import sys
import types
from importlib.machinery import ModuleSpec
from types import SimpleNamespace


def _install_fake_modules(monkeypatch):
    torch = types.ModuleType("torch")
    torch.__spec__ = ModuleSpec("torch", loader=None)
    torch.Tensor = object
    monkeypatch.setitem(sys.modules, "torch", torch)

    vllm = types.ModuleType("vllm")
    vllm.__path__ = []
    vllm.__version__ = "0.10.0"
    vllm.__spec__ = ModuleSpec("vllm", loader=None, is_package=True)
    monkeypatch.setitem(sys.modules, "vllm", vllm)

    v1 = types.ModuleType("vllm.v1")
    v1.__path__ = []
    v1.__spec__ = ModuleSpec("vllm.v1", loader=None, is_package=True)
    monkeypatch.setitem(sys.modules, "vllm.v1", v1)
    setattr(vllm, "v1", v1)

    kv_cache_interface = types.ModuleType("vllm.v1.kv_cache_interface")
    kv_cache_interface.__spec__ = ModuleSpec(kv_cache_interface.__name__, loader=None)

    class FullAttentionSpec:
        def __init__(self, block_size=16):
            self.block_size = block_size

    class MambaSpec:
        def __init__(self, block_size=4):
            self.block_size = block_size

    kv_cache_interface.FullAttentionSpec = FullAttentionSpec
    kv_cache_interface.MambaSpec = MambaSpec
    monkeypatch.setitem(sys.modules, kv_cache_interface.__name__, kv_cache_interface)
    setattr(v1, "kv_cache_interface", kv_cache_interface)

    utils = types.ModuleType("vllm.utils")
    utils.__path__ = []
    utils.__spec__ = ModuleSpec("vllm.utils", loader=None, is_package=True)
    monkeypatch.setitem(sys.modules, "vllm.utils", utils)

    torch_utils = types.ModuleType("vllm.utils.torch_utils")
    torch_utils.__spec__ = ModuleSpec(torch_utils.__name__, loader=None)
    torch_utils.get_dtype_size = lambda dtype: 1
    monkeypatch.setitem(sys.modules, torch_utils.__name__, torch_utils)
    setattr(utils, "torch_utils", torch_utils)

    return FullAttentionSpec, MambaSpec


def test_hybrid_reshape_uses_shared_pool_mapping_and_updates_layout(monkeypatch):
    FullAttentionSpec, MambaSpec = _install_fake_modules(monkeypatch)

    from kvcached.integration.vllm import patches

    def fake_reshape_mamba(raw_buffer, kv_cache_spec, get_dtype_size):
        return [f"mamba-state:{raw_buffer}"]

    monkeypatch.setattr(patches, "_reshape_mamba_non_contiguous", fake_reshape_mamba)

    class GPUModelRunner:
        def __init__(self):
            self._kvcached_mamba_raw_info = {
                "buffers": ["raw-attn-pool", "raw-mamba-pool"]
            }
            self.hybrid_layout_calls = []

        def _update_hybrid_attention_mamba_layout(self, kv_caches, kernel_block_sizes):
            self.hybrid_layout_calls.append((dict(kv_caches), list(kernel_block_sizes)))

    assert patches.GPUModelRunnerPatch().add_reshape_methods(GPUModelRunner) is True

    runner = GPUModelRunner()
    kv_cache_config = SimpleNamespace(
        runner_only_attn_layers={"attn.runner_only"},
        kv_cache_tensors=[
            SimpleNamespace(shared_by=["attn.0", "attn.runner_only"]),
            SimpleNamespace(shared_by=["mamba.0"]),
        ],
        kv_cache_groups=[
            SimpleNamespace(
                kv_cache_spec=FullAttentionSpec(block_size=16),
                layer_names=["attn.0", "attn.runner_only"],
            ),
            SimpleNamespace(
                kv_cache_spec=MambaSpec(block_size=4),
                layer_names=["mamba.0"],
            ),
        ],
    )

    kv_caches = runner._reshape_kv_cache_tensors_from_kvcached(
        kv_cache_config,
        ["attn-view", "unused-attn-pool"],
    )

    assert kv_caches == {
        "attn.0": "attn-view",
        "mamba.0": ["mamba-state:raw-mamba-pool"],
    }
    assert runner.hybrid_layout_calls == [
        (
            {
                "attn.0": "attn-view",
                "mamba.0": ["mamba-state:raw-mamba-pool"],
            },
            [16, 4],
        )
    ]
