# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0
"""vLLM 0.28 native prefix contracts and logical byte-page copies.

Run separately from tests that replace torch/vLLM globally with mocks.
"""

import hashlib
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from kvcached.integration.vllm import interfaces, patches


class Slots:
    def __init__(self, count=24):
        self.ids = list(range(1, count))

    def alloc(self, count):
        result, self.ids = self.ids[:count], self.ids[count:]
        return result

    def free(self, ids):
        self.ids.extend(ids)

    def available_size(self):
        return len(self.ids)


@pytest.fixture
def pool(monkeypatch, request):
    from vllm.v1.core import block_pool

    if not hasattr(block_pool.BlockPool, "cache_partial_block"):
        pytest.skip("Requires vLLM's partial-prefix contract")
    module = ModuleType("prefix_pool_test")
    setattr(module, "BlockPool", block_pool.BlockPool)
    setattr(module, "KVCacheBlock", block_pool.KVCacheBlock)
    assert patches.ElasticBlockPoolPatch().apply(module)
    monkeypatch.setattr(interfaces, "get_kv_cache_manager", lambda *a, **k: Slots())
    block_size, hash_block_size = getattr(request, "param", (64, 16))
    return module.ElasticBlockPool(
        24, block_size, 16, 1, True, hash_block_size=hash_block_size
    )


def request():
    return SimpleNamespace(block_hashes=[bytes([i]) * 32 for i in range(1, 9)])


@pytest.mark.parametrize("release", ["lru", "explicit", "reset"])
def test_partial_aliases_are_removed_with_their_owner(pool, release):
    req = request()
    block = pool.get_new_blocks(1)[0]
    pool.cache_partial_block(req, block, 48, 0, 64)
    pool.cache_partial_block(req, block, 32, 0, 64)
    assert pool.get_cached_block(req.block_hashes[2], [0]) == [block]
    assert pool.get_cached_block(req.block_hashes[1], [0]) == [block]
    assert block.block_id in pool.cached_block_hashes_by_block
    pool.free_blocks([block])
    if release == "lru":
        pool._evict_blocks_from_pool(1)
    elif release == "explicit":
        pool.evict_blocks({block.block_id})
    else:
        assert pool.reset_prefix_cache()
    assert not pool._cached_blocks
    assert not pool.cached_block_hashes_by_block
    assert block.block_hash is None
    assert block.block_hash_num_tokens is None


def test_partial_to_full_promotion_uses_hash_granularity(pool):
    req = request()
    block = pool.get_new_blocks(1)[0]
    pool.cache_partial_block(req, block, 48, 0, 64)
    pool.cache_full_blocks(req, [block], 0, 1, 64, 0)
    assert pool.get_cached_block(req.block_hashes[2], [0]) is None
    assert pool.get_cached_block(req.block_hashes[3], [0]) == [block]
    assert block.block_hash_num_tokens == 64
    assert not pool.cached_block_hashes_by_block


@pytest.mark.parametrize("pool", [(16, 4), (64, 16)], indirect=True)
def test_full_blocks_use_complete_chained_prefix_hashes(pool):
    from vllm.v1.core.kv_cache_utils import hash_block_tokens

    hash_size = pool.hash_block_size
    block_size = hash_size * 4

    def hashes(tokens):
        parent = b"prefix-test-parent"
        result = []
        for start in range(0, len(tokens), hash_size):
            parent = hash_block_tokens(
                lambda value: hashlib.sha256(repr(value).encode()).digest(),
                parent, tokens[start:start + hash_size],
            )
            result.append(parent)
        return result

    tokens = list(range(block_size * 2))
    req = SimpleNamespace(block_hashes=hashes(tokens))
    blocks = pool.get_new_blocks(2)
    pool.cache_full_blocks(req, blocks, 0, 1, block_size, 0)
    pool.cache_full_blocks(req, blocks, 1, 2, block_size, 0)

    for index, block in enumerate(blocks):
        assert pool.get_cached_block(req.block_hashes[(index + 1) * 4 - 1], [0]) == [block]
        assert block.block_hash_num_tokens == (index + 1) * block_size
    assert pool.get_cached_block(req.block_hashes[0], [0]) is None
    changed = tokens.copy()
    changed[hash_size] += 1000
    changed_hashes = hashes(changed)
    assert changed_hashes[0] == req.block_hashes[0]
    assert changed_hashes[3] != req.block_hashes[3]
    assert pool.get_cached_block(changed_hashes[3], [0]) is None


def test_move_preserves_aliases_and_duplicate_owners(pool):
    req = request()
    source, other, destination = pool.get_new_blocks(3)
    for block in (source, other):
        pool.cache_partial_block(req, block, 48, 0, 64)
        pool.cache_partial_block(req, block, 32, 0, 64)
    pool.move_block_hashes(source, destination)
    assert source.block_hash is None
    assert destination.block_hash_num_tokens == 48
    assert source.block_id not in pool._block_id_to_key
    assert destination.block_id in pool._block_id_to_key
    pool.evict_blocks({other.block_id})
    assert pool.get_cached_block(req.block_hashes[2], [0]) == [destination]
    assert pool.get_cached_block(req.block_hashes[1], [0]) == [destination]
    assert not pool.reset_prefix_cache()
    pool.free_blocks([source, other, destination])
    assert pool.reset_prefix_cache()


def test_coordinator_keeps_native_hash_size_before_subclass_assignment(monkeypatch):
    from vllm.v1.core import block_pool

    class Coordinator(SimpleNamespace):
        pass

    module = ModuleType("coordinator_test")
    setattr(module, "KVCacheCoordinator", Coordinator)
    adaptation = patches.KVCacheCoordinatorPatch()
    assert adaptation.patch_coordinator(module)
    config = SimpleNamespace(num_blocks=24)
    spec = SimpleNamespace(block_size=64)
    monkeypatch.setattr(patches, "_validate_kv_cache_groups", lambda _: None)
    monkeypatch.setattr(patches, "_get_first_attention_group",
                        lambda _: SimpleNamespace(kv_cache_spec=spec))
    monkeypatch.setattr(patches, "_infer_attention_type", lambda _: "MHA")
    monkeypatch.setattr(patches, "_get_kv_cache_params", lambda *a, **k: (16, 2))
    monkeypatch.setattr(patches, "_get_group_size", lambda _: 1)
    monkeypatch.setattr(interfaces, "init_kvcached", Mock())
    monkeypatch.setattr(interfaces, "get_world_size", lambda: 1)
    constructor = Mock(return_value=SimpleNamespace(null_block=object()))
    monkeypatch.setattr(block_pool, "ElasticBlockPool", constructor, raising=False)
    coordinator = Coordinator()
    coordinator.block_pool = SimpleNamespace(hash_block_size=16)
    coordinator.kv_cache_config = config
    coordinator.single_type_managers = []
    assert not hasattr(coordinator, "hash_block_size")
    coordinator._setup_kvcached_coordinator()
    assert constructor.call_args.kwargs["hash_block_size"] == 16


@pytest.mark.parametrize("contiguous", [False, True])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_copy_uses_logical_pages_and_deduplicates_shared_storage(contiguous, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    blocks, layers, page_bytes = 4, 3, 64
    stride = page_bytes * (layers if contiguous else 1)
    backings = [torch.full((blocks * stride + 49,), 77, dtype=torch.int8, device=device)
                for _ in range(1 if contiguous else layers)]
    info = dict(buffers=[raw[16:16 + blocks * stride] for raw in backings],
                num_blocks=blocks, num_pools=layers, page_size_bytes=page_bytes,
                is_contiguous=contiguous, block_stride_bytes=stride)
    views = [patches._reshape_mamba_page_tensor(
        info, SimpleNamespace(page_size_bytes=page_bytes), i) for i in range(layers)]
    for layer, view in enumerate(views):
        for block in range(blocks):
            view[block].fill_(layer * 10 + block)
    before = [view.clone() for view in views]
    # Swapping catches duplicate copies through shared layer aliases.
    interfaces._copy_kv_cache_blocks(views, blocks, [(0, 1), (1, 0)])
    for original, view in zip(before, views):
        assert torch.equal(view[0], original[1])
        assert torch.equal(view[1], original[0])
        assert torch.equal(view[2:], original[2:])
    for backing in backings:
        assert (backing[:16] == 77).all()
        assert (backing[16 + blocks * stride:] == 77).all()


@pytest.mark.parametrize("copies", [[(-1, 0)], [(0, 4)], [(0, 1), (2, 1)]])
def test_invalid_copy_does_not_write_any_pool(copies):
    raw = torch.arange(64, dtype=torch.uint8)
    tensor = raw.view(4, 16)
    interfaces._set_block_copy_view(tensor, raw, 4, 16)
    expected = raw.clone()
    with pytest.raises(ValueError):
        interfaces._copy_kv_cache_blocks([tensor], 4, copies)
    assert torch.equal(raw, expected)


def test_missing_metadata_fails_before_copying_first_pool():
    raw = torch.arange(64, dtype=torch.uint8)
    tensor = raw.view(4, 16)
    interfaces._set_block_copy_view(tensor, raw, 4, 16)
    expected = raw.clone()
    with pytest.raises(ValueError, match="metadata"):
        interfaces._copy_kv_cache_blocks([tensor, torch.zeros(1)], 4, [(0, 1)])
    assert torch.equal(raw, expected)


@pytest.mark.parametrize("layout", ["mla", "split", "unified", "contiguous"])
def test_legacy_layout_copy_preserves_logical_blocks(monkeypatch, layout):
    monkeypatch.setattr(interfaces, "_kvcached_initialized", True)
    monkeypatch.setattr(interfaces, "_contiguous_layout", layout == "contiguous")
    monkeypatch.setattr(interfaces, "PAGE_SIZE", 256)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_properties",
                        lambda _: SimpleNamespace(total_memory=24 * 1024))

    def create(size, itemsize, device, num_layers, **kwargs):
        return [torch.zeros(size * (num_layers if layout == "contiguous" else 1),
                            dtype=torch.uint8)
                for _ in range(1 if layout == "contiguous" else num_layers)]

    monkeypatch.setattr(interfaces, "create_kv_tensors", create)
    shape = (8, 16, 32) if layout == "mla" else (8, 2, 16, 2, 16)
    kind = "MLA" if layout == "mla" else (
        "HYBRID_LINEAR" if layout == "unified" else "MHA")
    result = interfaces.alloc_kv_cache(shape, 16, torch.float16, "cuda:0", 3,
                                       attention_type=kind)
    views = result[0] if kind == "HYBRID_LINEAR" else result
    for layer, view in enumerate(views):
        view[0].fill_(layer + 1)
        view[1].fill_(layer + 11)
    before = [view.clone() for view in views]
    interfaces._copy_kv_cache_blocks(views, views[0].shape[0], [(0, 1), (1, 0)])
    for view, original in zip(views, before):
        assert torch.equal(view[0], original[1])
        assert torch.equal(view[1], original[0])


def test_copy_patch_disabled_path_preserves_native_helper(monkeypatch):
    native = Mock()
    module = ModuleType("runner_copy_test")
    setattr(module, "copy_kv_cache_blocks_inplace", native)
    adaptation = patches.GPUModelRunnerPatch()
    adaptation.detected_version = "0.28.0"
    assert adaptation.patch_block_copy(module)
    monkeypatch.setattr(patches, "enable_kvcached", lambda: False)
    getattr(module, "copy_kv_cache_blocks_inplace")([], 4, [(0, 1)])
    native.assert_called_once_with([], 4, [(0, 1)])
