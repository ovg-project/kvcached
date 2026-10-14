# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

"""Native byte-storage and physical-page boundaries for DeepSeek-V4 SWA."""

import copy
import gc
import importlib
import importlib.metadata
import types
import weakref
from unittest import mock

import pytest
from test_vmm_transaction import _compiled_vmm_ops


@pytest.mark.parametrize(
    ("num_pages", "bytes_per_page", "physical_page_size"),
    [(n, 8192, 2 * 1024 * 1024) for n in (255, 256, 257, 513)]
    + [(5, 4 * 1024 * 1024, 4 * 1024 * 1024),
       (5, 5 * 1024 * 1024 // 2, 6 * 1024 * 1024)],
)
def test_dsv4_unsigned_views_and_null_block_lifecycle(
    monkeypatch, num_pages, bytes_per_page, physical_page_size
):
    _compiled_vmm_ops()
    import torch

    from kvcached import kv_cache_manager
    from kvcached.integration.sglang import interfaces

    assert interfaces.shutdown_kvcached()
    monkeypatch.delenv("KVCACHED_PAGE_SIZE_MB", raising=False)
    monkeypatch.setattr(interfaces, "PAGE_SIZE", 2 * 1024 * 1024)
    monkeypatch.setattr(interfaces, "_contiguous_layout", False)
    monkeypatch.setattr(kv_cache_manager, "CONTIGUOUS_LAYOUT", False)
    monkeypatch.setattr(kv_cache_manager, "PAGE_PREALLOC_ENABLED", False)
    interfaces.init_kvcached(device="cuda:0", async_sched=False)
    manager = None
    try:
        kwargs = dict(num_pages=num_pages, bytes_per_page=bytes_per_page, num_layers=1,
                      device="cuda:0", group_id=20011)
        if num_pages == 255:
            with pytest.raises(ValueError, match="at least one physical page"):
                interfaces.alloc_dsv4_swa_cache(**kwargs)
            return
        buffers, raw = interfaces.alloc_dsv4_swa_cache(**kwargs)
        assert buffers[0].dtype == torch.uint8
        assert buffers[0].data_ptr() == raw[0].data_ptr()
        assert buffers[0].shape == (num_pages, bytes_per_page)
        manager = interfaces.get_kv_cache_manager(
            num_pages, 1, bytes_per_page, 1, reserve_null_block=True, num_kv_buffers=1,
            group_id=20011, pool_name="dsv4_swa_kv")
        manager.wait_ready(timeout=8)
        assert manager.page_size == physical_page_size
        capacity = manager.available_size()
        # available_size() is an upper bound for non-tiling blocks. Two
        # usable blocks cross the first physical-page boundary in this case.
        count = 2 if physical_page_size % bytes_per_page else min(257, capacity)
        for _ in range(2):
            blocks = manager.alloc(count)
            assert blocks and 0 not in blocks
            assert max(blocks) < num_pages
            if num_pages == 513 or bytes_per_page > 2 * 1024 * 1024:
                assert max(blocks) * bytes_per_page >= physical_page_size
            for block in blocks:
                buffers[0][block].fill_(237)
            torch.cuda.synchronize()
            for block in blocks:
                assert torch.all(buffers[0][block] == 237).item()
            manager.free(blocks)
            # Releasing whole pages restores the initial capacity upper bound;
            # for non-tiling blocks it need not increase by len(blocks).
            assert manager.available_size() == capacity
        manager.clear()
        manager.wait_ready(timeout=8)
        assert manager.available_size() == capacity
    finally:
        if manager is not None:
            assert manager.shutdown()
        assert interfaces.shutdown_kvcached()


@pytest.mark.parametrize("enabled", [False, True], ids=["disabled", "enabled"])
@pytest.mark.parametrize("compress_ratio,fp4,with_state",
                         [(4, False, False), (4, True, False), (128, False, False), (4, False, True)])
def test_dsv4_real_runner_shared_draft_ownership(monkeypatch, enabled, compress_ratio, fp4, with_state):
    """Real SGLang pool construction and allocation; no model weights/kernels."""
    monkeypatch.setenv("ENABLE_KVCACHED", "true" if enabled else "false")
    _compiled_vmm_ops()
    pytest.importorskip("sglang")
    if importlib.metadata.version("sglang") != "0.5.20":
        pytest.skip("This native construction regression is pinned to SGLang 0.5.20")
    import torch

    from kvcached import kv_cache_manager
    from kvcached.integration.sglang import interfaces, patches

    assert interfaces.shutdown_kvcached()
    monkeypatch.setattr(interfaces, "_contiguous_layout", False)
    monkeypatch.setattr(patches, "CONTIGUOUS_LAYOUT", False)
    monkeypatch.setattr(kv_cache_manager, "CONTIGUOUS_LAYOUT", False)
    monkeypatch.setattr(kv_cache_manager, "PAGE_PREALLOC_ENABLED", False)
    monkeypatch.setattr(patches, "_resolve_sglang_parallel_context", lambda: (0, 1, 0))
    monkeypatch.setattr("sglang.srt.distributed.get_tensor_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr("sglang.srt.distributed.get_pipeline_model_parallel_world_size", lambda: 1)
    monkeypatch.setenv("SGLANG_OPT_USE_ONLINE_COMPRESS", "0")
    manager = None
    compressed_pools = []
    state_pools = []
    swa_size = 131072 if with_state else 16384
    state_size = 8192 if with_state else 64
    full_size = 524288 if compress_ratio == 128 else 262144
    c128_size = full_size // 128
    c4_size = full_size // 4
    try:
        # Match autopatch order: ModelRunner imports capture the pool aliases.
        for patch_class in (
            patches.ElasticAllocatorPatch, patches.ElasticSWAAllocatorPatch,
            patches.DeepSeekV4RuntimeReservationPatch, patches.DeepSeekV4KVPoolPatch,
            patches.DeepSeekV4SWAAllocatorPatch, patches.DeepSeekV4RunnerOwnershipPatch,
        ):
            patch = patch_class()
            assert patch.apply(importlib.import_module(patch.target_module))

        from sglang.srt.mem_cache.kv_cache_configurator import KVCacheConfigurator
        from sglang.srt.model_executor.model_runner import ModelRunner
        from sglang.srt.model_executor.pool_configurator import MemoryPoolConfig
        from sglang.srt.server_args import (
            ServerArgs,
            set_global_server_args_for_scheduler,
        )
        from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
        from transformers import PretrainedConfig

        # Native 0.5.20 inputs are a msgspec record. Resolution/model discovery
        # is outside this fixed-budget pool-construction regression.
        server_args = ServerArgs(model_path="unused-native-pool-fixture")
        monkeypatch.setattr(ServerArgs, "resolve_once", lambda self: None)
        server_args.page_size = 256
        server_args.enable_deepseek_v4_fp4_indexer = fp4
        server_args.speculative_algorithm = "EAGLE"
        server_args.speculative_num_draft_tokens = 4
        server_args.speculative_eagle_topk = 1
        server_args.enable_unified_memory = False
        server_args.enable_memory_saver = False
        server_args.enable_page_major_kv_layout = False
        server_args.prefill_only_disable_kv_cache = False
        server_args.disaggregation_mode = "null"
        set_global_server_args_for_scheduler(server_args)

        # Use the actual 0.5.20 runner/configurator construction path with a
        # fixed small pool budget. Profiling, model weights, and post-pool model
        # components are outside this storage/allocator regression.
        monkeypatch.setattr(
            KVCacheConfigurator, "_resolve_memory_pool_config",
            lambda self, _memory: self.memory_pool_config,
        )
        monkeypatch.setattr(ModelRunner, "_init_post_memory_pool_components", lambda self: None)
        # No graph capture or memory profiling is part of this fixed-budget
        # storage test. Keep native configurator/pool construction below.
        monkeypatch.setattr(
            importlib.import_module("sglang.srt.model_executor.model_runner"),
            "is_post_capture_kv_active", lambda *args, **kwargs: False,
        )

        def make_runner(target=None):
            runner = object.__new__(ModelRunner)
            runner.server_args = server_args
            hf_config = PretrainedConfig(architectures=["DeepseekV4ForCausalLM"])
            runner.model_config = types.SimpleNamespace(
                hf_config=hf_config, hf_text_config=hf_config,
                linear_attn_registry_result=None,
                context_len=32768, window_size=128, qk_nope_head_dim=448,
                qk_rope_head_dim=64, index_head_dim=128, compress_ratios=[compress_ratio],
            )
            runner.is_draft_worker = target is not None
            runner.req_to_token_pool = target.req_to_token_pool if target else None
            runner.token_to_kv_pool_allocator = (
                target.token_to_kv_pool_allocator if target else None
            )
            runner.device = "cuda"
            runner.dtype = torch.bfloat16
            runner.kv_cache_dtype = torch.float8_e4m3fn
            runner.c4_state_dtype = runner.c128_state_dtype = torch.bfloat16
            runner.max_running_requests = 2
            runner.page_size = 256
            runner.max_total_num_tokens = swa_size
            runner.full_max_total_num_tokens = full_size
            runner.swa_max_total_num_tokens = swa_size
            runner.c4_max_total_num_tokens = c4_size
            runner.c128_max_total_num_tokens = c128_size
            runner.c4_state_pool_size = state_size
            runner.c128_state_pool_size = 64
            runner.num_effective_layers = 1
            runner.start_layer, runner.end_layer = 0, 1
            runner.is_hybrid_swa = runner.is_hybrid_swa_compress = True
            runner.enable_hisparse = False
            runner.use_mla_backend = True
            runner.gpu_id = 0
            runner.ps = types.SimpleNamespace(tp_size=1, attn_dp_size=1, pp_size=1)
            runner.pp_group = None
            runner.model = None
            runner.kv_cache_dtype_str = "fp8_e4m3"
            runner.sliding_window_size = 128
            runner.spec_algorithm = SpeculativeAlgorithm.EAGLE
            runner.spec_aux_config = types.SimpleNamespace()
            runner.layer_info = types.SimpleNamespace(
                num_effective_layers=1, start_layer=0, end_layer=1,
            )
            runner.forward_stream = None
            runner.draft_model_idx = None
            runner.pre_model_load_memory = 0
            runner.memory_pool_config = (
                copy.deepcopy(target.memory_pool_config) if target else MemoryPoolConfig(
                    max_total_num_tokens=swa_size, max_running_requests=2,
                    full_max_total_num_tokens=full_size, swa_max_total_num_tokens=swa_size,
                    c4_max_total_num_tokens=c4_size, c128_max_total_num_tokens=c128_size,
                    c4_state_pool_size=state_size, c128_state_pool_size=64,
                )
            )
            return runner

        if not enabled:
            monkeypatch.setattr(patches, "_new_tp_scoped_kvcached_allocator",
                                mock.Mock(side_effect=AssertionError("Disabled backing was initialized")))
        target = make_runner()
        target.alloc_memory_pool()
        if not enabled:
            pools = [getattr(target.token_to_kv_pool, name, None) for name in
                     ("swa_kv_pool", "c4_kv_pool", "c128_kv_pool", "c4_indexer_kv_pool")]
            for name in ("compress_state_pools", "indexer_compress_state_pools"):
                pools.extend(getattr(target.token_to_kv_pool, name, None) or ())
            assert all(not getattr(pool, "_kvcached_managed", False) for pool in pools if pool is not None)
            assert not hasattr(target.req_to_token_pool, "_kvcached_dsv4_request_state")
            assert target.req_to_token_pool.available_size() == target.req_to_token_pool.size
            return
        manager = target.token_to_kv_pool.swa_kv_pool.kvcached_allocator
        manager.wait_ready(timeout=8)
        if with_state:
            state_pools = [target.token_to_kv_pool.compress_state_pools[0],
                           target.token_to_kv_pool.indexer_compress_state_pools[0]]
            assert all(p._kvcached_managed for p in state_pools)
            for pool in state_pools:
                assert torch.all(pool.kv_score_buffer[-1].kv == 0).item()
                assert torch.all(torch.isneginf(pool.kv_score_buffer[-1].score)).item()
            breakdown = target.token_to_kv_pool._kvcached_runtime_reservation_breakdown
            assert breakdown["dsv4.compress_state_pools"] == 0
            assert breakdown["dsv4.indexer_compress_state_pools"] == 0
        compressed_names = (["c128_kv_pool"] if compress_ratio == 128 else
                            ["c4_kv_pool", "c4_indexer_kv_pool"])
        for name in compressed_names:
            pool = getattr(target.token_to_kv_pool, name)
            compressed_pools.append(pool)
            pool.kvcached_allocator.wait_ready(timeout=8)
            assert target.token_to_kv_pool._kvcached_runtime_reservation_breakdown[
                "dsv4." + name] == 0
        assert (target.token_to_kv_pool._kvcached_runtime_reservation_breakdown[
            "dsv4.compress_state_pools"] == 0) is with_state
        if compress_ratio == 4:
            indexer = target.token_to_kv_pool.c4_indexer_kv_pool
            assert indexer.use_fp4_indexer is fp4
            assert indexer.get_bytes_per_token() == (68 if fp4 else 132)
        target_reservation = interfaces.get_runtime_owned_reservation_bytes("cuda:0")
        shared = target.token_to_kv_pool_allocator
        undersized = make_runner(target)
        undersized.memory_pool_config.swa_max_total_num_tokens -= 256
        with pytest.raises(patches.KVCachedConfigError, match="cover the shared target"):
            undersized.alloc_memory_pool()
        del undersized
        gc.collect()
        assert patches._dsv4_pool_context.get() is None
        assert interfaces.get_runtime_owned_reservation_bytes("cuda:0") == target_reservation

        draft = make_runner(target)
        draft.alloc_memory_pool()
        assert draft.token_to_kv_pool_allocator is shared
        assert not hasattr(draft.token_to_kv_pool.swa_kv_pool, "kvcached_allocator")
        assert draft.token_to_kv_pool.full_to_swa_index_mapping is shared.full_to_swa_index_mapping
        breakdown = draft.token_to_kv_pool._kvcached_runtime_reservation_breakdown
        assert breakdown["dsv4.swa_kv_pool"] == sum(
            t.nbytes for t in draft.token_to_kv_pool.swa_kv_pool.kv_buffer
        )
        assert breakdown["dsv4.swa_kv_pool"] > 0
        # Native MTP draft has no compressed layers in SGLang 0.5.20.
        for name in compressed_names:
            draft_pool = getattr(draft.token_to_kv_pool, name)
            assert breakdown["dsv4." + name] == 0
            assert not hasattr(draft_pool, "kvcached_allocator")
        assert patches._token_kv_managers(shared) == [
            manager, *(pool.kvcached_allocator for pool in compressed_pools),
            *(state._kvcached_dsv4_backing.manager for state in state_pools)]
        assert interfaces.get_runtime_owned_reservation_bytes("cuda:0") == (
            target_reservation + sum(breakdown.values())
        )

        # Full IDs remain native; physical pressure must roll them back before
        # the composite publishes any full-to-SWA mapping.
        logical = shared.full_attn_allocator
        assert not hasattr(logical, "kvcached_allocator")
        logical_capacity = logical.available_size()
        assert logical_capacity == target.memory_pool_config.full_max_total_num_tokens
        page_bytes = target.token_to_kv_pool.swa_kv_pool.bytes_per_page_padded
        first_crossing_page = manager.page_size // page_bytes
        assert first_crossing_page > 1
        prefix = torch.tensor([0], device="cuda", dtype=torch.int64)
        last = torch.tensor([-1], device="cuda", dtype=torch.int64)
        for operation in ("extend", "decode", "tail"):
            shared.clear()
            # Leave the next SWA page at the permanent-null backing boundary.
            # A one-token prefix exercises partial-page rollback on extend.
            count = 1 if operation == "extend" else (first_crossing_page - 1) * 256
            seq = torch.tensor([count], device="cuda", dtype=torch.int64)
            live = shared.alloc_extend(prefix, prefix.cpu(), seq, seq.cpu(), last, count)
            assert live is not None
            before_mapping = shared.full_to_swa_index_mapping.clone()
            before_capacity = logical.available_size()
            backing = target.token_to_kv_pool.swa_kv_pool._kvcached_dsv4_backing
            before_live, before_refs = set(backing.live), backing.refs.copy()
            claims = []

            def miss(ids):
                claims.append(list(ids))
                return None

            with monkeypatch.context() as pressure:
                pressure.setattr(manager, "alloc_exact", miss)
                if operation == "extend":
                    extended = torch.tensor([first_crossing_page * 256], device="cuda", dtype=torch.int64)
                    result = shared.alloc_extend(
                        seq, seq.cpu(), extended, extended.cpu(), live[-1:], int(extended.item()) - count,
                    )
                elif operation == "decode":
                    extended = seq + 1
                    result = shared.alloc_decode(extended, extended.cpu(), live[-1:])
                else:
                    extended = torch.tensor([257], device="cuda", dtype=torch.int64)
                    result = shared.alloc_extend_swa_tail(
                        prefix, prefix.cpu(), extended, extended.cpu(), last, 257, 128,
                    )
                assert result is None
            assert claims, f"{operation} must reach the physical exact-ID miss"
            assert logical.available_size() == before_capacity
            assert torch.equal(shared.full_to_swa_index_mapping, before_mapping)
            assert backing.live == before_live and backing.refs == before_refs
            shared.free(live)
            assert logical.available_size() == logical_capacity

        for _ in range(2):
            count = 40 * 256
            prefix = torch.tensor([0], device="cuda", dtype=torch.int64)
            seq = torch.tensor([count], device="cuda", dtype=torch.int64)
            last = torch.tensor([-1], device="cuda", dtype=torch.int64)
            full = shared.alloc_extend(prefix, prefix.cpu(), seq, seq.cpu(), last, count)
            assert full is not None
            seq += 1
            decoded = shared.alloc_decode(seq, seq.cpu(), full[-1:])
            assert decoded is not None
            full = torch.cat([full, decoded])
            swa = shared.translate_loc_from_full_to_swa(full)
            pages = (swa // 256).unique().tolist()
            assert min(pages) > 0
            bytes_per_page = target.token_to_kv_pool.swa_kv_pool.bytes_per_page_padded
            assert max(pages) * bytes_per_page >= manager.page_size
            for runner in (target, draft):
                buffer = runner.token_to_kv_pool.swa_kv_pool.kv_buffer[0]
                assert max(pages) < buffer.shape[0]
                for page in pages:
                    buffer[page].fill_(237)
                torch.cuda.synchronize()
                for page in pages:
                    assert torch.all(buffer[page] == 237).item()
            del runner, buffer
            shared.free(full)

        for compressed_pool in compressed_pools:
            # Full pages and compressed history outlive the sliding window.
            compressed_manager = compressed_pool.kvcached_allocator
            compressed_backing = compressed_pool._kvcached_dsv4_backing
            page_bytes = compressed_backing.native_page_bytes
            shared.clear()
            first_crossing = compressed_manager.page_size // page_bytes
            saved = []
            prefix = torch.tensor([0], device="cuda", dtype=torch.int64)
            last = torch.tensor([-1], device="cuda", dtype=torch.int64)
            for _ in range((first_crossing - 1) // 32):
                seq = torch.tensor([32 * 256], device="cuda", dtype=torch.int64)
                full = shared.alloc_extend(prefix, prefix.cpu(), seq, seq.cpu(), last, 32 * 256)
                assert full is not None
                saved.append(full)
                shared.free_swa(full)
            before = set(compressed_backing.live)
            mapping = shared.full_to_swa_index_mapping.clone()
            capacity = logical.available_size()
            claims = []

            def compressed_miss(ids):
                claims.append(ids)
                return None

            with monkeypatch.context() as pressure:
                pressure.setattr(compressed_manager, "alloc_exact", compressed_miss)
                assert shared.alloc_extend(
                    prefix, prefix.cpu(), seq, seq.cpu(), last, 32 * 256) is None
            assert claims and compressed_backing.live == before
            assert logical.available_size() == capacity
            assert torch.equal(shared.full_to_swa_index_mapping, mapping)
            full = shared.alloc_extend(prefix, prefix.cpu(), seq, seq.cpu(), last, 32 * 256)
            assert full is not None
            saved.append(full)
            pages = (full // 256).unique().tolist()
            assert max(pages) * page_bytes >= compressed_manager.page_size
            buffers = getattr(compressed_pool, "kv_buffer", None)
            if buffers is None:
                buffers = compressed_pool.contiguous_page_row_buffers()
            buffer = buffers[0]
            assert buffer.dtype == torch.uint8 and buffer.stride() == (page_bytes, 1)
            for page in pages:
                buffer[page].fill_(193)
            torch.cuda.synchronize()
            for page in pages:
                assert torch.all(buffer[page] == 193).item()
            del buffer
            shared.free_swa(full)
            shared.free_group_begin()
            all_full = torch.cat(saved)
            shared.free_full(all_full)
            assert compressed_backing.live
            shared.free_group_end()
            assert not compressed_backing.live and not compressed_backing.refs

        for state in state_pools:
            shared.clear()
            backing = state._kvcached_dsv4_backing
            state_manager = backing.manager
            first_crossing = backing.physical_page_bytes // backing.native_page_bytes
            count = (first_crossing + 1) * 256
            prefix = torch.tensor([0], device="cuda", dtype=torch.int64)
            seq = torch.tensor([count], device="cuda", dtype=torch.int64)
            last = torch.tensor([-1], device="cuda", dtype=torch.int64)
            mapping = shared.full_to_swa_index_mapping.clone()
            capacity = logical.available_size()
            with monkeypatch.context() as pressure:
                miss = mock.Mock(return_value=None)
                pressure.setattr(state_manager, "alloc_exact", miss)
                assert shared.alloc_extend(prefix, prefix.cpu(), seq, seq.cpu(), last, count) is None
                assert miss.called
            assert not backing.live and not backing.refs
            assert logical.available_size() == capacity
            assert torch.equal(shared.full_to_swa_index_mapping, mapping)
            full = shared.alloc_extend(prefix, prefix.cpu(), seq, seq.cpu(), last, count)
            assert full is not None
            swa = shared.translate_loc_from_full_to_swa(full)
            locs = state.translate_from_swa_loc_to_state_loc(swa)
            state.kv_score_buffer.kv_score[locs] = 29
            torch.cuda.synchronize()
            assert torch.all(state.kv_score_buffer.kv_score[locs] == 29).item()
            shared.free_swa(full)
            assert not backing.live and not backing.refs
            # The native sentinel remains mapped after all live SWA rows retire.
            state.kv_score_buffer[-1].clear()
            torch.cuda.synchronize()
            assert torch.all(torch.isneginf(state.kv_score_buffer[-1].score)).item()
            shared.free_full(full)

        draft_owner = weakref.ref(draft.token_to_kv_pool)
        del draft
        gc.collect()
        assert draft_owner() is None
        assert interfaces.get_runtime_owned_reservation_bytes("cuda:0") == target_reservation
        assert patches._dsv4_pool_context.get() is None
    finally:
        if manager is not None:
            assert manager.shutdown()
        for pool in compressed_pools:
            assert pool.kvcached_allocator.shutdown()
        for state in state_pools:
            assert state._kvcached_dsv4_backing.manager.shutdown()
        assert interfaces.shutdown_kvcached()
