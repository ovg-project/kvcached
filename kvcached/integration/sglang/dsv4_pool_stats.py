# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0
"""Keep DSV4 slot occupancy separate from physical allocation headroom."""

import functools
from dataclasses import replace

from kvcached.integration.patch_base import BasePatch
from kvcached.integration.version_utils import VersionAwarePatch


class DeepSeekV4PoolStatsPatch(VersionAwarePatch, BasePatch):
    library = "sglang"
    target_module = "sglang.srt.managers.scheduler_components.pool_stats_observer"
    target_class = "SchedulerPoolStatsObserver"
    patch_name = "dsv4_pool_stats"

    def apply(self, observer_mod) -> bool:
        if not self.initialize_version_info():
            return False
        observer_cls = self._get_target_class(observer_mod)
        original = getattr(observer_cls, "_get_swa_token_info", None)
        if not callable(original):
            return False
        if self._is_already_patched(original):
            return True

        @functools.wraps(original)
        def get_swa_token_info(observer, *args, **kwargs):
            stats = original(observer, *args, **kwargs)
            allocator = observer.token_to_kv_pool_allocator
            logical = getattr(allocator, "logical_attn_allocator", allocator)
            if getattr(logical, "_kvcached_dsv4_backing", None) is None:
                return stats
            hisparse = getattr(allocator, "hisparse_attn_allocator", None)
            for prefix in ("full", "swa"):
                logical_available = getattr(logical, f"_conserve_{prefix}_available_size", None)
                capacity = getattr(observer, f"{prefix}_tokens_per_layer")
                if not callable(logical_available) or capacity <= 0:
                    continue
                available = int(logical_available())
                if prefix == "full" and hisparse is not None:
                    # Native HiSparse device slots each cover compress_ratio FULL tokens.
                    available = min(available, hisparse.available_size() * allocator.compress_ratio)
                used = capacity - available - getattr(stats, f"{prefix}_evictable_size")
                # Preserve the native HiSparse clamp; other conservation
                # failures must remain visible to the native checker.
                if observer.enable_hisparse:
                    used = max(0, used)
                setattr(stats, f"{prefix}_available_size", available)
                setattr(stats, f"{prefix}_num_used", used)
                setattr(stats, f"{prefix}_token_usage", used / capacity)
            return stats

        self._mark_as_patched(get_swa_token_info)
        observer_cls._get_swa_token_info = get_swa_token_info
        return True


class DeepSeekV4FullPoolCheckPatch(VersionAwarePatch, BasePatch):
    library = "sglang"
    target_module = "sglang.srt.managers.scheduler_components.invariant_checker"
    target_class = "SchedulerInvariantChecker"
    patch_name = "dsv4_full_pool_check"

    def apply(self, checker_mod) -> bool:
        if not self.initialize_version_info():
            return False
        checker_cls = self._get_target_class(checker_mod)
        original = getattr(checker_cls, "_check_full_pool", None)
        if not callable(original):
            return False
        if self._is_already_patched(original):
            return True

        @functools.wraps(original)
        def check_full_pool(checker, ps, *args, **kwargs):
            allocator = checker.token_to_kv_pool_allocator
            logical = getattr(allocator, "logical_attn_allocator", None)
            if (getattr(allocator, "hisparse_attn_allocator", None) is not None
                    and getattr(logical, "_kvcached_dsv4_backing", None) is not None
                    and checker.full_tokens_per_layer):
                # The native invariant accounts for every logical slot. A device
                # limit affects admission and load reports, not slot ownership.
                logical_available = getattr(logical, "_conserve_full_available_size", None)
                if callable(logical_available):
                    ps = replace(ps, full_available_size=logical_available())
            return original(checker, ps, *args, **kwargs)

        self._mark_as_patched(check_full_pool)
        checker_cls._check_full_pool = check_full_pool
        return True
