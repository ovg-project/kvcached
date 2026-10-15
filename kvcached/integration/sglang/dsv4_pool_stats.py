# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0
"""Keep DSV4 slot occupancy separate from physical allocation headroom."""

import functools

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
            allocator = getattr(allocator, "logical_attn_allocator", allocator)
            if getattr(allocator, "_kvcached_dsv4_backing", None) is None:
                return stats
            for prefix in ("full", "swa"):
                logical_available = getattr(allocator, f"_conserve_{prefix}_available_size", None)
                capacity = getattr(observer, f"{prefix}_tokens_per_layer")
                if not callable(logical_available) or capacity <= 0:
                    continue
                available = int(logical_available())
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
