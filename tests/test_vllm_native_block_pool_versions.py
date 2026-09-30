# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0
"""Version routing for native prefix metadata without a vLLM installation."""

from types import ModuleType

import pytest

from kvcached.integration.vllm.native_block_pool import NativeBlockPoolMixin
from kvcached.integration.vllm.patches import ElasticBlockPoolPatch


@pytest.mark.parametrize(("version", "use_native"), [
    ("0.24.0", False),
    ("0.25.0", False),
    ("0.26.0", True),
    ("0.26.0+cu130", True),
    ("0.27.0", True),
    ("0.28.0", True),
    ("0.29.0", True),
])
def test_native_metadata_is_selected_from_026(version, use_native):
    target = ModuleType("_versioned_elastic_pool")
    setattr(target, "BlockPool", type("BlockPool", (), {}))
    setattr(target, "KVCacheBlock", type("KVCacheBlock", (), {}))
    patch = ElasticBlockPoolPatch()
    patch.detected_version = version
    assert patch.inject_elastic_block_pool(target)
    pool_cls = target.ElasticBlockPool
    assert issubclass(pool_cls, NativeBlockPoolMixin) is use_native
    if use_native:
        # Select native lookup, eviction cleanup, and pending-copy retention
        # together; selecting only the lookup path would leave stale owners.
        assert pool_cls.get_cached_block is NativeBlockPoolMixin.get_cached_block
        assert pool_cls._remove_cached_block is NativeBlockPoolMixin._remove_cached_block
        assert pool_cls.free_blocks is NativeBlockPoolMixin.free_blocks
