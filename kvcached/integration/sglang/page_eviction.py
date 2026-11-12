# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0
"""Page-aware prefix eviction using SGLang's native caches and eviction."""

import types
from dataclasses import replace
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    NamedTuple,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
    cast,
)

if TYPE_CHECKING:
    from sglang.srt.mem_cache.base_prefix_cache import EvictParams


def _tree(cache: Any) -> Any:
    return getattr(cache, "tree_core", cache)


def _component_type(cache: Any) -> Any:
    return getattr(cache, "tree_components", (None,))[0]


def _node_lock_ref(node: Any, component_type: Any) -> int:
    return node.lock_ref if component_type is None else node.component_data[component_type].lock_ref


def _evictable_size(cache: Any) -> int:
    return int(
        cache.evictable_size_ if _component_type(cache) is None else cache.full_evictable_size()
    )


def supports_page_aware_eviction(cache: Any) -> bool:
    """Support native RadixCache and FULL-only Unified caches on one rank."""
    import torch.distributed as dist

    # Rollback can change physical page placement independently on each rank.
    # Local victim selection would then diverge the logical prefix trees.
    # Use the engine's process topology: SGLang managers own local pools and
    # deliberately report world_size=1 even when TP/CP/PP is enabled.
    if dist.is_initialized() and dist.get_world_size() > 1:
        return False
    try:
        from sglang.srt.mem_cache.radix_cache import RadixCache
    except ImportError:
        return False
    if type(cache) is RadixCache:
        return True
    try:
        from sglang.srt.mem_cache.unified_radix_cache import (
            BASE_COMPONENT_TYPE,
            FullComponent,
            UnifiedRadixCache,
        )
    except ImportError:
        return False
    if type(cache) is not UnifiedRadixCache:
        return False

    # Hybrid and custom components retain native victim selection.
    if cache.tree_components != (BASE_COMPONENT_TYPE,):
        return False
    if type(cache.components.get(BASE_COMPONENT_TYPE)) is not FullComponent:
        return False

    # Host transfers, sessions and external linkers need their native policies.
    tree = _tree(cache)
    if (
        cache.cache_controller is not None
        or getattr(cache, "enable_session_radix_cache", False)
        or getattr(tree, "enable_hicache", False)
        or getattr(tree, "enable_external_cache_linker", False)
    ):
        return False
    if not hasattr(tree, "evictable_device_leaves"):
        return False

    # 0.5.16 owns its tree directly; 0.5.17+ must use the standard TreeCore.
    if tree is cache:
        return True
    try:
        from sglang.srt.mem_cache.unified_cache.unified_tree_core import UnifiedTreeCore
    except ImportError:
        return False
    return type(tree) is UnifiedTreeCore


def evict_prefix_cache(cache: Any, params: Union["EvictParams", int]) -> Any:
    """Evict KVCached prefix demands, preserving SGLang's result counters.

    Single-rank FULL-only RadixCache and UnifiedRadixCache requests use
    page-aware selection. Parallel execution, other caches and component
    requests keep native victim selection.
    Integer requests retain native eviction for older SGLang versions.
    """
    if (
        isinstance(params, int)
        or params.num_tokens <= 0
        or params.swa_num_tokens
        or params.mamba_num
        or not supports_page_aware_eviction(cache)
    ):
        return cache.evict(params)
    return _evict_radix_cache_page_aware(cache, params)


def _add_eviction_result(result: Any, additional: Any) -> None:
    for field in ("num_tokens_evicted", "swa_num_tokens_evicted", "mamba_num_evicted"):
        setattr(result, field, getattr(result, field, 0) + getattr(additional, field, 0))


class _RadixBlockIndex:
    def __init__(
        self,
        root_node: Any,
        logical_page_size: int,
        component_type: Any = None,
    ) -> None:
        self.root_node = root_node
        self.logical_page_size = logical_page_size
        self.component_type = component_type
        self.block_owners: Dict[int, Any] = {}
        self.block_offsets: Dict[int, int] = {}
        self.node_blocks: Dict[Any, Tuple[int, ...]] = {}

    def node_value(self, node: Any) -> Any:
        if self.component_type is None:
            return node.value
        return node.component_data[self.component_type].value

    def node_lock_ref(self, node: Any) -> int:
        return _node_lock_ref(node, self.component_type)

    def sync(self, nodes: Set[Any]) -> None:
        # node_blocks and the tensor attribute reference the same immutable
        # tuple.  Identity therefore detects unchanged nodes in O(1), without
        # comparing every cached block id on each eviction pass.
        changed_nodes = [
            node
            for node in nodes
            if node not in self.node_blocks
            or self.node_blocks[node]
            is not getattr(self.node_value(node), "_kvcached_block_ids", None)
        ]
        removed_nodes = self.node_blocks.keys() - nodes
        for node in removed_nodes:
            for block_id in self.node_blocks.pop(node, ()):
                if self.block_owners.get(block_id) is node:
                    del self.block_owners[block_id]
                    del self.block_offsets[block_id]
        for node in changed_nodes:
            if node not in self.node_blocks:
                continue
            for block_id in self.node_blocks.pop(node):
                if self.block_owners.get(block_id) is node:
                    del self.block_owners[block_id]
                    del self.block_offsets[block_id]

        uncached = [
            node
            for node in changed_nodes
            if getattr(self.node_value(node), "_kvcached_block_ids", None) is None
        ]
        if uncached:
            import torch

            values = [self.node_value(node) for node in uncached]
            lengths = [int(value.numel()) // self.logical_page_size for value in values]
            if self.logical_page_size > 1:
                flattened = torch.cat([value[:: self.logical_page_size] for value in values])
                flattened = flattened // self.logical_page_size
            else:
                flattened = torch.cat(values)
            new_block_ids = cast(List[int], cast(Any, flattened).tolist())
            offset = 0
            for node, length in zip(uncached, lengths):
                self.node_value(node)._kvcached_block_ids = tuple(
                    new_block_ids[offset : offset + length]
                )
                offset += length

        for node in changed_nodes:
            block_ids = cast(Tuple[int, ...], self.node_value(node)._kvcached_block_ids)
            self.node_blocks[node] = block_ids
            for offset, block_id in enumerate(block_ids):
                self.block_owners[block_id] = node
                self.block_offsets[block_id] = offset

    def count_tokens(self, nodes: Dict[Any, int]) -> int:
        return sum(
            int(self.node_value(node).numel()) - split_len for node, split_len in nodes.items()
        )


def _evictable_radix_nodes(radix_cache: Any) -> Set[Any]:
    """Return every node reachable by repeatedly evicting current leaves."""
    tree = _tree(radix_cache)
    root = tree.root_node
    component_type = _component_type(radix_cache)
    leaves = (
        radix_cache.evictable_leaves if component_type is None else tree.evictable_device_leaves
    )
    nodes: Set[Any] = set()
    for leaf in leaves:
        node = leaf
        while node is not root and _node_lock_ref(node, component_type) == 0:
            if node in nodes:
                break
            nodes.add(node)
            node = node.parent
    return nodes


_RadixSuffixPlan = Tuple[Any, int]


class _RadixCandidate(NamedTuple):
    sort_key: Tuple[Any, ...]
    suffix_plans: Tuple[_RadixSuffixPlan, ...]
    closure: Dict[Any, int]


class _RadixEvictionSelection(NamedTuple):
    suffix_plans: List[_RadixSuffixPlan]
    token_count: int


def _align_radix_eviction_budget(radix_cache: Any, num_tokens: int) -> int:
    """Round eviction up because radix nodes split only on logical pages.

    This is equivalent to rounding the effective cache cap down. It may evict
    up to one page minus one extra token, but avoids an unsafe partial-page
    split; the cache's evictable size remains the upper bound.
    """
    logical_page_size = max(1, int(radix_cache.page_size))
    aligned_tokens = (num_tokens + logical_page_size - 1) // logical_page_size * logical_page_size
    return min(aligned_tokens, _evictable_size(radix_cache))


def _radix_suffix_closure(
    index: _RadixBlockIndex,
    plans: Sequence[_RadixSuffixPlan],
) -> Optional[Tuple[List[_RadixSuffixPlan], Dict[Any, int]]]:
    """Normalize suffix roots and expand them to their descendant closure."""
    split_by_node: Dict[Any, int] = {}
    for node, split_len in plans:
        previous = split_by_node.get(node)
        split_by_node[node] = split_len if previous is None else min(previous, split_len)

    planned_ids = {id(node) for node in split_by_node}
    roots: List[_RadixSuffixPlan] = []
    for node, split_len in split_by_node.items():
        ancestor = node.parent
        while ancestor is not None and id(ancestor) not in planned_ids:
            ancestor = ancestor.parent
        if ancestor is None:
            roots.append((node, split_len))

    closure: Dict[Any, int] = {}
    stack = list(roots)
    while stack:
        node, retained_prefix = stack.pop()
        if node not in index.node_blocks or index.node_lock_ref(node) != 0:
            return None
        closure[node] = retained_prefix
        stack.extend((child, 0) for child in node.children.values())
    return roots, closure


def _select_page_aware_plan(
    radix_cache: Any,
    token_budget: int,
) -> _RadixEvictionSelection:
    empty = _RadixEvictionSelection([], 0)
    if token_budget <= 0:
        return empty

    allocator = getattr(radix_cache, "token_to_kv_pool_allocator", None)
    manager = getattr(allocator, "kvcached_allocator", None)
    if manager is None or getattr(manager, "page_allocator", None) is None:
        return empty

    logical_page_size = max(1, int(getattr(radix_cache, "page_size", 1)))
    if int(getattr(allocator, "page_size", logical_page_size)) != logical_page_size:
        return empty
    tree = _tree(radix_cache)
    can_split = callable(getattr(tree, "_split_node", None))

    evictable_nodes = _evictable_radix_nodes(radix_cache)
    if not evictable_nodes:
        return empty

    index = getattr(radix_cache, "_kvcached_radix_block_index", None)
    if (
        index is None
        or index.root_node is not tree.root_node
        or index.logical_page_size != logical_page_size
    ):
        index = _RadixBlockIndex(
            tree.root_node,
            logical_page_size,
            _component_type(radix_cache),
        )
        radix_cache._kvcached_radix_block_index = index
    index.sync(evictable_nodes)

    by_page = manager.page_allocator.group_indices_by_page(
        list(index.block_owners), manager.block_mem_size
    )
    occupancy = manager.get_page_occupancy(list(by_page))
    reclaimable_pages = [
        block_ids
        for page_id, block_ids in by_page.items()
        if len(block_ids) == occupancy.get(page_id, 0)
    ]
    if not reclaimable_pages:
        return empty

    candidates: List[_RadixCandidate] = []
    seen_plans: Set[frozenset[Tuple[Any, int]]] = set()

    for target_blocks in reclaimable_pages:
        split_by_node: Dict[Any, int] = {}
        for block_id in target_blocks:
            owner = index.block_owners[block_id]
            block_offset = index.block_offsets[block_id]
            previous = split_by_node.get(owner)
            split_by_node[owner] = block_offset if previous is None else min(previous, block_offset)
        plan = _radix_suffix_closure(
            index,
            [
                (node, block_offset * logical_page_size)
                for node, block_offset in split_by_node.items()
            ],
        )
        if plan is None:
            continue
        suffix_plans, closure = plan
        if not can_split and any(split_len > 0 for _, split_len in suffix_plans):
            continue
        plan_key = frozenset(suffix_plans)
        if plan_key in seen_plans:
            continue
        seen_plans.add(plan_key)

        token_cost = index.count_tokens(closure)
        if token_cost > token_budget:
            continue

        selected_blocks: Dict[int, int] = {}
        for node, split_len in closure.items():
            block_offset = split_len // logical_page_size
            suffix_blocks = index.node_blocks[node][block_offset:]
            for block_id in suffix_blocks:
                page_id = block_id * manager.block_mem_size // manager.page_size
                selected_blocks[page_id] = selected_blocks.get(page_id, 0) + 1

        completed_pages = [
            page_id
            for page_id, count in selected_blocks.items()
            if count == occupancy.get(page_id, 0)
        ]
        priority = max(tree.eviction_strategy.get_priority(node) for node in closure)
        sort_key = (
            token_cost / len(completed_pages),
            -len(completed_pages),
            token_cost,
            priority,
            min(completed_pages),
        )
        candidates.append(_RadixCandidate(sort_key, tuple(suffix_plans), closure))

    selected_plans: List[_RadixSuffixPlan] = []
    selected_closure: Dict[Any, int] = {}
    selected_tokens = 0
    for candidate in sorted(candidates, key=lambda item: item.sort_key):
        # Closures are already expanded and validated. Count only newly freed
        # tokens: overlapping suffixes keep the smaller retained prefix.
        additional_tokens = 0
        for node, split_len in candidate.closure.items():
            retained = selected_closure.get(node)
            if retained is None:
                retained = int(index.node_value(node).numel())
            additional_tokens += max(0, retained - split_len)
        if additional_tokens == 0 or selected_tokens + additional_tokens > token_budget:
            continue
        for node, split_len in candidate.closure.items():
            previous = selected_closure.get(node)
            selected_closure[node] = split_len if previous is None else min(previous, split_len)
        selected_plans.extend(candidate.suffix_plans)
        selected_tokens += additional_tokens
        if selected_tokens == token_budget:
            break

    # Normalize roots once, after selection, rather than repeatedly walking
    # the growing set of previously selected descendants for each candidate.
    plan = _radix_suffix_closure(index, selected_plans)
    assert plan is not None
    roots, _closure = plan
    return _RadixEvictionSelection(roots, selected_tokens)


def _split_eviction_suffixes(
    radix_cache: Any,
    plans: Sequence[_RadixSuffixPlan],
) -> Set[Any]:
    for node, split_len in plans:
        if split_len > 0:
            _tree(radix_cache)._split_node(node.key, node, split_len)

    suffix_closures: Set[Any] = set()
    stack = [node for node, _split_len in plans]
    while stack:
        current = stack.pop()
        if current in suffix_closures:
            continue
        suffix_closures.add(current)
        stack.extend(current.children.values())
    return suffix_closures


def _make_sglang_evict_arg(num_tokens: int, evict_params_cls: Optional[Callable[..., Any]]) -> Any:
    """Build the eviction argument used by the installed SGLang version."""
    if evict_params_cls is None:
        return num_tokens
    return evict_params_cls(num_tokens=num_tokens)


def _sync_radix_block_index(radix_cache: Any) -> None:
    index = getattr(radix_cache, "_kvcached_radix_block_index", None)
    if index is not None:
        index.sync(_evictable_radix_nodes(radix_cache))


def _evict_radix_cache_page_aware(
    radix_cache: Any,
    params: "EvictParams",
) -> Any:
    """Evict a FULL demand in legal logical-page units, counting all frees."""
    eviction_budget = _align_radix_eviction_budget(radix_cache, params.num_tokens)

    def evict_selected(selected: Set[Any], token_count: int) -> Any:
        tree = _tree(radix_cache)
        original_strategy = tree.eviction_strategy
        tree.eviction_strategy = types.SimpleNamespace(
            get_priority=lambda node: (
                node not in selected,
                original_strategy.get_priority(node),
            )
        )
        component_type = _component_type(radix_cache)
        full = radix_cache.components[component_type] if component_type is not None else None
        # FullComponent may already have cached the original bound priority.
        has_callback = hasattr(full, "session_ref_eviction_strategy")
        callback = getattr(full, "session_ref_eviction_strategy", None)
        if has_callback:
            full.session_ref_eviction_strategy = tree.eviction_strategy.get_priority
        try:
            result = radix_cache.evict(replace(params, num_tokens=token_count))
        finally:
            tree.eviction_strategy = original_strategy
            if has_callback:
                full.session_ref_eviction_strategy = callback
        if result is not None and result.num_tokens_evicted != token_count:
            raise RuntimeError("SGLang radix eviction did not honor the exact selected-node budget")
        return result

    try:
        if eviction_budget >= _evictable_size(radix_cache):
            return radix_cache.evict(replace(params, num_tokens=eviction_budget))

        selection = _select_page_aware_plan(radix_cache, eviction_budget)
        result = None
        if selection.suffix_plans:
            selected = _split_eviction_suffixes(radix_cache, selection.suffix_plans)
            result = evict_selected(selected, selection.token_count)

        remaining = eviction_budget - selection.token_count
        if remaining > 0:
            # Let SGLang choose the remaining victims. Native eviction removes
            # complete radix nodes, so the actual count may exceed `remaining`.
            native_result = radix_cache.evict(replace(params, num_tokens=remaining))
            if native_result is not None:
                if result is None:
                    result = native_result
                else:
                    _add_eviction_result(result, native_result)
        return result
    finally:
        _sync_radix_block_index(radix_cache)
