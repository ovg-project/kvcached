# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0
"""Agree device eviction victims before trimming a distributed HiCache."""

import functools
import hashlib
import heapq
import json
import struct
import time
import types
from array import array
from collections import Counter, OrderedDict, deque
from collections.abc import MutableSet
from dataclasses import replace
from enum import Enum, auto
from itertools import chain, islice
from typing import Any, Callable, Deque, Dict, Iterable, Iterator, List, Optional, Tuple
from weakref import ReferenceType, WeakKeyDictionary, ref

from kvcached import prefix_cache_retention
from kvcached.control import get_prefix_cache_retention_limit
from kvcached.integration.patch_base import BasePatch, enable_kvcached
from kvcached.integration.sglang.hicache import HICACHE_RANGE
from kvcached.integration.sglang.rank_groups import reduce_tensor, stage_cpu_groups
from kvcached.integration.sglang.scheduler.rank_plan import request_slot
from kvcached.integration.version_utils import VersionAwarePatch, version_range

# Limit synchronous scheduler work even when only part of a batch agrees.
_VICTIMS_PER_BATCH = 64
_ROUNDS_PER_POLL = 4
_INDEX_PER_POLL = _VICTIMS_PER_BATCH * _ROUNDS_PER_POLL
_RECHECK_POLLS = 32


class _DeviceLeaves(MutableSet):
    """Device-leaf membership with a bounded cursor that survives scheduler mutations."""

    def __init__(self, nodes: Iterable[Any] = ()) -> None:
        self._nodes = OrderedDict.fromkeys(nodes)

    def __len__(self) -> int:
        return len(self._nodes)

    def __contains__(self, node: Any) -> bool:
        return node in self._nodes

    def __iter__(self) -> Iterator[Any]:
        return iter(self._nodes)

    def add(self, node: Any) -> None:
        self._nodes[node] = None

    def discard(self, node: Any) -> None:
        self._nodes.pop(node, None)

    def clear(self) -> None:
        self._nodes.clear()

    def update(self, nodes: Iterable[Any]) -> None:
        self._nodes.update(OrderedDict.fromkeys(nodes))

    def pages(self, width: int) -> Iterator[List[Any]]:
        remaining = len(self)
        while remaining and self._nodes:
            page = list(islice(self._nodes, min(remaining, width)))
            for node in page:
                self._nodes.move_to_end(node)
            remaining -= len(page)
            yield page


def _leaf_pages(leaves: Any, width: int) -> Iterator[List[Any]]:
    if isinstance(leaves, _DeviceLeaves):
        yield from leaves.pages(width)
    else:
        # Some cache implementations expose their own iterable leaf collection.
        stream = iter(leaves)
        while page := list(islice(stream, width)):
            yield page


def _track_device_leaves(cache: Any) -> None:
    tree = getattr(cache, "tree_core", None)
    owner = cache if tree is None else tree
    field = "evictable_leaves" if tree is None else "evictable_device_leaves"
    leaves = getattr(owner, field, None)
    if type(leaves) is not set:
        return
    setattr(owner, field, _DeviceLeaves(leaves))
    reset = getattr(owner, "reset", None)
    if reset is not None:
        @functools.wraps(reset)
        def reset_leaves(*args: Any, **kwargs: Any) -> Any:
            result = reset(*args, **kwargs)
            current = getattr(owner, field)
            if type(current) is set:
                setattr(owner, field, _DeviceLeaves(current))
            return result

        owner.reset = reset_leaves


def parallel_cache(cache: Any) -> bool:
    if getattr(cache, "tp_world_size", 1) > 1 or getattr(cache, "pp_size", 1) > 1:
        return True
    return any(group is not None and group.size() > 1 for group in (
        getattr(cache, "attn_cp_group", None), getattr(cache, "attn_tp_group", None)))


def _write_through(cache: Any) -> bool:
    return (cache.cache_controller is not None and
            cache.cache_controller.write_policy in ("write_through", "write_through_selective"))


def _excess(cache: Any, limit: int) -> int:
    if limit < 0 or not _write_through(cache):
        return 0
    size = cache.full_evictable_size() if hasattr(cache, "tree_core") else cache.evictable_size_
    return max(0, size - limit)


def _eviction_order(cache: Any) -> Tuple[Any, Callable]:
    tree = getattr(cache, "tree_core", None)
    if tree is None:
        return cache.evictable_leaves, cache.eviction_strategy.get_priority
    from sglang.srt.mem_cache.unified_radix_cache import BASE_COMPONENT_TYPE

    full = cache.components[BASE_COMPONENT_TYPE]
    priority = getattr(full, "session_ref_eviction_strategy", None) or tree.eviction_strategy.get_priority
    return tree.evictable_device_leaves, priority


def _may_promote_parent(node: Any, candidates: Dict[Any, int]) -> bool:
    parent = node.parent
    if parent.parent is None:
        return False
    candidates[parent] -= 1
    if candidates[parent]:
        return False
    if hasattr(parent, "component_data"):
        from sglang.srt.mem_cache.unified_radix_cache import BASE_COMPONENT_TYPE
        return not parent.component_data[BASE_COMPONENT_TYPE].lock_ref
    return not parent.lock_ref


def _victim_batches(cache: Any, excess: int) -> Iterator[Tuple[List[Any], bool]]:
    if excess <= 0 or not _write_through(cache):
        return
    leaves, priority = _eviction_order(cache)

    heap: List[Tuple[Any, Any]] = []
    children: Counter = Counter()
    remaining = len(leaves)
    for page in _leaf_pages(leaves, _INDEX_PER_POLL):
        for node in page:
            if node.backuped or not node.children:
                heapq.heappush(heap, (priority(node), node))
                if node.parent.parent is not None:
                    children[node.parent] += 1
        remaining -= len(page)
        if remaining > 0:
            yield [], False  # Still indexing. No victim ballot yet.
    while heap and excess > 0:
        victims: List[Any] = []
        selected = set()
        parent = None
        while heap and excess > 0 and len(victims) < _VICTIMS_PER_BATCH:
            _, node = heapq.heappop(heap)
            if node not in leaves or node in selected:
                continue
            selected.add(node)
            victims.append(node)
            excess -= len(node.key)
            parent = node.parent
            if excess > 0 and _may_promote_parent(node, children):
                break
        yield victims, excess <= 0
        # The caller committed this batch. Use native membership for locks and
        # component eligibility before inserting the parent into the priority heap.
        if (parent is not None and parent.parent is not None and parent in leaves
                and (parent.backuped or not parent.children)):
            heapq.heappush(heap, (priority(parent), parent))
            if parent.parent.parent is not None:
                children[parent.parent] += 1


def _extend_path(segment: Any, path: Any, namespace: Any) -> Tuple[Any, Any]:
    key = segment.key
    raw = key.raw_token_ids()
    scope = (key.extra_key, key.cache_salt, key.is_bigram)
    path = hashlib.sha256(path.digest() + json.dumps(scope).encode()) if scope != namespace else path.copy()
    tokens = memoryview(raw if isinstance(raw, array) and raw.typecode == "q" else array("q", raw))
    # Adjacent bigram segments share one raw boundary token.
    path.update(tokens[1:] if key.is_bigram and segment.parent.parent is not None else tokens)
    return path, scope


def _remember_ancestor(ancestors: WeakKeyDictionary, node: Any, path: Any, scope: Any) -> None:
    # The sweep must not keep a detached subtree alive through its parent.
    parent = ref(node.parent) if node.parent.parent is not None else None
    ancestors[node] = node.key, parent, path, scope


def _leaf_signature(node: Any, paths: Dict[Any, Tuple[Any, Any]]) -> bytes:
    missing = []
    ancestor = node
    while ancestor.parent is not None and ancestor not in paths:
        missing.append(ancestor)
        ancestor = ancestor.parent
    path, namespace = (hashlib.sha256(), None) if ancestor.parent is None else paths[ancestor]
    for segment in reversed(missing):
        path, namespace = _extend_path(segment, path, namespace)
        paths[segment] = path, namespace
    digest = path.copy()
    digest.update(struct.pack("<Q", len(node.key)))
    digest.update(bytes([node.backuped]))
    # The device and host presence of each component determines what the
    # native demotion will retain. Physical slots and layer IDs are local.
    for cd in getattr(node, "component_data", ()):
        digest.update(bytes([cd.value is not None, cd.host_value is not None]))
    return digest.digest()


def _ballot_signature(node: Any, paths: Dict[Any, Tuple[Any, Any]], ancestors: WeakKeyDictionary,
                      edge_limit: int) -> Optional[bytes]:
    # A busy scheduler may have inserted an entire chain above this leaf.
    # Leave long repairs to the paged index; peers are waiting for this vote.
    missing: List[Any] = []
    cursor = node
    while cursor.parent is not None and cursor not in paths:
        saved = ancestors.get(cursor) if cursor is not node else None
        key = cursor.key
        if (saved is not None and saved[0] is key
                and (saved[1]() is cursor.parent if saved[1] is not None else cursor.parent.parent is None)
                and saved[3] == (key.extra_key, key.cache_salt, key.is_bigram)):
            paths[cursor] = saved[2], saved[3]
            break
        if len(missing) >= edge_limit:
            return None
        missing.append(cursor)
        cursor = cursor.parent
    signature = _leaf_signature(node, paths)
    for segment in missing[1:]:
        _remember_ancestor(ancestors, segment, *paths[segment])
    return signature


def _evict(cache: Any, nodes: List[Any]) -> int:
    if hasattr(cache, "tree_core"):
        from sglang.srt.mem_cache.unified_radix_cache import BASE_COMPONENT_TYPE
        tracker = {ct: 0 for ct in cache.tree_components}
        for node in nodes:
            cache._evict_device_leaf(node.id, tracker)
        freed = tracker[BASE_COMPONENT_TYPE]
    else:
        freed = sum(cache._evict_backuped(node) if node.backuped else cache._evict_regular(node)
                    for node in nodes)
    return freed


def _signature_pages(leaves: Any, ancestors: WeakKeyDictionary) -> Iterator[Tuple[List[Any], bool, int]]:
    """Index a leaf page across multiple polls if its radix paths are deep."""
    remaining = len(leaves)
    for page in _leaf_pages(leaves, _VICTIMS_PER_BATCH):
        paths: Dict[Any, Tuple[Any, Any]] = {}
        indexed: List[Any] = []
        steps = 0
        for leaf in page:
            if leaf not in leaves or not (leaf.backuped or not leaf.children):
                continue
            cursor = leaf
            chain = []
            while cursor.parent is not None and cursor not in paths:
                if steps >= _INDEX_PER_POLL:
                    yield indexed, True, steps
                    indexed, steps = [], 0
                chain.append(cursor)
                cursor = cursor.parent
                steps += 1
            for node in reversed(chain):
                if steps >= _INDEX_PER_POLL:
                    yield indexed, True, steps
                    indexed, steps = [], 0
                # The scheduler may have split the edge while we yielded.
                # Leave such a path for a later sweep.
                if node.parent is None or (node.parent.parent is not None and node.parent not in paths):
                    break
                base = (hashlib.sha256(), None) if node.parent.parent is None else paths[node.parent]
                paths[node] = _extend_path(node, *base)
                if node is not leaf:
                    _remember_ancestor(ancestors, node, *paths[node])
                steps += 1
            if leaf in paths and leaf in leaves:
                indexed.append((leaf, _leaf_signature(leaf, paths)))
            if len(paths) > _INDEX_PER_POLL:
                paths.clear()
        remaining -= len(page)
        yield indexed, remaining > 0, steps
    if remaining:
        yield [], False, 0


def _reduce_victims(data: Any, op: Any) -> None:
    reduce_tensor(data, stage_cpu_groups(), op)


class _OrderedVote(Enum):
    INDEXING = auto()
    CONTINUE = auto()
    DONE = auto()
    EMPTY = auto()
    DISAGREE = auto()


def _trim_batches(cache: Any, demand: int) -> Iterator[Tuple[_OrderedVote, int, int]]:
    import torch
    import torch.distributed as dist

    def vote_prefix(signatures: List[bytes], done: bool, building: bool) -> Tuple[int, int, bool, bool, bool]:
        # Compare each SHA-256 in the same round. The fixed 4 KiB payload
        # also works when a rank has fewer candidates or no candidates.
        length = len(signatures)
        width = _VICTIMS_PER_BATCH * hashlib.sha256().digest_size
        packed = bytearray(b"".join(signatures))
        packed.extend(bytes(width - len(packed)))
        raw = torch.frombuffer(packed, dtype=torch.uint8)
        # 2 marks a ready rank, 1 an unfinished index, 0 an exhausted rank.
        state = 1 if building else (2 if length or done else 0)
        vote = torch.cat((raw, 255 - raw, raw.new_tensor([length, 255 - length, done, state])))
        _reduce_victims(vote, dist.ReduceOp.MIN)
        shortest = int(vote[-4])
        low, high = vote[:2 * width].view(2, _VICTIMS_PER_BATCH, -1)[:, :shortest]
        mismatch = (low != 255 - high).any(dim=1).nonzero()
        common = int(mismatch[0].item()) if mismatch.numel() else shortest
        return common, shortest, shortest == 255 - int(vote[-3]), bool(vote[-2]), int(vote[-1]) == 1

    batches = _victim_batches(cache, demand)
    pending: List[Any] = []
    done = False
    # Retain this iterator only while the tree can change solely by our evictions.
    paths: Dict[Any, Tuple[Any, Any]] = {}
    while True:
        building = False
        if not pending:
            try:
                pending, done = next(batches)
                building = not pending and not done
            except StopIteration:
                done = demand <= 0
        signatures = [_leaf_signature(node, paths) for node in pending]
        common, shortest, same_length, all_done, indexing = vote_prefix(signatures, done, building)
        if indexing:
            yield _OrderedVote.INDEXING, demand, 0
            continue
        if not common:
            # A peer with no eligible victim cannot accept any proposal.
            if shortest:
                status = _OrderedVote.DISAGREE
            else:
                status = _OrderedVote.DONE if same_length and all_done else _OrderedVote.EMPTY
            yield status, demand, 0
            return
        agreed = pending[:common]
        # A shorter peer can expose a parent first. Vote on its next choice
        # against our remaining candidates before advancing our generator.
        pending = pending[common:]
        demand -= sum(len(node.key) for node in agreed)
        freed = _evict(cache, agreed)
        if common < shortest:
            yield _OrderedVote.DISAGREE, demand, freed
            return
        if same_length and all_done:
            yield _OrderedVote.DONE, demand, freed
            return
        # The native free can expose a shared parent. Vote again before
        # demoting it.
        yield _OrderedVote.CONTINUE, demand, freed


class _VictimPlan:
    def __init__(self) -> None:
        self.ordered: Optional[Iterator[Tuple[_OrderedVote, int, int]]] = None
        self.search: Optional[Iterator[Optional[bool]]] = None
        self.remaining = 0
        self.search_changed = False
        self.freed = 0
        self.active_seconds = 0.0

    def poll(self, cache: Any, demand: int, proposer: Optional[int], changed: bool,
             start: Optional[float] = None) -> Optional[bool]:
        start = time.perf_counter() if start is None else start
        result = self._advance(cache, demand, proposer, changed)
        end = time.perf_counter()
        self.active_seconds += end - start
        if self.freed:
            cache.update_eviction_metrics(self.freed, end - self.active_seconds)
            self.freed = 0
            self.active_seconds = 0.0
        return result

    def _advance(self, cache: Any, demand: int, proposer: Optional[int], changed: bool) -> Optional[bool]:
        if changed or self.remaining != demand:
            if self.ordered is not None and proposer is not None:
                # The ordered heap can only continue across a quiet slot.
                # A busy trim uses the paged search, which rechecks live nodes.
                self.search = self._shared_victims(cache, proposer)
            self.ordered = None
            self.search_changed = self.search is not None
        self.remaining = demand
        if proposer is None:
            self.search = None
        rounds = _ROUNDS_PER_POLL - 1  # Group demand already used one round.
        if self.search is None:
            if self.ordered is None:
                self.ordered = _trim_batches(cache, demand)
            while rounds:
                status, self.remaining, freed = next(self.ordered)
                self.freed += freed
                rounds -= 1
                if status is _OrderedVote.INDEXING:
                    return None
                if status is _OrderedVote.DISAGREE and proposer is not None:
                    self.ordered = None
                    self.search = self._shared_victims(cache, proposer)
                    break
                if status is not _OrderedVote.CONTINUE:
                    return status is _OrderedVote.DONE
        # Proposing signatures and confirming membership are one atomic pair.
        if self.search is not None and rounds >= 2:
            result = next(self.search)
            if result is not None:
                self.search = None
            return result
        return None

    def _shared_victims(self, cache: Any, proposer: int) -> Iterator[Optional[bool]]:
        """Scan in the proposer's best-effort native eviction order.

        An MRU hit between polls can move an indexed leaf ahead of the heap
        entry being considered. Continue the current sweep so the cap can
        be reached under load. Every proposed victim still requires a fresh
        eligibility and prefix-identity check on every participating rank.
        """
        import torch
        import torch.distributed as dist

        proposing = not dist.is_initialized() or dist.get_rank() == proposer
        width = hashlib.sha256().digest_size
        while True:
            leaves, priority = _eviction_order(cache)
            ancestors: WeakKeyDictionary = WeakKeyDictionary()
            pages = _signature_pages(leaves, ancestors)
            indexing = True
            by_signature: Dict[bytes, ReferenceType] = {}
            keys: List[Tuple[Any, bytes]] = []
            # Keep the lowest eviction-priority child as a live witness.
            # It normally outlasts all earlier pages from the same parent.
            witnesses: Dict[int, Tuple[Tuple[Any, bytes], ReferenceType]] = {}
            self.search_changed = False
            progressed = False
            reindex = False
            while True:
                leaves, priority = _eviction_order(cache)
                paths: Dict[Any, Tuple[Any, Any]] = {}
                candidates, indexing, steps = next(pages, ([], False, 0)) if indexing else ([], False, 0)
                ballot_budget = _INDEX_PER_POLL - steps
                for candidate, signature in candidates:
                    if candidate in leaves and (candidate.backuped or not candidate.children):
                        if proposing and signature not in by_signature:
                            entry = priority(candidate), signature
                            heapq.heappush(keys, entry)
                            if candidate.parent.parent is not None:
                                parent_id = id(candidate.parent)
                                previous = witnesses.get(parent_id)
                                if previous is None or entry > previous[0]:
                                    witnesses[parent_id] = entry, ref(candidate)
                        by_signature[signature] = ref(candidate)
                paths.clear()
                packet = bytearray(_VICTIMS_PER_BATCH * width + 4)
                page: List[Tuple[Any, bytes]] = []
                if proposing and not indexing:
                    may_promote = False
                    page_nodes = []
                    reprioritized = 0
                    for _ in range(min(len(keys), _VICTIMS_PER_BATCH)):
                        saved, key = heapq.heappop(keys)
                        node = by_signature[key]()
                        if node is None or node not in leaves:
                            continue
                        current = priority(node)
                        # Let a hit move behind a better candidate, but keep
                        # proposing leaves even if all priorities change on every poll.
                        if (reprioritized <= len(page) and current != saved and keys
                                and (current, key) > keys[0]):
                            heapq.heappush(keys, (current, key))
                            reprioritized += 1
                        else:
                            page.append((current, key))
                            page_nodes.append(node)
                    children = Counter(n.parent for n in page_nodes if n.parent.parent is not None)
                    proposed = set()
                    for index, node in enumerate(page_nodes):
                        proposed.add(node)
                        if _may_promote_parent(node, children):
                            witness = witnesses.get(id(node.parent))
                            other = witness[1]() if witness is not None else None
                            if (other is not None and other.parent is node.parent
                                    and other in leaves and other not in proposed):
                                continue
                            may_promote = True
                            for item in page[index + 1:]:
                                heapq.heappush(keys, item)
                            del page[index + 1:]
                            break
                    packet[:len(page) * width] = b"".join(key for _, key in page)
                    packet[-4:-1] = bytes((len(page), bool(keys), may_promote))
                # If the final index page used most of the slot, every rank
                # resumes the ballot next poll with its full budget.
                packet[-1] = indexing or ballot_budget < 2 * _VICTIMS_PER_BATCH
                _reduce_victims(torch.frombuffer(packet, dtype=torch.uint8), dist.ReduceOp.MAX)
                count, more, last_may_promote, building = packet[-4:]
                if building:
                    # Peers are still indexing. Recheck these priorities when
                    # the group can actually commit the proposal.
                    for item in page:
                        heapq.heappush(keys, item)
                    yield None
                    continue
                # A prior slot may have split, locked, evicted, or replaced
                # a node. Check the current tree and full signature afresh.
                leaves, _ = _eviction_order(cache)
                paths = {}
                nodes = []
                votes = []
                # Allow the same number of promoted parents after eviction.
                edge_limit = ballot_budget // (2 * max(1, count))
                for i in range(count):
                    signature = bytes(packet[i * width:(i + 1) * width])
                    reference = by_signature.get(signature)
                    node = reference() if reference is not None else None
                    vote = 0
                    if node is not None and node in leaves and (node.backuped or not node.children):
                        proof = _ballot_signature(node, paths, ancestors, edge_limit)
                        # MIN rejects an absent or changed leaf (0), defers an
                        # incomplete proof (1), and accepts only unanimity (2).
                        vote = 1 if proof is None else 2 * int(proof == signature)
                    votes.append(vote)
                    nodes.append(node)
                paths.clear()
                if count:
                    accepted = torch.tensor(votes, dtype=torch.uint8, device="cpu")
                    _reduce_victims(accepted, dist.ReduceOp.MIN)
                    votes = accepted.tolist()
                    reindex |= 1 in votes
                    # An accepted final child can refill the proposer's heap.
                    # Vote on its parent before re-indexing discarded leaves.
                    more = more or (last_may_promote and votes[-1] == 2)
                    agreed = []
                    for node, vote in zip(nodes, votes):
                        if vote == 2:
                            assert node is not None  # Local membership participated in MIN.
                            agreed.append(node)
                            self.remaining -= len(node.key)
                            if self.remaining <= 0:
                                break
                    if agreed:
                        parents = {node.parent for node in agreed if node.parent.parent is not None}
                        self.freed += _evict(cache, agreed)
                        progressed = True
                        for parent in parents:
                            if parent in leaves and (parent.backuped or not parent.children):
                                parent_key = _ballot_signature(parent, paths, ancestors, edge_limit)
                                if parent_key is None:
                                    continue
                                if proposing and parent_key not in by_signature:
                                    heapq.heappush(keys, (priority(parent), parent_key))
                                by_signature[parent_key] = ref(parent)
                        paths.clear()
                if self.remaining <= 0:
                    yield True
                    return
                if not more and not progressed and not reindex:
                    yield False
                    return
                yield None
                if not more:
                    # A committed leaf can expose a previously ineligible parent.
                    break


def _group_demand(cache: Any, limit: int, changed: bool) -> Tuple[int, bool, Optional[int]]:
    import torch
    import torch.distributed as dist

    rank = dist.get_rank() if dist.is_initialized() else 0
    # Each stage reports excess against its own configured token limit.
    # Stage zero relays whether to vote. An unlimited stage may still
    # release the same prefix to satisfy a peer's finite limit.
    election = -rank if _write_through(cache) else 1
    status = torch.tensor([_excess(cache, limit), int(changed), election], dtype=torch.int64, device="cpu")
    _reduce_victims(status, dist.ReduceOp.MAX)
    excess, changed_flag, election = status.tolist()
    return excess, bool(changed_flag), -election if election <= 0 else None


def _prepare_pipeline_vote(scheduler: Any, cache: Any) -> Callable[[], None]:
    # With three or more stages the last stage can have sent results that
    # PP0 will normally read in a later slot. Receive those results before a
    # vote so the last stage can retire its outstanding sends and join it.
    prefetch: Optional[Callable[[], None]] = None
    if cache.pp_size > 2 and cache.pp_rank == 0:
        from sglang.srt.managers.scheduler_pp_mixin import _pp_can_skip_output_comm

        send = scheduler._pp_send_output_to_next_stage
        recv = scheduler._pp_recv_dict_from_prev_stage
        buffered: Deque[Any] = deque()
        outstanding = 0

        @functools.wraps(send)
        def track_send(next_first_rank_mb_id: int, mbs: Any, last_rank_comm_queue: Any, pp_outputs: Any) -> Any:
            nonlocal outstanding
            target = mbs[next_first_rank_mb_id]
            if target is not None and not target.forward_mode.is_prebuilt() and not _pp_can_skip_output_comm(target):
                outstanding += 1
            return send(next_first_rank_mb_id, mbs, last_rank_comm_queue, pp_outputs)

        @functools.wraps(recv)
        def receive() -> Any:
            nonlocal outstanding
            if buffered:
                return buffered.popleft()
            value = recv()
            outstanding -= 1
            return value

        def prefetch_outputs() -> None:
            nonlocal outstanding
            while outstanding > 0:
                buffered.append(recv())
                outstanding -= 1

        scheduler._pp_send_output_to_next_stage = track_send
        scheduler._pp_recv_dict_from_prev_stage = receive
        prefetch = prefetch_outputs

    def prepare() -> None:
        if not scheduler.pp_group.is_last_rank and scheduler.pp_outputs:
            # The next stage may still need the previous slot's result.
            work = scheduler._pp_send_dict_to_next_stage(
                scheduler.pp_outputs.tensors, async_send=True, msg_type="output")
            scheduler.send_output_work.extend(work)
            scheduler.pp_outputs = None
        if prefetch is not None:
            prefetch()

    return prepare


def parallel_trim_cache(scheduler: Any) -> Optional[Any]:
    """The distributed HiCache whose ranks can agree on device victims, if any."""
    if not getattr(scheduler, "enable_hierarchical_cache", False):
        return None
    # PD multiplexing schedules split prefill directly, bypassing the
    # regular HiCache poll. Keep its existing request-finish enforcer.
    if getattr(scheduler, "enable_pdmux", False):
        return None
    # On 0.5.18/19 the diffusion loop bypasses the prefill HiCache poll.
    # 0.5.20 drains HiCache before choosing the diffusion or prefill path.
    if getattr(scheduler, "dllm_config", None) is not None and not hasattr(scheduler, "_process_hicache_events"):
        return None
    cache = scheduler.tree_cache
    cache = getattr(cache, "inner", cache)
    controller = getattr(cache, "cache_controller", None)
    if controller is None or getattr(cache, "linker", None) is not None or not parallel_cache(cache):
        return None
    if hasattr(cache, "tree_core") and not hasattr(cache.tree_core, "evictable_device_leaves"):
        # Opaque backends cannot enumerate an exact set of logical victims.
        return None
    return cache


def _device_leaves(cache: Any) -> Any:
    tree = getattr(cache, "tree_core", None)
    return getattr(cache if tree is None else tree,
                   "evictable_leaves" if tree is None else "evictable_device_leaves", None)


def shrink_trim_cache(scheduler: Any) -> Optional[Any]:
    """The HiCache a memory-limit shrink can trim with agreed victims, if any.

    Requires the condition evict_parallel_hicache needs: a write-through
    cache whose device leaves the victim protocol tracks. The protocol
    (DistributedHiCacheRetentionPatch, HICACHE_RANGE only) wraps inside the
    scheduler's idle-limit init, so it is installed before this runs. Older
    SGLang keeps the native leaf set, and the pool keeps the growth cap.
    """
    cache = parallel_trim_cache(scheduler)
    if cache is None or not _write_through(cache):
        return None
    return cache if isinstance(_device_leaves(cache), _DeviceLeaves) else None


class _ShrinkTrim:
    """A memory-limit trim in progress, kept across idle passes."""

    def __init__(self, limit: int) -> None:
        self.limit = limit
        self.plan = _VictimPlan()
        self.stamp: Optional[Tuple[Any, ...]] = None


def _shrink_state(cache: Any) -> Tuple[Any, ...]:
    if hasattr(cache, "tree_core"):
        eligible, locked = cache.full_evictable_size(), cache.full_protected_size()
    else:
        eligible, locked = cache.evictable_size_, cache.protected_size_
    return getattr(cache, "root_node", None), eligible, locked


def parallel_hicache_trim_pending(cache: Any) -> bool:
    return getattr(cache, "_kvcached_shrink_trim", None) is not None


def cancel_parallel_hicache_trim(cache: Any) -> None:
    """Drop a pending trim; every rank must drop it on the same pass."""
    cache._kvcached_shrink_trim = None


def evict_parallel_hicache(cache: Any, tokens: int) -> Optional[bool]:
    """Advance a trim of ``tokens`` device tokens with victims every rank agrees on.

    Every rank must call this together. ``tokens`` starts a trim when none
    is pending: each rank trims to its own evictable size less ``tokens``,
    so the group demand is ``tokens``. Each call does one bounded poll, as
    the retention path does, and returns None until the plan reaches a
    verdict; call again (any ``tokens``) to continue. The verdict comes from
    group votes, so every rank keeps or drops the trim on the same call.
    A cache whose device leaves are not tracked evicts nothing.
    """
    if not isinstance(_device_leaves(cache), _DeviceLeaves):
        return False
    trim = getattr(cache, "_kvcached_shrink_trim", None)
    if trim is None:
        if tokens <= 0:
            return True
        size = cache.full_evictable_size() if hasattr(cache, "tree_core") else cache.evictable_size_
        trim = cache._kvcached_shrink_trim = _ShrinkTrim(max(0, size - tokens))
    start = time.perf_counter()
    state = _shrink_state(cache)
    demand, changed, proposer = _group_demand(cache, trim.limit, state != trim.stamp)
    result = trim.plan.poll(cache, demand, proposer, changed, start) if demand else True
    trim.stamp = _shrink_state(cache)
    if result is not None:
        cache._kvcached_shrink_trim = None
    return result


def install_parallel_hicache_retention(scheduler: Any) -> None:
    cache = parallel_trim_cache(scheduler)
    if cache is None:
        return
    prepare_vote = None
    if cache.pp_size > 1:
        # DP-attention replicas can enter the vote independently, while control
        # broadcasts and MLP batch gathers synchronize the entire TP stage.
        if scheduler.ps.attn_dp_size > 1:
            return
        # The regular loop forwards the current requests before polling the
        # cache. The P/D loops forward more control traffic after the poll.
        if scheduler.disaggregation_mode.value != "null":
            return
    original_poll = cache.check_hicache_events
    original_reduce = cache._all_reduce
    count_has_digest = hasattr(getattr(cache, "tree_core", None), "write_back_duplicate_reclaim_digest")
    count_slots = 0
    needs_vote = False
    retry_delay = 0
    skip_votes = 0
    continue_work = False
    pp_probe_in = _RECHECK_POLLS
    enabled: Optional[bool] = None
    periodic_probe = False
    import torch
    import torch.distributed as dist

    # Dispatch can match or prefetch before the first ACK poll. Include
    # every stage here because subsequent PP ACKs only travel downstream.
    limit = get_prefix_cache_retention_limit()
    through = _write_through(cache)
    startup = torch.tensor([limit >= 0, limit == 0, through, not through], dtype=torch.uint8, device="cpu")
    _reduce_victims(startup, dist.ReduceOp.MAX)
    any_cap, any_zero, any_write_through, any_write_back = startup.tolist()
    if cache.pp_size > 1:
        # Runtime retention-limit updates support TP only. An unlimited
        # pipeline can keep its native leaf set and output exchange.
        if not any_cap:
            return
        prepare_vote = _prepare_pipeline_vote(scheduler, cache)
    _track_device_leaves(cache)
    pp_has_cap = cache.pp_size > 1 and bool(any_cap)
    group_zero = bool(any_zero)
    group_write_through = not any_write_back
    group_write_back = not any_write_through
    policy_pending = False
    cold_admission = group_zero and group_write_through
    plan: Optional[_VictimPlan] = None
    stamp: Optional[Tuple[Any, ...]] = None
    blocked_demand: Optional[int] = None
    sessions = getattr(getattr(scheduler, "session_controller", None), "sessions", {})

    def cache_state(limit: int) -> Tuple[Any, ...]:
        if hasattr(cache, "tree_core"):
            eligible, locked = cache.full_evictable_size(), cache.full_protected_size()
        else:
            eligible, locked = cache.evictable_size_, cache.protected_size_
        return (getattr(cache, "root_node", None), len(sessions), limit, eligible, locked)

    def trim(reusable: bool) -> Optional[bool]:
        nonlocal plan, stamp, blocked_demand, retry_delay
        start = time.perf_counter()
        limit = get_prefix_cache_retention_limit()
        state = cache_state(limit) if reusable else None
        changed = state is None or state != stamp
        demand, any_changed, proposer = _group_demand(cache, limit, changed)
        if not demand:
            forget_plan()
            return True
        if not any_changed and demand == blocked_demand:
            return False
        if plan is None:
            plan = _VictimPlan()
        result = plan.poll(cache, demand, proposer, any_changed, start)
        if plan.remaining < demand:
            retry_delay = 0
        stamp = cache_state(limit) if reusable else None
        blocked_demand = plan.remaining if result is False and not plan.search_changed else None
        if result is not None:
            plan = None
        return result

    session_cache = getattr(cache, "session", scheduler.tree_cache)

    def cold_request(req: Any) -> bool:
        # Cold admission is for new requests. Retraction releases the KV row,
        # but the resumed request may reuse its existing host prefix.
        if (not cold_admission or req is None or request_slot(req) is not None
                or req.extend_batch_idx or req.retracted_stain or req.output_ids):
            return False
        if not getattr(req.session, "streaming", False):
            return True
        slot = getattr(session_cache, "slots", {}).get(req.session.session_id)
        kv = getattr(slot, "kv", None)
        # Older slots use None when the row belongs to a request. Newer
        # slots retain the KV record and expose ownership through holds_kv.
        return kv is None or not getattr(kv, "holds_kv", True) or getattr(req, "to_finish", None) is not None

    def set_cold_admission(cold: bool) -> None:
        nonlocal cold_admission
        if cold == cold_admission:
            return
        cold_admission = cold
        ongoing = getattr(cache, "ongoing_prefetch", None)
        if not cold or not ongoing:
            return
        bootstrap = getattr(scheduler, "disagg_prefill_bootstrap_queue", None)
        for req in chain(scheduler.waiting_queue, getattr(bootstrap, "queue", ())):
            if cold_request(req):
                # Unified caches key reads by handle in 0.5.20 and rid in
                # 0.5.18. Legacy caches key them by rid in both versions.
                entry = ongoing.get(getattr(req, "cache_request_handle", None))
                if entry is None:
                    entry = ongoing.get(req.rid)
                if entry is not None:
                    operation = entry[3] if hasattr(cache, "tree_core") else entry[2]
                    # Mark the read for termination before wait_complete can
                    # delay admission. The normal poll owns consensus, pins,
                    # staging buffers and any bytes already transferred.
                    cache.cache_controller.terminate_prefetch(operation)

    def confirm_write_policy() -> None:
        nonlocal group_write_through, group_write_back
        through = _write_through(cache)
        policies = torch.tensor([through, not through], dtype=torch.uint8, device="cpu")
        _reduce_victims(policies, dist.ReduceOp.MAX)
        any_through, any_back = policies.tolist()
        group_write_through, group_write_back = not any_back, not any_through
        set_cold_admission(group_zero and group_write_through)

    route = getattr(scheduler, "_request_dispatcher", None)
    if route is not None:
        @functools.wraps(route)
        def dispatch_request(request: Any) -> Any:
            nonlocal policy_pending
            # The attach request reaches every rank even when a local idle
            # check rejects it before calling the backend.
            if getattr(request, "hicache_write_policy", None) is None:
                return route(request)
            policy_pending = cache.pp_size == 1 or pp_has_cap
            result = route(request)
            if cache.pp_size == 1:
                confirm_write_policy()
                policy_pending = False
            # PP forwards this batch only after dispatch returns.
            return result

        scheduler._request_dispatcher = dispatch_request

    pipeline = getattr(cache, "buffer_pipeline", None)
    rid_pipeline = pipeline if pipeline is not None and not hasattr(pipeline, "prepare_staged_prefetch") else None
    match = getattr(cache, "match_prefix", None)
    if match is not None:
        @functools.wraps(match)
        def match_prefix(params: Any) -> Any:
            nonlocal stamp
            # A match can split the tree or refresh native priority even when
            # the scheduler never admits the request.
            stamp = None
            if cold_request(params.req):
                # New requests start cold across the group, including when
                # another request pins a matching prefix on one stage.
                params = replace(params, key=params.key[:0])
                # 0.5.19 plans a splice by rid after this request match.
                # Drop any completed staging before it can reserve KV.
                if rid_pipeline is not None and params.req.rid in rid_pipeline.staged_prefetches:
                    rid_pipeline.release_staged_hold(params.req.rid)
                if getattr(params.req, "storage_hit_length", 0):
                    params.req.storage_hit_length = 0
                if hasattr(params.req, "storage_prefetch_last_match_len"):
                    params.req.storage_prefetch_last_match_len = None
            return match(params)

        cache.match_prefix = match_prefix

    prefetch = getattr(scheduler, "_prefetch_kvcache", None)
    if prefetch is not None:
        @functools.wraps(prefetch)
        def prefetch_request(req: Any, *args: Any, **kwargs: Any) -> Any:
            if not getattr(scheduler, "enable_hicache_storage", False) or not cold_request(req):
                return prefetch(req, *args, **kwargs)
            if policy_pending and cache.pp_size > 1:
                req._kvcached_pending_prefetch = (args, kwargs)
            return None

        scheduler._prefetch_kvcache = prefetch_request

    decode_queue = getattr(scheduler, "disagg_decode_prealloc_queue", None)
    if (getattr(scheduler, "enable_decode_hicache", False) and decode_queue is not None
            and hasattr(decode_queue, "_build_decode_prefix_match")):
        build_match = decode_queue._build_decode_prefix_match

        @functools.wraps(build_match)
        def decode_prefix(req: Any, result: Any) -> Any:
            if not cold_request(req):
                return build_match(req, result)
            from sglang.srt.disaggregation.decode import DecodePrefixMatch

            # A cold device match still has a root that can be probed for L3.
            # Reserve and advertise only the prefix the request will consume.
            return DecodePrefixMatch(prefix_indices=result.device_indices, l2_host_hit_length=0,
                                     l3_storage_hit_length=0, last_device_node=result.last_device_node)

        decode_queue._build_decode_prefix_match = decode_prefix

    if pipeline is not None and rid_pipeline is None:
        prepare_staged = pipeline.prepare_staged_prefetch

        @functools.wraps(prepare_staged)
        def prepare_prefetch(req: Any) -> bool:
            if cold_request(req):
                # A fetch begun under an older limit may have completed.
                # Release its anchor, host bounce slots and occupancy grant.
                handle = req.cache_request_handle
                if pipeline.has_staged(handle):
                    pipeline.release_staged_hold(handle, reason="dropped")
                    cache.pop_prefetch_loaded_span(handle)
                req.staged_prefetch_plan = None
                req.host_hit_length = req.swa_host_hit_length = 0
                pipeline._clear_storage_hit(req)
                return True
            return prepare_staged(req)

        pipeline.prepare_staged_prefetch = prepare_prefetch

    dispatch: Any = getattr(scheduler, "process_input_requests", None)
    if cache.pp_size > 1 and dispatch is not None:
        @functools.wraps(dispatch)
        def pipeline_requests(recv_reqs: Any) -> Any:
            nonlocal stamp
            if recv_reqs:
                stamp = None
            return dispatch(recv_reqs)

        scheduler.process_input_requests = pipeline_requests

    def forget_plan() -> None:
        nonlocal plan, stamp, blocked_demand, continue_work
        if plan is not None:
            plan.search = None
        plan = stamp = blocked_demand = None
        continue_work = False

    sleep = getattr(scheduler, "maybe_sleep_on_idle", None)
    if sleep is not None:
        @functools.wraps(sleep)
        def sleep_when_trimmed() -> None:
            if plan is None:
                sleep()

        scheduler.maybe_sleep_on_idle = sleep_when_trimmed

    if cache.pp_size == 1:
        # Runtime assignments arrive in the existing TP request broadcast.
        # Dispatch finishes before the normal and overlap loops check pause.
        dispatch = scheduler.process_input_requests
        revision = prefix_cache_retention._policy.applied_revision
        paused = scheduler._engine_paused
        skipped_paused = 0

        @functools.wraps(dispatch)
        def process_requests(recv_reqs: Any) -> Any:
            nonlocal enabled, revision, retry_delay, skip_votes, paused, skipped_paused, stamp, group_zero
            current = prefix_cache_retention._policy.applied_revision
            changed = current != revision
            if changed:
                revision = current
                limit = get_prefix_cache_retention_limit()
                enabled = limit >= 0
                if not enabled:
                    forget_plan()
                group_zero = limit == 0
                set_cold_admission(group_zero and group_write_through)
                retry_delay = skip_votes = 0
            result = dispatch(recv_reqs)
            if changed or paused != scheduler._engine_paused or recv_reqs:
                stamp = None
                skipped_paused = 0
            if scheduler._engine_paused and enabled is not False:
                # An empty paused dispatch can close sessions and complete a
                # deferred cache flush. Both invalidate local priority/path data.
                # Only shared inputs and the poll counter schedule collectives.
                if skipped_paused:
                    skipped_paused -= 1
                elif trim(True) is not None:
                    skipped_paused = _RECHECK_POLLS - 1
            paused = scheduler._engine_paused
            return result

        scheduler.process_input_requests = process_requests

    @functools.wraps(original_reduce)
    def reduce(data: Any, tp_reduce_op: Any) -> Any:
        nonlocal count_slots, needs_vote, enabled, periodic_probe, stamp
        nonlocal retry_delay, skip_votes
        if not count_slots:
            return original_reduce(data, tp_reduce_op)
        # The first reduction in the native poll carries ACK counts. Older
        # PP paths use a scalar write count and poll load ACKs separately.
        # Carry the invitation in that message so disabled and under-cap
        # schedulers add no collective. PP0 forwards the result to each stage.
        limit = get_prefix_cache_retention_limit()
        flat = data.reshape(-1)
        count_slots = max(0, count_slots - flat.numel())
        write_through = _write_through(cache)
        needed = False
        if limit >= 0 and write_through:
            if hasattr(cache, "tree_core"):
                eligible, locked = cache.full_evictable_size(), cache.full_protected_size()
            else:
                eligible, locked = cache.evictable_size_, cache.protected_size_
            needed = eligible > limit or (eligible + locked > limit and bool(flat[:2].any()))
        extended = data.new_empty(flat.numel() + 1)
        extended[:-1].copy_(flat)
        # MIN selects trim now (0), monitor a finite cap (1), or unlimited (2).
        extended[-1] = 0 if needed else 1 + int(limit < 0)
        original_reduce(extended, tp_reduce_op)
        data.copy_(extended[:-1].view_as(data))
        pressure = int(extended[-1])
        enabled = pp_has_cap or pressure < 2
        if not enabled:
            forget_plan()
        periodic_probe = enabled and group_write_through
        needs_vote |= periodic_probe and pressure == 0
        # A write/load completion can unlock a new common victim. Storage
        # queue activity also invalidates an idle index. The final digest pair
        # in unified counts describes persistent state, not a completion.
        activity = flat[:-2] if count_has_digest and flat.numel() >= 4 else flat
        if activity[:2].any():
            retry_delay = skip_votes = 0
        if stamp is not None and activity.any():
            stamp = None
        return None

    @functools.wraps(original_poll)
    def poll(*args: Any, **kwargs: Any) -> Any:
        nonlocal count_slots, needs_vote, retry_delay, skip_votes, plan, stamp, pp_probe_in, continue_work
        nonlocal policy_pending, periodic_probe
        if enabled is False:
            return original_poll(*args, **kwargs)
        count_slots, needs_vote = 2, False
        try:
            result = original_poll(*args, **kwargs)
        finally:
            count_slots = 0
        if policy_pending and prepare_vote is not None:
            prepare_vote()
            confirm_write_policy()
            policy_pending = False
            # A generation request can follow the attachment in the same
            # dispatch. Only a request still waiting can start its L3 read.
            for req in getattr(scheduler, "waiting_queue", ()):
                pending = getattr(req, "_kvcached_pending_prefetch", None)
                if pending is not None:
                    del req._kvcached_pending_prefetch
                    if not cold_admission and scheduler.enable_hicache_storage:
                        scheduler._prefetch_kvcache(req, *pending[0], **pending[1])
            forget_plan()
            retry_delay = skip_votes = 0
            periodic_probe = needs_vote = pp_has_cap and group_write_through
        if prepare_vote is not None and periodic_probe:
            # ACK readiness goes downstream from PP0. A follower can be the
            # only stage above its cap, so sample group demand on shared slots.
            pp_probe_in -= 1
            if pp_probe_in == 0:
                needs_vote = True
                pp_probe_in = _RECHECK_POLLS
        if not (needs_vote or continue_work):
            retry_delay = skip_votes = 0
            if cache.pp_size == 1:
                forget_plan()
        elif skip_votes:
            skip_votes -= 1
        else:
            if prepare_vote is not None:
                prepare_vote()
            # Native idle checks include in-flight host and storage operations.
            # Completed ACKs above and request dispatch invalidate the heap.
            idle = getattr(scheduler, "is_fully_idle", None)
            reusable = dispatch is not None and idle is not None and idle()
            verdict = trim(reusable)
            continue_work = verdict is None
            if verdict is True:
                retry_delay = skip_votes = 0
            elif verdict is False:
                # All ranks observed the same failed vote. Keep the shared poll
                # responsive to changing candidates without a PP vote every slot.
                retry_delay = min(32, 2 * retry_delay + 1)
                skip_votes = retry_delay
        return result

    # Request-finish, standalone load checks, and policy updates can run on
    # one rank. The shared scheduler poll consumes the current policy value.
    enforce = cache._enforce_prefix_cache_retention

    def defer_write_through(limit_tokens: int) -> None:
        # Request finish may run locally. Only unanimous write-back keeps
        # native enforcement while a write-through group uses shared victims.
        if group_write_back and not policy_pending and not _write_through(cache):
            enforce(limit_tokens)

    cache._enforce_prefix_cache_retention = defer_write_through
    cache._all_reduce = reduce
    cache.check_hicache_events = poll


class DistributedHiCacheRetentionPatch(VersionAwarePatch, BasePatch):
    library = "sglang"
    target_module = "sglang.srt.managers.scheduler"
    target_class = "Scheduler"
    patch_name = "distributed_hicache_retention"

    def apply(self, target_module: types.ModuleType) -> bool:
        return self.initialize_version_info() and self.patch_scheduler(target_module)

    @version_range(HICACHE_RANGE)
    def patch_scheduler(self, module: types.ModuleType) -> bool:
        cls = self._get_target_class(module)
        if cls is None:
            return False
        original = cls.__init__
        if self._is_already_patched(original):
            return True

        @functools.wraps(original)
        def initialize(scheduler: Any, *args: Any, **kwargs: Any) -> None:
            original(scheduler, *args, **kwargs)
            if enable_kvcached():
                install_parallel_hicache_retention(scheduler)

        self._mark_as_patched(initialize)
        cls.__init__ = initialize
        return True
