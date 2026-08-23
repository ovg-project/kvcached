# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0
"""Why page-aware eviction does nothing while traffic is running (CPU only).

A page can only be emptied if *every* allocated block on it is evictable. While
requests are in flight they hold blocks all over the pool, so most pages are
disqualified and the policy quietly falls back to LRU. That is invisible in a
serving run -- you just see memory not moving -- so this measures it directly.

It drives a real ElasticBlockPool with torch and the C extension mocked out (the
same trick tests/test_prefix_cache.py uses), builds a scattered prefix cache,
adds a controllable number of blocks held by "running requests", and reports for
each eviction budget: how many pages were eligible at all, how many came free,
and how many victims pure LRU would have kept.

    python sim_eviction.py                 # sweep active blocks
    python sim_eviction.py --active 1000   # one point
"""
import argparse
import json
import random
import sys
import types
from unittest import mock

BLOCKS_PER_PAGE = 64  # 2MiB page / 32KiB block, i.e. Qwen3-4B geometry

_torch = mock.MagicMock()
_torch.__version__ = "2.6.0"
_torch.cuda.mem_get_info.return_value = (8 * 1024**3, 16 * 1024**3)
sys.modules.setdefault("torch", _torch)
sys.modules.setdefault("torch.cuda", _torch.cuda)
sys.modules.setdefault("torch.utils", _torch.utils)
sys.modules.setdefault("torch.utils.cpp_extension", _torch.utils.cpp_extension)
sys.modules.setdefault("posix_ipc", mock.MagicMock())
sys.modules.setdefault("kvcached.vmm_ops", mock.MagicMock())
sys.modules.setdefault("kvcached.integration.vllm.interfaces", mock.MagicMock())


class Block:

    def __init__(self, block_id):
        self.block_id = block_id
        self.ref_cnt = 0
        self.is_null = False


class Request:

    def __init__(self, block_hashes):
        self.block_hashes = block_hashes


class PageAllocator:
    """Groups block ids by page the way the C++ allocator does."""

    def group_indices_by_page(self, indices, block_mem_size):
        by_page = {}
        for idx in indices:
            by_page.setdefault(idx // BLOCKS_PER_PAGE, []).append(idx)
        return by_page


class Manager:
    """Tracks which blocks are live on which page; that is all eviction reads."""

    def __init__(self, num_blocks):
        self._free = list(range(1, num_blocks))
        self._allocated = {0}  # block 0 is the reserved null block
        self._page_count = {0: 1}
        self.page_allocator = PageAllocator()
        self.block_mem_size = 32 * 1024
        self.page_size = BLOCKS_PER_PAGE * self.block_mem_size

    def _note_alloc(self, ids):
        self._allocated.update(ids)
        for i in ids:
            p = i // BLOCKS_PER_PAGE
            self._page_count[p] = self._page_count.get(p, 0) + 1

    def alloc(self, n, rng=None, spread_pages=None):
        """Hand out random free slots, as a churned pool would."""
        if len(self._free) < n:
            return None
        if rng is None:
            ids, self._free = self._free[:n], self._free[n:]
        else:
            limit = len(self._free)
            if spread_pages:
                limit = min(limit, spread_pages * BLOCKS_PER_PAGE)
            picks = set(rng.sample(range(limit), n))
            ids = [self._free[i] for i in picks]
            self._free = [b for i, b in enumerate(self._free) if i not in picks]
        self._note_alloc(ids)
        return ids

    def free(self, ids):
        for i in ids:
            if i in self._allocated:
                p = i // BLOCKS_PER_PAGE
                self._page_count[p] -= 1
                if not self._page_count[p]:
                    del self._page_count[p]
            self._allocated.discard(i)
            self._free.append(i)

    def available_size(self):
        return len(self._free)

    def get_page_occupancy(self, page_ids):
        return {p: self._page_count.get(p, 0) for p in page_ids}

    def pages_pinned(self):
        return len(self._page_count)


def make_pool(num_blocks, rng, spread_pages):
    mgr = Manager(num_blocks)
    mgr.alloc = lambda n: Manager.alloc(mgr, n, rng, spread_pages)

    mod = types.ModuleType("sim_block_pool")
    mod.BlockPool = object
    mod.KVCacheBlock = Block
    with mock.patch(
            "kvcached.integration.vllm.interfaces.get_kv_cache_manager",
            return_value=mgr):
        from kvcached.integration.vllm.patches import ElasticBlockPoolPatch
        ElasticBlockPoolPatch().inject_elastic_block_pool(mod)
        pool = mod.ElasticBlockPool(num_gpu_blocks=num_blocks, block_size=16,
                                    cell_size=2048, num_layers=36,
                                    enable_caching=True)
    pool.max_cached_blocks = -1  # keep the setup out of the way
    return pool, mgr


def build_cache(pool, cached, rng):
    got = []
    while len(got) < cached:
        n = min(rng.randint(8, 64), cached - len(got))
        blocks = pool.get_new_blocks(n)
        pool.cache_full_blocks(Request([f"h{b.block_id}" for b in blocks]),
                               blocks, 0, n, 16, 0)
        pool.free_blocks(blocks)
        got.extend(blocks)
    return got


def measure(budget, cached, active, num_blocks, spread_pages, seed):
    rng = random.Random(seed)
    pool, mgr = make_pool(num_blocks, rng, spread_pages)
    build_cache(pool, cached, rng)
    if active:
        # Blocks of requests that are still running: they never enter the
        # evictable pool, and a page holding one can never be emptied.
        pool.get_new_blocks(active)

    by_page = {}
    for bid in pool._evictable_blocks:
        by_page[bid // BLOCKS_PER_PAGE] = by_page.get(
            bid // BLOCKS_PER_PAGE, 0) + 1
    occ = mgr.get_page_occupancy(list(by_page))
    eligible = sum(1 for p, c in by_page.items() if c >= occ.get(p, 0))

    lru_window = set(list(pool._evictable_blocks)[:budget])
    before_pages = mgr.pages_pinned()
    before_ids = set(pool._evictable_blocks)
    evicted = pool._evict_blocks_from_pool(budget)
    victims = before_ids - set(pool._evictable_blocks)

    return {"budget": budget, "active": active, "evicted": evicted,
            "pages_total": before_pages, "pages_eligible": eligible,
            "pages_freed": before_pages - mgr.pages_pinned(),
            "victims_lru_would_have_kept": len(victims - lru_window)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cached", type=int, default=1000)
    ap.add_argument("--active", type=int, nargs="+", default=[0, 200, 1000])
    ap.add_argument("--budgets", type=int, nargs="+", default=[4, 16, 64, 128])
    ap.add_argument("--spread-pages", type=int, default=73,
                    help="confine the cache to this many pages, as a real "
                         "churned cache is")
    ap.add_argument("--num-blocks", type=int, default=14000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    rows = [measure(b, args.cached, a, args.num_blocks, args.spread_pages,
                    args.seed)
            for a in args.active for b in args.budgets]
    if args.json:
        print(json.dumps(rows, indent=1))
    else:
        print(f"{'active':>8} {'budget':>7} {'eligible/total':>16} "
              f"{'pages freed':>12} {'LRU would keep':>15}")
        for r in rows:
            print(f"{r['active']:>8} {r['budget']:>7} "
                  f"{str(r['pages_eligible']) + '/' + str(r['pages_total']):>16} "
                  f"{r['pages_freed']:>12} "
                  f"{r['victims_lru_would_have_kept']:>15}")
