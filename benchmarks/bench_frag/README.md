# bench_frag

Measures how much physical memory stays pinned when retained KV blocks are
scattered across pages ([#359](https://github.com/ovg-project/kvcached/issues/359)).

kvcached returns memory to the GPU one page at a time, and only once every block
on that page is free. A cache bound expressed in tokens therefore says nothing
about how much memory an idle instance actually holds: 1024 retained blocks
packed together occupy 8 pages, while the same 1024 blocks spread one per page
occupy 128. The `waste` column is that ratio.

## Run

```bash
python bench_frag.py
```

Defaults to 16 layers and 16384 blocks of 16 KiB, i.e. 128 blocks per 2 MB page.
Each row frees all but `KEEP` blocks at the given stride and reports the memory
still mapped afterwards.

| column | meaning |
|---|---|
| `stride` | spacing between retained blocks; 1 is packed, 16 is one per page |
| `kept` | blocks retained (constant across rows) |
| `held GB` | memory the retained blocks themselves need |
| `pinned GB` | memory still mapped, from `get_mapped_memory_size()` |
| `waste` | `pinned / held` — 1.0x is ideal, higher means fragmentation |

`stride=1` is the floor: it is what a page-aware policy converges toward. Rows
above it show what scattering costs.

## Scope

This drives `KVCacheManager` directly, so it measures the allocator's behaviour
rather than any engine's eviction policy. It reproduces the conditions #359
describes; it does not by itself exercise `ElasticBlockPool`'s block selection.

# bench_evict

Measures how much memory prefix-cache eviction actually returns. Caches 4096
blocks, touches every stride-th block so an age-only policy spares it, then
evicts down to 512 and reports the memory released.

```bash
python bench_evict.py
```

Measured on an RTX PRO 4000 Blackwell (24GB), 8 layers, 16 KiB blocks, 2 MB
pages (128 blocks per page). Both columns evict the same 3584 blocks:

| stride | freed before (LRU) | freed after (page-aware) |
|---|---|---|
| 1 | 0.84 GB | 0.88 GB |
| 2 | 0.75 GB | 0.88 GB |
| 4 | 0.50 GB | 0.88 GB |
| 8 | 0.03 GB | 0.88 GB |

Age-only eviction degrades as the retained blocks scatter: at stride 8 it evicts
3584 blocks and frees almost nothing, because each surviving block pins a page.
Page-aware selection holds flat, evicting the same count.
