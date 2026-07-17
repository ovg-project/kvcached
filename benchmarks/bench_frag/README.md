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
