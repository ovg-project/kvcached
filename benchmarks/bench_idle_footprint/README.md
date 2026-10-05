# bench_idle_footprint

How much physical memory does a kvcached instance still hold once it goes idle?

That is the question behind [#359](https://github.com/ovg-project/kvcached/issues/359).
`KVCACHED_MAX_CACHED_TOKENS` bounds the prefix cache in **blocks**, but memory
comes back a **page** at a time, and only once every block on that page is free.
So what an idle instance holds is decided by how many pages its surviving blocks
are spread over — not by how many blocks it kept.

## Scripts

| | |
|---|---|
| `run_idle_footprint.py` | how much memory is still held once the instance is idle — the headline number, plus a per-second timeline |
| `run_reuse_after_idle.py` | what the eviction policy's victim choice costs in hit rate |
| `sim_eviction.py` | why a page-aware policy does nothing while traffic is running (CPU only, no GPU or vLLM needed) |
| `workload.py`, `probe_mem.py` | the load generator and the memory probe the two runners drive |

## Run

```bash
MODEL=/path/to/Qwen3-4B ./run_idle_footprint.py
```

Run it once per branch and compare the `idle_gb` line. Useful knobs:

```bash
./run_idle_footprint.py --workload "--requests 1000 --concurrency 32"
./run_idle_footprint.py --serve-arg --enforce-eager
```

## What it measures

`idle_gb` comes from the number kvcached publishes about itself, in its MemInfo
shared-memory segment:

```
used_size = num_inuse_pages * num_layers * page_size * num_kv_buffers
```

`nvidia-smi` is no use here — it also counts weights and activations. Note the
`num_layers * num_kv_buffers` factor: one page id is mapped in every layer and
in both the K and V buffers, so on a 36-layer model a 2 MiB page id costs
**144 MiB** of physical memory.

`idle_prealloc_gb` is reported separately: `free_page()` parks up to
`KVCACHED_MAX_RESERVED_PAGES` (default 10) pages without unmapping them, and
only `trim()` returns those.

## The workload shape matters

The reproduction hinges on **staggered decode lengths**, which is easy to miss:

| workload | idle pages |
|---|---|
| 4000 requests, concurrency 32, `max_tokens=4` | 27 |
| 5000 requests, concurrency 96, varied prompts, `max_tokens=4` | 27 |
| **2500 requests, concurrency 96, varied prompts, `max_tokens` in [16,256]** | **73** |

Prefill asks for many blocks at once and the allocator drains one page before
moving to the next, so prefill-shaped traffic packs naturally. Decode asks for
**one block at a time**, and with many requests decoding at different rates and
finishing at different moments, those single-block allocations land in whatever
holes exist across many pages. A benchmark that only stresses prefill will
report kvcached as near-optimal and miss the problem entirely.

`workload.py` therefore mixes hot prefixes (revisited, so an age-only eviction
policy never drops them) with a long cold tail, and samples both prompt and
decode lengths. Prompts are sent as token-id lists so prefix sharing is exact.

## Reference numbers

NVIDIA L40, Qwen3-4B (36 layers), vLLM 0.22.1, default
`KVCACHED_MAX_CACHED_TOKENS=16000`, so the cap is 1000 blocks and one page holds
64 of them. Two runs per row, same workload and seed:

```
--requests 2500 --concurrency 96 --hot-prefixes 16 --hot-tokens 128
--suffix-tokens 64 --cold-tokens-min 192 --cold-tokens-max 1024
--hot-ratio 0.3 --max-tokens-min 16 --max-tokens-max 256 --seed 1234
```

| | idle GB | pages | vs packed floor |
|---|---|---|---|
| before #390 | 10.27 / 10.55 | 73 / 75 | 4.6x |
| \+ #390 page-aware eviction | 5.20 / 5.34 | 37 / 38 | 2.3x |
| \+ page-aware allocation | 3.09 / 3.09 | 22 / 22 | **1.4x** |
| packed floor | 2.25 | 16 | 1.0x |

The **packed floor** is what the cap would cost with no holes at all:
`ceil(1000 blocks / 64 per page) = 16 pages = 2.25 GB`. No eviction or
allocation policy can go below it without caching fewer blocks than the cap
allows. (It equals the raw KV bytes of 1000 blocks, which is the same statement:
packed means no holes.)

Prefix-cache hits were identical across all rows (93,952) and throughput was
11.7–11.9 req/s throughout.

## What eviction costs in hit rate

That last point needs care, which is what `run_reuse_after_idle.py` is for. Two
eviction policies make **identical** choices while traffic is running: live
requests hold blocks all over the pool, so hardly any page is fully evictable
and a page-aware policy has nothing to choose from. `sim_eviction.py` shows this
directly — with 1000 blocks held by running requests, **0 of 89** pages are
eligible at any eviction budget:

```
  active  budget   eligible/total  pages freed
       0      64            86/87           14
     200      64             8/89            5
    1000      64             0/89            0
```

The policies only diverge once the pool goes quiet. So the measurement is: send
traffic, let it go idle so the trim runs, then send more — and have that second
round ask for the **cold** prompts the first round cached. Hot prefixes survive
under any policy and fresh prompts were never cached, so neither can tell two
policies apart; `--cold-pool N` gives both rounds one fixed set of prompts to
draw from, and the hit rate afterwards reads out what the trim kept.

| | memory while idle | hit rate after idle |
|---|---|---|
| before #390 | 9.00 / 8.86 GB | 0.4200 / 0.4200 |
| \+ #390 page-aware eviction | 4.92 / 4.78 GB | 0.4138 / 0.4121 |

Do not carry that ~0.70 pp anywhere: the prompts are random token ids and the
reuse pattern is uniform sampling from a 200-prompt pool, nothing a real
deployment produces. It establishes only that the cost is non-zero and small
here.
