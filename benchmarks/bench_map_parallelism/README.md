# kvcached Map Parallelism Benchmark

This benchmark measures how **CUDA VMM mapping** scales with process-level parallelism using *kvcached*'s memory ops—**without** the TP IPC layer. This benchmark is motivated by an observation that the async TP IPC implementation doesn't not achieve near ×TP speedup in VMM operations. It compares:

- **Serial:** 1 process maps **N** pages.
- **Parallel:** **P** processes map **N/P** pages **concurrently**.
- **Single Process:** 1 process maps **N/P** pages

Each worker process
1. initializes kvcached,
2. allocates a virtual KV-cache, which reserves some virtual memory,
3. maps a specified range of pages per iteration. Workers use a **per-iteration barrier** so each iteration’s mapping starts at the same time across processes.

---

## Quick Start

```bash
# From repository root
cd benchmarks/bench_map_parallelism

# Example Run: 
# 100 pages total, 4 procs map 25 each on distinct GPUs (cuda:rank)
# 20 trials 
python kvcached_map_parallel_benchmark.py --pages-total 100 --procs 4 --iters 20
```

## CLI Flags

| Flag        |  Default  | Description      |
|-------------|-----------|------------------|
| --pages-total N | —         | Total pages to map per iteration.        |
| --procs P       | 1         | Number of worker processes.      |
| --iters K       | 20        | Number of benchmark iterations.      |
| --not-contiguous       | store_true        | KV tensor layout.      |

## Example Output
Tests are conducted on 4 L40S GPUs with PCIe interconnect.

```bash
python kvcached_map_parallel_benchmark.py --pages-total 100 --procs 4 --iters 20
```

```bash
Using layout contiguous=True

========== Single Process case (1 proc maps N/P pages) ==========
Processes           : 1
Iterations          : 20
Pages total / iter  : 25
Pages per proc / iter: 25 

+-----------------+-------------+
| Metric          | ms          |
+-----------------+-------------+
| mean wall       |     14.249 |
| p95 wall        |     15.447 |
| max wall        |     18.132 |
| min wall        |     3.977 |
+-----------------+-------------+

========== Serial case (1 proc maps N pages) ==========
Processes           : 1
Iterations          : 20
Pages total / iter  : 100
Pages per proc / iter: 100 

+-----------------+-------------+
| Metric          | ms          |
+-----------------+-------------+
| mean wall       |    52.083 |
| p95 wall        |    56.886 |
| max wall        |    57.245 |
| min wall        |    15.631 |
+-----------------+-------------+

========== Parallel case (4 procs map N/P each) ==========
Processes           : 4
Iterations          : 20
Pages total / iter  : 100
Pages per proc / iter: 25 

+-----------------+-------------+
| Metric          | ms          |
+-----------------+-------------+
| mean wall       |     26.650 |
| p95 wall        |     30.373 |
| max wall        |     31.055 |
| min wall        |     17.700 |
+-----------------+-------------+

Speedup (parallel vs serial): mean x1.94   (ideal being approximately 4.00 if perfect overlap)
```

```bash
python kvcached_map_parallel_benchmark.py --pages-total 100 --procs 4 --iters 20 --not-contiguous
```

```bash
Using layout contiguous=False

========== Single Process case (1 proc maps N/P pages) ==========
Processes           : 1
Iterations          : 20
Pages total / iter  : 25
Pages per proc / iter: 25 

+-----------------+-------------+
| Metric          | ms          |
+-----------------+-------------+
| mean wall       |     261.651 |
| p95 wall        |     270.385 |
| max wall        |     273.408 |
| min wall        |     255.804 |
+-----------------+-------------+

========== Serial case (1 proc maps N pages) ==========
Processes           : 1
Iterations          : 20
Pages total / iter  : 100
Pages per proc / iter: 100 

+-----------------+-------------+
| Metric          | ms          |
+-----------------+-------------+
| mean wall       |    1055.820 |
| p95 wall        |    1070.830 |
| max wall        |    1083.050 |
| min wall        |    1044.460 |
+-----------------+-------------+

========== Parallel case (4 procs map N/P each) ==========
Processes           : 4
Iterations          : 20
Pages total / iter  : 100
Pages per proc / iter: 25 

+-----------------+-------------+
| Metric          | ms          |
+-----------------+-------------+
| mean wall       |     950.014 |
| p95 wall        |    1006.295 |
| max wall        |    1014.937 |
| min wall        |     869.605 |
+-----------------+-------------+

Speedup (parallel vs serial): mean x1.11   (ideal being approximately 4.00 if perfect overlap)
```

### Interpretation
- **wall** is per-iteration wall-clock, computed as the **max** across ranks (the slowest rank determines completion).

## Side Notes
### Valid Page Numbers
`--pages-total` should respect the maximum number of pages calculated in kvcached:

```python
num_pages = gpu_mem_size // num_layers // 2 // PAGE_SIZE
```

That is, for each one page regarding `--pages-total`, the `map_to_kv_tensors` operation of `kvcached` will map one PAGE_SIZE-sized (say, 2MB) page for each of K and V tensors and for each layer.
