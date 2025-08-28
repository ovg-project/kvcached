# Tensor-Parallel IPC Page Map/Unmap Benchmark

This benchmark measures **overhead** of the *kvcached vmm ops* when using inter-process-communication (IPC) to map / unmap KV cache pages across **tensor-parallel (TP) workers** in vLLM.

The script launches a minimal TP layout:
* A main process that will broadcast vmm ops to all workers, emulating the **scheduler process**
* *tp_size* processes spawned by the main process, which listen on and execute vmm ops, emulating the **GPU worker processes**

For each iteration the scheduler

1. broadcasts a *map* command with a list of page offsets
2. broadcasts an *unmap* command,
3. records timing statistics (mean, p95, max, min, per-page ms).

Different **broadcast strategies** are pluggable (`sequential loop`, `thread pool`, `asyncio`, ...). You can compare their latency under identical workload.

---

## Quick start

```bash
# From repository root
cd benchmarks/bench_tp_ipc

# Run on 4 GPUs, map 1 kv cache page per iteration, run 20 iterations
# using the Python async-await broadcast implementation
python kvcached_tp_ipc_benchmark.py --tp-size 4 --pages-per-iter 1 --iters 20 --map-impl async
```

### CLI Flags

| Flag                |  Default  | Description                                                        |
|---------------------|-----------|--------------------------------------------------------------------|
| --tp-size           | —         | Number of tensor-parallel workers to employ.                   |
| --pages-per-iter    | 1         | How many pages to map per iteration.                |
| --iters             | 20        | Number of benchmark iterations.                                    |
| --map-impl          | seq       | Mapping broadcast implementation to benchmark. **{seq,thread,async}**            |
| -v, --verbose       | off       | Show per-iteration timings.             |
| --num-layers        | 32        | Dummy layer count for alloc_kv_cache(). Only affect the number of virtual pages    |
| --async-sched       | off       | Whether asynchronous scheduling is enabled, input arg of init_kvcached
| --block-size        | 16        | Number of tokens in a cache block. Actually not used by vLLM interfaces.    |
| --not-contiguous        | store_true        | KV tensor layout.    |

### Example
Test different implementations on 4 L40S GPUs with PCIe interconnect.
#### Sequential socket

```bash
python kvcached_tp_ipc_benchmark.py --tp-size 4 --pages-per-iter 1 --iters 20 --map-impl seq
```

```
=== IPC Benchmark Summary ===
Broadcast impl       : seq (sequential_sync.py)
TP size                 : 4
Iterations              : 20
Pages Per Iteration     : 1

+---------------+-----------+------------+
| Metric        |  Map (ms) | Unmap (ms) |
|---------------|-----------|------------|
| mean          |     3.15 |      2.43 |
| p95           |     3.55 |      2.85 |
| max           |     5.95 |      3.38 |
| min           |     2.71 |      2.09 |
|---------------|-----------|------------|
| per-page mean |     3.15 |      2.43 |
+---------------+-----------+------------+

```

```bash
python kvcached_tp_ipc_benchmark.py --tp-size 4 --pages-per-iter 1 --iters 20 --map-impl seq --not-contiguous
```

```
=== IPC Benchmark Summary ===
Broadcast impl       : seq (sequential_sync.py)
TP size                 : 4
Iterations              : 20
Pages Per Iteration     : 1

+---------------+-----------+------------+
| Metric        |  Map (ms) | Unmap (ms) |
|---------------|-----------|------------|
| mean          |     83.93 |      66.93 |
| p95           |     97.70 |      71.24 |
| max           |     134.70 |      79.72 |
| min           |     70.37 |      60.35 |
|---------------|-----------|------------|
| per-page mean |     83.93 |      66.93 |
+---------------+-----------+------------+

```

#### Thread pool

```bash
python kvcached_tp_ipc_benchmark.py --tp-size 4 --pages-per-iter 1 --iters 20 --map-impl thread
```

```
=== IPC Benchmark Summary ===
Broadcast impl       : thread (threadpool.py)
TP size                 : 4
Iterations              : 20
Pages Per Iteration     : 1

+---------------+-----------+------------+
| Metric        |  Map (ms) | Unmap (ms) |
|---------------|-----------|------------|
| mean          |     2.57 |      2.05 |
| p95           |     3.44 |      2.63 |
| max           |     3.94 |      3.08 |
| min           |     2.21 |      1.57 |
|---------------|-----------|------------|
| per-page mean |     2.57 |      2.05 |
+---------------+-----------+------------+

```

```bash
python kvcached_tp_ipc_benchmark.py --tp-size 4 --pages-per-iter 1 --iters 20 --map-impl thread --not-contiguous
```

```
=== IPC Benchmark Summary ===
Broadcast impl       : thread (threadpool.py)
TP size                 : 4
Iterations              : 20
Pages Per Iteration     : 1

+---------------+-----------+------------+
| Metric        |  Map (ms) | Unmap (ms) |
|---------------|-----------|------------|
| mean          |     42.74 |      38.71 |
| p95           |     55.14 |      42.00 |
| max           |     63.71 |      62.54 |
| min           |     35.86 |      33.35 |
|---------------|-----------|------------|
| per-page mean |     42.74 |      38.71 |
+---------------+-----------+------------+

```

#### Asyncio

```bash
python kvcached_tp_ipc_benchmark.py --tp-size 4 --pages-per-iter 1 --iters 20 --map-impl async
```

```
=== IPC Benchmark Summary ===
Broadcast impl       : async (__main__.py)
TP size                 : 4
Iterations              : 20
Pages Per Iteration     : 1

+---------------+-----------+------------+
| Metric        |  Map (ms) | Unmap (ms) |
|---------------|-----------|------------|
| mean          |     2.10 |      1.59 |
| p95           |     2.12 |      2.57 |
| max           |     2.21 |      3.17 |
| min           |     2.04 |      1.40 |
|---------------|-----------|------------|
| per-page mean |     2.10 |      1.59 |
+---------------+-----------+------------+

```

```bash
python kvcached_tp_ipc_benchmark.py --tp-size 4 --pages-per-iter 1 --iters 20 --map-impl async --not-contiguous
```

```
=== IPC Benchmark Summary ===
Broadcast impl       : async (__main__.py)
TP size                 : 4
Iterations              : 20
Pages Per Iteration     : 1

+---------------+-----------+------------+
| Metric        |  Map (ms) | Unmap (ms) |
|---------------|-----------|------------|
| mean          |     35.96 |      40.63 |
| p95           |     61.85 |      58.95 |
| max           |     67.46 |      63.93 |
| min           |     27.70 |      32.40 |
|---------------|-----------|------------|
| per-page mean |     35.96 |      40.63 |
+---------------+-----------+------------+

```

## Adding a new broadcast strategy
* Drop a my_impl.py file into broadcast_map_impl/ that defines:

```python
def broadcast_map_to_kv_tensors(tp_size: int,
                                           offsets: list[int]) -> None:
    ...
# If you use async def, the factory auto-wraps it with asyncio.run().
```

* Register it in get_broadcast_impl.name_map, e.g.

```python
"my_impl":  "my_impl",
```

* Run:

```bash
python ... --map-impl my_impl  ...
```

## Benchmark Notes

### Worker-side Temporal Behavior
Workers handle vmm ops in a synchronous manner -- since each worker executes one operation a time independently. They get instructions via UNIX-domain sockets.

#### Proper --pages-per-iter value
In the benchmark script, the code below

```python
page_ids = [i for i in range(pages_per_iter)] 
offsets  = [pid * PAGE_SIZE for pid in page_ids]
```

synthesize `page_ids` for the mapping operation.
`page_ids` can be arbitrary for the benchmarking, as long as it respects the virtual memory size.

In `kvcached`, alloc_kv_cache() will infer the virtual page count according to the GPU memory capacity and num_layers argument. For example, there are 354 pages on my testing machine (A40 GPU), and it is from:
> 2 (for K/V) x 354 x 2MB x 32 (num_layers) = 45312 MB

So, if `page_ids` has any page_id > 353, the benchmark script will fail due to the CUDA error:
`cuMemUnmap(reinterpret_cast<CUdeviceptr>(vaddr), kPageSize) failed in CUDA driver (1): invalid argument`

Please make sure to have good `page_ids` in your settings.
