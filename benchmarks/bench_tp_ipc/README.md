# Tensor-Parallel IPC Benchmark for *kvcached*

This benchmark measures **overhead** of the *kvcached vmm ops* when using inter-process-communication (IPC) to map / unmap KV cache pages across **tensor-parallel (TP) workers** in vLLM.

The script launches a minimal TP layout:
* A main process that will broadcast vmm ops to all workers, emulating the **scheduler process**
* *tp_size* processes spawned by the main process, who listen on and execute vmm ops, emulating the **GPU worker processes**

For each iteration the scheduler

1. broadcasts a *map* command with a list of page offsets
2. broadcasts an *unmap* command,
3. records timing statistics (mean, p95, max, min, per-page ms).

Different **broadcast strategies** are pluggable (`sequential loop`, `thread pool`, `asyncio`, ...). You can compare their latency under identical workload.

---

## Quick start

```bash
# From repository root
cd kvcached/benchmarks/bench_tp_ipc

# Run on 4 GPUs, map 1 kv cache page per iteration, run 20 iterations
# using the Python async-await broadcast implementation
python kvcached_tp_ipc_benchmark.py --tp-size 4 --pages-per-iter 1 --iters 20 --map-impl async
```

### CLI Flags

| Flag                |  Default  | Description                                                        |
|---------------------|-----------|--------------------------------------------------------------------|
| --tp-size           | —         | Number of tensor-parallel workers to employ.                   |
| --pages-per-iter    | 1         | How many *pages* to map per iteration.                |
| --iters             | 20        | Number of benchmark iterations.                                    |
| --map-impl          | seq       | Mapping broadcast implementation to benchmark. {seq,thread,async}            |
| -v, --verbose       | off       | Show per-iteration timings.             |
| --num-layers        | 32        | Dummy layer count for alloc_kv_cache(). Only affect the number of virtual pages    |
| --async-sched       | off       | Whether asynchronous scheduling is enabled, input arg of init_kvcached
| --block-size        | 16        | Number of tokens in a cache block. Actually not used by vLLM interfaces.    |

### Example
#### Command

```bash
# test with 8 A40 GPUs with PCIe interconnect
python kvcached_tp_ipc_benchmark.py --tp-size 8 --pages-per-iter 1 --iters 20 --map-impl seq
```

#### Output

```

=== IPC Benchmark Summary ===
Broadcast impl       : seq (sequential_sync.py)
TP size                 : 8
Iterations              : 20
Pages Per Iteration     : 1

+---------------+-----------+------------+
| Metric        |  Map (ms) | Unmap (ms) |
|---------------|-----------|------------|
| mean          |     90.25 |     112.66 |
| p95           |    109.56 |     116.83 |
| max           |    140.68 |     144.85 |
| min           |     82.63 |     109.08 |
|---------------|-----------|------------|
| per-page mean |     90.25 |     112.66 |
+---------------+-----------+------------+

```

## Adding a new broadcast strategy
* Drop a my_impl.py file into broadcast_map_impl/ that defines:

```python
def broadcast_map_to_kv_tensors_to_workers(tp_size: int,
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
# Synthetic offsets: contiguous pages → byte-offset list
# 50 is just arbitrary --- but it cannot exceed the number of virtual pages
page_ids = [50 + i for i in range(pages_per_iter)] 
offsets  = [pid * PAGE_SIZE for pid in page_ids]
```

synthesize `page_ids` for the mapping operation.
`page_ids` can be arbitrary for the benchmarking, except that it needs to respect the virtual memory size.

In `kvcached`, alloc_kv_cache() will infer the virtual page count according to the GPU memory capacity and num_layers argument. For example, there are 354 pages on my testing machine (A40 GPU), and it is from:
> 2 (for K/V) x 354 x 2MB x 32 (num_layers) = 45312 MB

So, if `page_ids` has any page_id > 353, the benchmark script will fail due to the CUDA error:
`cuMemUnmap(reinterpret_cast<CUdeviceptr>(vaddr), kPageSize) failed in CUDA driver (1): invalid argument`

Please make sure to have good `page_ids` in your settings.
