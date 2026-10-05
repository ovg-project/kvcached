# VMM Benchmark

This benchmark measures the latency of various GPU Virtual Memory Management (VMM) operations on both NVIDIA (CUDA) and AMD (ROCm/HIP) GPUs.

## Description

The tool benchmarks the following VMM API calls:
- `address_reserve`: Reserving a virtual address range.
- `mem_create`: Allocating physical memory.
- `mem_map`: Mapping physical memory to a virtual address.
- `set_access`: Setting access permissions for a mapped region.
- `mem_unmap`: Unmapping physical memory.

It uses multiple CPU threads to issue these commands in parallel and reports latency statistics (average, p50, p90, p99, and max).

## Building the Benchmark

You need a GPU with VMM support and the corresponding toolkit installed (CUDA Toolkit or ROCm).

For NVIDIA GPUs (default):

```bash
make
```

For AMD GPUs:

```bash
make KVCACHED_BACKEND=hip
```

## Running the Benchmark

Execute the compiled binary:

```bash
./bench_vmm.bin
```

The benchmark parameters (number of threads, page size, etc.) are defined as `constexpr` values at the top of `bench_vmm.cpp` and can be modified before compilation.

## Sample Output on A100

```
Backend: CUDA
Total Free Memory: 84.5442GB
====== VMM Benchmark ======

address_reserve (8GB) latency: 19 us

Benchmarking with 1 threads and 4096 pages of size 2MB.
---------------------------------------------------------------------------
Operation      avg (us)       p50 (us)       p90 (us)       p99 (us)       max (us)
---------------------------------------------------------------------------
mem_create     193.32         195.00         339.00         381.00         493.00
mem_map        1.45           0.00           4.00           5.00           105.00
set_access     35.99          35.00          42.00          54.00          169.00
mem_unmap      25.63          25.00          27.00          39.00          126.00
```
