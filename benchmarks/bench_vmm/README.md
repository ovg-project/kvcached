# VMM Benchmark

This benchmark measures the latency of various CUDA Virtual Memory Management (VMM) operations.

## Description

The tool benchmarks the following CUDA driver API calls:
- `cuMemAddressReserve`: Reserving a virtual address range.
- `cuMemCreate`: Allocating physical memory.
- `cuMemMap`: Mapping physical memory to a virtual address.
- `cuMemSetAccess`: Setting access permissions for a mapped region.
- `cuMemUnmap`: Unmapping physical memory.

It uses multiple CPU threads to issue these commands in parallel and reports latency statistics (average, p50, p90, p99, and max).

## Building the Benchmark

You need a CUDA-enabled GPU and the CUDA Toolkit installed.

Compile the benchmark`:

```bash
make
```

## Running the Benchmark

Execute the compiled binary:

```bash
./bench_vmm.bin
```

The benchmark parameters (number of threads, page size, etc.) are defined as `constexpr` values at the top of `bench_vmm.cpp` and can be modified before compilation.

## Sample Output on A100

```
Total Free Memory: 84.5442GB
====== cuMemMap ElemSz=1 ======

cuMemAddressReserve (8GB) latency: 19 us

Benchmarking with 1 threads and 4096 pages of size 2MB.
---------------------------------------------------------------------------
Operation      avg (us)       p50 (us)       p90 (us)       p99 (us)       max (us)
---------------------------------------------------------------------------
cuMemCreate    193.32         195.00         339.00         381.00         493.00
cuMemMap       1.45           0.00           4.00           5.00           105.00
cuMemSetAccess 35.99          35.00          42.00          54.00          169.00
cuMemUnmap     25.63          25.00          27.00          39.00          126.00
```
