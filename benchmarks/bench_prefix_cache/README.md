# vLLM Prefix Cache Tests

This directory contains comprehensive tests for validating prefix caching functionality with kvcached's ElasticBlockPool integration.

## Overview

Prefix caching allows vLLM to reuse KV cache blocks from previously processed prompts. When multiple requests share a common prefix (e.g., few-shot examples), only the first request computes the full KV cache. Subsequent requests reuse the cached blocks for the shared prefix, significantly reducing latency.

**Key Features Tested:**
- ✅ Basic prefix caching and cache hits
- ✅ Speedup with long shared prefixes (1.2-1.5x typical)
- ✅ LRU eviction with small cache sizes

---

## Quick Start

### Basic Prefix Cache Test

The simplest way to verify prefix caching works:

```bash
cd benchmarks/bench_prefix_cache

# Option 1: Automated (recommended)
./run_test.sh [path/to/venv]

# Option 2: Manual
cd ../simple_bench
./start_server.sh vllm --venv-path ../../engine_integration/vllm-pip-venv \
  --model meta-llama/Llama-3.2-1B --port 12346 --tp 1

cd ../bench_prefix_cache
python test_prefix_cache.py --num-questions 10
```

---

## Available Tests

### Test 1: Basic Prefix Cache (`test_prefix_cache.py`)

**Purpose:** Verify basic prefix caching functionality

**Configuration:**
- Default cache size (1000 blocks)
- Short prefix (~500 tokens)
- 5-10 test questions

**Run:**

```bash
./run_test.sh
# or
python test_prefix_cache.py --num-questions 10
```

---

### Test 2: High Speedup with Long Prefix (`test_high_speedup.py`)

**Purpose:** Demonstrate higher speedup with very long shared prefixes

**Configuration:**
- Default cache size (1000 blocks)
- Long prefix (~2000 tokens with 40+ examples)
- 15-20 test questions

**Run:**

```bash
bash run_advanced_tests.sh 1 --num-questions 15
```

**Expected:** 1.2-1.5x speedup (higher with larger models)

---

### Test 3: Cache Eviction (LRU) (`test_eviction.py`)

**Purpose:** Verify LRU eviction works correctly with small cache

**Configuration:**
- **Small cache size (10 blocks)** - must configure before starting server
- 4-5 distinct prefixes
- Multiple requests per prefix

**Run:**

```bash
# Stop any running server
pkill -f "vllm serve"

# Start server with small cache
bash start_server_small_cache.sh > server_eviction.log 2>&1 &
sleep 25  # Wait for server to initialize
curl http://127.0.0.1:12346/health

# Run test (simple or repeated mode)
python test_eviction.py --mode simple --cache-size 10
python test_eviction.py --mode repeated --cache-size 10

# Or use the runner
bash run_advanced_tests.sh 2 --mode repeated
```

---

## Test Modes for Eviction Test

### Simple Mode
- 5 prefixes (A, B, C, D, E)
- 1 request per prefix + 1 revisit to prefix A
- Fast execution (~5 seconds)

```bash
python test_eviction.py --mode simple --cache-size 10
```

### Repeated Mode
- 4 prefixes (A, B, C, D)
- 3 requests per prefix + 1 revisit to prefix A
- More comprehensive test (~15 seconds)

```bash
python test_eviction.py --mode repeated --cache-size 10
```

---

## Configuration

### Environment Variables

Set before starting the vLLM server:

| Variable | Default | Purpose |
|----------|---------|---------|
| `KVCACHED_PREFIX_CACHE_MAX_SIZE` | 1000 | Maximum cache size in blocks |
| `KVCACHED_LOG_LEVEL` | INFO | Logging level (use DEBUG to see cache operations) |
| `KVCACHED_PREFIX_CACHE_ENABLED` | true | Enable/disable prefix caching |

**Example:**

```bash
export KVCACHED_PREFIX_CACHE_MAX_SIZE=10
export KVCACHED_LOG_LEVEL=DEBUG

# Then start server
bash start_server_small_cache.sh > server.log 2>&1 &
```

### Command-Line Arguments

#### test_prefix_cache.py

```bash
python test_prefix_cache.py \
  --model meta-llama/Llama-3.2-1B \
  --port 12346 \
  --host 127.0.0.1 \
  --num-questions 10
```

#### test_high_speedup.py

```bash
python test_high_speedup.py \
  --num-questions 15 \
  --model meta-llama/Llama-3.2-1B \
  --port 12346
```

#### test_eviction.py

```bash
python test_eviction.py \
  --mode simple \              # or 'repeated'
  --cache-size 10 \
  --model meta-llama/Llama-3.2-1B \
  --port 12346
```

---

## Verification & Debugging

### 1. Check Test Output

All tests report success/failure and key metrics:

```
✓ EVICTION DETECTED: Prefix A was likely evicted from cache
  Second request is 1.16x as slow as first (near 1.0 = evicted)
```

### 2. Enable Debug Logging

```bash
export KVCACHED_LOG_LEVEL=DEBUG
# Restart server
```

**Cache population (first request):**

```
[kvcached][DEBUG] Cached block 0 with hash b'...', cache_size=1/10
[kvcached][DEBUG] Cached block 1 with hash b'...', cache_size=2/10
```

**Cache hits (subsequent requests):**

```
[kvcached][DEBUG] Cache hit for hash b'...': block_id=5, refcount=2
```

**Refcount management:**

```
[kvcached][DEBUG] Block 17: decremented prefix cache refcount for hash b'...', can_free=True
[kvcached][DEBUG] Decremented refcount for hash b'...': refcount=0
```

**Successful evictions:**

```
[kvcached][DEBUG] Evicting LRU entry with hash b'...': block_id=0
[kvcached][DEBUG] Evicting LRU entry with hash b'...': block_id=1
```

### 3. Check Server Logs

```bash
# View cache activity
grep -E '(Cached block|Cache hit)' server.log | head -20

# View evictions (for small cache tests)
grep -i 'evicting lru entry' server_eviction.log | head -20

# Count eviction attempts
grep -c 'Failed to evict' server_eviction.log

# Verify cache size configuration
grep 'PrefixCacheManager initialized' server.log
```

### 4. vLLM Metrics

Check cache hit rate from vLLM logs:

```bash
grep 'Prefix cache hit rate' server.log | tail -5
```

---

## Troubleshooting

### No Cache Hits Detected

1. **Verify prefix caching is enabled:**

   ```bash
   grep "enable-prefix-caching" ../simple_bench/start_server.sh
   # Should see: --enable-prefix-caching (line 141)
   # NOT: --no-enable-prefix-caching
   ```

2. **Check cache was initialized:**

```bash
   grep 'PrefixCacheManager initialized' server.log
   ```

1. **Enable debug logging:**

   ```bash
export KVCACHED_LOG_LEVEL=DEBUG
# Restart server and re-run test

   ```

### Low or No Speedup

**Possible causes:**
- Short prefix (less benefit from caching)
- Fast model (smaller absolute time savings)
- CPU-bound generation (caching only helps prefill phase)
- Variance in latency measurements

**Expected speedup:**
- Short prefixes (~500 tokens): 1.2-1.5x
- Long prefixes (~2000 tokens): 1.5-2.5x
- Larger models (7B+): Higher speedup

### Eviction Not Working

1. **Verify you're running the latest code:**

   ```bash
   grep "decrement_refcount" kvcached/kv_cache_manager.py
   # Should find the synchronized refcount decrement in free() method
   ```

1. **Check debug logs show decrements:**

```bash
   grep "decremented prefix cache refcount" server_eviction.log | head -5
   ```

1. **Verify evictions are occurring:**

   ```bash
grep "Evicting LRU entry" server_eviction.log | wc -l
# Should see 10+ successful evictions

   ```

### Server Fails to Start

```bash
# Check port availability
lsof -i :12346

# Kill existing processes
pkill -f "vllm serve"

# Verify model is downloaded
huggingface-cli download meta-llama/Llama-3.2-1B

# Check venv path
ls ../../engine_integration/vllm-pip-venv/bin/vllm
```

## Cache Size Not Applied

```bash
# Check environment variable was set
ps aux | grep vllm
ps eww <PID> | tr ' ' '\n' | grep KVCACHED_PREFIX_CACHE_MAX_SIZE

# Verify in server logs
grep "max_cache_size=" server.log
# Should show: PrefixCacheManager initialized with max_cache_size=10
```

---

## Understanding Test Results

### Latency Patterns

**Cache Miss (First Request):**
- Full KV cache computation
- Blocks are cached with hash

**Cache Hit (Subsequent Requests):**
- Prefix blocks reused from cache
- Only new tokens computed

### Eviction Behavior

**Normal eviction warnings (expected):**
- Occur when all cached blocks have refcount > 0 (actively in use)
- Resolve when requests complete

**Successful evictions (working correctly):**
- Appear as "Evicting LRU entry with hash..." in DEBUG logs
- Cache size stays near configured limit
- Oldest (least recently used) entries removed first

---

## References

- **vLLM Prefix Caching:** https://docs.vllm.ai/en/latest/automatic_prefix_caching
- **kvcached Integration:** `kvcached/integration/vllm/patches.py`
- **PrefixCacheManager:** `kvcached/prefix_cache_manager.py`
- **KVCacheManager:** `kvcached/kv_cache_manager.py`

---

When adding new tests:

1. Follow the naming convention: `test_<feature>.py`
2. Add command-line arguments for model, port, host
3. Print clear success/failure messages
4. Save detailed results to `results/` directory
5. Update this README with test description and usage
6. Include expected output and troubleshooting tips
