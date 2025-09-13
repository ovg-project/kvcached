## Two-model colocation with kvcached (no controller)

This example shows the minimal, end-to-end setup to colocate two vLLM model servers on the same GPU using kvcached. It focuses on mechanics and observability, not benchmarking.

### What this is (vs benchmarks)
- **This example**: Start two vLLM servers that share a kvcached segment and send a couple requests to each; good for getting started and verifying colocation works.
- **Benchmarks**: Located under `benchmarks/`, they drive request rates, download datasets, and collect metrics for performance analysis.

### Prerequisites
- A working vLLM installation available on PATH (or an activated virtualenv).
- GPU with enough memory for the selected small models (defaults fit on a single consumer GPU).
- Optional: `jq` for pretty-printing responses in `send_requests.sh`.

### Files
- `start_two_models.sh`: Launch two servers with selectable engines per model (`vllm` or `sglang`). Defaults: vLLM+vLLM (shared segment). vLLM+sglang isolates by default.
- `send_requests.sh`: Sends simple completion requests to both servers.

### Quickstart
1) Start two servers (choose engine per model):

```
# Two vLLM instances sharing kvcached
bash start_two_models.sh \
  --engine-a vllm --engine-b vllm \
  --venv-vllm-path ../../engine_integration/vllm-v0.9.2/.venv

# Cross-engine (isolated by default)
bash start_two_models.sh \
  --engine-a vllm --engine-b sglang \
  --venv-vllm-path ../../engine_integration/vllm-v0.9.2/.venv \
  --venv-sgl-path  ../../engine_integration/sglang-v0.4.9/.venv
```

Customize models/ports if needed (works for either case):

```
bash start_two_models.sh \
  --engine-a vllm --engine-b vllm \
  --model-a meta-llama/Llama-3.2-1B --port-a 12346 \
  --model-b Qwen/Qwen3-0.6B        --port-b 12347
```

1) In a separate terminal, send simple requests:

```
bash send_requests.sh 12346 12347
```

You should see responses printed for each server. With default settings, both servers share the same kvcached segment, demonstrating colocation.

### Notes and knobs
- `KVCACHED_IPC_NAME`: Both processes must use the same value to share the same kvcached segment.
- `ENABLE_KVCACHED=true`: Enables kvcached integration in the engine.
- `--enable-sleep-mode`: Enabled by default in the script to make memory behavior easier to observe.
- NVIDIA L4 GPUs: The script adds `--enforce-eager` for vLLM to improve stability on L4.
- Cross-engine sharing: If you want vLLM and sglang to share the same segment, export the same `KVCACHED_IPC_NAME` for both before starting them. Be mindful of allocator and memory pressure interactions.

### Troubleshooting
- If `vllm` is not found, activate your vLLM virtualenv or use `--venv-path`.
- If you hit OOM, switch to smaller models or adjust GPU memory use in vLLM (e.g., `--gpu-memory-utilization 0.5`).
- If `jq` is not installed, remove the `| jq -r` parts in `send_requests.sh`.
