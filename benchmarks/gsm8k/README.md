# GSM8K Benchmarking with kvcached

This folder contains scripts to evaluate GSM8K accuracy and throughput against engines with kvcached enabled.

## Start the engine server

Start the vLLM or SGLang engine server with the same commands as in `benchmarks/simple_bench/start_server.sh`. Can use math models like `Qwen/Qwen2.5-Math-1.5B`.

## Run GSM8K client

Open a new terminal at the repo root and go to this folder:

```bash
cd benchmarks/gsm8k
```

### vLLM client

Use `bench_vllm.py` to hit an OpenAI-compatible vLLM server (`/v1/completions`). You must specify `--model` matching the served model. Need to be the same model as server, like `Qwen/Qwen2.5-Math-1.5B`. 

```bash
# Activate the same venv as the server if needed
source /path/to/your/vllm-venv/bin/activate

python bench_vllm.py \
  --model meta-llama/Llama-3.2-1B \
  --port 12346 \
  --num-questions 100 \
  --num-shots 5 \
  --parallel 8 \
  --max-tokens 512 \
  --result-file result.jsonl \
  --raw-result-file eval_raw.jsonl
```

### SGLang client

Use `bench_sglang.py` which integrates with SGLang's testing utilities and backends. Provide standard SGLang connection flags via its common args.

```bash
# Activate the same venv as the server if needed
source /path/to/your/sglang-venv/bin/activate

python bench_sglang.py \
  --port 30000 \
  --num-questions 100 \
  --num-shots 5 \
  --parallel 8 \
  --result-file result.jsonl \
  --raw-result-file eval_raw.jsonl
```

### Data and outputs

- Input dataset default: `test.jsonl` in this directory; auto-downloaded if missing.
- Results summary: appended to `result.jsonl` by default.
- Raw per-request output: written to `eval_raw.jsonl` by default.

You can override these paths with `--data-path`, `--result-file`, and `--raw-result-file`.

### Reusing the same venv as simple bench

- Always use the same engine venv for both server and client.
- Prefer activating the venv before running clients, or pass `--venv-path` when using the simple bench scripts for servers.
- Example venvs (from `engine_integration/scripts/setup.sh`):
  - vLLM: `/home/you/kvcached/engine_integration/vllm-pip-venv`
  - SGLang: `/home/you/kvcached/engine_integration/sglang-pip-venv`

## Troubleshooting

- Ensure the server is reachable: vLLM default `127.0.0.1:12346`, SGLang default `127.0.0.1:30000`.
- Match `--model` in the client to the exact model served by the server.
- On NVIDIA L4 GPUs, the simple bench server script automatically sets engine-specific flags for compatibility.
