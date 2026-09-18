# Testing

## Running Tests

### With Docker

Using the engine-specific Docker images, run original engine benchmarks:

```bash
# vLLM
export ENABLE_KVCACHED=true && export KVCACHED_AUTOPATCH=1
vllm serve meta-llama/Llama-3.2-1B-Instruct --port=12346
vllm bench serve --model meta-llama/Llama-3.2-1B-Instruct --request-rate 10 --num-prompts 1000 --port 12346

# SGLang
export ENABLE_KVCACHED=true && export KVCACHED_AUTOPATCH=1
python -m sglang.launch_server --model meta-llama/Llama-3.2-1B-Instruct --port 30000
python -m sglang.bench_serving --backend sglang-oai --model meta-llama/Llama-3.2-1B-Instruct --request-rate 10 --num-prompts 1000 --port 30000
```

### From Source

```bash
cd benchmarks/simple_bench
./start_server.sh [sglang|vllm] --venv-path $VENV_PATH --model meta-llama/Llama-3.2-1B-Instruct
./start_client.sh [sglang|vllm] --venv-path $VENV_PATH --model meta-llama/Llama-3.2-1B-Instruct
```

### Controller Tests

```bash
cd tests
python test_sleep_manager.py
python test_traffic_monitor.py
```

## CI Pipeline

The project uses GitHub Actions for:

- Pre-commit hook validation
- MyPy type checking across Python 3.9–3.13
