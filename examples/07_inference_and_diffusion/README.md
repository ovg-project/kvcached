# Diffusion and kvcached Integration Example

This example demonstrates running diffusion models alongside LLM servers using kvcached for efficient GPU memory sharing.

## Scripts Overview

### `setup.sh`
Sets up the environment and installs required dependencies for diffusion models.

### `start_inference_and_diffusion.sh`
Runs both diffusion and LLM server concurrently with shared GPU memory.

```bash
./start_inference_and_diffusion.sh [--llm-engine ENGINE] [--llm-model MODEL] [--llm-port PORT] [--llm-venv-path PATH] [--llm-tp-size TP_SIZE] [--diff-model MODEL] [--diff-num-inference-steps N] [--diff-save-images]
# Example: ./start_inference_and_diffusion.sh --llm-engine vllm --diff-num-inference-steps 20 --diff-save-images
```

### `start_diffusion.sh`
Runs the diffusion model standalone.

```bash
./start_diffusion.sh [--venv-path PATH] [--model MODEL] [--num-inference-steps N] [--save-images]
# Example: ./start_diffusion.sh --model stabilityai/stable-diffusion-3.5-medium --num-inference-steps 30 --save-images
```

### `start_llm_server.sh`
Launches an LLM server (vLLM or SGLang) with kvcached integration.

```bash
./start_llm_server.sh <engine> [--venv-path PATH] [--port PORT] [--model MODEL_ID] [--tp TP_SIZE]
# Example: ./start_llm_server.sh vllm --model meta-llama/Llama-3.2-1B --port 12346
```

### `start_llm_client.sh`
Benchmarks the LLM server with ShareGPT dataset.

```bash
./start_llm_client.sh <engine> [--venv-path PATH] [--port PORT] [--model MODEL_ID] [--num-prompts N] [--request-rate R]
# Example: ./start_llm_client.sh vllm --num-prompts 50 --request-rate 2 --port 12346
```

## Usage Examples

1. **Setup environment:**

   ```bash
   ./setup.sh
   ```

2. **Run both concurrently (recommended):**

   ```bash
   ./start_inference_and_diffusion.sh --llm-engine vllm --diff-num-inference-steps 20 --diff-save-images
   # Waiting for both server to start, then
   ./start_llm_client.sh vllm
   ```

3. **Run diffusion only:**

   ```bash
   ./start_diffusion.sh --num-inference-steps 20 --save-images
   ```

4. **Run LLM server only:**

   ```bash
   ./start_llm_server.sh [sglang|vllm] --model meta-llama/Llama-3.2-1B --port 12346
   # Must launch LLM client manually to start sending requests to the LLM server:
   # e.g.,
   # ./start_llm_client.sh [sglang|vllm]
   ```

5. **Run LLM client:**

   ```bash
   ./start_llm_client.sh [sglang|vllm]
   # Test LLM server with a single run
   # ./start_llm_client.sh [sglang|vllm] --num-prompts 10 --request-rate 1 --port 12346
   ```

## Dataset

The example uses the VidProm dataset (`datasets/vidprom.txt`) for diffusion prompts and ShareGPT dataset for LLM benchmarking (automatically downloaded).
