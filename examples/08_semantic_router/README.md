# KVCached with vLLM Semantic Router Example

This example demonstrates running KVCached with the vLLM Semantic Router for intelligent model routing based on query content. The semantic router automatically routes requests to the most appropriate model endpoint using BERT-based classification.

## Scripts Overview

### `setup_semantic_router.sh`
Sets up the semantic router by cloning the repository, downloading classification models (≈1.5GB, first run only), and applying necessary patches.

```bash
./setup_semantic_router.sh --model-a MODEL_A --model-b MODEL_B
# Example: ./setup_semantic_router.sh --model-a "Qwen/Qwen2.5-1.5B-Instruct" --model-b "Qwen/Qwen1.5-1.8B"
```

### `start_vllm_servers.sh`
Starts two vLLM servers with KVCached integration for the specified models.

```bash
./start_vllm_servers.sh --model-a MODEL_A --model-b MODEL_B
# Example: ./start_vllm_servers.sh --model-a "Qwen/Qwen2.5-1.5B-Instruct" --model-b "Qwen/Qwen1.5-1.8B"
```

### `test_routing.sh`
Tests the semantic router by sending various queries to verify correct model routing.

```bash
./test_routing.sh --model-a-name MODEL_A_NAME --model-b-name MODEL_B_NAME
# Example: ./test_routing.sh --model-a-name "Qwen2.5-1.5B-Instruct" --model-b-name "Qwen1.5-1.8B"
```

## Usage Examples

1. **Setup semantic router:**

   ```bash
   ./setup_semantic_router.sh --model-a "Qwen/Qwen2.5-1.5B-Instruct" --model-b "Qwen/Qwen1.5-1.8B"
   ```

2. **Start semantic router stack:**

   ```bash
   cd semantic-router
   docker compose up --build
   ```

3. **Start vLLM servers (in new terminal):**

   ```bash
   ./start_vllm_servers.sh --model-a "Qwen/Qwen2.5-1.5B-Instruct" --model-b "Qwen/Qwen1.5-1.8B"
   ```

4. **Test routing:**

   ```bash
   ./test_routing.sh --model-a-name "Qwen2.5-1.5B-Instruct" --model-b-name "Qwen1.5-1.8B"
   ```

5. **Manual test with curl:**

   ```bash
   curl -X POST http://localhost:8801/v1/chat/completions \
     -H "Content-Type: application/json" \
     -d '{"model": "auto", "messages": [{"role": "user", "content": "What is the derivative of x^4?"}]}'
   ```

## Model Routing

The semantic router uses BERT-based classification to route requests:

- **Model A (port 12346)**: Handles business, law, psychology, biology, chemistry, history, health, and general queries
- **Model B (port 12347)**: Specializes in economics, math, physics, computer science, philosophy, and engineering

## Monitoring

- **Grafana Dashboard**: http://localhost:3000 (admin/admin)
- **Prometheus**: http://localhost:9090
