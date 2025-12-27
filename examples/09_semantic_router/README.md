# Semantic Router Example

This example demonstrates how to build a semantic router that directs requests to specialized models based on query content.

## Overview

Instead of routing by model name, a semantic router analyzes the user's query and routes to the most appropriate model:

- **Code queries** -> Code-specialized model (e.g., CodeLlama, DeepSeek-Coder)
- **Math queries** -> Math-specialized model (e.g., Qwen-Math, DeepSeek-Math)
- **General queries** -> General-purpose model (e.g., Llama, Qwen)

## Architecture

```
User Request
     |
     v
+------------------+
| Semantic Router  |  <- Classifies query intent
+------------------+
     |
     +---> Code Model (port 12346)
     |
     +---> Math Model (port 12347)
     |
     +---> General Model (port 12348)
```

## Prerequisites

1. Install sentence-transformers for semantic similarity:
   ```bash
   pip install sentence-transformers
   ```

2. Start multiple model servers with kvcached:
   ```bash
   export ENABLE_KVCACHED=true
   export KVCACHED_AUTOPATCH=1

   # Start code model
   vllm serve deepseek-ai/deepseek-coder-1.3b-instruct --port 12346 &

   # Start math model
   vllm serve Qwen/Qwen2.5-Math-1.5B-Instruct --port 12347 &

   # Start general model
   vllm serve Qwen/Qwen2.5-1.5B-Instruct --port 12348 &
   ```

## Usage

1. Start the semantic router:
   ```bash
   python semantic_router.py --port 8080
   ```

2. Send requests (model selection is automatic):
   ```bash
   # This will route to the code model
   curl http://localhost:8080/v1/chat/completions \
     -H "Content-Type: application/json" \
     -d '{"messages": [{"role": "user", "content": "Write a Python function to sort a list"}]}'

   # This will route to the math model
   curl http://localhost:8080/v1/chat/completions \
     -H "Content-Type: application/json" \
     -d '{"messages": [{"role": "user", "content": "Solve the integral of x^2 dx"}]}'

   # This will route to the general model
   curl http://localhost:8080/v1/chat/completions \
     -H "Content-Type: application/json" \
     -d '{"messages": [{"role": "user", "content": "What is the capital of France?"}]}'
   ```

## Configuration

Edit `config.yaml` to customize:
- Model endpoints
- Category keywords/embeddings
- Similarity thresholds

## How It Works

1. **Query Embedding**: The router embeds the user query using a lightweight model
2. **Category Matching**: Compares query embedding to category embeddings
3. **Routing Decision**: Routes to the model with highest similarity score
4. **Fallback**: Uses general model if no category exceeds threshold

## Monitoring

Check routing statistics:
```bash
curl http://localhost:8080/stats
```

Output:
```json
{
  "total_requests": 100,
  "routes": {
    "code": 35,
    "math": 25,
    "general": 40
  }
}
```
