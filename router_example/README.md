# LLM Router Example

This directory contains a complete example of an LLM router system that provides OpenAI-compatible API endpoints and can route requests to multiple backend model servers. The system is designed to work with SGLang and provides load balancing capabilities for serving multiple large language models.

## Architecture

The system consists of four main components:

- **`server.py`** - HTTP server providing OpenAI-compatible API endpoints
- **`router.py`** - Core routing logic that directs requests to backend model servers
- **`benchmark.py`** - Concurrent benchmarking tool with terminal UI
- **`two_models_config.json`** - Configuration file defining model endpoints

## Features

- **OpenAI-Compatible API** - Support for `/v1/completions` and `/v1/chat/completions` endpoints
- **Multi-Model Support** - Route requests to different models based on request parameters
- **Flexible Configuration** - JSON-based model and endpoint configuration

## Quick Start

### 1. Configuration

Edit `two_models_config.json` to define your models and their endpoints:

```json
{
  "models": {
    "meta-llama/Llama-3.1-8B": {
      "endpoint": {
        "host": "localhost",
        "port": 30101
      },
      "start_command": "../engine_integration/benchmark/start_server.sh sglang 30101 meta-llama/Llama-3.1-8B"
    },
    "Qwen/Qwen2.5-7B-Instruct": {
      "endpoint": {
        "host": "localhost",
        "port": 30102
      },
      "start_command": "../engine_integration/benchmark/start_server.sh sglang 30102 Qwen/Qwen2.5-7B-Instruct"
    }
  }
}
```

### 2. Start Backend Model Servers

Before starting the router, ensure your backend model servers are running on the configured ports. Based on the configuration above:

```bash
# Start first model server
../engine_integration/benchmark/start_server.sh sglang 30101 meta-llama/Llama-3.1-8B

# !!! Wait for the first model server to be ready, you could see the log in the terminal
# [2025-07-21 07:34:00] The server is fired up and ready to roll!

# Start second model server
../engine_integration/benchmark/start_server.sh sglang 30102 Qwen/Qwen2.5-7B-Instruct
```
> Note: If you start two servers at the same time, you may not able to get response

### 3. Start the Router Server

```bash
python server.py --config two_models_config.json --port 8080
```

### Health Checks
```bash
# Overall system health
curl http://localhost:8080/health

# Specific model health (URL encode model names with forward slashes)
curl "http://localhost:8080/health/meta-llama%2FLlama-3.1-8B"
curl "http://localhost:8080/health/Qwen%2FQwen2.5-7B-Instruct"

# Check all models at once
curl http://localhost:8080/health/all

# List available models
curl http://localhost:8080/models
```

**Important:** Model names containing forward slashes (like `Qwen/Qwen2.5-7B-Instruct`) must be URL-encoded when used in path parameters. Replace "/" with "%2F" in URLs.

## Benchmarking

The included benchmarking tool allows you to test multiple models simultaneously with a real-time terminal interface.

### Run Benchmarks manually

```bash
# Start first model client
../engine_integration/benchmark/start_client.sh sglang 8080 meta-llama/Llama-3.1-8B

# Start second model client
../engine_integration/benchmark/start_client.sh sglang 8080 Qwen/Qwen2.5-7B-Instruct
```

### Run Benchmarks with Terminal UI

```bash
python benchmark.py 8080 --config two_models_config.json
```

This will launch a split-screen terminal interface showing benchmark results for both models side by side.

## Configuration Reference

### Model Configuration Structure

```json
{
  "models": {
    "<model-name>": {
      "endpoint": {
        "host": "<hostname>",
        "port": <port-number>,
        "health_check_path": "/health"  // optional, defaults to /health
      },
      "start_command": "<command-to-start-model-server>"  // optional, for documentation
    }
  }
}
```

### Server Command Line Options

```bash
python server.py [OPTIONS]

Options:
  --config PATH        Path to router configuration file (required)
  --port INTEGER       Port to run the server on (default: 8080)
  --log-level LEVEL    Log level (default: INFO)
```

### Benchmark Command Line Options

```bash
python benchmark.py PORT [OPTIONS]

Arguments:
  PORT                 Server port to connect to

Options:
  --config PATH        Configuration file path (default: two_models_config.json)
  --no-curses         Run without terminal interface for debugging
```

## Architecture Details

### Request Flow

1. Client sends request to router server (`/v1/completions` or `/v1/chat/completions`)
2. Router extracts model name from request
3. Router looks up backend endpoint for the specified model
4. Router forwards request to appropriate backend server
5. Router returns response to client (handles both streaming and non-streaming)
