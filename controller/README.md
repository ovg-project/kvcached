# kvcached Multi-LLM Controller & Router

This directory contains a complete example of a **multi-LLM serving stack** built on top of kvcached.
It exposes unified **OpenAI-compatible** HTTP endpoints that transparently route requests to one of many backend model servers (SGLang or vLLM).

## Components

| File | Purpose |
|------|---------|
| `example-config.yaml` | Single YAML file that defines **all** engines, environment variables, and router options. |
| `frontend.py` | HTTP server that implements the OpenAI API (`/v1/completions`, `/v1/chat/completions`, …). |
| `router.py`   | Lightweight routing layer that forwards each request to the correct backend model based on the `model` field. |
| `launch.py`   | One-shot controller that spins up every configured model **and** the router in their own *tmux* sessions. |
| `benchmark.py`| Utility that launches load-generation clients against the router (each in its own *tmux* session) for benchmarking. |

## Features

* **Declarative YAML configuration** – Define engines, ports, environment overrides, virtual-envs, and router settings in a single place.
* **OpenAI API compatibility** – Supports both `/v1/completions` and `/v1/chat/completions` with streaming and non-streaming responses.
* **Multi-model routing** – Provides a unified IP and port as frontend. All requests can be sent to this unified frontend. The router will route the request to the corresponding backend based on the model named in the request.
* **tmux-based process management** – Every engine instance, the router, and optional benchmark clients run in isolated *tmux* sessions.

---

## Quick Start

### 1. Create/adjust your configuration

Change `example-config.yaml` to match your hardware and model choices.

### 2. Launch everything (engines + frontend with router)

```bash
python launch.py --config example-config.yaml
```

`launch.py` will:
1. Create one *tmux* session per engine (`kvcached-<name>`).
2. Optionally create a `kvcached-frontend` session that runs the router (if `enable_router: true`).

Attach with `tmux attach -t kvcached-<name>` or `tmux ls` to inspect logs.

### 3. Talk to the router

```bash
curl http://localhost:8080/v1/completions \
     -H "Content-Type: application/json" \
     -d '{"model": "meta-llama/Llama-3.2-1B", "prompt": "Hello"}'
```

### Health & introspection

```bash
# Router health
curl http://localhost:8080/health

# Per-model health (URL-*encode* slashes)
curl "http://localhost:8080/health/meta-llama%2FLlama-3.2-1B"

# See which models the router currently knows about
curl http://localhost:8080/models

# Get server information
curl http://localhost:8080/get_server_info
```

### Traffic Monitoring Examples

```bash
# Get traffic statistics for all models
curl http://localhost:8080/model/traffic/stats

# Get traffic statistics for specific model with 120-second window
curl "http://localhost:8080/model/traffic/stats/meta-llama%2FLlama-3.2-1B?window=120"

# Get idle models (idle for more than 5 minutes)
curl "http://localhost:8080/model/traffic/idle?threshold=300"

# Get active models with 60-second rate calculation window
curl "http://localhost:8080/model/traffic/active?threshold=300&window=60"
```

### Sleep Management Examples

```bash
# Get sleep status of all models
curl http://localhost:8080/model/sleep/status

# Put a model to sleep
curl "http://localhost:8080/model/sleep/meta-llama%2FLlama-3.2-1B" -X POST

# Wake up a sleeping model
curl "http://localhost:8080/model/wake/meta-llama%2FLlama-3.2-1B" -X POST

# Get sleep candidates
curl http://localhost:8080/model/sleep/candidates
```

---

## Stand-alone router (skip `launch.py`)
Already have your engines running?  Spin up only the router:

```bash
python frontend.py --config example-config.yaml --port 8080
```

`frontend.py` will parse the YAML, extract the host/port for each instance, and start routing immediately.

---

## Benchmarking

```bash
python benchmark.py --config example-config.yaml
```

A separate *tmux* session (`benchmark-<name>`) is created for every model so you can watch latency/throughput side by side.

## Testing Scripts

### Basic Functionality Test

1. **Start the controller server:**

```bash
# Activate the vLLM virtual environment
source engine_integration/vllm-v0.9.2/.venv/bin/activate

# Navigate to controller directory
cd controller

# Launch all services (engines + frontend router)
python launch.py --config example-config.yaml
```

1. **Test a simple completion request:**

```bash
curl http://localhost:8081/v1/completions \
     -H "Content-Type: application/json" \
     -d '{"model": "Qwen/Qwen3-0.6B", "prompt": "what is the result of 1+1=?"}'
```

**Expected response:**

```json
{
  "id": "cmpl-e8ec1629d85441d195f39770a2a48184",
  "object": "text_completion",
  "created": 1756218965,
  "model": "Qwen/Qwen3-0.6B",
  "choices": [
    {
      "index": 0,
      "text": "?\n\nIt's a simple math problem, right? So, if I have ",
      "logprobs": null,
      "finish_reason": "length",
      "stop_reason": null,
      "prompt_logprobs": null
    }
  ],
  "usage": {
    "prompt_tokens": 10,
    "total_tokens": 26,
    "completion_tokens": 16,
    "prompt_tokens_details": null
  },
  "kv_transfer_params": null
}
```

### Traffic Monitoring Test

1. **Run comprehensive traffic monitoring tests:**

```bash
# Navigate to test directory
cd test

# Run traffic monitoring test suite
python test_traffic_monitor.py > test_monitor.txt

# View test results
cat test_monitor.txt
```

1. **Monitor server logs:**

```bash
# Attach to the frontend tmux session to view logs
tmux attach -t kvcached-frontend

# Or list all running sessions
tmux ls
```

## Traffic Statistics (traffic_monitor.py)

### Core Monitoring Features
* **Request tracking**: Records all requests per model with timestamps
* **Request rate calculation**: Calculates requests/second over configurable time windows
* **Response time monitoring**: Tracks average response times
* **Success/failure rates**: Monitors request success rates
* **Idle time detection**: Identifies models with no recent activity

### Data Collection
* **Real-time statistics**: Updates statistics for each request
* **Historical data**: Maintains request history for trend analysis
* **Configurable windows**: Supports custom time windows for rate calculations
* **Thread-safe operations**: Uses locks to ensure data consistency

## Sleep Management (sleep_manager.py)

### Sleep Configuration
* **Idle threshold**: Configurable idle time before sleep (default: 300s)
* **Check interval**: How often to check for idle models (default: 60s)
* **Auto-sleep enabled**: Toggle automatic sleep functionality
* **Wake-on-request**: Toggle automatic wake functionality
* **Minimum sleep duration**: Minimum time to keep model asleep (default: 60s)

## API Endpoints
(frontend.py)

### Core OpenAI-Compatible Endpoints
* **POST /v1/completions** - Text completion API
* **POST /v1/chat/completions** - Chat completion API
* **GET /health** - Router health check
* **GET /models** - List all configured models
* **GET /health/{model_name}** - Specific model health check (URL encode model names)
* **GET /get_server_info** - Server information

### Traffic Monitoring Endpoints
* **GET /model/traffic/stats?window=60** - All models' traffic statistics (window: time window in seconds for rate calculation)
* **GET /model/traffic/stats/{model_name}?window=60** - Specific model statistics (URL encode model names, window: time window in seconds)
* **GET /model/traffic/idle?threshold=300** - Models idle longer than threshold (threshold: idle time threshold in seconds)
* **GET /model/traffic/active?threshold=300&window=60** - Currently active models (threshold: idle time threshold, window: time window for rate calculation)

### Sleep Management Endpoints
* **GET /model/sleep/status** - Current sleep status of all models
* **POST /model/sleep/{model_name}** - Manually put a model to sleep
* **POST /model/wake/{model_name}** - Manually wake up a sleeping model
* **GET /model/sleep/candidates** - Models that are candidates for sleep mode

### Response Examples

#### Traffic Statistics Response (All Models)

```json
{
  "traffic_stats": {
    "meta-llama/Llama-3.2-1B": {
      "total_requests": 150,
      "successful_requests": 148,
      "failed_requests": 2,
      "request_rate": 0.25,
      "avg_response_time": 1.2,
      "last_activity_time": 1703123456.789,
      "idle_time_seconds": 45.2,
      "is_idle": false
    }
  },
  "window_seconds": 60
}
```

#### Single Model Traffic Statistics Response

```json
{
  "model_stats": {
    "model_name": "meta-llama/Llama-3.2-1B",
    "total_requests": 150,
    "successful_requests": 148,
    "failed_requests": 2,
    "request_rate": 0.25,
    "avg_response_time": 1.2,
    "last_activity_time": 1703123456.789,
    "idle_time_seconds": 45.2,
    "is_idle": false
  },
  "window_seconds": 60
}
```

#### Idle Models Response

```json
{
  "idle_models": ["meta-llama/Llama-3.2-1B"],
  "idle_threshold_seconds": 300,
  "idle_model_details": {
    "meta-llama/Llama-3.2-1B": {
      "idle_time_seconds": 456.7,
      "total_requests": 150,
      "last_activity_time": 1703123000.0
    }
  },
  "sleep_mode_candidates": 1
}
```

#### Active Models Response

```json
{
  "active_models": ["Qwen/Qwen3-0.6B"],
  "idle_threshold_seconds": 300,
  "window_seconds": 60,
  "active_model_details": {
    "Qwen/Qwen3-0.6B": {
      "request_rate": 2.5,
      "total_requests": 300,
      "avg_response_time": 0.8,
      "last_activity_time": 1703123456.789
    }
  },
  "active_count": 1
}
```

#### Sleep Status Response

```json
{
  "sleeping_models": {
    "Qwen/Qwen3-0.6B": {
      "sleep_start_time": 1703123000.0,
      "sleep_duration": 456.7,
      "manual_sleep": false
    }
  },
  "sleep_candidates": ["meta-llama/Llama-3.2-1B"],
  "auto_sleep_enabled": true,
  "idle_threshold_seconds": 300,
  "wake_on_request": true
}
```

#### Sleep/Wake Response

```json
{
  "model_name": "meta-llama/Llama-3.2-1B",
  "success": true,
  "message": "Model meta-llama/Llama-3.2-1B sleep request successful"
}
```

#### Sleep Candidates Response

```json
{
  "sleep_candidates": ["meta-llama/Llama-3.2-1B"],
  "candidate_details": {
    "meta-llama/Llama-3.2-1B": {
      "idle_time_seconds": 456.7,
      "total_requests": 150,
      "last_activity_time": 1703123000.0,
      "can_sleep": true
    }
  },
  "idle_threshold_seconds": 300,
  "auto_sleep_enabled": true
}
```

### Note: URL Encoding for Model Names
When using endpoints with model names containing slashes (e.g., `meta-llama/Llama-3.2-1B`), URL encode the slashes:
* Original: `meta-llama/Llama-3.2-1B`
* Encoded: `meta-llama%2FLlama-3.2-1B`

Example:

```bash
# Traffic statistics for specific model
curl "http://localhost:8081/model/traffic/stats/meta-llama%2FLlama-3.2-1B"

# Put model to sleep
curl "http://localhost:8081/model/sleep/meta-llama%2FLlama-3.2-1B" -X POST

# Wake up model
curl "http://localhost:8081/model/wake/meta-llama%2FLlama-3.2-1B" -X POST

# Model health check
curl "http://localhost:8081/health/meta-llama%2FLlama-3.2-1B"
```
