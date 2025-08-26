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

# Check every configured model
curl http://localhost:8080/health/all

# See which models the router currently knows about
curl http://localhost:8080/models
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

2. **Test a simple completion request:**
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

3. **Run comprehensive traffic monitoring tests:**
```bash
# Navigate to test directory
cd test

# Run traffic monitoring test suite
python test_traffic_monitor.py > test_monitor.txt

# View test results
cat test_monitor.txt
```

4. **Monitor server logs:**
```bash
# Attach to the frontend tmux session to view logs
tmux attach -t kvcached-frontend

# Or list all running sessions
tmux ls
```


## Traffic Statistics 
(traffic_monitor.py)

### Current Implementation Status
✅ **Fully Implemented**: The traffic monitoring system is fully functional and tracks real request data.

### Core Monitoring Features
- **Request tracking**: Records all requests per model with timestamps
- **Request rate calculation**: Calculates requests/second over configurable time windows
- **Response time monitoring**: Tracks average response times
- **Success/failure rates**: Monitors request success rates
- **Idle time detection**: Identifies models with no recent activity

### Data Collection
- **Real-time statistics**: Updates statistics for each request
- **Historical data**: Maintains request history for trend analysis
- **Configurable windows**: Supports custom time windows for rate calculations
- **Thread-safe operations**: Uses locks to ensure data consistency

## Sleep Management
(sleep_manager.py)

### Current Implementation Status
⚠️ **Note**: The current sleep management is a **signal-based framework** that only tracks sleep states. It does not actually free resources or pause model processes.

### Current Features (Signal Only)
- **Sleep state tracking**: Records which models are marked as "sleeping"
- **Manual sleep/wake operations**: Allows setting/clearing sleep flags
- **Automatic sleep detection**: Identifies models that should be sleeping based on idle time
- **Wake-on-request detection**: Detects when requests arrive for sleeping models
- **Configuration management**: Manages sleep-related settings

### Real Implementation Requirements
For a production-ready sleep management system, the following would need to be implemented:

#### Resource Management
- **GPU memory deallocation**: Move model weights from GPU to CPU/storage
- **Process suspension**: Pause model service processes
- **Memory optimization**: Release computational resources
- **Weight storage**: Efficient storage and retrieval of model weights

#### Process Control
- **Model process management**: Start/stop/suspend model service processes
- **Resource monitoring**: Track GPU memory usage and computational load
- **Graceful shutdown**: Proper cleanup when putting models to sleep
- **Fast recovery**: Quick model reloading when waking up

#### Advanced Features
- **Partial offloading**: Keep frequently used layers in memory
- **Predictive loading**: Pre-load models based on usage patterns
- **Resource scheduling**: Intelligent allocation of limited resources
- **Failure recovery**: Handle wake-up failures gracefully

### Sleep Configuration
- **Idle threshold**: Configurable idle time before sleep (default: 300s)
- **Check interval**: How often to check for idle models (default: 60s)
- **Auto-sleep enabled**: Toggle automatic sleep functionality
- **Wake-on-request**: Toggle automatic wake functionality
- **Minimum sleep duration**: Minimum time to keep model asleep (default: 60s)

## New API Endpoints
(frontend.py)

### Traffic Monitoring Endpoints
- **GET /traffic/stats** - All models' traffic statistics
- **GET /traffic/stats/model/{model_name}** - Specific model statistics (URL encode model names)
- **GET /traffic/idle?threshold=300** - Models idle longer than threshold
- **GET /traffic/active?threshold=300** - Currently active models

### Sleep Management Endpoints
- **GET /sleep/status** - Current sleep status of all models
- **POST /sleep/model/{model_name}** - Manually put a model to sleep
- **POST /wake/model/{model_name}** - Manually wake up a sleeping model
- **GET /sleep/candidates** - Models that are candidates for sleep mode

### Response Examples

#### Traffic Statistics Response
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
  }
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

### URL Encoding for Model Names
When using endpoints with model names containing slashes (e.g., `meta-llama/Llama-3.2-1B`), URL encode the slashes:
- Original: `meta-llama/Llama-3.2-1B`
- Encoded: `meta-llama%2FLlama-3.2-1B`

Example:
```bash
curl "http://localhost:8081/traffic/stats/model/meta-llama%2FLlama-3.2-1B"
curl "http://localhost:8081/sleep/model/meta-llama%2FLlama-3.2-1B" -X POST
```

## Implementation Status Summary

### ✅ Fully Implemented
- **Traffic Monitoring**: Complete request tracking, statistics, and analysis
- **Multi-model Routing**: Full OpenAI-compatible API with request routing
- **Process Management**: tmux-based service management
- **Configuration System**: YAML-based declarative configuration

### ⚠️ Signal-Based Framework (Not Fully Implemented)
- **Sleep Management**: Currently only tracks sleep states, does not actually free resources
- **Resource Optimization**: Framework exists but no actual GPU memory management
- **Process Control**: Sleep/wake signals are recorded but not acted upon

### 🔄 Future Implementation Requirements
For production deployment, the following would need to be implemented:

#### Sleep Management Enhancement
```python
# Example of what real implementation would look like:
async def put_model_to_sleep(self, model_name: str):
    # 1. Send signal to model process to reduce resources
    await self.send_sleep_signal(model_name)
    
    # 2. Move model weights to storage
    await self.offload_model_weights(model_name)
    
    # 3. Pause request processing
    await self.pause_model_service(model_name)
    
    # 4. Release GPU memory
    await self.free_gpu_memory(model_name)
```

#### Resource Management
- **GPU Memory Management**: Actual deallocation and reallocation
- **Process Lifecycle**: Start/stop/suspend model processes
- **Weight Storage**: Efficient model weight persistence
- **Performance Optimization**: Minimize wake-up latency
