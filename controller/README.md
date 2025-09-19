# kvcached Multi-LLM Controller & Router

This directory contains a complete example of a **multi-LLM serving stack** built on top of kvcached.
It exposes unified **OpenAI-compatible** HTTP endpoints that transparently route requests to one of many backend model servers (SGLang or vLLM).

## Features

* **Declarative YAML configuration** – Define engines, ports, environment overrides, virtual-envs, and router settings in a single place.
* **OpenAI API compatibility** – Supports both `/v1/completions` and `/v1/chat/completions` with streaming and non-streaming responses.
* **Multi-model routing** – Provides a unified IP and port as frontend. All requests can be sent to this unified frontend. The router will route the request to the corresponding backend based on the model named in the request.
* **tmux-based process management** – Every engine instance, the router, and optional benchmark clients run in isolated *tmux* sessions.
* **Traffic monitoring** – Monitors the request status of each model, e.g., request rate, idle time.
* **Sleep and wakeup** – Idle models can be deactivated to release model weights, and waked up upon receiving requests.

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

### 2.1 Stand-alone router (skip `launch.py`)
Already have your engines running?  Spin up only the router:

```bash
python frontend.py --config example-config.yaml --port 8080
```

`frontend.py` will parse the YAML, extract the host/port for each instance, and start routing immediately.

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

# Get server information
curl http://localhost:8080/get_server_info
```

### Traffic Monitoring Examples

```bash
# Get traffic statistics for all models
curl http://localhost:8080/traffic/stats

# Get traffic statistics for specific model with 120-second window
curl "http://localhost:8080/traffic/stats/meta-llama%2FLlama-3.2-1B?window=120"

# Get idle models (idle for more than 5 minutes)
curl "http://localhost:8080/models/idle?threshold=300"

# Get active models with 60-second rate calculation window
curl "http://localhost:8080/models/active?threshold=300&window=60"
```

### Sleep Management Examples

```bash
# Get sleep status of all models
curl http://localhost:8080/sleep/status

# Put a model to sleep
curl "http://localhost:8080/action/sleep/meta-llama%2FLlama-3.2-1B" -X POST

# Wake up a sleeping model
curl "http://localhost:8080/action/wakeup/meta-llama%2FLlama-3.2-1B" -X POST

# Get sleep candidates
curl http://localhost:8080/sleep/candidates
```

### Note: URL Encoding for Model Names
When using endpoints with model names containing slashes (e.g., `meta-llama/Llama-3.2-1B`), URL encode the slashes:
* Original: `meta-llama/Llama-3.2-1B`
* Encoded: `meta-llama%2FLlama-3.2-1B`

---

## Benchmarking

```bash
python benchmark.py --config example-config.yaml
```

A separate *tmux* session (`benchmark-<name>`) is created for every model so you can watch latency/throughput side by side.

---

## Testing

```bash
cd tests
python test_sleep_manager.py
python test_traffic_monitor.py
```

---

## API Endpoints

### Core OpenAI-Compatible
| Endpoint | Description |
| --- | --- |
| `/v1/completions` | Text completion API |
| `/v1/chat/completions` | Chat completion API |
| `/health` | Router health check |
| `/models` | List all configured models |
| `/health/{model_name}` | Specific model health check (URL encode model names) |
| `/get_server_info` | Server information |

### Traffic Monitoring
| Endpoint | Description |
| --- | --- |
| `/traffic/stats?window=60` | All models' traffic statistics (window: time window in seconds for rate calculation) |
| `/traffic/stats/{model_name}?window=60` | Specific model statistics (URL encode model names, window: time window in seconds) |
| `/models/idle?threshold=300` | Models idle longer than threshold (threshold: idle time threshold in seconds) |
| `/models/active?threshold=300&window=60` | Currently active models (threshold: idle time threshold, window: time window for rate calculation) |

### Sleep Management
| Endpoint | Description |
| --- | --- |
| `/sleep/status` | Current sleep status of all models |
| `/action/sleep/{model_name}` | Manually put a model to sleep |
| `/action/wakeup/{model_name}` | Manually wake up a sleeping model |
| `/sleep/candidates` | Models that are candidates for sleep mode |
