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
