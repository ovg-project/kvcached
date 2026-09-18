# Router & Sleep Management

## Overview

The kvcached controller provides a production-ready multi-LLM serving stack with OpenAI-compatible routing, traffic monitoring, and automatic sleep management.

## Quick Start

### 1. Configure

Edit `controller/example-config.yaml` to match your models and hardware.

### 2. Launch

```bash
cd controller
python launch.py --config example-config.yaml
```

This creates tmux sessions for each engine instance and optionally a frontend router.

### 3. Send Requests

```bash
curl http://localhost:8080/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "meta-llama/Llama-3.2-1B", "prompt": "Hello"}'
```

## Features

- **Declarative YAML configuration** — Define engines, ports, environment overrides in one file
- **OpenAI API compatibility** — `/v1/completions` and `/v1/chat/completions` with streaming
- **Multi-model routing** — Unified frontend routes requests by model name
- **Traffic monitoring** — Per-model request rates, idle time tracking
- **Sleep/wake management** — Idle models release memory, wake on request arrival

## API Endpoints

| Endpoint | Description |
|----------|-------------|
| `/v1/completions` | Text completion API |
| `/v1/chat/completions` | Chat completion API |
| `/health` | Router health check |
| `/models` | List configured models |
| `/traffic/stats` | Traffic statistics |
| `/sleep/status` | Sleep status of all models |
| `/action/sleep/{model}` | Put model to sleep |
| `/action/wakeup/{model}` | Wake up model |

## Session Management

```bash
python launch.py --list-sessions    # List running sessions
python launch.py --kill-all         # Kill all sessions
```
