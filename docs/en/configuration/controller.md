# Controller Configuration

The kvcached controller uses a YAML configuration file to declare models, engines, and routing settings.

## Configuration File Format

```yaml
# Global kvcached settings
kvcached:
  ENABLE_KVCACHED: "true"
  KVCACHED_AUTOPATCH: "1"

# Model instances
instances:
  - name: llama-1b
    model: meta-llama/Llama-3.2-1B
    engine: vllm
    engine_args:
      - "--port 12346"
      - "--no-enable-prefix-caching"
    using_venv: true
    venv_path: /path/to/venv

  - name: qwen-0.6b
    model: Qwen/Qwen3-0.6B
    engine: sglang
    engine_args:
      - "--port 30000"
      - "--disable-radix-cache"

# Router settings
router:
  enable_router: true
  router_port: 8080

# Sleep manager settings
sleep_manager:
  idle_threshold_seconds: 300
  auto_sleep_enabled: true

# Delay between launching instances (seconds)
launch_delay_seconds: 30
```

## Key Fields

| Field | Required | Description |
|-------|----------|-------------|
| `instances[].model` | Yes | HuggingFace model identifier |
| `instances[].engine` | Yes | `vllm` or `sglang` |
| `instances[].engine_args` | No | Additional engine arguments |
| `instances[].using_venv` | No | Whether to use a virtual environment |
| `router.enable_router` | No | Enable the frontend router |
| `sleep_manager.idle_threshold_seconds` | No | Seconds before idle model sleeps |
