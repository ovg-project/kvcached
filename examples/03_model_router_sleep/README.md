# Model Router with Sleep Management

Automatically manage idle models to save GPU resources.

## What It Does

- **TrafficMonitor**: Tracks which models are being used
- **SleepManager**: Puts idle models to sleep and wakes them when needed

## Quick Start

### 1. Start Controller

```bash
cd /kvcached
source engine_integration/vllm-v0.9.2/.venv/bin/activate
cd controller
python launch.py --config example-config.yaml
```

### 2. Monitor Traffic

**Using TrafficMonitor in code:**

```python
from controller.traffic_monitor import TrafficMonitor

# Create monitor
monitor = TrafficMonitor(idle_threshold_seconds=300)
await monitor.start()

# Record requests
request_stats = monitor.record_request_start(
    model_name="meta-llama/Llama-3.2-1B",
    endpoint_path="/v1/completions"
)
# ... process request ...
monitor.record_request_end(request_stats, success=True)

# Get idle/active models
idle_models = monitor.get_idle_models(idle_threshold_seconds=300)
active_models = monitor.get_active_models()

# Get traffic summary
summary = monitor.get_traffic_summary(window_seconds=60)
```

**Using HTTP API:**

```bash
# Send test requests to generate traffic
curl -X POST http://localhost:8080/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "meta-llama/Llama-3.2-1B", "prompt": "Hello", "max_tokens": 50}'

# Check idle models (threshold in seconds)
curl http://localhost:8080/models/idle?threshold=300

# Check active models
curl http://localhost:8080/models/active

# View traffic stats (all models)
curl http://localhost:8080/traffic/stats

# View stats for specific model
curl http://localhost:8080/traffic/stats/meta-llama%2FLlama-3.2-1B
```

### 3. Control Sleep

**Using SleepManager in code:**

```python
from controller.sleep_manager import SleepManager, SleepConfig
from controller.traffic_monitor import TrafficMonitor

# Create sleep manager
config = SleepConfig(
    idle_threshold_seconds=300,
    auto_sleep_enabled=True,
    wakeup_on_request=True
)
monitor = TrafficMonitor()
sleep_manager = SleepManager(config, traffic_monitor=monitor)

# Add models
sleep_manager.add_vllm_model("meta-llama/Llama-3.2-1B", "localhost", "12346")
sleep_manager.add_sglang_model("Qwen/Qwen3-0.6B", "localhost", "30000")

await sleep_manager.start()

# Put model to sleep
await sleep_manager.put_model_to_sleep("meta-llama/Llama-3.2-1B", manual=True)

# Wake model up
await sleep_manager.wakeup_model("meta-llama/Llama-3.2-1B")

# Check if sleeping
is_sleeping = sleep_manager.is_model_sleeping("meta-llama/Llama-3.2-1B")
```

**Using HTTP API:**

```bash
# Via controller router
curl -X POST http://localhost:8080/models/meta-llama/Llama-3.2-1B/sleep
curl -X POST http://localhost:8080/models/meta-llama/Llama-3.2-1B/wake

# Or directly to engine
curl -X POST http://localhost:12346/sleep       # vLLM
curl -X POST http://localhost:12346/wake_up     # vLLM
```

## Configuration

Edit `example-config.yaml`:

```yaml
sleep_manager:
  idle_threshold_seconds: 300      # Sleep after 5 min idle
  check_interval_seconds: 60       # Check every minute
  auto_sleep_enabled: false        # Enable auto-sleep
  wakeup_on_request: true          # Auto wake on requests
  min_sleep_duration: 80           # Min sleep time (seconds)
```

## Testing

### Test Traffic Monitor

```bash
cd /kvcached/tests
python test_traffic_monitor.py
```

Tests traffic monitoring and checks idle/active models.

### Test Sleep Manager

```bash
cd /kvcached/tests
python test_sleep_manager.py
```

Tests sleep/wake operations:
- **vLLM**: Calls `put_model_to_sleep()` → `/sleep` API, then `wakeup_model()` → `/wake_up` API
- **SGLang**: Calls `put_model_to_sleep()` → `/release_memory_occupation` API, then `wakeup_model()` → `/resume_memory_occupation` API

## Files

- [controller/traffic_monitor.py](../../controller/traffic_monitor.py) - Traffic monitoring
- [controller/sleep_manager.py](../../controller/sleep_manager.py) - Sleep management
- [tests/test_traffic_monitor.py](../../tests/test_traffic_monitor.py) - Traffic tests
- [tests/test_sleep_manager.py](../../tests/test_sleep_manager.py) - Sleep tests

## Engine Support

- **vLLM**: `/sleep`, `/wake_up` APIs
- **SGLang**: `/release_memory_occupation`, `/resume_memory_occupation` APIs (see [#95](https://github.com/ovg-project/kvcached/issues/95))
