# Model Router with Sleep Management

Automatically traffic monitor and idled models sleep.

## Quick Start

```bash
# 1. Start controller
cd /kvcached
source engine_integration/vllm-v0.9.2/.venv/bin/activate
cd controller
python launch.py --config example-config.yaml
```

## Usage Examples

### Monitor Traffic

```python
from controller.traffic_monitor import TrafficMonitor

monitor = TrafficMonitor(idle_threshold_seconds=300)
await monitor.start()

# Record requests
stats = monitor.record_request_start("meta-llama/Llama-3.2-1B", "/v1/completions")
monitor.record_request_end(stats, success=True)

# Check status
idle_models = monitor.get_idle_models(idle_threshold_seconds=300)
# Returns: ['meta-llama/Llama-3.2-1B']

summary = monitor.get_traffic_summary(window_seconds=60)
# Returns: {
#   'meta-llama/Llama-3.2-1B': {
#     'total_requests': 10,
#     'successful_requests': 10,
#     'failed_requests': 0,
#     'last_request_time': datetime(...)
#   }
# }
```

### Manage Sleep

```python
from controller.sleep_manager import SleepManager, SleepConfig

config = SleepConfig(idle_threshold_seconds=300, auto_sleep_enabled=True)
sleep_manager = SleepManager(config, traffic_monitor=monitor)

sleep_manager.add_vllm_model("meta-llama/Llama-3.2-1B", "localhost", "12346")
await sleep_manager.start()

# Manual control
await sleep_manager.put_model_to_sleep("meta-llama/Llama-3.2-1B", manual=True)
# Returns: True (success)

await sleep_manager.wakeup_model("meta-llama/Llama-3.2-1B")
# Returns: True (success)

is_sleeping = sleep_manager.is_model_sleeping("meta-llama/Llama-3.2-1B")
# Returns: False (after wakeup)
```

## Testing

```bash
cd /kvcached/tests
python test_traffic_monitor.py
# Testing traffic monitoring...
# ✓ Test idle detection passed
# ✓ Test traffic stats passed

python test_sleep_manager.py
# Testing sleep manager...
# INFO xx-xx xx:xx:xx [api_server.py:958] check whether the engine is sleeping
# INFO:     xxx.xxx.xxx.xxx:xxxxx - "GET /is_sleeping HTTP/1.1" 200 OK
# INFO xx-xx xx:xx:xx [gpu_worker.py:98] Sleep mode freed 3.99 GiB memory, 6.37 GiB memory is still in use.
# INFO xx-xx xx:xx:xx [executor_base.py:211] It took 2.218816 seconds to fall asleep.
# INFO:     xxx.xxx.xxx.xxx:xxxxx - "POST /sleep HTTP/1.1" 200 OK
# INFO xx-xx xx:xx:xx [api_server.py:958] check whether the engine is sleeping
# INFO:     xxx.xxx.xxx.xxx:xxxxx - "GET /is_sleeping HTTP/1.1" 200 OK
# INFO xx-xx xx:xx:xx [api_server.py:950] wake up the engine with tags: None
# INFO xx-xx xx:xx:xx [executor_base.py:227] It took 0.121931 seconds to wake up tags {'weights', 'kv_cache'}.
# INFO:     xxx.xxx.xxx.xxx:xxxxx - "POST /wake_up HTTP/1.1" 200 OK
# ✓ vLLM sleep/wake test passed
# ✓ SGLang sleep/wake test passed
```
