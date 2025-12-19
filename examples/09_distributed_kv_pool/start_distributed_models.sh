#!/bin/bash
set -e

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

# Configuration
IPC_NAME_A="kvcached_dist_a"
IPC_NAME_B="kvcached_dist_b"

# Configurable GPU IDs (defaults: 4, 5, 6)
GPU_A=${GPU_A:-4}
GPU_B=${GPU_B:-5}
POOL_GPU_ID=${POOL_GPU_ID:-6}

# Memory limit for each pool on GPU 'POOL_GPU_ID'.
# Total used on GPU 'POOL_GPU_ID' = POOL_SIZE_A + POOL_SIZE_B
# Adjust based on your GPU memory. Here we set each to ~10GB.
POOL_SIZE_BYTES_A=$((60 * 1024 * 1024 * 1024))
POOL_SIZE_BYTES_B=$((10 * 1024 * 1024 * 1024))

# Conda environment
VENV_PATH="/root/miniforge3/envs/sglang-kvcached"
PYTHON="$VENV_PATH/bin/python3"

# Check if python exists
if [[ ! -x "$PYTHON" ]]; then
    echo "Python not found at $PYTHON"
    exit 1
fi
export PATH="$VENV_PATH/bin:$PATH"

# Ports
PORT_A=12348
PORT_B=12349

# Models
# Using small models for demonstration
MODEL_A="/workspace/models/Qwen3-32B"
MODEL_B="/workspace/models/Llama-3.2-1B"

# Cleanup function
cleanup() {
    echo "Stopping servers..."
    kill $(jobs -p) 2>/dev/null || true
    echo "Cleaning up IPC..."
    $PYTHON -m kvcached.cli.kvctl delete "$IPC_NAME_A" || true
    $PYTHON -m kvcached.cli.kvctl delete "$IPC_NAME_B" || true
}
trap cleanup EXIT

# 1. Configure the shared memory limit for the pool GPU
echo "Configuring KV Cache Pool A on GPU $POOL_GPU_ID with size $POOL_SIZE_BYTES_A bytes..."
# Configure shared memory limit. Note: Inside the model process, the pool GPU will be
# the SECOND visible device (index 1), so we configure the limit for device 1.
# But kvctl config-shared sets the limit for a specific key in the config map.
# PageAllocator reads this config. If PageAllocator sees "1" in the config, it will try to use device 1.
# So we must use "1" as the key here.
$PYTHON -m kvcached.cli.kvctl config-shared "$IPC_NAME_A" "1" "$POOL_SIZE_BYTES_A"

echo "Configuring KV Cache Pool B on GPU $POOL_GPU_ID with size $POOL_SIZE_BYTES_B bytes..."
$PYTHON -m kvcached.cli.kvctl config-shared "$IPC_NAME_B" "1" "$POOL_SIZE_BYTES_B"

export ENABLE_KVCACHED=true
export KVCACHED_AUTOPATCH=1
# Force extremely low local utilization (0.01) to simulate full local memory
# and trigger spillover to the remote KV cache pool for demonstration purposes.
export KVCACHED_GPU_UTILIZATION=0.99

# 2. Start Model A on GPU $GPU_A
echo "Starting Model A ($MODEL_A) on GPU $GPU_A with IPC $IPC_NAME_A..."
# Expose both the Local GPU and the Pool GPU
CUDA_VISIBLE_DEVICES=$GPU_A,$POOL_GPU_ID \
KVCACHED_IPC_NAME="$IPC_NAME_A" \
$PYTHON -m sglang.launch_server \
    --model "$MODEL_A" \
    --port "$PORT_A" \
    --disable-radix-cache \
    --trust-remote-code \
    --cuda-graph-max-bs 64 > server_a.log 2>&1 &

PID_A=$!
echo "Model A PID: $PID_A"

# 3. Start Model B on GPU $GPU_B
echo "Starting Model B ($MODEL_B) on GPU $GPU_B with IPC $IPC_NAME_B..."
# Using CUDA_VISIBLE_DEVICES=$GPU_B for the second GPU
CUDA_VISIBLE_DEVICES=$GPU_B,$POOL_GPU_ID \
KVCACHED_IPC_NAME="$IPC_NAME_B" \
$PYTHON -m sglang.launch_server \
    --model "$MODEL_B" \
    --port "$PORT_B" \
    --disable-radix-cache \
    --trust-remote-code \
    --cuda-graph-max-bs 64 > server_b.log 2>&1 &

PID_B=$!
echo "Model B PID: $PID_B"

# 4. Wait for readiness
wait_for_server() {
    local PORT=$1
    echo "Waiting for server at port $PORT..."
    while ! curl -s "http://127.0.0.1:$PORT/health" > /dev/null; do
        sleep 1
    done
    echo "Server at port $PORT is ready!"
}

# Sglang might not have /health endpoint exactly like vllm, usually /health_generate or just check /v1/models
wait_for_server_v1() {
  local PORT=$1
  echo "Waiting for server at port $PORT..."
  while ! curl -s "http://127.0.0.1:$PORT/v1/models" > /dev/null; do
      sleep 2
      if ! kill -0 "$PID_A" 2>/dev/null; then echo "Model A failed to start"; exit 1; fi
      if ! kill -0 "$PID_B" 2>/dev/null; then echo "Model B failed to start"; exit 1; fi
  done
  echo "Server at port $PORT is ready!"
}

wait_for_server_v1 "$PORT_A"
wait_for_server_v1 "$PORT_B"

echo "Both servers are running."
echo "You can check logs in server_a.log and server_b.log"
echo "Press Ctrl+C to stop."

wait
