#!/bin/bash
set -e

# =============================================================================
# Automated test runner for vLLM prefix caching with kvcached
# 
# This script:
# 1. Starts vLLM server with prefix caching enabled
# 2. Waits for server readiness
# 3. Runs prefix cache tests
# 4. Collects and displays results
# 5. Cleans up server process
# =============================================================================

# Configuration
MODEL="${MODEL:-meta-llama/Llama-3.2-1B}"
PORT="${PORT:-12346}"
VENV_PATH="${1:-../../engine_integration/vllm-pip-venv}"
NUM_QUESTIONS="${NUM_QUESTIONS:-10}"

# Enable debug logging to see cache hits
export KVCACHED_LOG_LEVEL=DEBUG

# Ensure we're in the test directory
cd "$(dirname "$0")"

echo "============================================================================="
echo "  vLLM Prefix Cache Test Runner"
echo "============================================================================="
echo ""
echo "Configuration:"
echo "  Model:         $MODEL"
echo "  Port:          $PORT"
echo "  Venv:          $VENV_PATH"
echo "  Questions:     $NUM_QUESTIONS"
echo "  Log Level:     $KVCACHED_LOG_LEVEL"
echo ""

# Check if server is already running
if curl -s "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
    echo "⚠ WARNING: Server already running on port $PORT"
    echo "Using existing server. To restart, kill it first with:"
    echo "  kill \$(lsof -t -i:${PORT})"
    echo ""
    SERVER_PID=""
else
    # Start vLLM server with prefix caching
    echo "Starting vLLM server with prefix caching enabled..."
    echo ""
    
    bash ../simple_bench/start_server.sh vllm \
        --venv-path "$VENV_PATH" \
        --model "$MODEL" \
        --port "$PORT" \
        --tp 1 \
        > server.log 2>&1 &
    
    SERVER_PID=$!
    echo "Server PID: $SERVER_PID"
    echo ""
    
    # Wait for server to be ready
    echo "Waiting for server to be ready (max 120 seconds)..."
    READY=false
    for i in {1..60}; do
        if curl -s "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
            echo "✓ Server ready after $((i * 2)) seconds!"
            READY=true
            break
        fi
        echo -n "."
        sleep 2
    done
    echo ""
    
    if [ "$READY" = false ]; then
        echo "ERROR: Server failed to start within 120 seconds"
        echo "Check server.log for details:"
        echo ""
        tail -50 server.log
        exit 1
    fi
fi

# Run the test
echo ""
echo "============================================================================="
echo "  Running Prefix Cache Test"
echo "============================================================================="
echo ""

python test_prefix_cache.py \
    --model "$MODEL" \
    --port "$PORT" \
    --num-questions "$NUM_QUESTIONS"

TEST_EXIT_CODE=$?

echo ""
echo "============================================================================="
echo "  Test Complete"
echo "============================================================================="
echo ""

# Show relevant server logs if we started the server
if [ -n "$SERVER_PID" ]; then
    echo "Looking for cache-related messages in server logs..."
    echo ""
    
    if grep -i "cache hit\|cached block" server.log 2>/dev/null | head -20; then
        echo ""
        echo "✓ Cache activity detected in logs"
    else
        echo "⚠ WARNING: No cache activity found in logs"
        echo "  This may indicate prefix caching is not working properly"
        echo "  Check server.log for details"
    fi
    echo ""
fi

# Cleanup if we started the server
if [ -n "$SERVER_PID" ]; then
    echo "Cleaning up (killing server PID $SERVER_PID)..."
    kill $SERVER_PID 2>/dev/null || true
    
    # Give it a moment to shutdown gracefully
    sleep 2
    
    # Force kill if still running
    kill -9 $SERVER_PID 2>/dev/null || true
    
    echo "Server stopped."
fi

echo ""
if [ $TEST_EXIT_CODE -eq 0 ]; then
    echo "✓ Test completed successfully"
    exit 0
else
    echo "✗ Test failed with exit code $TEST_EXIT_CODE"
    exit $TEST_EXIT_CODE
fi
