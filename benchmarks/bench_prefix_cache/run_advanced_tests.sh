#!/bin/bash
#
# Runner script for advanced prefix cache tests
#
# IMPORTANT: These tests CANNOT run continuously because they require
# different server configurations:
#
# Test 1 (High Speedup):  Needs default cache size (1000 blocks)
# Test 2 (Eviction):      Needs small cache size (10 blocks)
#
# The cache size is set via KVCACHED_PREFIX_CACHE_MAX_SIZE environment variable
# and MUST be set before starting the server. It cannot be changed at runtime.
#
# Usage:
#   bash run_advanced_tests.sh <test_number>
#
# Arguments:
#   test_number    Which test to run (1 or 2) [required]
#                  1 = High speedup test (long prefix)
#                  2 = Cache eviction test (requires KVCACHED_PREFIX_CACHE_MAX_SIZE=10)
#
# Options:
#   --num-questions N    For test 1: Number of questions (default: 15)
#   --mode MODE          For test 2: Test mode (simple/repeated, default: repeated)
#
# Examples:
#   # Run high speedup test
#   bash run_advanced_tests.sh 1 --num-questions 15
#
#   # Run eviction test (server must be started with small cache first!)
#   bash run_advanced_tests.sh 2 --mode repeated
#

set -e

# Default configuration
NUM_QUESTIONS=15
TEST_MODE="repeated"
MODEL="meta-llama/Llama-3.2-1B"
PORT=12346
HOST="127.0.0.1"

# Check for test number (positional argument)
if [[ $# -lt 1 ]]; then
    echo "ERROR: Test number is required (1 or 2)"
    echo ""
    echo "Usage: $0 <test_number> [options]"
    echo "  test_number: 1 (high speedup) or 2 (eviction)"
    echo ""
    exit 1
fi

TEST_NUM="$1"
shift

# Validate test number
if [[ "$TEST_NUM" != "1" && "$TEST_NUM" != "2" ]]; then
    echo "ERROR: Test number must be 1 or 2"
    echo "  1 = High speedup test"
    echo "  2 = Cache eviction test"
    exit 1
fi

# Parse remaining arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --num-questions)
            NUM_QUESTIONS="$2"
            shift 2
            ;;
        --mode)
            TEST_MODE="$2"
            shift 2
            ;;
        --model)
            MODEL="$2"
            shift 2
            ;;
        --port)
            PORT="$2"
            shift 2
            ;;
        --host)
            HOST="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 <test_number> [--num-questions N] [--mode MODE] [--model MODEL] [--port PORT] [--host HOST]"
            exit 1
            ;;
    esac
done

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo "=============================================================================="
echo "                  ADVANCED PREFIX CACHE TEST #${TEST_NUM}"
echo "=============================================================================="
echo ""
echo "Configuration:"
echo "  Server: http://${HOST}:${PORT}"
echo "  Model: ${MODEL}"

if [[ "$TEST_NUM" == "1" ]]; then
    echo "  Questions: ${NUM_QUESTIONS}"
    echo "  Required cache config: Default (1000 blocks or any large size)"
else
    echo "  Test mode: ${TEST_MODE}"
    echo "  Required cache config: KVCACHED_PREFIX_CACHE_MAX_SIZE=10"
    echo "  Current cache config: ${KVCACHED_PREFIX_CACHE_MAX_SIZE:-NOT SET (will use default 1000)}"
fi

echo ""
echo "=============================================================================="
echo ""

# Check if server is running
echo "Checking server health..."
if ! curl -s --max-time 5 "http://${HOST}:${PORT}/health" > /dev/null 2>&1; then
    echo ""
    echo "ERROR: Cannot connect to vLLM server at http://${HOST}:${PORT}"
    echo ""
    echo "Please start the server first:"
    if [[ "$TEST_NUM" == "2" ]]; then
        echo "  export KVCACHED_PREFIX_CACHE_MAX_SIZE=10"
    fi
    echo "  bash ../simple_bench/start_server.sh vllm --venv-path ../../engine_integration/vllm-pip-venv --model ${MODEL} --port ${PORT} --tp 1"
    echo ""
    exit 1
fi

echo "✓ Server is running"
echo ""

# Run the appropriate test
if [[ "$TEST_NUM" == "1" ]]; then
    #
    # TEST 1: High Speedup Test
    #
    echo "=============================================================================="
    echo "TEST 1: HIGH LATENCY IMPROVEMENT (Long Prefix)"
    echo "=============================================================================="
    echo ""
    echo "This test demonstrates speedup using a very long shared prefix"
    echo "with 25+ few-shot examples (~2,000 tokens)."
    echo ""

    python3 test_high_speedup.py \
        --model "$MODEL" \
        --port "$PORT" \
        --host "$HOST" \
        --num-questions "$NUM_QUESTIONS"

    TEST_EXIT=$?

elif [[ "$TEST_NUM" == "2" ]]; then
    #
    # TEST 2: Cache Eviction Test
    #
    echo "=============================================================================="
    echo "TEST 2: CACHE EVICTION (LRU Behavior)"
    echo "=============================================================================="
    echo ""
    echo "This test demonstrates cache eviction using multiple distinct prefixes."
    echo ""

    # Check if cache size is properly configured
    if [[ -z "$KVCACHED_PREFIX_CACHE_MAX_SIZE" ]]; then
        echo "⚠️  WARNING: KVCACHED_PREFIX_CACHE_MAX_SIZE is not set!"
        echo ""
        echo "For eviction to occur, you must:"
        echo "  1. Stop the current server: pkill -f 'vllm serve'"
        echo "  2. Export the variable: export KVCACHED_PREFIX_CACHE_MAX_SIZE=10"
        echo "  3. Restart the server with the environment variable"
        echo ""
        echo "Continuing anyway (eviction may not occur)..."
        echo ""
    elif [[ "$KVCACHED_PREFIX_CACHE_MAX_SIZE" -gt 20 ]]; then
        echo "⚠️  WARNING: KVCACHED_PREFIX_CACHE_MAX_SIZE=${KVCACHED_PREFIX_CACHE_MAX_SIZE} is too large!"
        echo "  For eviction, use 10-20 blocks. Current value may prevent eviction."
        echo ""
        echo "Continuing anyway..."
        echo ""
    else
        echo "✓ KVCACHED_PREFIX_CACHE_MAX_SIZE=${KVCACHED_PREFIX_CACHE_MAX_SIZE} (good for eviction test)"
        echo ""
    fi

    python3 test_eviction.py \
        --model "$MODEL" \
        --port "$PORT" \
        --host "$HOST" \
        --mode "$TEST_MODE" \
        --cache-size 10

    TEST_EXIT=$?
fi

if [ $TEST_EXIT -ne 0 ]; then
    echo ""
    echo "ERROR: Test ${TEST_NUM} failed with exit code $TEST_EXIT"
    echo ""
    exit $TEST_EXIT
fi

echo ""
echo "=============================================================================="
echo "                        TEST ${TEST_NUM} COMPLETED"
echo "=============================================================================="
echo ""
echo "Next steps:"
echo "  1. Review the results above"
echo "  2. Check server logs for detailed cache behavior:"
echo "     tail -100 server.log | grep -E '(Cached|Cache hit|Evicting)'"

if [[ "$TEST_NUM" == "2" ]]; then
    echo "  3. Verify eviction in logs:"
    echo "     grep -i 'evict' server.log"
fi

echo ""
echo "To run the other test, you must reconfigure and restart the server:"
if [[ "$TEST_NUM" == "1" ]]; then
    echo "  # To run Test 2 (eviction):"
    echo "  pkill -f 'vllm serve'"
    echo "  export KVCACHED_PREFIX_CACHE_MAX_SIZE=10"
    echo "  bash ../simple_bench/start_server.sh vllm --venv-path ../../engine_integration/vllm-pip-venv --model ${MODEL} --port ${PORT} --tp 1"
    echo "  bash run_advanced_tests.sh 2"
else
    echo "  # To run Test 1 (high speedup):"
    echo "  pkill -f 'vllm serve'"
    echo "  unset KVCACHED_PREFIX_CACHE_MAX_SIZE  # Use default cache size"
    echo "  bash ../simple_bench/start_server.sh vllm --venv-path ../../engine_integration/vllm-pip-venv --model ${MODEL} --port ${PORT} --tp 1"
    echo "  bash run_advanced_tests.sh 1"
fi
echo ""
echo "=============================================================================="
echo ""
