#!/bin/bash
#
# Start vLLM server with small cache size for eviction testing
#
# This script sets KVCACHED_PREFIX_CACHE_MAX_SIZE=10 and starts the server
#

set -e

# Set small cache size BEFORE starting server
export KVCACHED_PREFIX_CACHE_MAX_SIZE=10

echo "=============================================================================="
echo "Starting vLLM server with SMALL cache for eviction testing"
echo "=============================================================================="
echo ""
echo "Configuration:"
echo "  KVCACHED_PREFIX_CACHE_MAX_SIZE = ${KVCACHED_PREFIX_CACHE_MAX_SIZE}"
echo "  Model: meta-llama/Llama-3.2-1B"
echo "  Port: 12346"
echo ""
echo "This configuration is for EVICTION TESTING only."
echo "For normal testing, use the default cache size (1000)."
echo ""
echo "=============================================================================="
echo ""

cd "$(dirname "${BASH_SOURCE[0]}")/../simple_bench"

bash start_server.sh vllm \
    --venv-path ../../engine_integration/vllm-pip-venv \
    --model meta-llama/Llama-3.2-1B \
    --port 12346 \
    --tp 1
