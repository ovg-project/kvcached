#!/bin/bash

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Default models
MODEL_A="${MODEL_A:-Qwen/Qwen2.5-1.5B-Instruct}"
MODEL_B="${MODEL_B:-Qwen/Qwen1.5-1.8B}"

# Function to show usage
usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Start two vLLM servers with KVCached for semantic router example"
    echo ""
    echo "Options:"
    echo "  --model-a MODEL     Model for endpoint1 (port 12346) [default: $MODEL_A]"
    echo "  --model-b MODEL     Model for endpoint2 (port 12347) [default: $MODEL_B]"
    echo "  --help              Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0"
    echo "  $0 --model-a 'meta-llama/Llama-3.2-3B-Instruct' --model-b 'Qwen/Qwen1.5-1.8B'"
    exit 1
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model-a)
            MODEL_A="$2"
            shift 2
            ;;
        --model-b)
            MODEL_B="$2"
            shift 2
            ;;
        --help)
            usage
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            usage
            ;;
    esac
done

echo -e "${GREEN}Starting vLLM servers with KVCached for semantic router example${NC}"
echo -e "${YELLOW}Model A (port 12346): ${MODEL_A}${NC}"
echo -e "${YELLOW}Model B (port 12347): ${MODEL_B}${NC}"

# Check if ENABLE_KVCACHED is set
if [ -z "$ENABLE_KVCACHED" ]; then
    echo -e "${YELLOW}Warning: ENABLE_KVCACHED is not set. Setting it to true.${NC}"
    export ENABLE_KVCACHED=true
fi

# Function to start a vLLM server in background
start_vllm_server() {
    local model=$1
    local port=$2
    local server_name=$3

    echo -e "${GREEN}Starting ${server_name} on port ${port}...${NC}"

    # Start vLLM server in background with KVCached venv activated
    (
        source ../../engine_integration/vllm-v0.9.2/.venv/bin/activate && \
        vllm serve "${model}" \
            --disable-log-requests \
            --no-enable-prefix-caching \
            --port="${port}" \
            --tensor-parallel-size=1
    ) &
}

# Extract model names for display
MODEL_A_NAME=$(basename "$MODEL_A")
MODEL_B_NAME=$(basename "$MODEL_B")

# Start first server
start_vllm_server "$MODEL_A" 12346 "${MODEL_A_NAME} Server"

# Wait a moment for the first server to start
sleep 5

# Start second server
start_vllm_server "$MODEL_B" 12347 "${MODEL_B_NAME} Server"

echo -e "${GREEN}Both vLLM servers started successfully!${NC}"
echo -e "${YELLOW}Servers are running in the background.${NC}"
echo -e "${YELLOW}Use 'jobs' to see running processes or 'pkill -f vllm' to stop them.${NC}"
echo ""
echo -e "${GREEN}Semantic router should now be able to route requests to:${NC}"
echo "  - ${MODEL_A} on port 12346 (endpoint1)"
echo "  - ${MODEL_B} on port 12347 (endpoint2)"
echo ""
echo -e "${GREEN}Test with:${NC}"
echo 'curl -X POST http://localhost:8801/v1/chat/completions \'
echo '  -H "Content-Type: application/json" \'
echo '  -d "{\"model\": \"auto\", \"messages\": [{\"role\": \"user\", \"content\": \"What is the derivative of x^4?\"}]}'

# Wait for all background processes
wait
