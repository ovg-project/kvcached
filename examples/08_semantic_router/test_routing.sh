#!/bin/bash

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default models (should match what was used in setup)
MODEL_A_NAME="${MODEL_A_NAME:-Qwen2.5-1.5B-Instruct}"
MODEL_B_NAME="${MODEL_B_NAME:-Qwen1.5-1.8B}"

# Function to show usage
usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Test semantic routing with various query categories"
    echo ""
    echo "Options:"
    echo "  --model-a-name NAME    Expected name for model A (endpoint1) [default: $MODEL_A_NAME]"
    echo "  --model-b-name NAME    Expected name for model B (endpoint2) [default: $MODEL_B_NAME]"
    echo "  --help                 Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0"
    echo "  $0 --model-a-name 'Llama-3.2-3B-Instruct' --model-b-name 'Qwen1.5-1.8B'"
    exit 1
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model-a-name)
            MODEL_A_NAME="$2"
            shift 2
            ;;
        --model-b-name)
            MODEL_B_NAME="$2"
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

SEMANTIC_ROUTER_URL="http://localhost:8801/v1/chat/completions"

echo -e "${GREEN}Testing Semantic Router with KVCached${NC}"
echo "=========================================="

# Function to send a test request
send_request() {
    local query="$1"
    local expected_model="$2"
    local description="$3"

    echo -e "${BLUE}Testing: ${description}${NC}"
    echo -e "${YELLOW}Query: ${query}${NC}"
    echo -e "${YELLOW}Expected model: ${expected_model}${NC}"

    # Send request and capture response
    response=$(curl -s -X POST "${SEMANTIC_ROUTER_URL}" \
        -H "Content-Type: application/json" \
        -d "{\"model\": \"auto\", \"messages\": [{\"role\": \"user\", \"content\": \"${query}\"}], \"max_tokens\": 50}")

    # Extract model from response (if available)
    selected_model=$(echo "$response" | grep -o '"model":"[^"]*"' | head -1 | sed 's/"model":"//' | sed 's/"//')

    if [ -n "$selected_model" ]; then
        echo -e "${GREEN}✓ Routed to model: ${selected_model}${NC}"
        if [[ "$selected_model" == *"$expected_model"* ]]; then
            echo -e "${GREEN}✓ Correct routing!${NC}"
        else
            echo -e "${RED}✗ Unexpected routing${NC}"
        fi
    else
        echo -e "${YELLOW}Response received (model info not in response)${NC}"
    fi

    echo "Response preview: $(echo "$response" | head -c 200)..."
    echo ""
}

echo ""

# Test cases for different categories
echo -e "${GREEN}Testing different query categories:${NC}"
echo "=================================="

# Math queries (should route to MODEL_B - endpoint2)
send_request "What is the derivative of x^4?" "$MODEL_B_NAME" "Math derivative question"

# Physics queries (should route to MODEL_B - endpoint2)
send_request "Explain Newton's laws of motion" "$MODEL_B_NAME" "Physics question"

# Economics queries (should route to MODEL_B - endpoint2)
send_request "What is supply and demand?" "$MODEL_B_NAME" "Economics question"

# Business queries (should route to MODEL_A - endpoint1)
send_request "How to write a business plan?" "$MODEL_A_NAME" "Business question"

# Biology queries (should route to MODEL_A - endpoint1)
send_request "Explain how photosynthesis works" "$MODEL_A_NAME" "Biology question"

# Chemistry queries (should route to MODEL_A - endpoint1)
send_request "What is the periodic table?" "$MODEL_A_NAME" "Chemistry question"

# General/other queries (should route to MODEL_A - endpoint1 as default)
send_request "Tell me a joke" "$MODEL_A_NAME" "General question"

echo -e "${GREEN}Testing complete!${NC}"
echo ""
echo -e "${YELLOW}Note: The semantic router uses BERT-based classification to route requests.${NC}"
echo -e "${YELLOW}Expected routing: endpoint1 ($MODEL_A_NAME) for general/business queries, endpoint2 ($MODEL_B_NAME) for technical/math queries.${NC}"
echo -e "${YELLOW}Actual routing may vary based on classification confidence scores.${NC}"
echo ""
echo -e "${BLUE}Monitor routing logs in the semantic-router container with:${NC}"
echo "docker compose logs -f semantic-router"
