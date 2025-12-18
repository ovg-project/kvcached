#!/bin/bash
set -e

PORT_A=12348
PORT_B=12349

PROMPT="Explain the theory of relativity in one sentence."

echo "Sending request to Model A (Port $PORT_A)..."
curl -s "http://localhost:$PORT_A/v1/completions" \
  -H "Content-Type: application/json" \
  -d "{\"model\": \"meta-llama/Llama-3.2-1B\", \"prompt\": \"$PROMPT\", \"max_tokens\": 64}" | python3 -m json.tool

echo ""
echo "Sending request to Model B (Port $PORT_B)..."
curl -s "http://localhost:$PORT_B/v1/completions" \
  -H "Content-Type: application/json" \
  -d "{\"model\": \"Qwen/Qwen3.5-0.6B\", \"prompt\": \"$PROMPT\", \"max_tokens\": 64}" | python3 -m json.tool
