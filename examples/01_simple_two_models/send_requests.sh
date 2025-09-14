#!/bin/bash
set -e

PORT_A=12346
PORT_B=12347

# Parse flags: --port-a, --port-b
print_usage() {
  echo "Usage: $0 [--port-a PORT] [--port-b PORT]" >&2
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --port-a)
      PORT_A="$2"; shift 2;;
    --port-b)
      PORT_B="$2"; shift 2;;
    -h|--help)
      print_usage; exit 0;;
    --)
      shift; break;;
    -*)
      echo "Unknown option: $1" >&2; print_usage; exit 1;;
    *)
      echo "Unexpected positional argument: $1" >&2; print_usage; exit 1;;
  esac
done

PROMPT_A=${PROMPT_A:-"Explain how LLM works."}
PROMPT_B=${PROMPT_B:-"Summarize LLM in one sentence."}

# Discover served model IDs from each server to avoid 404 (model not found)
MODEL_A=$(curl -s http://127.0.0.1:$PORT_A/v1/models | python3 -c 'import sys,json; d=json.load(sys.stdin); data=d.get("data") or []; print(data[0].get("id","")) if data else print("")' 2>/dev/null || echo "")
MODEL_B=$(curl -s http://127.0.0.1:$PORT_B/v1/models | python3 -c 'import sys,json; d=json.load(sys.stdin); data=d.get("data") or []; print(data[0].get("id","")) if data else print("")' 2>/dev/null || echo "")

request_and_print() {
  local NAME="$1"
  local PORT="$2"
  local MODEL="$3"
  local PROMPT="$4"

  echo "========================================"
  echo "Server ${NAME} (port ${PORT})"
  if [[ -z "$MODEL" ]]; then
    echo "Model: (not detected)"
    echo "Status: Could not GET /v1/models on port ${PORT}; skipping request."
    return 0
  fi
  echo "Model: ${MODEL}"
  echo "Prompt:"
  echo "  ${PROMPT}"
  echo "Response:"
  local RESP
  RESP=$(curl -s -X POST http://127.0.0.1:${PORT}/v1/completions \
    -H "Content-Type: application/json" \
    -d "{\"model\":\"${MODEL}\",\"prompt\":\"${PROMPT}\",\"max_tokens\":64,\"top_p\":1,\"seed\":0}" \
    | python3 -c 'import sys,json; d=json.load(sys.stdin); print((d.get("choices") or [{}])[0].get("text","" ).strip())' || true)
  if [[ -z "$RESP" ]]; then
    echo "  (no text returned)"
  else
    echo "$RESP" | sed 's/^/  /'
  fi
}

request_and_print "A" "$PORT_A" "$MODEL_A" "$PROMPT_A"
echo
request_and_print "B" "$PORT_B" "$MODEL_B" "$PROMPT_B"


