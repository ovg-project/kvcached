#!/bin/bash
set -e

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)

# Default values
DEFAULT_RESEARCH_PORT=12346
DEFAULT_WRITING_PORT=12347
DEFAULT_VENV_PATH="../../engine_integration/vllm-v0.9.2/.venv"

# CLI variables
research_port=""
writing_port=""
langchain_venv_path=""
single_topic=""
streaming=false

usage() {
    cat <<EOF
Multi-Agent System Client (LangChain-powered)

Usage: $0 [OPTIONS]

OPTIONS:
  --research-port PORT       Research Agent port (default: $DEFAULT_RESEARCH_PORT)
  --writing-port PORT        Writing Agent port (default: $DEFAULT_WRITING_PORT)
  --langchain-venv-path PATH Path to LangChain virtual environment (default: $DEFAULT_LANGCHAIN_VENV_PATH)
  --topic "TOPIC"            Run single topic instead of examples
  --streaming                Enable streaming mode
  -h, --help                 Show this help and exit

EXAMPLES:
  # Run all example topics
  $0

  # Run with custom ports
  $0 --research-port 12348 --writing-port 12349

  # Run single topic
  $0 --topic "artificial intelligence"

  # Run with streaming mode (real-time responses)
  $0 --topic "blockchain technology" --streaming
EOF
}

# GNU getopt parsing
TEMP=$(getopt \
    --options h \
    --longoptions research-port:,writing-port:,langchain-venv-path:,topic:,streaming,help \
    --name "$0" -- "$@")

if [[ $? -ne 0 ]]; then
    exit 1
fi

eval set -- "$TEMP"

while true; do
    case "$1" in
        --research-port) research_port="$2"; shift 2;;
        --writing-port) writing_port="$2"; shift 2;;
        --langchain-venv-path) langchain_venv_path="$2"; shift 2;;
        --topic) single_topic="$2"; shift 2;;
        --streaming) streaming=true; shift;;
        --help|-h) usage; exit 0;;
        --) shift; break;;
        *) echo "Unknown option: $1" >&2; usage; exit 1;;
    esac
done

# Apply defaults
RESEARCH_PORT=${research_port:-$DEFAULT_RESEARCH_PORT}
WRITING_PORT=${writing_port:-$DEFAULT_WRITING_PORT}
LANGCHAIN_VENV_PATH=${langchain_venv_path:-$DEFAULT_LANGCHAIN_VENV_PATH}

# Validate LangChain venv path
if [[ ! -f "$LANGCHAIN_VENV_PATH/bin/activate" ]]; then
    echo "Error: LangChain virtual environment not found at '$LANGCHAIN_VENV_PATH'"
    echo "Please run setup.sh first to create the LangChain environment"
    echo "Or specify correct path with --langchain-venv-path"
    exit 1
fi
source "$LANGCHAIN_VENV_PATH/bin/activate"

# Check if model servers are running
echo "Checking model servers..."
for port in "$RESEARCH_PORT" "$WRITING_PORT"; do
    if ! curl -s "http://127.0.0.1:$port/v1/models" >/dev/null 2>&1; then
        echo "Error: Model server on port $port is not responding"
        echo "Please start the model servers first:"
        echo "  ./start_multi_agent_models.sh"
        exit 1
    fi
done
echo "Model servers are ready"

cd "$SCRIPT_DIR"

# Example topics
example_topics=(
    "artificial intelligence and machine learning"
    "renewable energy technologies"
    "quantum computing basics"
)

if [[ -n "$single_topic" ]]; then
    # Run single topic
    echo ""
    if [[ "$streaming" == "true" ]]; then
        echo "Running single topic (streaming mode)..."
        python3 multi_agent_system.py --research-port "$RESEARCH_PORT" --writing-port "$WRITING_PORT" --topic "$single_topic" --streaming
    else
        echo "Running single topic (non-streaming mode)..."
        python3 multi_agent_system.py --research-port "$RESEARCH_PORT" --writing-port "$WRITING_PORT" --topic "$single_topic"
    fi
else
    # Run all example topics
    echo ""
    echo "Running example topics..."
    echo "Total topics: ${#example_topics[@]}"
    echo ""

    for i in "${!example_topics[@]}"; do
        topic="${example_topics[$i]}"
        echo "==============================================="
        echo "Example $((i+1))/${#example_topics[@]}: $topic"
        echo "==============================================="
        if [[ "$streaming" == "true" ]]; then
            python3 multi_agent_system.py --research-port "$RESEARCH_PORT" --writing-port "$WRITING_PORT" --topic "$topic" --streaming
        else
            python3 multi_agent_system.py --research-port "$RESEARCH_PORT" --writing-port "$WRITING_PORT" --topic "$topic"
        fi

        # Add separator between topics (except for the last one)
        if [[ $((i+1)) -lt ${#example_topics[@]} ]]; then
            echo ""
            echo "Press Enter to continue to next topic..."
            read -r
        fi
    done
fi

echo "Multi-agent system client completed!"