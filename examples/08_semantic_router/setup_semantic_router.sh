#!/bin/bash

set -e

# Default models
MODEL_A="${MODEL_A:-Qwen/Qwen2.5-1.5B-Instruct}"
MODEL_B="${MODEL_B:-Qwen/Qwen1.5-1.8B}"

# Function to show usage
usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Setup semantic router with configurable models for KVCached integration"
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
            echo "Unknown option: $1"
            usage
            ;;
    esac
done

echo "Setting up Semantic Router with KVCached integration..."
echo "Model A (endpoint1): $MODEL_A"
echo "Model B (endpoint2): $MODEL_B"

# Clone semantic router
if [ ! -d "semantic-router" ]; then
    echo "Cloning semantic router..."
    git clone https://github.com/vllm-project/semantic-router.git
else
    echo "Semantic router already exists, skipping clone..."
fi

cd semantic-router

# Install Hugging Face CLI if not present
echo "Checking for Hugging Face CLI..."
if ! command -v hf &> /dev/null; then
    echo "Installing Hugging Face CLI..."
    # Create virtual environment for model downloads
    python3 -m venv .hf_venv
    source .hf_venv/bin/activate
    pip install huggingface-hub[hf-transfer]
    if ! command -v hf &> /dev/null; then
        echo "✗ Failed to install Hugging Face CLI"
        deactivate
        exit 1
    fi
    echo "✓ Hugging Face CLI installed successfully"
else
    echo "✓ Hugging Face CLI already installed"
fi

# Download classification models (≈1.5GB, first run only)
echo "Downloading classification models..."
if make download-models; then
    echo "✓ Classification models downloaded successfully"
    # Deactivate virtual environment after successful download (only if we activated it)
    if [[ "$VIRTUAL_ENV" != "" ]]; then
        deactivate 2>/dev/null || true
    fi
else
    echo "✗ Failed to download classification models"
    # Deactivate virtual environment on failure (only if we activated it)
    if [[ "$VIRTUAL_ENV" != "" ]]; then
        deactivate 2>/dev/null || true
    fi
    exit 1
fi

# Create config directory if it doesn't exist
mkdir -p config

# Apply patches using the diff provided
echo "Applying patches..."


# Update config/config.yaml using template
echo "Updating config/config.yaml with custom models..."

# Create backup of original config
cp config/config.yaml config/config.yaml.backup

# Copy template and generate config with variable substitution
cp ../templates/config.yaml.template .
# Escape forward slashes in model names for sed
MODEL_A_ESCAPED=$(echo "$MODEL_A" | sed 's/\//\\\//g')
MODEL_B_ESCAPED=$(echo "$MODEL_B" | sed 's/\//\\\//g')
sed -i "s/@MODEL_A@/$MODEL_A_ESCAPED/g; s/@MODEL_B@/$MODEL_B_ESCAPED/g" config.yaml.template
mv config.yaml.template config/config.yaml

# Apply config/envoy-docker.yaml patch
echo "Applying config/envoy-docker.yaml patch..."
patch -p1 < ../patches/envoy_docker_yaml.patch


# Apply the Go code patch for Docker gateway IP mapping
echo "Applying required Docker gateway IP mapping patch..."
if patch -p1 < ../patches/request_handler_docker_gateway.patch; then
    echo "✓ Docker gateway IP mapping patch applied successfully"
else
    echo "✗ Docker gateway IP mapping patch failed - this is required for your setup"
    echo "  Please check the patch manually or update the line numbers"
    exit 1
fi

# Clean up generated patch files (keep the ones in patches/ directory)
rm -f config_yaml.patch envoy_docker_yaml.patch docker_compose_yml.patch request_handler_go.patch config.yaml.template

echo "Semantic router setup complete!"
echo ""
echo "Next steps:"
echo "1. Start the semantic router services:"
echo "   cd semantic-router"
echo "   docker compose up --build"
echo ""
echo "2. In separate terminals, start vLLM servers (activate vLLM KVCached venv first):"
echo "   source ../../engine_integration/vllm-v0.9.2/.venv/bin/activate"
echo "   export ENABLE_KVCACHED=true"
echo "   vllm serve $MODEL_A --disable-log-requests --no-enable-prefix-caching --port=12346 --tensor-parallel-size=1"
echo "   vllm serve $MODEL_B --disable-log-requests --no-enable-prefix-caching --port=12347 --tensor-parallel-size=1"
echo "   OR use the convenience script: ./start_vllm_servers.sh --model-a '$MODEL_A' --model-b '$MODEL_B'"
echo ""
echo "3. Test with: curl -X POST http://localhost:8801/v1/chat/completions -H 'Content-Type: application/json' -d '{\"model\": \"auto\", \"messages\": [{\"role\": \"user\", \"content\": \"What is the derivative of x^4?\"}]}'"
