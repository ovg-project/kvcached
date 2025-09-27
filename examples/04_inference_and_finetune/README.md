# LLM Inference and Finetuning with KVCached

This example demonstrates how to run LLM inference and LLaMA Factory finetuning concurrently, sharing GPU memory through the KVCached system.

## Setup

First, run the setup script to install dependencies and create the virtual environment:

```bash
./setup.sh
```

This will:
- Install `uv` package manager if not present
- Create a Python 3.11 virtual environment for LLaMA Factory
- Clone and install LLaMA Factory with required dependencies

## Scripts Overview

### Individual Scripts

- **`setup.sh`**: Sets up the LLaMA Factory environment
- **`start_llm_server.sh`**: Starts an LLM inference server (vLLM or SGLang)
- **`start_llm_client.sh`**: Runs benchmark tests against the LLM server
- **`start_finetune.sh`**: Starts LLaMA Factory finetuning
- **`start_inference_and_finetune.sh`**: Main orchestration script that runs both inference and finetuning

### Configuration Files

- **`llama3_lora_sft.yaml`**: LLaMA Factory configuration for LoRA finetuning

## Usage

### Running Inference and Finetuning Together

The main script `start_inference_and_finetune.sh` runs both LLM inference and finetuning concurrently:

```bash
# Basic usage with defaults
./start_inference_and_finetune.sh

# With custom parameters
./start_inference_and_finetune.sh \
  --llm-engine vllm \
  --llm-model meta-llama/Llama-3.2-1B \
  --llm-port 12346 \
  --finetune-config llama3_lora_sft.yaml \
  --finetune-gpus "0"
```

#### Options

**LLM Server Options:**
- `--llm-engine`: LLM engine (vllm | sglang) (default: vllm)
- `--llm-model`: Model identifier (default: meta-llama/Llama-3.2-1B)
- `--llm-port`: Port for LLM server (default: vllm=12346, sglang=30000)
- `--llm-venv-path`: Path to virtual environment for LLM engine (optional)
- `--llm-tp-size`: Tensor parallel size (default: 1)

**Finetuning Options:**
- `--llama-factory-venv-path`: Path to LLaMA Factory virtual environment (default: ./llama-factory-venv)
- `--finetune-config`: Finetuning configuration file (default: llama3_lora_sft.yaml)
- `--finetune-gpus`: GPU IDs for finetuning (default: "0")

### Running Individual Components

#### LLM Server Only

```bash
# Start vLLM server
./start_llm_server.sh vllm --model meta-llama/Llama-3.2-1B

# Start SGLang server
./start_llm_server.sh sglang --model meta-llama/Llama-3.2-1B --port 30000
```

#### LLM Client Benchmarking

```bash
# Test vLLM server
./start_llm_client.sh vllm --num-prompts 100 --request-rate 5

# Test SGLang server
./start_llm_client.sh sglang --port 30000 --num-prompts 50
```

#### Finetuning Only

```bash
# Run finetuning with default config
./start_finetune.sh

# Run finetuning with specific GPUs
./start_finetune.sh GPUS=1
```

## Configuration

### Finetuning Configuration

The `llama3_lora_sft.yaml` file contains the LLaMA Factory configuration. Key parameters:

- **Model**: `meta-llama/Llama-3.1-8B-Instruct`
- **Method**: LoRA finetuning with rank 16
- **Dataset**: `alpaca_en_demo` (30k samples)
- **Training**: 100 epochs with cosine learning rate schedule

You can modify this file or create new configuration files for different finetuning scenarios.

### KVCached Integration

The scripts automatically set up KVCached environment variables:

- `ENABLE_KVCACHED=true`: Enables KVCached memory sharing
- `KVCACHED_IPC_NAME`: Sets unique IPC names for different processes
  - `VLLM` for vLLM servers
  - `SGLANG` for SGLang servers
  - `LLAMAFACTORY` for finetuning processes

## Monitoring

### Logs

- **LLM Server**: `vllm.log` or `sglang.log`
- **Finetuning**: `finetuning.log`
- **Model Outputs**: `llama_factory_saves/` directory

### Testing the LLM Server

While both processes are running, you can test the LLM server:

```bash
# Test with curl
curl -X POST "http://localhost:12346/v1/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Llama-3.2-1B",
    "prompt": "Hello, how are you?",
    "max_tokens": 50
  }'

# Or use the benchmark client
./start_llm_client.sh vllm --port 12346 --num-prompts 10
```

## Troubleshooting

### Common Issues

1. **Virtual environment not found**: Run `./setup.sh` first
2. **GPU memory issues**: Adjust model size or tensor parallel settings
3. **Port conflicts**: Use different ports for different services
4. **Permission errors**: Ensure scripts are executable (`chmod +x *.sh`)

### GPU Requirements

- **Minimum**: 1 GPU with 8GB+ VRAM for small models
- **Recommended**: 2+ GPUs for larger models and concurrent workloads
- **L4 GPUs**: Automatically detected and configured with appropriate settings

### Environment Variables

Key environment variables that affect behavior:

- `CUDA_VISIBLE_DEVICES`: Controls GPU visibility
- `ENABLE_KVCACHED`: Enables memory sharing (set automatically)
- `PYTHON`: Python executable to use (default: python3)

## Examples

### Example 1: Basic Setup

```bash
# Setup environment
./setup.sh

# Run with defaults
./start_inference_and_finetune.sh
```

### Example 2: Multi-GPU Setup

```bash
# Use multiple GPUs for finetuning
./start_inference_and_finetune.sh \
  --llm-engine vllm \
  --llm-tp-size 2 \
  --finetune-gpus "2,3"
```

### Example 3: Custom Model and Config

```bash
# Use different model and custom config
./start_inference_and_finetune.sh \
  --llm-model meta-llama/Llama-3.1-8B-Instruct \
  --finetune-config custom_config.yaml
```

This setup enables efficient GPU memory sharing between inference and training workloads, maximizing hardware utilization.
