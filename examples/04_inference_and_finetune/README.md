# LLM Inference and Finetuning with kvcached

This example demonstrates how to run LLM inference and LLaMA Factory finetuning concurrently, sharing GPU memory through kvcached.

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

### Dataset Files

- **`data/`**: datasets for LLaMA Factory finetuning

## Usage

### Running Inference and Finetuning Together (**Recommended**)

It is recommended to use the main script `start_inference_and_finetune.sh` to run both LLM inference and finetuning concurrently:

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

The script will first launch the vLLM server, followed by LLaMA Factory. Execution logs for each server are saved to `[sglang|vllm].log` and `finetuning.log` respectively. Once both servers have started successfully, you should see terminal output similar to:

```txt
...
(APIServer pid=934211) INFO:     Started server process [934211]
(APIServer pid=934211) INFO:     Waiting for application startup.
(APIServer pid=934211) INFO:     Application startup complete.
...
[INFO|trainer.py:2530] 2025-09-29 09:30:24,040 >>   Gradient Accumulation steps = 12
[INFO|trainer.py:2531] 2025-09-29 09:30:24,040 >>   Total optimization steps = 700
[INFO|trainer.py:2532] 2025-09-29 09:30:24,042 >>   Number of trainable parameters = 41,943,040
  1%|          | 6/700 [00:54<1:42:30,  8.86s/it]
```

Next, you need to run `./start_llm_client.sh [sglang|vllm]` to launch the LLM client, which will send requests to the LLM server. You should observe progress from both the LLM server and LLaMA Factory, confirming that they are running concurrently.

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
./start_finetune.sh 1
```

## Configuration

### Finetuning Configuration

The `llama3_lora_sft.yaml` file contains the LLaMA Factory configuration. Key parameters:

- **Model**: `meta-llama/Llama-3.1-8B-Instruct`
- **Method**: LoRA finetuning with rank 16
- **Dataset**: `alpaca_en_demo` (30k samples)
- **Training**: 100 epochs with cosine learning rate schedule

You can modify this file or create new configuration files for different finetuning scenarios.

### kvcached Integration

The scripts automatically set up kvcached environment variables:

- `ENABLE_KVCACHED=true`: Enables kvcached memory sharing
- `KVCACHED_AUTOPATCH=1`: Enables kvcached autopatching
- `KVCACHED_IPC_NAME`: Sets unique IPC names for different processes, e.g.,
  - `VLLM` for vLLM servers
  - `SGLANG` for SGLang servers

## Datasets

The example uses the alpaca dataset (`data/alpaca_en_demo.json`) from the original LLaMA Factory repository for finetuning and ShareGPT dataset for LLM benchmarking (automatically downloaded).

## Monitoring

### Logs

- **LLM Server**: `vllm.log` or `sglang.log`
- **Finetuning**: `finetuning.log`
- **Model Outputs**: `llama_factory_saves/` directory

### Testing the LLM Server

While both processes are running, you can test the LLM server:

```bash
./start_llm_client.sh vllm
```

## Troubleshooting

### Common Issues

1. **Virtual environment not found**: Run `./setup.sh` first
2. **GPU memory issues**: Adjust model size or tensor parallel settings
3. **Port conflicts**: Use different ports for different services
4. **Permission errors**: Ensure scripts are executable (`chmod +x *.sh`)

### Environment Variables

Key environment variables that affect behavior:

- `CUDA_VISIBLE_DEVICES`: Controls GPU visibility
- `ENABLE_KVCACHED`: Enables memory sharing (set automatically)
- `KVCACHED_AUTOPATCH`: Enables autopatching (set automatically)
- `PYTHON`: Python executable to use (default: python3)
