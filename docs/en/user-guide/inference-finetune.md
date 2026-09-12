# Inference & Fine-tuning Colocation

Run LLM inference serving and model fine-tuning concurrently on the same GPU with kvcached memory sharing.

## Quick Start

```bash
cd examples/04_inference_and_finetune
./setup.sh
./start_inference_and_finetune.sh --llm-engine vllm --llm-model meta-llama/Llama-3.2-1B
```

## How It Works

kvcached enables both workloads to share GPU memory elastically:

- The inference server allocates KV cache memory on demand
- The fine-tuning job uses remaining GPU memory for gradients and activations
- Memory is dynamically balanced between the two workloads

## Options

| Option | Default | Description |
|--------|---------|-------------|
| `--llm-engine` | `vllm` | Engine choice: `vllm` or `sglang` |
| `--llm-model` | `meta-llama/Llama-3.2-1B` | Model for inference |
| `--finetune-config` | `llama3_lora_sft.yaml` | LLaMA Factory config |
| `--finetune-gpus` | `"0"` | GPU IDs for fine-tuning |

## Further Reading

- See `examples/04_inference_and_finetune/` for the full example with LLaMA Factory integration
