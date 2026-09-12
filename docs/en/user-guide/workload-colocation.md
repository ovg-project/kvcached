# Workload Colocation

kvcached allows LLM inference to coexist with other GPU workloads such as training, fine-tuning, or diffusion models on the same GPU.

## Supported Colocation Scenarios

| Workload A | Workload B | Example |
|-----------|-----------|---------|
| LLM Inference | LLM Inference | Two chat models |
| LLM Inference | Fine-tuning | Serving + LoRA training |
| LLM Inference | Diffusion | Chat + image generation |
| LLM Inference | Vision models | Multi-modal pipeline |

## Example: Inference + Diffusion

```bash
cd examples/07_inference_and_diffusion
./setup.sh
./start_inference_and_diffusion.sh --llm-engine vllm --diff-num-inference-steps 20
```

This runs a vLLM inference server alongside a Stable Diffusion pipeline, sharing GPU memory elastically.

## Further Reading

- See `examples/07_inference_and_diffusion/` for diffusion model colocation
- See `examples/04_inference_and_finetune/` for fine-tuning colocation
