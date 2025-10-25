#!/bin/bash
# Adapted from GVM project

SCRIPT_DIR=$(cd $(dirname ${BASH_SOURCE[0]:-$0}) && pwd -P)

GPUS=${1:-"0"}
CONFIG=${2:-"$SCRIPT_DIR/llama3_lora_sft.yaml"}

export CUDA_VISIBLE_DEVICES=$GPUS

# Disable transformers version check for llama-factory
export DISABLE_VERSION_CHECK=1

# Clean up previous runs
rm -rf ${SCRIPT_DIR}/llama_factory_saves

echo "Finetuning with config: ${CONFIG}"
echo "Finetuning with GPUs: ${GPUS}"
rm -f $SCRIPT_DIR/finetuning.log
llamafactory-cli train ${CONFIG} 2>&1 | tee $SCRIPT_DIR/finetuning.log
