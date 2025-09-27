#!/bin/bash
# Adapted from GVM project

SCRIPT_DIR=$(cd $(dirname ${BASH_SOURCE[0]:-$0}) && pwd -P)

GPUS=${2:-0}

export CUDA_VISIBLE_DEVICES=$GPUS

# Disable transformers version check for llama-factory
export DISABLE_VERSION_CHECK=1

# Clean up previous runs
rm -rf ${SCRIPT_DIR}/llama_factory_saves
llamafactory-cli train ${SCRIPT_DIR}/llama3_lora_sft.yaml