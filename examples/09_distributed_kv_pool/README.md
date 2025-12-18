# Distributed KV Cache Pool Example

This example demonstrates how to run two models on different GPUs (e.g., GPU 0 and GPU 1) while sharing a KV cache pool located on a third, shared GPU (e.g., GPU 2). This setup leverages NVLink for efficient peer-to-peer memory access.

## Topology

- **Model A**: Running on GPU 0.
- **Model B**: Running on GPU 1.
- **KV Cache Pool**: Configured on GPU 2.

## Prerequisites

- Multiple GPUs (at least 3 recommended for full isolation, but can overlap if needed).
- `sglang` (and `kvcached` installed).
- NVLink connectivity between the GPUs.

## Usage

1. **Start the distributed setup**:

   ```bash
   bash start_distributed_models.sh
   ```

   This script will:
   - Configure a shared memory limit for GPU 2 using `kvcached`.
   - Launch Model A on GPU 0.
   - Launch Model B on GPU 1.
   - **Note**: It exports `KVCACHED_GPU_UTILIZATION=0.01` to force `kvcached` to think local memory is full, ensuring the remote shared pool is used for demonstration purposes.

2. **Send requests**:

   In a separate terminal:

   ```bash
   bash send_requests.sh
   ```

## Configuration

The `start_distributed_models.sh` script contains configuration variables you can adjust:

- `POOL_GPU_ID`: The ID of the GPU to use as the shared pool (default: 6).
- `GPU_A` / `GPU_B`: The GPU IDs for Model A and Model B (default: 4 and 5).
- `POOL_SIZE_BYTES_A` / `POOL_SIZE_BYTES_B`: The size of the shared pool for each model in bytes.
- `IPC_NAME_A` / `IPC_NAME_B`: The IPC names for the shared memory segments.
- `MODEL_A` / `MODEL_B`: The models to serve.
