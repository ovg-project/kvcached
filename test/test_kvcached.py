import torch

from kvcached.integration.sglang.interfaces import (alloc_kv_cache,
                                                    init_kvcached)
from kvcached.utils import _get_page_size
from kvcached.vmm_ops import (map_to_kv_tensors, shutdown_kvcached,
                              unmap_from_kv_tensors)

num_blocks = 2864
head_num = 8
head_dim = 4096

block_size = 1
block_mem_size = head_num * head_dim * block_size

num_layers = 32
device = "cuda:0"
dtype = torch.float16
contiguous_layout = True

layer_mem_size = num_blocks * block_mem_size * dtype.itemsize
total_mem_size = num_layers * 2 * layer_mem_size

print(f"layer_mem_size: {layer_mem_size / 1024 / 1024} MB")
print(f"total_mem_size: {total_mem_size / 1024 / 1024 / 1024} GB")

print("Initializing kvcached")
torch.cuda.set_device(0)
init_kvcached(contiguous_layout=contiguous_layout)
print("Initialized kvcached")

print("Creating KV tensors")
kv_shape = (num_blocks, head_num, head_dim)

page_size = _get_page_size()
k_tensors, v_tensors = alloc_kv_cache(kv_shape, dtype, device, num_layers)

print("Created KV tensors")

print("Mapping to KV tensors")
slab_size = _get_page_size()
slab_size = slab_size * 2 * num_layers if contiguous_layout else slab_size
num_pages = 10  # 10 pages
# offsets = [slab_size * i for i in range(num_pages)]
offsets = [slab_size * i for i in [1, 3, 5, 7, 9]]

# Add CUDA event timing for accurate measurement
print("Testing map_to_kv_tensors timing with CUDA events...")
print("Running 10 iterations to calculate average...")

num_iterations = 10
execution_times = []

for iteration in range(num_iterations):
    print(f"Iteration {iteration + 1}/{num_iterations}")

    # Unmap before each map (except first iteration if nothing is mapped yet)
    if iteration > 0:
        print("  Unmapping offsets...")
        unmap_from_kv_tensors(offsets)
        torch.cuda.synchronize()

    # Create CUDA events for this iteration
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    # Time the mapping operation
    start_event.record()
    map_to_kv_tensors(offsets)
    end_event.record()
    torch.cuda.synchronize()  # Wait for all CUDA operations to complete

    elapsed_time = start_event.elapsed_time(end_event)
    execution_times.append(elapsed_time)
    print(f"  Execution time: {elapsed_time:.4f} ms")

# Calculate statistics
average_time = sum(execution_times) / len(execution_times)
min_time = min(execution_times)
max_time = max(execution_times)

# Final unmap to clean up
print("Final cleanup: unmapping offsets...")
unmap_from_kv_tensors(offsets)
print("Mapping benchmark completed")

print("\nMap to KV tensors Timing Results:")
print(f"Average execution time: {average_time:.4f} ms")
print(f"Min execution time: {min_time:.4f} ms")
print(f"Max execution time: {max_time:.4f} ms")
print(f"All times: {[f'{t:.4f}' for t in execution_times]} ms")

# Convert byte offsets to block indices
# Each block has (head_num * head_dim * block_size * dtype.itemsize) bytes
bytes_per_block = head_num * head_dim * block_size * dtype.itemsize
ele_idxs = [o // bytes_per_block for o in offsets]
print("Accessing KV tensors")
for i in range(num_layers):
    # Create tensors with proper shape to match the indexing result [5, 8, 4096]
    k_value = torch.tensor(list(range(i, i + 5)), device=device, dtype=dtype)
    k_value = k_value.view(5, 1, 1).expand(5, head_num, head_dim)
    k_tensors[i][ele_idxs] = k_value

    v_value = torch.tensor(list(range(i + 5, i + 10)),
                           device=device,
                           dtype=dtype)
    v_value = v_value.view(5, 1, 1).expand(5, head_num, head_dim)
    v_tensors[i][ele_idxs] = v_value

# Suppress verbose tensor output during benchmark
# for i in range(num_layers):
#     print(k_tensors[i][ele_idxs], v_tensors[i][ele_idxs])
print("Access done")

shutdown_kvcached()
