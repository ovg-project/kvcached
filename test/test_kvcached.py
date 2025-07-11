import torch

from kvcached.integration.vllm.ops import alloc_kv_cache, init_kvcached
from kvcached.vmm_ops import map_to_kv_tensors, shutdown_kvcached

num_blocks = 2864
block_mem_size = 65536
block_size = 16
num_layers = 32
device = "cuda:0"
dtype = torch.float16

layer_mem_size = num_blocks * block_mem_size * dtype.itemsize
total_mem_size = num_layers * 2 * layer_mem_size

print("Initializing kvcached")
torch.cuda.set_device(0)
init_kvcached()
print("Initialized kvcached")

print("Creating KV tensors")
# ktensors = kvcached_ops.create_ktensors(
#     layer_mem_size, dtype, device, num_layers
# )
# vtensors = kvcached_ops.create_vtensors(
#     layer_mem_size, dtype, device, num_layers
# )
kv_shape = (2, num_blocks, block_mem_size)
kv_tensors = alloc_kv_cache(kv_shape, block_size, dtype, device, num_layers)

ktensors = []
vtensors = []
num_elements = layer_mem_size // 2
for i in range(num_layers):
    print(kv_tensors[i].shape)
    ktensors.append(kv_tensors[i].narrow(0, 0, 1).view(-1))
    vtensors.append(kv_tensors[i].narrow(0, 1, 1).view(-1))
print("Created KV tensors")

for i in range(num_layers):
    print(ktensors[i].shape, vtensors[i].shape)

print("Mapping to KV tensors")
slab_size = 2 * 1024 * 1024  # 2MB
offsets = [slab_size * i for i in [1, 3, 5, 7, 9]]
map_to_kv_tensors(offsets)
print("Mapped to KV tensors")

ele_idxs = [o // 2 for o in offsets]
print("Accessing KV tensors")
for i in range(num_layers):
    ktensors[i][ele_idxs] = torch.tensor(list(range(i, i + 5)),
                                         device=device,
                                         dtype=dtype)
    vtensors[i][ele_idxs] = torch.tensor(list(range(i + 5, i + 10)),
                                         device=device,
                                         dtype=dtype)

for i in range(num_layers):
    print(ktensors[i][ele_idxs], vtensors[i][ele_idxs])
print("Access done")

shutdown_kvcached()
