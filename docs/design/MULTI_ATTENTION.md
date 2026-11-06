# Multi-Attention Type Support RFC

**Status**: Proposed
**Issues**: #198, #202
**Author**: Community
**Last Updated**: December 2024

## Summary

This RFC proposes extending kvcached to support multiple attention types beyond the current MHA (Multi-Head Attention) implementation, including GQA, MQA, MLA, and hybrid architectures.

## Motivation

### Current Limitations

kvcached currently supports:
- **MHA** (Multi-Head Attention): Standard transformer attention
- **GQA** (Grouped-Query Attention): Limited support via MHA path

Not yet supported:
- **MLA** (Multi-head Latent Attention): Used in DeepSeek-V2
- **MQA** (Multi-Query Attention): Single KV head shared across all Q heads
- **Hybrid**: Models with different attention types per layer (e.g., gpt-oss)

### Models Affected

| Model | Attention Type | Status |
|-------|---------------|--------|
| Llama 2/3 | GQA | Supported |
| Mistral | GQA | Supported |
| DeepSeek-V2 | MLA | Not Supported |
| Falcon | MQA | Partial |
| gpt-oss-20b | Hybrid | Not Supported |
| Qwen-MoE | Mixed | Not Supported |

## Design Goals

1. **Unified Interface**: Single API for all attention types
2. **Memory Efficiency**: Optimize memory layout per attention type
3. **Backward Compatibility**: Existing MHA code continues to work
4. **Extensibility**: Easy to add new attention types

## Attention Type Analysis

### MHA (Multi-Head Attention)

```
KV Shape: [num_heads, seq_len, head_dim]
Memory: 2 * num_heads * seq_len * head_dim * dtype_size
```

### GQA (Grouped-Query Attention)

```
KV Shape: [num_kv_heads, seq_len, head_dim]
Memory: 2 * num_kv_heads * seq_len * head_dim * dtype_size
Ratio: num_heads / num_kv_heads (e.g., 8:1)
```

### MQA (Multi-Query Attention)

```
KV Shape: [1, seq_len, head_dim]
Memory: 2 * seq_len * head_dim * dtype_size
Effectively: GQA with num_kv_heads=1
```

### MLA (Multi-head Latent Attention)

```
Compressed KV: [seq_len, latent_dim]
Decompressed at attention time
Memory: seq_len * latent_dim * dtype_size (much smaller)
```

## Proposed Architecture

### Attention Type Registry

```python
class AttentionType(Enum):
    MHA = "mha"
    GQA = "gqa"
    MQA = "mqa"
    MLA = "mla"

@dataclass
class AttentionSpec:
    """Specification for an attention type."""
    attention_type: AttentionType
    num_heads: int
    num_kv_heads: int
    head_dim: int
    latent_dim: Optional[int] = None  # For MLA

    @property
    def kv_size_per_token(self) -> int:
        """Calculate KV cache size per token."""
        if self.attention_type == AttentionType.MLA:
            return self.latent_dim * 2  # Compressed K and V
        else:
            return self.num_kv_heads * self.head_dim * 2
```

### KV Cache Factory

```python
class KVCacheFactory:
    """Creates appropriate KV cache manager for attention type."""

    @staticmethod
    def create(
        attention_spec: AttentionSpec,
        num_layers: int,
        max_seq_len: int,
        dtype: torch.dtype,
        device: str,
    ) -> "BaseKVCacheManager":
        if attention_spec.attention_type == AttentionType.MLA:
            return MLAKVCacheManager(
                latent_dim=attention_spec.latent_dim,
                num_layers=num_layers,
                max_seq_len=max_seq_len,
                dtype=dtype,
                device=device,
            )
        else:
            return StandardKVCacheManager(
                num_kv_heads=attention_spec.num_kv_heads,
                head_dim=attention_spec.head_dim,
                num_layers=num_layers,
                max_seq_len=max_seq_len,
                dtype=dtype,
                device=device,
            )
```

### Multi-Type Manager

```python
class HybridKVCacheManager:
    """Manages KV caches for models with multiple attention types."""

    def __init__(
        self,
        layer_specs: Dict[int, AttentionSpec],
        max_seq_len: int,
        dtype: torch.dtype,
        device: str,
    ):
        self.managers: Dict[int, BaseKVCacheManager] = {}

        # Group layers by attention type
        type_groups: Dict[AttentionType, List[int]] = defaultdict(list)
        for layer_idx, spec in layer_specs.items():
            type_groups[spec.attention_type].append(layer_idx)

        # Create manager for each group
        for attn_type, layers in type_groups.items():
            spec = layer_specs[layers[0]]
            manager = KVCacheFactory.create(
                attention_spec=spec,
                num_layers=len(layers),
                max_seq_len=max_seq_len,
                dtype=dtype,
                device=device,
            )
            for layer_idx in layers:
                self.managers[layer_idx] = manager
```

## SGLang Integration (Issue #198)

### Current SGLang Code

```python
# kvcached/integration/sglang/patches.py
class ElasticMHATokenToKVPool(MHATokenToKVPool):
    # Only handles MHA
    def _create_buffers(self):
        self.k_buffer, self.v_buffer = kvi.alloc_kv_cache(
            kvcache_shape=(...),
            attention_type="MHA",  # Hardcoded
        )
```

### Proposed Changes

```python
class ElasticTokenToKVPool(BaseTokenToKVPool):
    """Universal KV pool supporting all attention types."""

    def __init__(
        self,
        attention_type: str,  # "MHA", "GQA", "MLA", etc.
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.attention_type = attention_type

    def _create_buffers(self):
        if self.attention_type == "MLA":
            self._create_mla_buffers()
        else:
            self._create_standard_buffers()

    def _create_mla_buffers(self):
        # MLA-specific buffer creation
        self.kv_buffer = kvi.alloc_kv_cache(
            kvcache_shape=(self.size, self.latent_dim),
            attention_type="MLA",
        )

    def _create_standard_buffers(self):
        self.k_buffer, self.v_buffer = kvi.alloc_kv_cache(
            kvcache_shape=(self.size, self.head_num, self.head_dim),
            attention_type=self.attention_type,
        )
```

### Auto-Detection

```python
def detect_attention_type(model_config) -> str:
    """Auto-detect attention type from model configuration."""
    if hasattr(model_config, "kv_lora_rank"):
        return "MLA"
    elif model_config.num_attention_heads == model_config.num_key_value_heads:
        return "MHA"
    elif model_config.num_key_value_heads == 1:
        return "MQA"
    else:
        return "GQA"
```

## vLLM Integration

### Current Limitation

```python
# kvcached/integration/vllm/patches.py
if len(kv_cache_config.kv_cache_groups) > 1:
    raise NotImplementedError(
        "Hybrid models with more than one KV cache type are not supported yet."
    )
```

### Proposed Fix

```python
def _allocate_kv_cache_from_kvcached(self, kv_cache_config):
    kv_cache_tensors = {}

    for group_idx, kv_cache_group in enumerate(kv_cache_config.kv_cache_groups):
        kv_cache_spec = kv_cache_group.kv_cache_spec
        attention_type = self._detect_attention_type(kv_cache_spec)

        group_tensors = kvi.alloc_kv_cache(
            kvcache_shape=self._get_shape(kv_cache_spec),
            attention_type=attention_type,
            num_layers=len(kv_cache_group.layer_names),
        )

        for idx, layer_name in enumerate(kv_cache_group.layer_names):
            kv_cache_tensors[layer_name] = group_tensors[idx]

    return kv_cache_tensors
```

## Memory Layout Optimization

### Standard Layout (MHA/GQA)

```
[Layer 0 K] [Layer 0 V] [Layer 1 K] [Layer 1 V] ...
```

### Interleaved Layout (Better Cache Locality)

```
[Layer 0 K token 0-N] [Layer 0 V token 0-N]
[Layer 1 K token 0-N] [Layer 1 V token 0-N]
...
```

### MLA Layout (Compressed)

```
[Layer 0 Compressed KV] [Layer 1 Compressed KV] ...
```

## Configuration

### Environment Variables

```bash
# Override auto-detection
export KVCACHED_ATTENTION_TYPE=MLA

# Enable hybrid model support
export KVCACHED_HYBRID_SUPPORT=true
```

### API

```python
from kvcached import configure

configure(
    attention_types={
        "layer_0-15": "MHA",
        "layer_16-31": "GQA",
    }
)
```

## Implementation Plan

### Phase 1: GQA/MQA Full Support
- [x] Basic GQA via MHA path
- [ ] Optimize GQA memory layout
- [ ] Add MQA detection

### Phase 2: MLA Support
- [ ] Implement MLA buffer allocation
- [ ] Add compression/decompression hooks
- [ ] Test with DeepSeek-V2

### Phase 3: Hybrid Models
- [ ] Multi-group KV cache allocation
- [ ] Per-layer attention type detection
- [ ] Test with gpt-oss-20b

### Phase 4: Optimization
- [ ] Memory layout optimization
- [ ] Benchmark all attention types
- [ ] Documentation

## Testing Strategy

### Unit Tests
- Attention type detection
- Memory size calculations
- Buffer allocation per type

### Integration Tests
- DeepSeek-V2 (MLA)
- gpt-oss-20b (Hybrid)
- Falcon (MQA)

### Performance Tests
- Memory efficiency comparison
- Latency per attention type
- Multi-model with different types

## Open Questions

1. How to handle dynamic attention patterns (e.g., sparse attention)?
2. Should we support custom attention implementations?
3. How to optimize for flash attention with different types?
4. What's the memory overhead of supporting multiple types?

## References

- [Multi-head Latent Attention (DeepSeek-V2)](https://arxiv.org/abs/2405.04434)
- [GQA: Training Generalized Multi-Query Transformers](https://arxiv.org/abs/2305.13245)
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
