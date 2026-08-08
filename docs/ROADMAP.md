# kvcached Roadmap

This document outlines the planned features and improvements for kvcached.

## Current Status (v0.1.x)

### Supported Features
- Dynamic GPU memory management for KV caches
- vLLM integration (v0.8.4 - v0.10.x)
- SGLang integration (v0.4.x - v0.5.x)
- Multi-model serving on single GPU
- Tensor parallel support
- Real-time memory monitoring (kvctl)

### Supported Attention Types
- Multi-Head Attention (MHA)
- Grouped-Query Attention (GQA)

## Short-Term Goals (Q1 2025)

### Engine Support
- [ ] **Ollama Integration** (Issue #81)
  - Investigate Ollama's memory management architecture
  - Design integration approach
  - Implement basic support

- [ ] **TensorRT-LLM Integration** (Issue #199)
  - Evaluate TensorRT-LLM's KV cache implementation
  - Design compatibility layer
  - Prototype integration

### Attention Types
- [ ] **Multi-head Latent Attention (MLA)** (Issue #198, #202)
  - Add support for DeepSeek-V2 style MLA
  - Extend SGLang integration for MLA models
  - Document attention type configuration

### Quantization Support
- [ ] **FP8 Quantization** (Issue #203, #214)
  - Fix type mismatch errors with FP8 models
  - Test with Qwen-FP8 and similar models
  - Add FP8 to compatibility matrix

- [ ] **FP4/NF4 Quantization**
  - Investigate nvfp4 support requirements
  - Implement dtype handling for 4-bit types

### Performance
- [ ] **Prefix Caching Support**
  - Design prefix cache integration
  - Implement shared prefix blocks
  - Benchmark prefix cache performance

## Medium-Term Goals (Q2-Q3 2025)

### Hardware Support
- [ ] **AMD GPU Support** (Issue #94)
  - Port VMM operations to ROCm
  - Test on MI250/MI300 GPUs
  - Document AMD-specific configuration

- [ ] **ARM64 Support** (Issue #225)
  - Build and test on ARM platforms
  - Optimize for Graviton/Grace CPUs
  - Add ARM to CI pipeline

### Memory Management
- [ ] **CPU Memory Offloading** (Issue #93)
  - Design tiered memory hierarchy
  - Implement CPU-GPU page migration
  - Add offloading policies

- [ ] **Speculative Decoding Support**
  - Handle draft model KV caches
  - Optimize for speculation patterns

### Scalability
- [ ] **Multi-Node Support**
  - Extend IPC for distributed systems
  - Design cross-node memory sharing
  - Implement network-aware allocation

## Long-Term Goals (2025+)

### Advanced Features
- [ ] **Automatic Model Placement**
  - ML-based model placement optimization
  - Dynamic model loading/unloading
  - SLA-aware scheduling

- [ ] **KV Cache Compression**
  - Implement lossy KV compression
  - Design quality-memory tradeoffs
  - Integrate with attention computation

- [ ] **Continuous Batching Optimization**
  - Memory-aware batch scheduling
  - Predictive preallocation
  - Request prioritization

### Ecosystem
- [ ] **Kubernetes Operator** (Issue #87)
  - Design CRD for kvcached resources
  - Implement controller for auto-scaling
  - GPU memory-aware pod scheduling

- [ ] **Prometheus Metrics**
  - Export memory usage metrics
  - Add alerting thresholds
  - Create Grafana dashboards

## Design Documents

For complex features, we maintain design documents:

- [CPU Offloading Design](design/CPU_OFFLOADING.md)
- [Ollama Integration Design](design/OLLAMA_INTEGRATION.md)
- [Multi-Attention Support RFC](design/MULTI_ATTENTION.md)
- [TensorRT-LLM Integration](design/TENSORRT_LLM.md)

## Contributing

We welcome contributions! Priority areas:

1. **Documentation**: Improve guides and examples
2. **Testing**: Add test coverage for edge cases
3. **Bug Fixes**: Address open issues
4. **Integrations**: Help with new engine support

See [CONTRIBUTING.md](../CONTRIBUTING.md) for guidelines.

## Version History

### v0.1.0 (Initial Release)
- Core VMM-based memory management
- vLLM and SGLang integration
- Basic multi-model support
- kvctl monitoring tool

## Feedback

Have suggestions for the roadmap? Open an issue or discussion on GitHub:
- Feature requests: Use the "enhancement" label
- Bug reports: Use the "bug" label
- Questions: Use the "question" label

---

*Last updated: December 2024*
