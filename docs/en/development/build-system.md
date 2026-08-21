# Build System

## Overview

kvcached uses setuptools with a C++ extension (`vmm_ops`) compiled via PyTorch's build utilities.

## Components

- `pyproject.toml` — Project metadata, dependencies, and build system configuration
- `setup.py` — C++ extension build logic with CUDA/HIP detection
- `kvcached_autopatch.pth` — Python path configuration file for autopatch

## C++ Extension (vmm_ops)

The `csrc/` directory contains the C++ source for GPU virtual memory operations:

- Built with `torch.utils.cpp_extension.CUDAExtension` (NVIDIA) or `CppExtension` (AMD)
- Links against `libcuda` (NVIDIA) or `libamdhip64` (AMD)
- Requires C++17 standard

## Building

```bash
pip install -e . --no-build-isolation --no-cache-dir
```

The `--no-build-isolation` flag is required because the extension links against your installed PyTorch's CUDA libraries.

## .pth File Installation

The `kvcached_autopatch.pth` file must be in Python's `site-packages` to enable autopatch. For development:

```bash
python tools/dev_copy_pth.py
```
