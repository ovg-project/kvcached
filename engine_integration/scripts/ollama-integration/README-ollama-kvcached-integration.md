# Ollama + kvcached Integration

This integration adds kvcached KV cache management to Ollama for improved memory efficiency and performance.

## Features

- **Direct Source Code Modification**: Patches Ollama's source code to integrate kvcached directly
- **CGO Integration**: Uses CGO to call kvcached Python interface functions from Go
- **Automated Setup**: Dedicated script to clone, patch, and build modified Ollama

## Architecture

```
┌─────────────┐    CGO Bridge    ┌─────────────┐    Python    ┌─────────────┐
│   Ollama    │ ◄──────────────► │ C Bridge    │ ◄──────────► │  kvcached   │
│   (Go)      │                  │   (C)       │              │ (Python/C++)│
└─────────────┘                  └─────────────┘              └─────────────┘
```

1. **Ollama** calls C bridge functions via CGO
2. **C Bridge** calls Python interface functions
3. **Python Interface** calls kvcached C++ functions
4. **kvcached** handles the actual KV cache memory management

## Quick Start

### 1. Automated Setup

From the kvcached project root directory:

```bash
cd engine_integration/scripts/ollama-integration
./setup_ollama.sh
```

This will:
- Install Go and CMake if not present
- Clone Ollama source code from GitHub
- Build kvcached C bridge library
- Apply kvcached patch to Ollama
- Build modified Ollama with CMake (following official guide)
- Install kvcached Python package

### 2. Use Modified Ollama

Following the [Ollama developer guide](https://github.com/ollama/ollama#development):

```bash
cd engine_integration/ollama-v0.11.8

# Start the server with kvcached enabled
LD_LIBRARY_PATH=./kvcached_bridge:$LD_LIBRARY_PATH \
PYTHONPATH=/path/to/kvcached:$PYTHONPATH \
ENABLE_KVCACHED=true \
go run . serve

# In another terminal, run a model
LD_LIBRARY_PATH=./kvcached_bridge:$LD_LIBRARY_PATH \
PYTHONPATH=/path/to/kvcached:$PYTHONPATH \
ENABLE_KVCACHED=true \
go run . run gemma3
```

## How It Works

The integration works by:

- **Patching Ollama**: Modifies Ollama's Go source code to include kvcached calls
- **CGO Bridge**: Uses CGO to call Python interface functions from Go
- **Python Interface**: Provides the same interface as vLLM/SGLang integrations
- **Conditional Initialization**: kvcached starts when `ENABLE_KVCACHED=true` is set
- **Memory Management**: Efficiently manages KV cache memory with virtual memory patterns

## Environment Variables

The following environment variables control the kvcached integration:

- `ENABLE_KVCACHED=true/false`: Enable or disable kvcached integration (default: false)
- `LD_LIBRARY_PATH`: Must include the path to the kvcached bridge library (`./kvcached_bridge`)
- `PYTHONPATH`: Must include the path to the kvcached Python package

## CGO Bridge Implementation

The integration uses a C bridge to connect Go (Ollama) with Python (kvcached):

### C Bridge Enums (`kvcached_bridge.h`)

```c
// Operation types for the bridge message
typedef enum {
    BRIDGE_OP_INIT = 0,
    BRIDGE_OP_ALLOC_KV_CACHE = 1,
    BRIDGE_OP_ALLOC_KV_BRIDGE = 2,
    BRIDGE_OP_FREE_KV = 3,
    BRIDGE_OP_SHUTDOWN = 4
} bridge_operation_t;

// Logging levels for the bridge
typedef enum {
    LOG_DEBUG = 0,
    LOG_INFO = 1,
    LOG_WARN = 2,
    LOG_ERROR = 3
} log_level_t;
```

### C Bridge Functions (`kvcached_bridge.h`)

```c
// Initialize the Python bridge
int kvcached_bridge_init();

// Call Python init_kvcached function (Stage 1)
int kvcached_bridge_init_kvcached(const char* device, int async_sched);

// Call Python alloc_kv_cache function (Stage 2)
int kvcached_bridge_alloc_kv_cache(int num_blocks, int block_size, int head_num, int head_dim, int num_layers, const char* device);

// Call Python alloc_kv_bridge function (Stage 3 - allocate blocks for a request)
long long* kvcached_bridge_alloc_kv(int num_blocks);

// Call Python free_kv function (free blocks when needed)
int kvcached_bridge_free_kv(long long* block_ids, int num_blocks);

// Call Python shutdown_kvcached function
int kvcached_bridge_shutdown_kvcached();

// Cleanup function
void kvcached_bridge_cleanup();

// Set logging level (0=DEBUG, 1=INFO, 2=WARN, 3=ERROR)
void kvcached_bridge_set_log_level(int level);
```

### Python Interface Functions (`kvcached/integration/ollama/interfaces.py`)

The C bridge calls these Python functions:

```python
# Initialize kvcached (Stage 1)
init_kvcached(tp_rank=0, tp_size=1, is_worker=False, device="cuda:0", async_sched=True)

# Allocate KV cache tensors (Stage 2)
# Automatically calculates optimal num_blocks based on GPU memory
kv_tensors = alloc_kv_cache(
    kvcache_shape=(2, num_blocks, head_num, head_dim),
    block_size=block_size,
    dtype="float16",  # or torch.float16
    device="cuda:0",
    num_layers=num_layers,
    attention_type="MHA",
    kv_layout="NHD"
)

# Allocate specific blocks for a request (Stage 3)
block_ids = alloc_kv_bridge(num_blocks)

# Free blocks when necessary
result = free_kv_bridge(block_ids)

# Shutdown kvcached
shutdown_kvcached()
```

## Manual Setup

If you prefer to set up components manually instead of using the automated `setup_ollama.sh` script, follow these steps:

### 1. Install Prerequisites

Following the [Ollama developer guide](https://github.com/ollama/ollama#development):

```bash
# Install uv package manager
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install Go and CMake for Ollama
sudo apt install golang-go cmake

# Optional: Install CUDA for NVIDIA GPU support
# Follow: https://developer.nvidia.com/cuda-downloads
```

### 2. Clone Ollama Source Code

```bash
cd engine_integration

# Clone Ollama source code (or update if it exists)
git clone -b v0.11.8 https://github.com/ollama/ollama.git ollama-v0.11.8
cd ollama-v0.11.8
```

### 3. Setup Python Environment and Dependencies

```bash
# Create Python virtual environment
uv venv --python=python3.11
source .venv/bin/activate
uv pip install --upgrade pip

# Install kvcached requirements (from project root)
uv pip install -r ../../../requirements.txt
```

### 4. Apply kvcached Patch

```bash
# Apply the kvcached integration patch (creates C bridge files)
git apply ../scripts/kvcached-ollama-v0.11.8.patch
```

### 5. Build C Bridge Library

```bash
# Build the kvcached C bridge library
gcc -shared -fPIC -o kvcached_bridge/libkvcached_bridge.so kvcached_bridge/kvcached_bridge.c \
    $(python3-config --cflags --ldflags) -lpython3.11 -lpthread
```

### 6. Build Ollama

```bash
# Build Ollama with CMake
cmake -B build
cmake --build build
```

### 7. Install kvcached

Choose one of the following installation methods:

**For Development (Editable Install):**

```bash
# Install kvcached in editable mode (recommended for development)
pip wheel ../../../kvcached -w /tmp/kvcached_wheel --no-build-isolation --no-cache-dir
uv pip install /tmp/kvcached_wheel/kvcached-*.whl --no-cache-dir

# Create editable install (hybrid approach)
site_packages=$(python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')
installed_pkg_dir="$site_packages/kvcached"

# Save compiled binaries
tmp_so_dir=$(mktemp -d)
find "$installed_pkg_dir" -name 'vmm_ops*.so' -exec mv {} "$tmp_so_dir/" \; || true

# Uninstall Python files
uv pip uninstall kvcached

# Restore binaries and create proxy
mkdir -p "$installed_pkg_dir"
mv "$tmp_so_dir"/*.so "$installed_pkg_dir/"
cat > "$installed_pkg_dir/__init__.py" <<EOF
import os
import sys
__path__.insert(0, os.path.abspath(os.path.join("../../../..", "kvcached")))
EOF
```

**For Production (Release Install):**

```bash
# Install released kvcached package
uv pip install kvcached --no-build-isolation --no-cache-dir
```

### 8. Test the Integration

```bash
# Test the integration with kvcached enabled
OLLAMA_DEBUG=1 LD_LIBRARY_PATH=./kvcached_bridge:$LD_LIBRARY_PATH \
PYTHONPATH=/path/to/kvcached:$PYTHONPATH \
ENABLE_KVCACHED=true \
go run . --version

# Start the server
OLLAMA_DEBUG=1 LD_LIBRARY_PATH=./kvcached_bridge:$LD_LIBRARY_PATH \
PYTHONPATH=/path/to/kvcached:$PYTHONPATH \
ENABLE_KVCACHED=true \
go run . serve

# In another terminal, test with a model
OLLAMA_DEBUG=1 LD_LIBRARY_PATH=./kvcached_bridge:$LD_LIBRARY_PATH \
PYTHONPATH=/path/to/kvcached:$PYTHONPATH \
ENABLE_KVCACHED=true \
go run . run gemma3
```
