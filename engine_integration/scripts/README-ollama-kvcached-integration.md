# Ollama + kvcached Integration

This integration adds kvcached KV cache management to Ollama for improved memory efficiency and performance.

## Features

- **Direct Source Code Modification**: Patches Ollama's source code to integrate kvcached directly
- **CGO Integration**: Uses CGO to call kvcached Python interface functions from Go
- **Automatic Setup**: Scripts to clone, patch, and build modified Ollama
- **Real Parameters**: Uses Ollama's actual model configuration, not fake parameters

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

### 1. Install Prerequisites

Following the [Ollama developer guide](https://github.com/ollama/ollama#development) for Linux:

```bash
# Install Go
sudo apt install golang-go

# Install CMake
sudo apt install cmake

# Optional: Install CUDA for NVIDIA GPU support
# Follow instructions at: https://developer.nvidia.com/cuda-downloads
```

### 2. Automated Setup (Recommended)

```bash
cd engine_integration/scripts
./setup.sh ollama
```

This will:
- Install Go and CMake if not present
- Clone Ollama source code from GitHub
- Build kvcached C bridge library
- Apply kvcached patch to Ollama
- Build modified Ollama with CMake (following official guide)
- Install kvcached Python package

### 3. Use Modified Ollama

Following the [Ollama developer guide](https://github.com/ollama/ollama#development):

```bash
cd engine_integration/ollama-v0.11.8

# Start the server
LD_LIBRARY_PATH=./kvcached_bridge:$LD_LIBRARY_PATH \
PYTHONPATH=/path/to/kvcached:$PYTHONPATH \
go run . serve

# In another terminal, run a model
LD_LIBRARY_PATH=./kvcached_bridge:$LD_LIBRARY_PATH \
PYTHONPATH=/path/to/kvcached:$PYTHONPATH \
go run . run gemma3
```

**Note**: The CMake build creates optimized binaries in the `build/` directory. For development and testing, `go run .` works perfectly.

The kvcached integration will automatically initialize when you start Ollama with a model.

## How It Works

The integration works by:

- **Patching Ollama**: Modifies Ollama's Go source code to include kvcached calls
- **CGO Bridge**: Uses CGO to call Python interface functions from Go
- **Python Interface**: Provides the same interface as vLLM/SGLang integrations
- **Automatic Initialization**: kvcached starts automatically when Ollama loads a model
- **Memory Management**: Efficiently manages KV cache memory with virtual memory patterns
- **Performance Monitoring**: Provides real-time cache statistics and performance metrics

## CGO Bridge Implementation

The integration uses a C bridge to connect Go (Ollama) with Python (kvcached):

### C Bridge Functions (`kvcached_bridge.c`)

```c
// Initialize Python interpreter and import kvcached module
int kvcached_bridge_init();

// Call Python init_kvcached function
int kvcached_bridge_init_kvcached(const char* device, int async_sched);

// Call Python alloc_kv_cache function
long long kvcached_bridge_alloc_kv_cache(
    int* kvcache_shape, int shape_len,
    int block_size,
    const char* dtype_str,
    const char* device,
    int num_layers
);

// Call Python get_kv_cache_manager function
long long kvcached_bridge_get_kv_cache_manager(
    int num_blocks,
    int block_size,
    int cell_size,
    int num_layers
);

// Call Python shutdown_kvcached function
int kvcached_bridge_shutdown_kvcached();
```

### Go CGO Integration

```go
/*
#cgo CFLAGS: -I../kvcached_bridge
#cgo LDFLAGS: -L../kvcached_bridge -lkvcached_bridge
#include "kvcached_bridge.h"
*/
import "C"

func (l *LLMInstance) initKVCache() error {
    device := C.CString("cuda:0")
    defer C.free(unsafe.Pointer(device))
    
    result := C.kvcached_bridge_init_kvcached(device, 1) // async_sched = true
    if result != 0 {
        return fmt.Errorf("failed to initialize kvcached")
    }
    l.kvCacheInitialized = true
    return nil
}
```

### Python Interface Methods

The C bridge calls these Python functions:

```python
# kvcached/integration/ollama/interfaces.py

# Initialize kvcached
init_kvcached(device="cuda:0", async_sched=True)

# Allocate KV cache tensors
kv_tensors = alloc_kv_cache(
    kvcache_shape=(2, num_blocks, num_heads, head_dim),
    block_size=block_size,
    dtype=torch.float16,
    device="cuda:0",
    num_layers=num_layers
)

# Get KV cache manager
kv_cache_manager = get_kv_cache_manager(
    num_blocks=num_blocks,
    block_size=block_size,
    cell_size=cell_size,
    num_layers=num_layers
)

# Shutdown kvcached
shutdown_kvcached()
```

## Manual Setup (Advanced)

If you prefer to set up components manually instead of using `setup.sh`:

### 1. Install Prerequisites

Following the [Ollama developer guide](https://github.com/ollama/ollama#development):

```bash
# Install Go and CMake
sudo apt install golang-go cmake

# Optional: Install CUDA for NVIDIA GPU support
# Follow: https://developer.nvidia.com/cuda-downloads
```

### 2. Clone and Patch Ollama

```bash
cd engine_integration

# Clone Ollama source code
git clone -b v0.11.8 https://github.com/ollama/ollama.git ollama-v0.11.8
cd ollama-v0.11.8

# Apply the kvcached integration patch (creates C bridge files)
git apply ../scripts/kvcached-ollama-v0.11.8.patch
```

### 3. Build C Bridge and Ollama

```bash
# Build the C bridge library
PYTHON_INCLUDES=$(python3-config --includes)
PYTHON_LDFLAGS=$(python3-config --ldflags)
gcc -shared -fPIC -o kvcached_bridge/libkvcached_bridge.so kvcached_bridge/kvcached_bridge.c \
    $PYTHON_INCLUDES $PYTHON_LDFLAGS

# Build Ollama with CMake
cmake -B build
cmake --build build
cp build/ollama ollama-kvcached
```

### 4. Install Python Dependencies

```bash
# Install kvcached
uv venv --python=python3.11
source .venv/bin/activate
uv pip install -e ../../kvcached
```

### 5. Test the Integration

```bash
cd ollama-v0.11.8

# Test the integration
LD_LIBRARY_PATH=./kvcached_bridge:$LD_LIBRARY_PATH \
PYTHONPATH=/path/to/kvcached:$PYTHONPATH \
go run . --version

# Start the server
LD_LIBRARY_PATH=./kvcached_bridge:$LD_LIBRARY_PATH \
PYTHONPATH=/path/to/kvcached:$PYTHONPATH \
go run . serve

# In another terminal, test with a model
LD_LIBRARY_PATH=./kvcached_bridge:$LD_LIBRARY_PATH \
PYTHONPATH=/path/to/kvcached:$PYTHONPATH \
go run . run gemma3
```

**Note**: The CMake build creates optimized binaries in `build/ollama`. You can also run them directly:

```bash
LD_LIBRARY_PATH=./kvcached_bridge:$LD_LIBRARY_PATH \
PYTHONPATH=/path/to/kvcached:$PYTHONPATH \
./build/ollama serve
```

**Note**: This README is for the kvcached integration. The original Ollama README is in the `ollama-v0.11.8/` directory.

## Configuration

### Model Configuration

The integration automatically detects model parameters from Ollama's internal model loading process. No manual configuration is needed - the model parameters are obtained directly from the loaded model.

## Performance Monitoring

The integration provides real-time monitoring of:
- **Cache Hit Rate**: Percentage of tokens found in cache
- **Memory Usage**: Current memory consumption
- **Token Throughput**: Tokens processed per second
- **Session Statistics**: Per-model performance metrics

## Troubleshooting

### Common Issues

1. **CGO Build Failed**
   - Ensure Python development headers are installed
   - Check that the C bridge library was built successfully
   - Verify Go CGO environment is properly configured

2. **Model Not Supported**
   - Add custom model configuration
   - Check model name matching logic

3. **Memory Issues**
   - Adjust block size and number of blocks
   - Monitor GPU memory usage
   - Consider reducing model batch size

### Debug Mode

Enable debug logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Health Checks

Test the integration:

```bash
# Test the modified Ollama binary
./ollama-kvcached --version

# Test with a model
./ollama-kvcached run llama2:7b
```

## Development

### Adding New Features

1. **Extend C Bridge**: Add new functions to `kvcached_bridge.c`
2. **Update Go Code**: Add new methods to LLM instances
3. **Modify Interfaces**: Update Python integration functions as needed

### Testing

```bash
# Test the C bridge library
cd kvcached_bridge
gcc -o test_bridge test_bridge.c -lkvcached_bridge

# Test the modified Ollama
cd ../ollama
go test ./...
```

## License

This integration is part of the kvcached project and follows the same license terms.
