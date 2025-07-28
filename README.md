# kvcached

kvcached is a new KV cache management system that supports on-demand KV cache allocation. It implements the concept of GPU virtual memory, allowing applications to reserve virtual address space without immediately committing physical memory. Physical memory is then automatically allocated and mapped as needed at runtime. This capability allows multiple LLMs to run concurrently on a single GPU or a group of GPUs (TP) and flexibly share the GPU memory, significantly improving GPU utilization and reducing memory fragmentation.

kvcached is compatible with popular LLM serving engines, including SGLang and vLLM.

## kvcached Installation

### Prerequisites

* Python (tested with 3.11)
* PyTorch (tested with 2.6.0 and 2.7.0)

kvcached can be installed as simple as

```bash
pip install kvcached --no-build-isolation
```

Note that `--no-build-isolation` is required for kvcached to find the right PyTorch in the current virtual environment. If PyTorch is re-installed or upgraded, kvcached also needs re-installation.

kvcached now supports both **SGLang** and **vLLM**. While we are in the process of upstreaming the integration interfaces, we provide temporary support via code patches.

To apply a patch:

```bash
cd $PATH_TO_ENGINE_SRC # Where the source code of installed SGLang or vLLM is
git apply $PATH_TO_KVCACHED/engine_integration/scripts/$PATCH_FILE
```

## All-in-One Installation (Recommended for Development)

You can install everything (SGLang+vLLM+kvcached) by running the following commands:

```bash
cd engine_integration/scripts
./setup.sh all
```

This script will download the specified versions of SGLang and vLLM, create separate venv environments (using `uv`), compile the code, apply the necessary patches, and install kvcached.

## Run kvcached with Docker

You can test kvcached with Docker. We provide Dockerfile for both SGLang and vLLM.

Take SGLang as an example. To build the image (will take a few minutes):

```bash
docker build -f docker/Dockerfile.sglang -t kvcached-sglang:0.4.9 .
```

Run it:

```bash
docker run -itd --shm-size 32g --gpus all \
    -v ~/.cache:/root/.cache \
    -v /dev/shm:/shm \
    --ipc=host \
    --network=host \
    --privileged \
    --name kvcached-sglang \
    kvcached-sglang:0.4.9 \
    bash
```

## Testing

kvcached can be enabled or disabled by `export ENABLE_KVCACHED=true` or `false`. To verify the successful installation and benchmark the performance of SGLang/vLLM with kvcached, run:

```bash
cd engine_integration/benchmark
./start_server.sh [sgl|vllm]
# Wait until LLM server is ready
./start_client.sh [sgl|vllm]
```

The benchmark scripts automatically set `ENABLE_KVCACHED=true`. Please refer to each script for instructions on how to run SGLang/vLLM with kvcached.

## Memory monitoring and control via kvcached CLI

kvcached includes a built-in CLI tool that allows you to monitor GPU memory usage and manage memory limits across different applications. A command `kvctl` is installed along with kvcached package:

```bash
kvctl
```

Once inside the CLI, type `help` to view all supported commands:

```
kvcached> help
Available commands:
  list [ipc ...]               List IPC segments and usage
  limit <ipc> <size>           Set absolute limit (e.g. 512M, 2G)
  limit-percent <ipc> <pct>    Set limit as percentage of total GPU RAM
  watch [-n sec] [ipc ...]     Continuously display usage table
  kvtop [ipc ...] [--refresh r]  Launch curses kvtop UI (q to quit)
  !<shell cmd>                 Run command in system shell
  help                         Show this help message
  delete <ipc>                 Delete IPC segment and its limit entry
  exit | quit                  Exit the shell

kvcached>
```

Use the `kvtop` command for real-time visualization of memory usage:

<!-- KVCache memory monitor (muted colours) -->
<pre>
<span style="color:#009ACD; font-weight:bold;">KVCache Memory Usage</span>

<span style="color:#009ACD;">IPC: SGLANG</span>
<span style="color:#009ACD;">[</span><span style="color:#B7A800;">==</span><span style="color:#009E8F;">##################</span><span style="color:#666666;">----------------------------------------</span><span style="color:#009ACD;">]</span>
Prealloc: 792.0&nbsp;MB | Used: 11.2&nbsp;GB / 39.9&nbsp;GB (30.1%) | Free: 27.9&nbsp;GB

<span style="color:#009ACD;">IPC: VLLM</span>
<span style="color:#009ACD;">[</span><span style="color:#B7A800;">==</span><span style="color:#009E8F;">#######</span><span style="color:#666666;">--------------------------------------------------- </span><span style="color:#009ACD;">]</span>
Prealloc: 768.0&nbsp;MB | Used: 3.6&nbsp;GB / 37.4&nbsp;GB (11.7%) | Free: 33.0&nbsp;GB

<span style="color:#009ACD;">GPU Memory Usage</span>
<span style="color:#009ACD;">[</span><span style="color:#B7A800;">########################################</span><span style="color:#666666;">--------------------</span><span style="color:#009ACD;">]</span>
Used: 52.9&nbsp;GB / 79.2&nbsp;GB (66.8%) | Free: 26.3&nbsp;GB

<span style="color:#555555;">Press 'q' to quit</span>
</pre>

## Contributing

We are grateful for and open to contributions and collaborations of any kind.

We use pre-commit to ensure a consistent coding style. You can set it up by

```
pip install pre-commit
pre-commit install
```

Before pushing your code, please run the following check and make sure your code passes all checks.

```
pre-commit run --all-files
```

## Contacts

Feel free to contact us for contributions and collaborations.

```
Jiarong Xing (jxing@rice.edu)
Yifan Qiao (yifanqiao@berkeley.edu)
Shan Yu (shanyu1@g.ucla.edu)
```
