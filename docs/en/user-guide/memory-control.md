# Memory Control CLI

kvcached includes built-in CLI tools for monitoring and controlling GPU memory usage.

## kvctl — Interactive Memory Manager

```bash
kvctl
```

### Available Commands

| Command | Description |
|---------|-------------|
| `list [ipc ...]` | List IPC segments and usage |
| `limit <ipc> <size>` | Set absolute limit (e.g., `512M`, `2G`) |
| `limit-percent <ipc> <pct>` | Set limit as percentage of total GPU RAM |
| `watch [-n sec] [ipc ...]` | Continuously display usage table |
| `kvtop [ipc ...] [--refresh r]` | Launch curses UI (q to quit) |
| `delete <ipc>` | Delete IPC segment and its limit entry |
| `!<shell cmd>` | Run command in system shell |

### Example Usage

```bash
kvcached> list
kvcached> limit-percent VLLM 50
kvcached> limit-percent SGLANG 50
kvcached> watch -n 2
```

## kvtop — Real-Time Memory Monitor

```bash
kvtop
```

Shows a live curses UI with per-model memory bars:

```
KVCache Memory Usage

IPC: SGLANG
[==##################----------------------------------------]
Prealloc: 792.0 MB | Used: 11.2 GB / 39.9 GB (30.1%) | Free: 27.9 GB

IPC: VLLM
[==#######---------------------------------------------------]
Prealloc: 768.0 MB | Used: 3.6 GB / 37.4 GB (11.7%) | Free: 33.0 GB

GPU Memory Usage
[########################################--------------------]
Used: 52.9 GB / 79.2 GB (66.8%) | Free: 26.3 GB

Press 'q' to quit
```
