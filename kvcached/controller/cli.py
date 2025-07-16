import logging

import posix_ipc

from kvcached.utils import (DEFAULT_IPC_NAME, MemInfoStruct, RwLockedShm,
                            get_ipc_name)

logger = logging.getLogger(__name__)


def get_kv_cache_limit(ipc_name: str) -> MemInfoStruct:
    """
    Get the kv cache limit for the current process.
    """
    try:
        with RwLockedShm(get_ipc_name(ipc_name), MemInfoStruct.SHM_SIZE,
                         RwLockedShm.RLOCK) as mm:
            return MemInfoStruct.from_buffer(mm)
    except FileNotFoundError:
        return None


def init_kv_cache_limit(ipc_name: str, kv_cache_limit: int):
    """
    Set the kv cache limit for the current process.
    Creates a persistent shared memory file that remains even after the script exits.
    """
    shm = posix_ipc.SharedMemory(get_ipc_name(ipc_name),
                                 posix_ipc.O_CREAT,
                                 size=MemInfoStruct.SHM_SIZE,
                                 mode=0o666)
    shm.close_fd()

    # Now we can safely memory map and write the values
    with RwLockedShm(get_ipc_name(ipc_name), MemInfoStruct.SHM_SIZE,
                     RwLockedShm.WLOCK) as mm:
        mem_info = MemInfoStruct.from_buffer(mm)
        mem_info.total_size = kv_cache_limit
        mem_info.used_size = 0
        mem_info.write_to_buffer(mm)
        return mem_info


def update_kv_cache_limit(ipc_name: str, kv_cache_limit: int):
    """
    Update the kv cache limit for the current process.
    """
    try:
        with RwLockedShm(get_ipc_name(ipc_name), MemInfoStruct.SHM_SIZE,
                         RwLockedShm.WLOCK) as mm:
            mem_info = MemInfoStruct.from_buffer(mm)
            delta = kv_cache_limit - mem_info.total_size
            if delta < 0:
                if mem_info.total_size - mem_info.used_size + delta < 0:
                    logger.warning(
                        f"No enough free space to decrease for the new kv_cache_limit for {ipc_name}"
                    )
            mem_info.total_size = kv_cache_limit
            mem_info.write_to_buffer(mm)
            return mem_info
    except FileNotFoundError:
        return None


def get_total_gpu_memory() -> int:
    """Return total memory of CUDA device 0 or 0 if CUDA unavailable."""
    try:
        import torch  # imported lazily to avoid heavy import cost when not needed

        if torch.cuda.is_available():
            return torch.cuda.get_device_properties(0).total_memory
    except Exception:  # pragma: no cover – best-effort helper
        pass
    return 0


def main(args):
    if args.action == "init":
        init_kv_cache_limit(args.ipc_name, args.size)
        print(f"Initialized kv cache limit to {args.size}")
    elif args.action == "get":
        mem_info = get_kv_cache_limit(args.ipc_name)
        if mem_info is None:
            print("No kv cache limit set")
        else:
            print(
                f"{{kv_cache_limit: {mem_info.total_size}, in_use: {mem_info.used_size}}}"
            )
    elif args.action == "update":
        update_kv_cache_limit(args.ipc_name, args.size)
        print(f"Updated kv cache limit to {args.size}")
    else:
        raise ValueError(f"Invalid action: {args.action}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--action', choices=['init', 'get', 'update'])
    args, remaining = parser.parse_known_args()

    parser = argparse.ArgumentParser()
    parser.add_argument("--ipc_name", type=str, default=DEFAULT_IPC_NAME)
    parser.add_argument("--action",
                        type=str,
                        required=True,
                        choices=["init", "get", "update"])

    if args.action == "init" or args.action == "update":
        parser.add_argument("--size", type=str, required=True)
        args = parser.parse_args()

        # Convert size string to bytes
        size_str = args.size
        if size_str.endswith("M") or size_str.endswith("m"):
            args.size = int(size_str[:-1]) * 1024 * 1024
        elif size_str.endswith("G") or size_str.endswith("g"):
            args.size = int(size_str[:-1]) * 1024 * 1024 * 1024
        else:
            args.size = int(size_str)
    elif args.action == "get":
        args = parser.parse_args()
    else:
        raise ValueError(f"Invalid action: {args.action}")

    main(args)
