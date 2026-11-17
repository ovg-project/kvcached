# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0
"""Normal vLLM requests, public control APIs and reset; observe ZMQ while idle."""
from __future__ import annotations

import argparse
import fcntl
import gc
import hashlib
import json
import os
import sys
import tempfile
import threading
import time
from collections import Counter
from pathlib import Path
from typing import Any


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--events", type=int, required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model", required=True, help="Model repository ID or local directory")
    args = parser.parse_args()
    out = args.out
    out.mkdir(parents=True, exist_ok=False)

    # A normal supported controller environment is required. Do not substitute
    # constants, mutate engine state, or manufacture a private callback sequence.
    required = ("F_OFD_SETLK", "F_OFD_GETLK")
    if any(not hasattr(fcntl, name) for name in required):
        raise SystemExit("Controller Python lacks Linux OFD constants; use a standard Linux Python build. No compatibility patch is applied.")

    import msgspec
    import torch
    import vllm
    import zmq
    from vllm import LLM, SamplingParams
    from vllm.config.kv_events import KVEventsConfig
    from vllm.distributed.kv_events import KVEventBatch
    from zmq.utils.monitor import recv_monitor_message

    import kvcached
    from kvcached import control_segment
    from kvcached.cli.utils import get_kv_cache_limit, update_kv_cache_limit
    with tempfile.TemporaryFile() as lock_probe:
        control_segment.lock(lock_probe.fileno())

    done = threading.Event()
    listening = threading.Event()
    connected = threading.Event()
    batches: list[tuple[int, Any]] = []
    errors: list[str] = []

    def collect():
        socket = zmq.Context.instance().socket(zmq.SUB)
        socket.setsockopt(zmq.SUBSCRIBE, b"")
        monitor = socket.get_monitor_socket(events=zmq.EVENT_CONNECTED)
        socket.connect(f"tcp://127.0.0.1:{args.port}")
        listening.set()
        try:
            with (out / "events.jsonl").open("wb") as log:
                while not done.is_set():
                    if monitor.poll(0):
                        recv_monitor_message(monitor)
                        connected.set()
                    if not socket.poll(100):
                        continue
                    topic, sequence, payload = socket.recv_multipart()
                    batch = msgspec.msgpack.decode(payload, type=KVEventBatch)
                    batches.append((int.from_bytes(sequence, "big"), batch))
                    log.write(msgspec.json.encode(batch) + b"\n")
                    log.flush()
        except Exception as error:
            errors.append(repr(error))
        finally:
            monitor.close(linger=0)
            socket.close(linger=0)

    def counts():
        return Counter(type(event).__name__ for _, batch in tuple(batches) for event in batch.events)

    def memory():
        state = get_kv_cache_limit(os.environ["KVCACHED_IPC_NAME"])
        assert state is not None
        return dict(total=state.total_size, used=state.used_size, preallocated=state.prealloc_size)

    reader = threading.Thread(target=collect, name="idle-event-consumer", daemon=True)
    reader.start()
    assert listening.wait(5)
    model = args.model
    config = dict(model=model, tokenizer=model, trust_remote_code=False,
        dtype="float16", enforce_eager=True, max_model_len=512, max_num_seqs=1,
        max_num_batched_tokens=512, gpu_memory_utilization=0.20, block_size=16, seed=42,
        enable_prefix_caching=True,
        kv_events_config=KVEventsConfig(enable_kv_cache_events=bool(args.events),
            publisher="zmq", endpoint=f"tcp://*:{args.port}",
            replay_endpoint=f"tcp://*:{args.port + 1}"))
    source = Path(kvcached.__file__).parent / "integration/vllm/patches.py"
    (out / "settings.json").write_text(json.dumps(dict(name=args.name,
        config={key: str(value) for key, value in config.items()},
        vllm=vllm.__version__, torch=torch.__version__, cuda=torch.version.cuda,
        gpu=torch.cuda.get_device_name(0), source=str(source),
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        ipc=os.environ["KVCACHED_IPC_NAME"], python=sys.version,
        CUDA_VISIBLE_DEVICES=os.getenv("CUDA_VISIBLE_DEVICES"), controller_kernel_lock_probe="passed"), indent=2) + "\n")
    outputs = []
    observed: dict[str, Any] = {}
    llm = None
    original_limit = None
    try:
        llm = LLM(**config)
        if args.events:
            assert connected.wait(15), "Subscriber did not connect"
            time.sleep(5)
        tokenizer = llm.get_tokenizer()
        common = tokenizer.encode(" A cache stores immutable prefixes and preserves request ownership.") * 80
        prompts = [(tokenizer.encode(f"Independent engineering document number {i}:\n") + common)[:384]
                   for i in range(20)]
        assert all(len(prompt) == 384 for prompt in prompts)
        (out / "prompts.json").write_text(json.dumps(prompts) + "\n")
        sampling = SamplingParams(temperature=0, seed=42, max_tokens=4, min_tokens=4, ignore_eos=True)

        def generate(tokens, phase):
            result, = llm.generate([dict(prompt_token_ids=tokens)], sampling, use_tqdm=False)
            generated = result.outputs[0]
            assert len(generated.token_ids) == 4
            outputs.append(dict(phase=phase, token_ids=list(generated.token_ids),
                text=generated.text, num_cached_tokens=getattr(result, "num_cached_tokens", None)))

        for index, prompt in enumerate(prompts):
            generate(prompt, f"fill-{index}")
        generate(prompts[-1], "reuse")
        time.sleep(0.5)
        initial_counts = counts()
        before = memory()
        original_limit = before["total"]
        target = before["used"] // 2
        assert 0 < target < before["used"]
        observed.update(memory_before=before, counts_before=dict(initial_counts), target=target)
        assert update_kv_cache_limit(os.environ["KVCACHED_IPC_NAME"], target) is not None
        deadline = time.monotonic() + 6
        while time.monotonic() < deadline:
            state = memory()
            removed = counts()["BlockRemoved"] - initial_counts["BlockRemoved"]
            if state["used"] <= target and (removed > 0 or not args.events):
                break
            time.sleep(0.05)
        observed.update(memory_after_idle=memory(), idle_removed=counts()["BlockRemoved"] - initial_counts["BlockRemoved"],
                        requests_during_idle=0)
        # Restore capacity before generating anything else, even in the failing baseline.
        assert update_kv_cache_limit(os.environ["KVCACHED_IPC_NAME"], original_limit) is not None
        time.sleep(0.6)
        generate(prompts[-1], "recovery")
        time.sleep(0.3)
        observed["counts_after_recovery"] = dict(counts())
        before_reset = counts()["AllBlocksCleared"]
        observed["reset_result"] = llm.reset_prefix_cache()
        deadline = time.monotonic() + (4 if args.events else 0.4)
        while time.monotonic() < deadline and counts()["AllBlocksCleared"] == before_reset:
            time.sleep(0.05)
        observed["idle_cleared"] = counts()["AllBlocksCleared"] - before_reset
        observed["requests_after_reset"] = 0
        observed["memory_after_reset"] = memory()
    finally:
        try:
            if original_limit is not None:
                update_kv_cache_limit(os.environ["KVCACHED_IPC_NAME"], original_limit)
        finally:
            if llm is not None:
                del llm
            gc.collect()
        time.sleep(0.3)
        done.set()
        reader.join(timeout=3)
        (out / "outputs.json").write_text(json.dumps(outputs, indent=2) + "\n")
        observed.update(completed=len(outputs), events=dict(counts()), listener_errors=errors,
                        sequences=[sequence for sequence, _ in batches],
                        data_parallel_ranks=sorted({batch.data_parallel_rank for _, batch in batches}))
        (out / "result.json").write_text(json.dumps(observed, indent=2) + "\n")
    assert not reader.is_alive() and not errors, errors
    assert observed["memory_after_idle"]["used"] <= target
    assert observed["memory_after_idle"]["used"] < before["used"]
    sequences = observed["sequences"]
    assert not sequences or sequences == list(range(sequences[0], sequences[-1] + 1))
    if args.events:
        assert initial_counts["BlockStored"] > 0
        assert observed["idle_removed"] > 0, "physically reclaimed while idle without removal publication"
        assert observed["idle_cleared"] == 1, "idle reset did not publish AllBlocksCleared"
    else:
        assert not batches
    print("IDLE_MODEL_RESULT", json.dumps(observed), flush=True)


if __name__ == "__main__":
    main()
