# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0
"""Normal AsyncLLM flow from PR #584 review; no private pool calls or patches.

Execution results are recorded separately; the script alone is not validation.
Use --check-observer for a dependency-free check of the subscriber estimator.
"""

import argparse
import asyncio
import hashlib
import json
import os
import threading
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from typing import Any


class Residency:
    def __init__(self):
        self.prefixes: dict[tuple[int, Any], tuple[int, ...]] = {}
        self.resident: set[tuple[int, Any]] = set()

    def apply(self, kind, event):
        if kind == "AllBlocksCleared":
            self.resident.clear()
        elif kind == "BlockRemoved":
            for key in event.block_hashes:
                self.resident.discard((event.group_idx, key))
        elif kind == "BlockStored":
            group = event.group_idx
            parent = () if event.parent_block_hash is None else self.prefixes[
                (group, event.parent_block_hash)
            ]
            hashes = event.block_hashes or []
            assert len(event.token_ids) == len(hashes) * event.block_size
            for index, key in enumerate(hashes):
                start = index * event.block_size
                parent += tuple(event.token_ids[start:start + event.block_size])
                self.prefixes[group, key] = parent
                self.resident.add((group, key))

    def estimate(self, tokens):
        # vLLM recomputes the final prompt token, so an exactly full prompt
        # cannot be counted as a complete cache hit.
        prefixes = {self.prefixes[key] for key in self.resident if key[0] == 0}
        hit = 0
        for end in range(16, len(tokens), 16):
            if tuple(tokens[:end]) not in prefixes:
                break
            hit = end
        return hit


def check_observer():
    tokens = list(range(192))
    tracker = Residency()
    tracker.apply("BlockStored", SimpleNamespace(
        group_idx=0, parent_block_hash=None, block_hashes=list(range(12)),
        token_ids=tokens, block_size=16,
    ))
    assert tracker.estimate(tokens) == 176
    tracker.apply("BlockRemoved", SimpleNamespace(group_idx=0, block_hashes=[9]))
    assert tracker.estimate(tokens) == 144
    print("observer arithmetic check passed; no engine/GPU execution")


async def run(args):
    import msgspec
    import torch
    import vllm
    import zmq
    from transformers import AutoTokenizer
    from vllm import SamplingParams
    from vllm.config.kv_events import KVEventsConfig
    from vllm.distributed.kv_events import KVEventBatch
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.sampling_params import RequestOutputKind
    from vllm.v1.engine.async_llm import AsyncLLM
    from zmq.utils.monitor import recv_monitor_message

    import kvcached

    assert os.environ.get("CUDA_VISIBLE_DEVICES"), "Set CUDA_VISIBLE_DEVICES explicitly"
    assert torch.cuda.device_count() == 1
    assert os.environ.get("KVCACHED_MAX_CACHED_TOKENS") == "0"
    args.out.mkdir(parents=True, exist_ok=False)
    root = Path(kvcached.__file__).parent
    config = dict(
        model=args.model, dtype="float16", enforce_eager=True,
        block_size=16, max_model_len=512, max_num_seqs=4,
        max_num_batched_tokens=1024, gpu_memory_utilization=0.20,
        enable_prefix_caching=True, tensor_parallel_size=1,
        pipeline_parallel_size=1, seed=42,
        kv_events_config=KVEventsConfig(
            enable_kv_cache_events=True, publisher="zmq",
            endpoint=f"tcp://*:{args.port}",
        ),
    )
    settings = {
        "config": {key: str(value) for key, value in config.items()},
        "vllm": vllm.__version__, "torch": torch.__version__,
        "CUDA_VISIBLE_DEVICES": os.environ["CUDA_VISIBLE_DEVICES"],
        "ENABLE_KVCACHED": os.getenv("ENABLE_KVCACHED"),
        "KVCACHED_AUTOPATCH": os.getenv("KVCACHED_AUTOPATCH"),
        "KVCACHED_IPC_NAME": os.getenv("KVCACHED_IPC_NAME"),
        "KVCACHED_MAX_CACHED_TOKENS": os.environ["KVCACHED_MAX_CACHED_TOKENS"],
        "source": str(root),
        "probe_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "source_sha256": {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                          for name in ("integration/vllm/patches.py", "integration/vllm/native_block_pool.py")},
    }
    (args.out / "settings.json").write_text(json.dumps(settings, indent=2) + "\n")

    lock = threading.Lock()
    tracker = Residency()
    ready, connected, stop = threading.Event(), threading.Event(), threading.Event()
    errors: list[str] = []
    counts: Counter[str] = Counter()
    sequences: list[int] = []

    def subscribe():
        context = zmq.Context()
        socket = context.socket(zmq.SUB)
        socket.setsockopt(zmq.SUBSCRIBE, b"")
        monitor = socket.get_monitor_socket(events=zmq.EVENT_CONNECTED)
        socket.connect(f"tcp://127.0.0.1:{args.port}")
        ready.set()
        try:
            with (args.out / "events.jsonl").open("wb") as output:
                while not stop.is_set():
                    if monitor.poll(0):
                        recv_monitor_message(monitor)
                        connected.set()
                    if not socket.poll(25):
                        continue
                    _, sequence, payload = socket.recv_multipart()
                    number = int.from_bytes(sequence, "big")
                    batch = msgspec.msgpack.decode(payload, type=KVEventBatch)
                    output.write(msgspec.json.encode(batch) + b"\n")
                    output.flush()
                    with lock:
                        if sequences:
                            assert number == sequences[-1] + 1, "ZMQ sequence gap"
                        sequences.append(number)
                        for event in batch.events:
                            kind = type(event).__name__
                            counts[kind] += 1
                            tracker.apply(kind, event)
        except Exception as error:
            errors.append(repr(error))
        finally:
            monitor.close(linger=0)
            socket.close(linger=0)
            context.term()

    reader = threading.Thread(target=subscribe, daemon=True)
    reader.start()
    assert ready.wait(5)
    engine = None
    tasks: list[asyncio.Task[Any]] = []
    result: dict[str, Any] = {"completed": 0}
    try:
        engine = AsyncLLM.from_engine_args(AsyncEngineArgs(**config))
        assert await asyncio.to_thread(connected.wait, 15), "Subscriber did not connect"
        await asyncio.sleep(5)
        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=False)
        base = tokenizer.encode("Shared engineering notes about cache residency and request ownership. ",
                                add_special_tokens=False)
        prompt = (base * (160 // len(base) + 1))[:160]
        (args.out / "prompt.json").write_text(json.dumps(prompt) + "\n")

        async def consume(request_id, tokens, length):
            assert engine is not None
            final = None
            sampling = SamplingParams(temperature=0, seed=42, ignore_eos=True,
                                      min_tokens=length, max_tokens=length,
                                      output_kind=RequestOutputKind.FINAL_ONLY)
            async for value in engine.generate({"prompt_token_ids": tokens}, sampling, request_id):
                final = value
            assert final is not None and final.finished
            generated = list(final.outputs[0].token_ids)
            assert len(generated) == length
            return {"request_id": request_id, "token_ids": generated,
                    "num_cached_tokens": final.num_cached_tokens}

        await engine.pause_generation(mode="keep", clear_cache=False)
        short = asyncio.create_task(consume("short", prompt, 32))
        long = asyncio.create_task(consume("long", prompt, 256))
        tasks += [short, long]
        await asyncio.sleep(0.1)  # Let public generate calls enqueue while paused.
        await engine.resume_generation()
        short_output = await asyncio.wait_for(short, 120)
        await asyncio.sleep(0.250)
        assert not long.done(), "Long request already finished; timing scenario was not reached"
        result["long_active_before_probe"] = True
        probe_tokens = prompt + short_output["token_ids"]
        result["probe_input_tokens"] = len(probe_tokens)
        result["wait_after_short_seconds"] = 0.250
        with lock:
            assert not errors, errors
            estimate = tracker.estimate(probe_tokens)
            result["sequence_before_probe"] = sequences[-1] if sequences else None
        probe = asyncio.create_task(consume("probe", probe_tokens, 1))
        tasks.append(probe)
        probe_output, long_output = await asyncio.wait_for(asyncio.gather(probe, long), 120)
        outputs = [short_output, long_output, probe_output]
        (args.out / "outputs.json").write_text(json.dumps(outputs, indent=2) + "\n")
        result.update(completed=3, subscriber_cached_tokens=estimate,
                      actual_cached_tokens=probe_output["num_cached_tokens"],
                      consistent=estimate == probe_output["num_cached_tokens"])
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        if engine is not None:
            engine.shutdown()
        stop.set()
        reader.join(timeout=5)
        result.update(listener_errors=errors, event_counts=dict(counts), sequences=sequences)
        (args.out / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    assert not reader.is_alive() and not errors, errors
    print(json.dumps(result, indent=2))
    if args.expect_consistent:
        assert result["consistent"], "Subscriber underestimates the engine's cache hit"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--port", type=int, default=19341)
    parser.add_argument("--expect-consistent", action="store_true")
    parser.add_argument("--check-observer", action="store_true")
    args = parser.parse_args()
    if args.check_observer:
        check_observer()
        return
    if args.model is None or args.out is None:
        parser.error("--model and --out are required for a GPU run")
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
