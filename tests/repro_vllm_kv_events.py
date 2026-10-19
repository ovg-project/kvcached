# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0
"""Normal vLLM LLM/generate/reset with an actual ZMQ subscriber; no pool calls."""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import threading
import time
from collections import Counter
from pathlib import Path
from typing import Any


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--events", type=int, required=True)
    parser.add_argument("--port", type=int, default=19161)
    parser.add_argument("--out", required=True)
    parser.add_argument("--model", required=True, help="Model repository ID or local directory")
    parser.add_argument("--expect-events", action="store_true", help="Fail if normal requests/reset publish no events")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)

    import msgspec
    import torch
    import vllm
    import zmq
    from vllm import LLM, SamplingParams
    from vllm.config.kv_events import KVEventsConfig
    from vllm.distributed.kv_events import KVEventBatch
    from zmq.utils.monitor import recv_monitor_message

    import kvcached

    done, listening, connected = threading.Event(), threading.Event(), threading.Event()
    batches: list[tuple[int, Any]] = []
    listener_errors: list[str] = []

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
            listener_errors.append(repr(error))
        finally:
            monitor.close(linger=0)
            socket.close(linger=0)

    reader = threading.Thread(target=collect, name="native-event-consumer", daemon=True)
    reader.start()
    assert listening.wait(5)
    model = args.model
    config = dict(model=model, tokenizer=model, trust_remote_code=False,
                  dtype="float16", enforce_eager=True, max_model_len=512,
                  max_num_seqs=1, max_num_batched_tokens=512,
                  gpu_memory_utilization=0.20, block_size=16, seed=42,
                  enable_prefix_caching=True, disable_log_stats=False,
                  kv_events_config=KVEventsConfig(
                      enable_kv_cache_events=bool(args.events), publisher="zmq",
                      endpoint=f"tcp://*:{args.port}",
                      replay_endpoint=f"tcp://*:{args.port+1}"))
    settings: dict[str, Any] = {k: str(v) for k, v in config.items()}
    settings.update(name=args.name, vllm=vllm.__version__, torch=torch.__version__,
                    kvcached_source=kvcached.__file__,
                    cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(0),
                    enable_kvcached=os.getenv("ENABLE_KVCACHED"),
                    pythonpath=os.getenv("PYTHONPATH"),
                    cache_cap=os.getenv("KVCACHED_MAX_CACHED_TOKENS"))
    settings.update(python=sys.version, CUDA_VISIBLE_DEVICES=os.getenv("CUDA_VISIBLE_DEVICES"),
                    KVCACHED_IPC_NAME=os.getenv("KVCACHED_IPC_NAME"))
    (out / "settings.json").write_text(json.dumps(settings, indent=2) + "\n")
    shared = ("These are fictional engineering notes. A cache stores immutable prefixes. "
              "A running request owns its writable state. Freed capacity can be reused. ") * 6
    prompts = [shared + text for text in (
        "\nWrite the integers from one to eight:",
        "\nExplain a cache hit in one short sentence:",
        "\nComplete this list: red, green, blue,",
        "\nAnswer with one sentence about memory:",
    )]
    sampling = SamplingParams(temperature=0, top_p=1.0, max_tokens=16,
                              min_tokens=16, ignore_eos=True, seed=42)
    outputs = []
    llm = None
    try:
        llm = LLM(**config)
        if args.events:
            assert connected.wait(15), "Subscriber did not connect"
            time.sleep(5)
        for round_index in range(2):
            for prompt_index, prompt in enumerate(prompts):
                result, = llm.generate([prompt], sampling, use_tqdm=False)
                generated = result.outputs[0]
                assert len(generated.token_ids) == 16
                outputs.append(dict(round=round_index, prompt=prompt_index,
                    token_ids=list(generated.token_ids), text=generated.text,
                    num_cached_tokens=getattr(result, "num_cached_tokens", None)))
        reset = llm.reset_prefix_cache()
        result, = llm.generate([prompts[0]], sampling, use_tqdm=False)
        generated = result.outputs[0]
        assert len(generated.token_ids) == 16
        outputs.append(dict(round=2, prompt=0, token_ids=list(generated.token_ids),
                            text=generated.text,
                            num_cached_tokens=getattr(result, "num_cached_tokens", None)))
        (out / "outputs.json").write_text(json.dumps(outputs, indent=2) + "\n")
        (out / "reset.json").write_text(json.dumps({"result":reset}) + "\n")
    finally:
        if llm is not None:
            del llm
            gc.collect()
        time.sleep(0.3)
        done.set()
        reader.join(timeout=3)
    assert not reader.is_alive() and not listener_errors, listener_errors
    sequences = [number for number, _ in batches]
    assert not sequences or sequences == list(range(sequences[0], sequences[-1] + 1))
    counts = Counter(type(event).__name__ for _, batch in batches for event in batch.events)
    result = dict(name=args.name, completed=len(outputs), event_batches=len(batches),
                  event_counts=dict(counts), sequences=sequences,
                  data_parallel_ranks=sorted({batch.data_parallel_rank for _,batch in batches}),
                  listener_errors=listener_errors)
    (out / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print("MODEL_EVENT_RESULT",json.dumps(result),flush=True)
    if args.expect_events:
        assert counts["BlockStored"] > 0, "normal inference completed but no BlockStored events arrived"
        assert counts["AllBlocksCleared"] > 0, "public reset completed but no AllBlocksCleared event arrived"


if __name__ == "__main__":
    main()
