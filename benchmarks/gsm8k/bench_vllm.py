# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

import argparse
import ast
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np  # type: ignore
import requests
from tqdm import tqdm  # type: ignore

INVALID = -9999999


def get_one_example(lines, i, include_answer):
    ret = "Question: " + lines[i]["question"] + "\nAnswer:"
    if include_answer:
        ret += " " + lines[i]["answer"]
    return ret


def get_few_shot_examples(lines, k):
    ret = ""
    for i in range(k):
        ret += get_one_example(lines, i, True) + "\n\n"
    return ret


def get_answer_value(answer_str):
    answer_str = answer_str.replace(",", "")
    numbers = re.findall(r"\d+", answer_str)
    if len(numbers) < 1:
        return INVALID
    try:
        return ast.literal_eval(numbers[-1])
    except SyntaxError:
        return INVALID


def read_gsm8k_lines(data_path: str) -> List[Dict[str, Any]]:
    """Read GSM8K test split, downloading if needed (no hard sglang dep)."""
    url = "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl"
    if not os.path.isfile(data_path):
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        tmp_path = os.path.join(os.getcwd(), "_tmp_gsm8k_test.jsonl")
        with open(tmp_path, "w", encoding="utf-8") as f:
            f.write(r.text)
        with open(tmp_path, "r", encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]
    else:
        with open(data_path, "r", encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]


def _vllm_worker(base_url: str, model: str, prompt: str, max_tokens: int, stop: Iterable[str]) -> Tuple[str, int]:
    url = f"{base_url}/v1/completions"
    body = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "top_p": 1.0,
        "n": 1,
        "stop": list(stop),
    }
    r = requests.post(url, json=body, timeout=120)
    if r.status_code != 200:
        raise RuntimeError(f"Completion failed: {r.status_code} {r.text}")
    data = r.json()
    text = data["choices"][0]["text"]
    completion_tokens = 0
    usage = data.get("usage")
    if isinstance(usage, dict):
        completion_tokens = int(usage.get("completion_tokens", 0))
    return text, completion_tokens


def main(args):
    # Resolve connection
    base_url = args.base_url or f"http://{args.host}:{args.port}"
    if not args.model:
        raise ValueError("--model is required for vLLM benchmarking")

    # Read data
    lines = read_gsm8k_lines(args.data_path)

    # Construct prompts
    num_questions = args.num_questions
    num_shots = args.num_shots
    few_shot_examples = get_few_shot_examples(lines, num_shots)

    questions: List[str] = []
    labels: List[int] = []
    for i in range(len(lines[:num_questions])):
        questions.append(get_one_example(lines, i, False))
        labels.append(get_answer_value(lines[i]["answer"]))
    assert all(label_value != INVALID for label_value in labels)
    prompts = [few_shot_examples + q for q in questions]

    # Run requests against vLLM server
    preds: List[int] = [INVALID] * len(prompts)
    texts: List[str] = [""] * len(prompts)
    comp_tokens: List[int] = [0] * len(prompts)
    num_output_tokens = 0
    stop = ["Question", "Assistant:", "<|separator|>"]

    tic = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.parallel or 1) as pool:
        futures = {
            pool.submit(_vllm_worker, base_url, args.model, p, args.max_tokens, stop): idx
            for idx, p in enumerate(prompts)
        }
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Benchmarking", unit="req"):
            idx = futures[fut]
            text, ctoks = fut.result()
            texts[idx] = text
            comp_tokens[idx] = ctoks
            preds[idx] = get_answer_value(text)
            num_output_tokens += ctoks
    latency = time.perf_counter() - tic

    # Metrics
    acc = float(np.mean(np.array(preds) == np.array(labels))) if len(labels) > 0 else 0.0
    invalid = float(np.mean(np.array(preds) == INVALID)) if len(preds) > 0 else 0.0
    output_throughput = (num_output_tokens / latency) if latency > 0 else 0.0

    # Print results
    print(f"Accuracy: {acc:.3f}")
    print(f"Invalid: {invalid:.3f}")
    print(f"Latency: {latency:.3f} s")
    print(f"Output throughput: {output_throughput:.3f} token/s")

    # Optional raw dump (jsonl with text)
    if args.raw_result_file:
        try:
            with open(args.raw_result_file, "w", encoding="utf-8") as fout:
                for i in range(len(prompts)):
                    rec = {
                        "idx": i,
                        "question": questions[i],
                        "label": labels[i],
                        "prediction": preds[i],
                        "text": texts[i],
                        "meta": {"completion_tokens": comp_tokens[i]},
                    }
                    fout.write(json.dumps(rec) + "\n")
        except Exception:
            pass

    # Summary line
    value = {
        "task": "gsm8k",
        "backend": "vllm",
        "num_gpus": 1,
        "latency": round(latency, 3),
        "accuracy": round(acc, 3),
        "num_requests": args.num_questions,
        "other": {
            "num_questions": args.num_questions,
            "parallel": args.parallel,
        },
    }
    if args.result_file:
        with open(args.result_file, "a", encoding="utf-8") as fout:
            fout.write(json.dumps(value) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Common
    parser.add_argument("--num-shots", type=int, default=5)
    parser.add_argument("--data-path", type=str, default="test.jsonl")
    parser.add_argument("--num-questions", type=int, default=100)
    parser.add_argument("--parallel", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--result-file", type=str, default="eval_results.jsonl")
    parser.add_argument("--raw-result-file", type=str, default="eval_raw.jsonl")

    # vLLM connection
    parser.add_argument("--model", type=str, required=True, help="HuggingFace model id served by vLLM")
    parser.add_argument("--base-url", type=str, default=None, help="Base URL of OpenAI-compatible server (e.g., http://127.0.0.1:12346)")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=12346)

    main(parser.parse_args())