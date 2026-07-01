#!/usr/bin/env python
"""Byte-exact correctness check for kvcached on a heterogeneous-KV-group model.

Greedy-decode a long-context chat request (prompt > sliding window, so the
sliding-window layers are exercised) and save the output token ids. Run once
with kvcached OFF and once ON, then diff — identical token ids prove kvcached's
per-group KV views are correct.

  # baseline
  python correctness_check.py --tag baseline
  # kvcached (heterogeneous -> non-contiguous)
  ENABLE_KVCACHED=true KVCACHED_AUTOPATCH=1 KVCACHED_CONTIGUOUS_LAYOUT=false \
    python correctness_check.py --tag kvcached
  # compare
  python correctness_check.py --compare baseline kvcached

Gemma is gated + multimodal: authenticate first, and this runs it text-only
(limit_mm_per_prompt=0). Requires vLLM >=0.24 for Gemma 4.
"""
import argparse
import json
import sys

MODEL_DEFAULT = "google/gemma-4-12B-it"
# ~1500-token document (> the 1024 sliding window) + a question over it.
_DOC = ("In the kingdom of Eldoria seven cities each guarded a colored gem: "
        "Aurora red, Belfor orange, Caldera yellow, Dawnhaven green, Evermoor "
        "blue, Frosthold indigo, Grimstone violet. They traded along the river. ") * 60
MSGS = [
    [{"role": "user", "content": _DOC + " List the seven cities with their gem colors, then explain the trade."}],
    [{"role": "user", "content": _DOC + " Which city held the blue gem and which the violet? Describe the route between them."}],
]


def run(tag: str, model: str):
    from vllm import LLM, SamplingParams
    llm = LLM(model=model, enforce_eager=True, max_model_len=8192,
              gpu_memory_utilization=0.5, enable_prefix_caching=False,
              limit_mm_per_prompt={"image": 0, "video": 0, "audio": 0}, max_num_seqs=8)
    outs = llm.chat(MSGS, SamplingParams(max_tokens=128, temperature=0.0))
    toks = [list(o.outputs[0].token_ids) for o in outs]
    json.dump(toks, open(f"toks_{tag}.json", "w"))
    print(f"saved toks_{tag}.json | uniq[0]={len(set(toks[0]))} | "
          f"sample={outs[0].outputs[0].text[:80]!r}")


def compare(a: str, b: str) -> int:
    ta = json.load(open(f"toks_{a}.json"))
    tb = json.load(open(f"toks_{b}.json"))
    ok = ta == tb
    for i, (x, y) in enumerate(zip(ta, tb)):
        print(f"  prompt {i}: {'IDENTICAL' if x == y else 'DIFF'} "
              f"({len(x)} vs {len(y)} toks)")
    print("RESULT:", "PASS - byte-exact identical" if ok else "FAIL - token mismatch")
    return 0 if ok else 1


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--tag")
    p.add_argument("--model", default=MODEL_DEFAULT)
    p.add_argument("--compare", nargs=2, metavar=("A", "B"))
    args = p.parse_args()
    if args.compare:
        sys.exit(compare(*args.compare))
    run(args.tag, args.model)
