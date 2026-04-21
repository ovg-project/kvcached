#!/usr/bin/env python3
"""Compare KVCACHED_LAYOUT_DUMP / KVCACHED_BLOCK_SHA lines across the
prefill and decode logs of the equivalence harness (experiments/09_equivalence.sh).

Exits 0 on match, 1 on any mismatch.
"""

import json
import re
import sys
from collections import defaultdict

LAYOUT_RE = re.compile(r"KVCACHED_LAYOUT_DUMP=(\{.*\})\s*$")
SHA_RE = re.compile(r"KVCACHED_BLOCK_SHA=(\{.*\})\s*$")

# Per-layer fields that can legitimately differ across GPUs (pointer values).
IGNORED_LAYER_FIELDS = {"data_ptr", "device"}


def parse_log(path):
    layouts = defaultdict(list)  # tag -> [payload...]
    shas = defaultdict(list)     # (tag, layer, block) -> [sha...]
    with open(path) as f:
        for line in f:
            m = LAYOUT_RE.search(line)
            if m:
                try:
                    p = json.loads(m.group(1))
                    layouts[p["tag"]].append(p)
                except Exception:
                    pass
                continue
            m = SHA_RE.search(line)
            if m:
                try:
                    p = json.loads(m.group(1))
                    shas[(p["tag"], p["layer"], p["block"])].append(p["sha256"])
                except Exception:
                    pass
    return layouts, shas


def _strip_ignored(layer_entry):
    return {k: v for k, v in layer_entry.items() if k not in IGNORED_LAYER_FIELDS}


def compare_layouts(a_payload, b_payload, tag, side_a, side_b):
    issues = []
    if a_payload["num_layers"] != b_payload["num_layers"]:
        issues.append(
            f"[{tag}] num_layers differs: {side_a}={a_payload['num_layers']} vs {side_b}={b_payload['num_layers']}"
        )
    if a_payload.get("blocks_dim_idx") != b_payload.get("blocks_dim_idx"):
        issues.append(
            f"[{tag}] blocks_dim_idx differs: {side_a}={a_payload.get('blocks_dim_idx')} vs {side_b}={b_payload.get('blocks_dim_idx')}"
        )
    for al, bl in zip(a_payload["layers"], b_payload["layers"]):
        if _strip_ignored(al) != _strip_ignored(bl):
            issues.append(
                f"[{tag}] layer {al.get('layer')} metadata differs: {side_a}={_strip_ignored(al)} vs {side_b}={_strip_ignored(bl)}"
            )
    a_extra = a_payload.get("extra", {})
    b_extra = b_payload.get("extra", {})
    for k in set(a_extra) | set(b_extra):
        if a_extra.get(k) != b_extra.get(k):
            issues.append(
                f"[{tag}] extra.{k} differs: {side_a}={a_extra.get(k)} vs {side_b}={b_extra.get(k)}"
            )
    return issues


def main():
    if len(sys.argv) != 3:
        print("usage: _compare_layout_dumps.py <prefill_log> <decode_log>", file=sys.stderr)
        return 2
    prefill_log, decode_log = sys.argv[1], sys.argv[2]

    p_layouts, p_shas = parse_log(prefill_log)
    d_layouts, d_shas = parse_log(decode_log)

    if not p_layouts:
        print(f"FAIL: no KVCACHED_LAYOUT_DUMP lines in {prefill_log}")
        print("      (is KVCACHED_DUMP_LAYOUT=1 set on the vllm serve invocation?)")
        return 1
    if not d_layouts:
        print(f"FAIL: no KVCACHED_LAYOUT_DUMP lines in {decode_log}")
        return 1

    all_issues = []
    for tag in sorted(set(p_layouts) | set(d_layouts)):
        p_entries = p_layouts.get(tag, [])
        d_entries = d_layouts.get(tag, [])
        if not p_entries:
            all_issues.append(f"[{tag}] missing from prefill log")
            continue
        if not d_entries:
            all_issues.append(f"[{tag}] missing from decode log")
            continue
        # Compare the first occurrence on each side.
        all_issues.extend(compare_layouts(p_entries[0], d_entries[0], tag, "prefill", "decode"))

    # SHA comparison: for tags present in both logs at the same (tag, layer, block),
    # every prefill SHA must have a matching decode SHA.
    common_keys = set(p_shas) & set(d_shas)
    if not common_keys:
        print("INFO: no (tag, layer, block) SHA entries overlap between prefill and decode")
    for key in sorted(common_keys):
        tag, layer, block = key
        # Compare the set of distinct observed SHAs. One side may poll a hook
        # more times than the other, which is fine as long as the set of
        # byte-states seen on each side agrees.
        pa, da = set(p_shas[key]), set(d_shas[key])
        if pa != da:
            all_issues.append(
                f"[{tag}] layer {layer} block {block} SHA set differs: "
                f"prefill_only={sorted(pa - da)} decode_only={sorted(da - pa)}"
            )

    if all_issues:
        print("FAIL: layout/SHA mismatches:")
        for issue in all_issues:
            print(f"  - {issue}")
        return 1

    print(f"PASS: {len(p_layouts)} tag(s) compared, {len(common_keys)} SHA key(s) matched")
    return 0


if __name__ == "__main__":
    sys.exit(main())
