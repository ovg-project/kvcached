#!/bin/bash
# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0
#
# Run one full benchmark cycle for a given mode: start servers, run the
# workload, stop servers. Results land in logs/<mode>/results.jsonl.
#
# Usage: ./run_one.sh <static|offload|kvcached>
set -e
cd "$(dirname "$0")"

MODE=${1:?usage: $0 <static|offload|kvcached>}
LOGDIR=logs/$MODE
ENV_BIN=/home/exouser/miniconda3/envs/kvc-bench/bin

cleanup() {
    echo "stopping servers..."
    for f in "$LOGDIR"/server-*.pid; do
        [ -f "$f" ] && kill "$(cat "$f")" 2>/dev/null || true
    done
    sleep 5
    pkill -f "vllm serve" 2>/dev/null || true
    sleep 5
}
trap cleanup EXIT

rm -rf "$LOGDIR"
./start_servers.sh "$MODE" "$LOGDIR"

echo "warmup: one short request to each instance"
for port in 8100 8200; do
    curl -sf "http://localhost:$port/v1/completions" \
        -H 'Content-Type: application/json' \
        -d '{"model":"Qwen/Qwen3-4B","prompt":[9906,1917,11,420,374,264,1296],"max_tokens":8}' >/dev/null
done
sleep 5

"$ENV_BIN/python" workload.py --output "$LOGDIR/results.jsonl" "${@:2}"

# Snapshot server metrics (e.g. KV offload transfer counters) before shutdown
for inst in A:8100 B:8200; do
    name=${inst%%:*}; port=${inst##*:}
    curl -s "http://localhost:$port/metrics" > "$LOGDIR/metrics-$name.prom" || true
done

echo "done; results in $LOGDIR/results.jsonl"
