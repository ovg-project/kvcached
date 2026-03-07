#!/bin/bash
# Trace-based workload sweep for 6x Qwen2.5-7B-Instruct.
#
# For each R_peak, generates a realistic mixed-workload trace via
# plot_workload_design.py, then replays it across all 6 instances
# simultaneously — all sharing the exact same START_AT timestamp.
# No inter-instance stagger in start time; the trace itself encodes
# each instance's arrival schedule.
#
# Usage:
#   bash run_trace_sweep_7b.sh [tag] [results_dir] [options]
#
# R_peak options (pick one):
#   --r-peaks "15 20 25 30"           explicit list
#   --r-peak-start N --r-peak-end N --r-peak-step N   range (default)
#
# Examples:
#   bash run_trace_sweep_7b.sh kvcached-7b results/trace-kvcached
#   bash run_trace_sweep_7b.sh baseline-7b results/trace-baseline --r-peaks "20 25 30"
#   bash run_trace_sweep_7b.sh kvcached-7b results/trace-kvcached --r-peak-start 15 --r-peak-end 35 --r-peak-step 5

set -e

PYTHON="${PYTHON:-/home/amd-rice-collab/home/kvcached/py12-vllm-0.14.0-pip/bin/python}"
export PYTHONPATH="../../:../../benchmarks:$PYTHONPATH"

# ── Positional args ────────────────────────────────────────────────────────────
RUN_TAG="${1:-kvcached-7b}"
RESULTS_DIR="${2:-results/trace-sweep}"
shift 2 2>/dev/null || true

# ── Workload / request params ──────────────────────────────────────────────────
PROMPT_LEN=256
COMPLETION_LEN=512
BACKEND="vllm"

# ── Trace generation params ────────────────────────────────────────────────────
TRACE_T=300          # experiment duration (s)
SEED=42
N_INSTANCES=6        # must match number of entries in MODELS

# ── R_peak range defaults ──────────────────────────────────────────────────────
R_PEAK_START=15
R_PEAK_END=30
R_PEAK_STEP=5
R_PEAKS_LIST=""

# ── Parse optional flags ───────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --r-peaks)       R_PEAKS_LIST="$2";    shift 2 ;;
        --start)  R_PEAK_START="$2";    shift 2 ;;
        --end)    R_PEAK_END="$2";      shift 2 ;;
        --step)   R_PEAK_STEP="$2";     shift 2 ;;
        --trace-t)       TRACE_T="$2";         shift 2 ;;
        --seed)          SEED="$2";            shift 2 ;;
        *) echo "Unknown arg: $1"; shift ;;
    esac
done

# Build R_peaks array
if [ -n "$R_PEAKS_LIST" ]; then
    read -ra R_PEAKS <<< "$R_PEAKS_LIST"
else
    R_PEAKS=()
    for rp in $(seq "$R_PEAK_START" "$R_PEAK_STEP" "$R_PEAK_END"); do
        R_PEAKS+=("$rp")
    done
fi

# ── Instance → port mapping ────────────────────────────────────────────────────
MODELS=(
    "Qwen/Qwen2.5-7B-Instruct:12346"
    "Qwen/Qwen2.5-7B-Instruct:30000"
    "Qwen/Qwen2.5-7B-Instruct:40000"
    "Qwen/Qwen2.5-7B-Instruct:50000"
    "Qwen/Qwen2.5-7B-Instruct:60000"
    "Qwen/Qwen2.5-7B-Instruct:65000"
)
NUM_MODELS=${#MODELS[@]}

# ── Timing ─────────────────────────────────────────────────────────────────────
# All instances share the SAME START_AT (trace already encodes per-instance timing).
# We still stagger process launches to avoid CPU contention during init/warmup.
# START_AT = sweep_start + last_launch_time + warmup_est + buffer
LAUNCH_STAGGER=20    # seconds between consecutive process launches
WARMUP_EST=15        # estimated warmup request duration (seconds)
INIT_BUFFER_EXTRA=20 # slack after warmup completes on last-launched instance
LAST_LAUNCH_OFFSET=$(( (NUM_MODELS - 1) * LAUNCH_STAGGER ))
INIT_BUFFER=$(( LAST_LAUNCH_OFFSET + WARMUP_EST + INIT_BUFFER_EXTRA ))
# inst6 finishes warmup at ~LAST_LAUNCH_OFFSET+WARMUP_EST = 100+15=115s
# START_AT at 135s → 20s slack for all instances ✓

TRACE_DIR="${RESULTS_DIR}/traces"
mkdir -p "$RESULTS_DIR" "$TRACE_DIR"

echo "=== Trace sweep: R_peaks=(${R_PEAKS[*]}), tag=${RUN_TAG} ==="
echo "    Results: ${RESULTS_DIR}"
echo "    Instances: ${NUM_MODELS}, prompt=${PROMPT_LEN}, completion=${COMPLETION_LEN}"
echo "    T=${TRACE_T}s, seed=${SEED}"
echo "    Launch stagger: ${LAUNCH_STAGGER}s | Init buffer: ${INIT_BUFFER}s (all instances same start_at)"

for R_PEAK in "${R_PEAKS[@]}"; do
    echo ""
    echo "══════════════════════════════════════════"
    echo "  R_peak = ${R_PEAK} req/s"
    echo "══════════════════════════════════════════"

    # ── 1. Generate workload trace ─────────────────────────────────────────────
    TRACE_FILE="${TRACE_DIR}/trace_Rpeak${R_PEAK}_T${TRACE_T}_seed${SEED}.npz"
    if [ -f "$TRACE_FILE" ]; then
        echo "  Trace already exists, reusing: $TRACE_FILE"
    else
        echo "  Generating trace -> $TRACE_FILE"
        "$PYTHON" workload_design.py \
            --R-peak  "$R_PEAK" \
            --T       "$TRACE_T" \
            --N       "$N_INSTANCES" \
            --seed    "$SEED" \
            --out-dir "$TRACE_DIR" \
            --cap-enabled
        mv "${TRACE_DIR}/workload_trace.npz" "$TRACE_FILE"
    fi

    # Print per-instance expected arrival counts from trace
    "$PYTHON" - <<EOF
import numpy as np
d = np.load("$TRACE_FILE")
arr = d["arrivals"]
dt  = float(d["dt"])
print(f"  Trace: dt={dt}s, duration={arr.shape[1]*dt:.0f}s")
for i in range(arr.shape[0]):
    print(f"    inst{i+1}: {int(arr[i].sum())} arrivals, mean rate={arr[i].mean()/dt:.2f} req/s")
print(f"    TOTAL mean rate: {arr.sum(axis=0).mean()/dt:.2f} req/s  (target: $R_PEAK)")
EOF

    # ── 2. Compute shared START_AT ─────────────────────────────────────────────
    SWEEP_START_MS=$(date +%s%3N)
    SWEEP_START_S=$(echo "scale=3; $SWEEP_START_MS / 1000" | bc)
    START_AT=$(echo "scale=3; $SWEEP_START_S + $INIT_BUFFER" | bc)
    echo "  Shared start_at: t=+${INIT_BUFFER}s from now (${START_AT})"

    # ── 3. Launch all instances (staggered launches, same start_at) ────────────
    PIDS=()
    RESULT_FILES=()

    for i in "${!MODELS[@]}"; do
        MODEL=$(echo "${MODELS[$i]}" | cut -d':' -f1)
        PORT=$(echo "${MODELS[$i]}"  | cut -d':' -f2)
        INST=$((i + 1))
        RESULT_FILE="${RESULTS_DIR}/${RUN_TAG}-${BACKEND}-Rpeak${R_PEAK}-prompt${PROMPT_LEN}-completion${COMPLETION_LEN}-inst${INST}.json"

        # Stagger process launches to reduce CPU contention during tokenizer load
        if [ "$i" -gt 0 ]; then
            sleep "$LAUNCH_STAGGER"
        fi

        "$PYTHON" bench_kvcached_vllm.py \
            --backend         "$BACKEND" \
            --model           "$MODEL" \
            --dataset-name    random \
            --random-input-len  "$PROMPT_LEN" \
            --random-output-len "$COMPLETION_LEN" \
            --num-prompts     1 \
            --host            localhost \
            --port            "$PORT" \
            --endpoint        /v1/completions \
            --save-result \
            --result-filename "$RESULT_FILE" \
            --metadata        "R_peak=${R_PEAK},tag=${RUN_TAG},num_models=${NUM_MODELS}" \
            --trace           "$TRACE_FILE" \
            --inst            "$INST" \
            --start-at        "$START_AT" &

        PIDS+=($!)
        RESULT_FILES+=("$RESULT_FILE")
        SECS_UNTIL=$(echo "scale=1; $START_AT - $(date +%s%3N)/1000" | bc)
        echo "  Launched inst${INST} (port ${PORT}) PID=${PIDS[$i]} | start_at in ~${SECS_UNTIL}s"
    done

    # ── 4. Wait for completion ─────────────────────────────────────────────────
    echo "  Waiting for all instances to finish..."
    FAILED=0
    for i in "${!PIDS[@]}"; do
        wait "${PIDS[$i]}" || { echo "  WARNING: instance $((i+1)) failed (PID=${PIDS[$i]})"; FAILED=1; }
    done

    # ── 5. Print summary ───────────────────────────────────────────────────────
    if [ "$FAILED" -eq 0 ]; then
        echo "  R_peak=${R_PEAK} complete:"
        for f in "${RESULT_FILES[@]}"; do
            "$PYTHON" -c "
import json
try:
    d = json.load(open('$f'))
    name = '$f'.split('/')[-1]
    print(f'    {name}: mean_ttft={d.get(\"mean_ttft_ms\", float(\"nan\")):.1f}ms  p99_ttft={d.get(\"p99_ttft_ms\", float(\"nan\")):.1f}ms  throughput={d.get(\"request_throughput\", float(\"nan\")):.2f} req/s')
except Exception as e:
    print(f'    (could not parse $f: {e})')
" 2>/dev/null || true
        done
    else
        echo "  R_peak=${R_PEAK} had failures."
    fi
done

echo ""
echo "=== Sweep complete. All results in ${RESULTS_DIR}/ ==="
