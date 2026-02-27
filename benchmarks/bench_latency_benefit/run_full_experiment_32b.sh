#!/bin/bash
# Full 32B kvcached vs baseline TTFT sweep experiment.
# Launches each stack, waits for health, runs sweep, then tears down.
#
# Usage: bash run_full_experiment_32b.sh [start_rps] [end_rps] [step]
# Example: bash run_full_experiment_32b.sh 1 20 2
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

START_RPS="${1:-1}"
END_RPS="${2:-20}"
STEP="${3:-1}"

KVCACHED_CONFIG="$SCRIPT_DIR/bench-config-amd-32b.yaml"
BASELINE_CONFIG="$SCRIPT_DIR/bench-config-amd-32b-baseline.yaml"
LAUNCH_PY="$REPO_ROOT/controller/launch.py"

ROUTER_PORT=8080
INSTANCE_PORTS=(12346 30000)
HEALTH_TIMEOUT=300   # seconds to wait for all instances to be healthy
HEALTH_INTERVAL=10   # seconds between health checks

VENV="$REPO_ROOT/py12-vllm-0.14.0-pip"
# shellcheck disable=SC1091
source "$VENV/bin/activate"

export PYTHONPATH="$REPO_ROOT:$REPO_ROOT/benchmarks:$PYTHONPATH"

# ---------------------------------------------------------------------------
wait_for_healthy() {
    local tag="$1"
    local deadline=$(( $(date +%s) + HEALTH_TIMEOUT ))
    echo "[$tag] Waiting for all instances to be healthy (timeout ${HEALTH_TIMEOUT}s)..."

    while [ "$(date +%s)" -lt "$deadline" ]; do
        local all_ok=1
        for port in "${INSTANCE_PORTS[@]}"; do
            local status
            status=$(curl -s --max-time 5 "http://localhost:${port}/health" 2>/dev/null || echo "")
            if [ -z "$status" ]; then
                all_ok=0
                break
            fi
        done

        if [ "$all_ok" -eq 1 ]; then
            echo "[$tag] All instances healthy."
            return 0
        fi

        echo "[$tag] Not all healthy yet, retrying in ${HEALTH_INTERVAL}s..."
        sleep "$HEALTH_INTERVAL"
    done

    echo "[$tag] ERROR: instances did not become healthy within ${HEALTH_TIMEOUT}s. Aborting."
    python "$LAUNCH_PY" --config "$KVCACHED_CONFIG" --kill-all 2>/dev/null || true
    exit 1
}

kill_stack() {
    local config="$1"
    echo "Killing stack..."
    python "$LAUNCH_PY" --config "$config" --kill-all 2>/dev/null || true
    sleep 10  # give processes time to fully exit and free GPU memory
}

# ---------------------------------------------------------------------------
echo "========================================================"
echo " 32B TTFT Sweep: RPS ${START_RPS} to ${END_RPS} step ${STEP}"
echo " Results → $SCRIPT_DIR/results/sweep/"
echo "========================================================"

mkdir -p "$SCRIPT_DIR/results/sweep"

# ── kvcached run ──────────────────────────────────────────
echo ""
echo "=== [1/2] kvcached run ==="
kill_stack "$KVCACHED_CONFIG"   # clean slate

echo "Launching kvcached stack..."
python "$LAUNCH_PY" --config "$KVCACHED_CONFIG" &
LAUNCH_PID=$!

wait_for_healthy "kvcached"

cd "$SCRIPT_DIR"
bash run_sweep_32b.sh kvcached-32b "$START_RPS" "$END_RPS" "$STEP"

kill_stack "$KVCACHED_CONFIG"
wait "$LAUNCH_PID" 2>/dev/null || true

# ── baseline run ──────────────────────────────────────────
echo ""
echo "=== [2/2] baseline run ==="

echo "Launching baseline stack..."
python "$LAUNCH_PY" --config "$BASELINE_CONFIG" &
LAUNCH_PID=$!

wait_for_healthy "baseline"

cd "$SCRIPT_DIR"
bash run_sweep_32b.sh baseline-32b "$START_RPS" "$END_RPS" "$STEP"

kill_stack "$BASELINE_CONFIG"
wait "$LAUNCH_PID" 2>/dev/null || true

# ── summary ──────────────────────────────────────────────
echo ""
echo "========================================================"
echo " All done. Results in results/sweep/"
echo "========================================================"
ls -lh "$SCRIPT_DIR/results/sweep/"

echo ""
echo "Quick TTFT summary:"
python3 - <<'EOF'
import json, glob, os

results = {}
for f in sorted(glob.glob("results/sweep/*.json")):
    d = json.load(open(f))
    name = os.path.basename(f)
    # extract tag and rps from filename
    parts = name.split("-")
    tag = "kvcached" if name.startswith("kvcached") else "baseline"
    rps_part = [p for p in parts if p.startswith("rps")]
    rps = rps_part[0].replace("rps","") if rps_part else "?"
    inst_part = [p for p in parts if p.startswith("inst")]
    inst = inst_part[0] if inst_part else "?"
    key = (tag, rps, inst)
    results[key] = d.get("mean_ttft_ms")

print(f"{'Tag':<15} {'RPS':>5} {'Inst':>6}  {'mean_ttft_ms':>14}")
print("-" * 45)
for key in sorted(results):
    tag, rps, inst = key
    val = results[key]
    val_str = f"{val:.1f}" if val else "N/A"
    print(f"{tag:<15} {rps:>5} {inst:>6}  {val_str:>14}")
EOF
