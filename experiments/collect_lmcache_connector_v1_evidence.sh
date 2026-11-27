#!/usr/bin/env bash
# =============================================================================
# Collect LMCacheConnectorV1 PD debug evidence for PRs/issues.
#
# Usage:
#   ./experiments/collect_lmcache_connector_v1_evidence.sh \
#       experiments/logs_lmcache_v1_debug/plain_lmcache_hits_1
#
# Output:
#   <run_dir>/../evidence_<run_id>_<timestamp>.tar.gz
# =============================================================================

set -euo pipefail

RUN_DIR="${1:-}"
if [ -z "$RUN_DIR" ]; then
    echo "usage: $0 <run_dir>" >&2
    exit 2
fi

RUN_DIR="${RUN_DIR%/}"
if [ ! -d "$RUN_DIR" ]; then
    echo "run dir not found: $RUN_DIR" >&2
    exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
RUN_DIR_ABS="$(cd "$RUN_DIR" && pwd)"
RUN_ID="$(basename "$RUN_DIR_ABS")"
OUT_ROOT="$(dirname "$RUN_DIR_ABS")"
EVIDENCE_DIR="$OUT_ROOT/evidence_${RUN_ID}_$(date +%Y%m%d_%H%M%S)"

mkdir -p "$EVIDENCE_DIR"

copy_if_present() {
    local src=$1
    local dst=$2
    if [ -e "$src" ]; then
        mkdir -p "$(dirname "$dst")"
        cp -a "$src" "$dst"
    fi
}

cp -a "$RUN_DIR_ABS" "$EVIDENCE_DIR/run"

{
    echo "# LMCacheConnectorV1 Evidence"
    echo
    echo "- run_id: $RUN_ID"
    echo "- collected_at_utc: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "- repo_dir: $REPO_DIR"
    echo "- run_dir: $RUN_DIR_ABS"
    echo
    echo "## Git"
    git -C "$REPO_DIR" status --short --branch || true
    git -C "$REPO_DIR" log --oneline --decorate -5 || true
    echo
    echo "## Key Classifiers"
    grep -R -E \
        "Classifier:|LMCache hit tokens: [1-9][0-9]*|Retrieved [1-9][0-9]* out of|Stored [1-9][0-9]* out of|EngineCore encountered|Traceback|HTTP 500|kv-layout-diag|stride=|storage_offset=" \
        "$RUN_DIR_ABS"/*.log "$RUN_DIR_ABS"/*.layout_diag.log 2>/dev/null || true
    echo
    echo "## Request/Response Files"
    ls -la "$RUN_DIR_ABS"/*_request.json "$RUN_DIR_ABS"/*_response.json 2>/dev/null || true
} > "$EVIDENCE_DIR/summary.md"

{
    echo "## uname"
    uname -a || true
    echo
    echo "## nvidia-smi"
    nvidia-smi || true
    echo
    echo "## nvidia-smi query"
    nvidia-smi --query-gpu=index,name,driver_version,memory.total,memory.used,utilization.gpu --format=csv || true
} > "$EVIDENCE_DIR/system.txt"

{
    echo "## python"
    python3 --version || true
    echo
    echo "## pip"
    python3 -m pip --version || true
    echo
    echo "## package versions"
    python3 - <<'PY' || true
import importlib

for name in ("vllm", "lmcache", "torch", "transformers", "aiohttp"):
    try:
        mod = importlib.import_module(name)
        print(f"{name} {getattr(mod, '__version__', '<no __version__>')} {getattr(mod, '__file__', '')}")
    except Exception as exc:
        print(f"{name} import failed: {exc!r}")

for stmt in ("from nixl._api import nixl_agent", "from rixl._api import nixl_agent"):
    try:
        exec(stmt)
        print(f"{stmt}: ok")
    except Exception as exc:
        print(f"{stmt}: failed: {exc!r}")
PY
    echo
    echo "## selected pip freeze"
    python3 -m pip freeze | grep -E '^(vllm|lmcache|torch|transformers|aiohttp|nixl|rixl|flashinfer|triton|nvidia-)' || true
} > "$EVIDENCE_DIR/python_packages.txt"

{
    echo "## generated configs"
    for file in "$RUN_DIR_ABS"/configs/*; do
        [ -f "$file" ] || continue
        echo "### $(basename "$file")"
        sed -n '1,220p' "$file"
        echo
    done
    echo "## kv transfer configs"
    for file in "$RUN_DIR_ABS"/*.kv_transfer_config.json; do
        [ -f "$file" ] || continue
        echo "### $(basename "$file")"
        python3 -m json.tool "$file" || cat "$file"
        echo
    done
} > "$EVIDENCE_DIR/configs.txt"

for file in request_1_request.json request_1_response.json request_2_request.json request_2_response.json; do
    copy_if_present "$RUN_DIR_ABS/$file" "$EVIDENCE_DIR/json/$file"
done

tarball="${EVIDENCE_DIR}.tar.gz"
tar -C "$OUT_ROOT" -czf "$tarball" "$(basename "$EVIDENCE_DIR")"

echo "Evidence directory: $EVIDENCE_DIR"
echo "Evidence tarball:   $tarball"
