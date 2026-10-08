#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0
#
# Run the e2e suite on a CI VM over IAP SSH and bring the results back.
#
# usage: remote.sh <vm-name> <zone> <profile> <local-results-dir> [run.py arguments...]
#
# The run is started detached on the VM, so a dropped SSH session does not
# kill it; its log is streamed here by polling. Exits with the suite's status.

set -euo pipefail

name=$1
zone=$2
profile=$3
results=$4
shift 4
args=""
if [ $# -gt 0 ]; then
  args=$(printf ' %q' "$@")
fi

repo=$(git rev-parse --show-toplevel)
# The CI connects through IAP. GCE_SSH_IAP=0 uses the VM's external IP
# instead, for a manual run by someone without IAP tunnel access.
tunnel=(--tunnel-through-iap)
if [ "${GCE_SSH_IAP:-1}" = 0 ]; then
  tunnel=()
fi
ssh_vm() {
  gcloud compute ssh "$name" --zone "$zone" "${tunnel[@]}" --quiet \
    --command "$1" -- -o ServerAliveInterval=30 -o ConnectTimeout=30
}

echo "Waiting for SSH on $name"
for _ in $(seq 60); do
  if ssh_vm true 2>/dev/null; then
    break
  fi
  sleep 10
done
ssh_vm true

src=$(mktemp --suffix=.tar.gz)
git -C "$repo" archive --format=tar.gz -o "$src" HEAD
gcloud compute scp --zone "$zone" "${tunnel[@]}" --quiet "$src" "$name":src.tar.gz
rm -f "$src"
ssh_vm "rm -rf kvcached && mkdir kvcached && tar -xzf src.tar.gz -C kvcached"

# The VM gets an exported tree without .git, so pass the commit along.
sha=$(git -C "$repo" rev-parse HEAD)
ssh_vm "nohup bash -c 'KVCACHED_SHA=$sha bash kvcached/tools/ci/gcp/vm_run.sh $profile$args; \
  echo \$? > run.rc' > run.log 2>&1 < /dev/null &"

offset=0
chunk=$(mktemp)
trap 'rm -f "$chunk"' EXIT
while true; do
  sleep 30
  if ssh_vm "tail -c +$((offset + 1)) run.log" >"$chunk" 2>/dev/null; then
    cat "$chunk"
    offset=$((offset + $(stat -c %s "$chunk")))
  fi
  if ssh_vm "test -f run.rc" 2>/dev/null; then
    break
  fi
done
ssh_vm "tail -c +$((offset + 1)) run.log" || true
rc=$(ssh_vm "cat run.rc")

mkdir -p "$results"
ssh_vm "mkdir -p e2e-out && cp run.log e2e-out/run.log; tar -czf e2e-out.tar.gz -C e2e-out . || true"
gcloud compute scp --zone "$zone" "${tunnel[@]}" --quiet "$name":e2e-out.tar.gz \
  "$results/e2e-out.tar.gz" || true
tar -xzf "$results/e2e-out.tar.gz" -C "$results" 2>/dev/null || true

echo "Suite exit status: $rc"
exit "$rc"
