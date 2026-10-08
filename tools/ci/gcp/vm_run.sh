#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0
#
# Runs on the CI VM: make sure the NVIDIA driver and Docker with the NVIDIA
# runtime work, pull the engine images, and run the e2e suite.
#
# usage: vm_run.sh <profile> [run.py arguments...]
#   KVCACHED_SRC  checkout to test (default ~/kvcached)
#   E2E_OUT       results directory (default ~/e2e-out)

set -euo pipefail

profile=$1
shift
src=${KVCACHED_SRC:-$HOME/kvcached}
out=${E2E_OUT:-$HOME/e2e-out}
export DEBIAN_FRONTEND=noninteractive

if ! nvidia-smi >/dev/null 2>&1; then
  # The engine images use CUDA 13, which needs driver 580. Install the
  # prebuilt module for the running kernel: the -gcp meta package may point
  # at a newer kernel than the one booted.
  sudo apt-get update -q
  sudo apt-get install -y -q "linux-modules-nvidia-580-server-open-$(uname -r)" \
    nvidia-utils-580-server
  sudo modprobe nvidia
  sudo modprobe nvidia_uvm
fi
nvidia-smi

if ! command -v docker >/dev/null 2>&1; then
  curl -fsSL https://get.docker.com | sudo sh
fi
if ! sudo docker info 2>/dev/null | grep -q nvidia; then
  curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey |
    sudo gpg --dearmor --yes -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
  curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list |
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#' |
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list >/dev/null
  sudo apt-get update -q
  sudo apt-get install -y -q nvidia-container-toolkit
  sudo nvidia-ctk runtime configure --runtime=docker
  sudo systemctl restart docker
fi

cd "$src"
images=$(python3 -c "import sys; sys.path.insert(0, 'tests'); \
from e2e.matrix import ENGINES, PROFILES; \
print(' '.join(ENGINES[e].image for e in PROFILES['$profile'].engines))")
pids=()
for image in $images; do
  sudo docker pull -q "$image" &
  pids+=($!)
done
for pid in "${pids[@]}"; do
  wait "$pid"
done

rc=0
E2E_DOCKER="sudo docker" python3 tests/e2e/run.py --profile "$profile" --out "$out" "$@" || rc=$?

# GPU state at the end, for runs that fail on the hardware or the driver.
mkdir -p "$out"
nvidia-smi -q > "$out/nvidia-smi-q.txt" 2>&1 || true
sudo dmesg -T 2>/dev/null | grep -iE "NVRM|Xid" | tail -n 200 > "$out/dmesg-nvidia.txt" || true
exit "$rc"
