#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0
#
# Set up the GPU CI machine: a Linux machine with NVIDIA GPUs and a working
# driver >= 580 (the engine images use CUDA 13), which runs the jobs of
# gpu-correctness.yml and gpu-pr.yml as a self-hosted runner of this
# repository. Run it once on the machine, from a kvcached checkout, as a user
# with sudo; running it again updates the job hook and restarts the runner.
#
# usage: RUNNER_TOKEN=<token> bash tools/ci/gpu-runner/setup.sh
#   RUNNER_TOKEN  a runner registration token, valid for an hour, which a
#                 repository admin gets with
#                 gh api -X POST repos/ovg-project/kvcached/actions/runners/registration-token --jq .token
#
# The runner only has the label kvcached-gpu, and job_started.sh refuses
# every job but the GPU CI jobs of main.

set -euo pipefail

repo=ovg-project/kvcached
version=2.338.0
sha256=af4b794c1bc41d73d40535e3fe092a39f9679cd8d965954c2aca25a05ca41d32
user=gha-runner
runner=/opt/actions-runner
root=/opt/kvcached-ci
here=$(cd "$(dirname "$0")" && pwd)
export DEBIAN_FRONTEND=noninteractive

# Unattended upgrades would change packages under a CI run, hold apt's lock,
# or install a kernel the NVIDIA module is not built for: upgrade by hand.
sudo systemctl disable --now unattended-upgrades.service apt-daily.timer \
  apt-daily-upgrade.timer >/dev/null 2>&1 || true
for _ in $(seq 120); do
  pgrep -f unattended-upgrade >/dev/null || break
  sleep 5
done

nvidia-smi

# The containers share the host's network and so its hostname. torch's Gloo
# reads it into a HOST_NAME_MAX buffer, which a 64-character name such as a
# cloud VM's full domain name overflows ("File name too long").
name=$(hostname)
if [ "${#name}" -ge 64 ]; then
  sudo hostnamectl set-hostname "${name%%.*}"
fi

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
  # Some images hold nvidia-container-toolkit-base at their own version;
  # install the toolkit of that version instead of upgrading the held package.
  toolkit=nvidia-container-toolkit
  held=$(dpkg-query -W -f='${Version}' nvidia-container-toolkit-base 2>/dev/null || true)
  if [ -n "$held" ]; then
    toolkit=nvidia-container-toolkit=$held
  fi
  sudo apt-get install -y -q "$toolkit"
  sudo nvidia-ctk runtime configure --runtime=docker
  sudo systemctl restart docker
fi

# The runner's own user, which may use Docker and read the kernel log.
if ! id "$user" >/dev/null 2>&1; then
  sudo useradd --system --create-home --shell /bin/bash "$user"
fi
sudo usermod -aG docker,adm "$user"
# The model weights the runs share, kept between them.
sudo mkdir -p "$root/hf"
sudo install -m 0755 -o root -g root "$here/job_started.sh" "$root/job_started.sh"

if [ ! -x "$runner/config.sh" ]; then
  tarball=$(mktemp)
  curl -fsSL -o "$tarball" \
    "https://github.com/actions/runner/releases/download/v$version/actions-runner-linux-x64-$version.tar.gz"
  echo "$sha256  $tarball" | sha256sum -c -
  sudo mkdir -p "$runner"
  sudo tar -xzf "$tarball" -C "$runner"
  rm -f "$tarball"
  sudo "$runner/bin/installdependencies.sh"
  sudo chown -R "$user:" "$runner"
fi

cd "$runner"
if [ ! -f .runner ]; then
  : "${RUNNER_TOKEN:?set RUNNER_TOKEN to a runner registration token}"
  sudo -u "$user" ./config.sh --unattended --url "https://github.com/$repo" \
    --token "$RUNNER_TOKEN" --name "$(hostname)" --labels kvcached-gpu --no-default-labels
fi
if ! sudo grep -q '^ACTIONS_RUNNER_HOOK_JOB_STARTED=' .env 2>/dev/null; then
  echo "ACTIONS_RUNNER_HOOK_JOB_STARTED=$root/job_started.sh" | sudo -u "$user" tee -a .env >/dev/null
fi
if [ ! -f .service ]; then
  sudo ./svc.sh install "$user"
fi
sudo ./svc.sh stop >/dev/null 2>&1 || true
sudo ./svc.sh start
