#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0
#
# Create the GPU VM for one CI run, trying each zone until one has capacity.
#
# usage: launch.sh <vm-name> <machine-type> <zone,zone,...>
#
# The VM has no service account, so code running on it holds no GCP
# credentials. It deletes itself after GCE_MAX_RUN_DURATION even if the
# workflow never gets to delete it. Prints zone=<zone> to $GITHUB_OUTPUT.

set -euo pipefail

name=$1
machine_type=$2
zones=$3

: "${GCE_IMAGE_FAMILY:?set GCE_IMAGE_FAMILY (an image with NVIDIA driver >= 580)}"
: "${GCE_IMAGE_PROJECT:?set GCE_IMAGE_PROJECT}"
max_run_duration=${GCE_MAX_RUN_DURATION:-5h}
disk_gb=${GCE_DISK_GB:-400}
output=${GITHUB_OUTPUT:-/dev/stdout}
err=$(mktemp)
trap 'rm -f "$err"' EXIT

for zone in ${zones//,/ }; do
  echo "Creating $name ($machine_type) in $zone"
  if gcloud compute instances create "$name" \
      --zone "$zone" \
      --machine-type "$machine_type" \
      --image-family "$GCE_IMAGE_FAMILY" \
      --image-project "$GCE_IMAGE_PROJECT" \
      --boot-disk-size "${disk_gb}GB" \
      --boot-disk-type pd-ssd \
      --maintenance-policy TERMINATE \
      --provisioning-model STANDARD \
      --max-run-duration "$max_run_duration" \
      --instance-termination-action DELETE \
      --no-service-account --no-scopes \
      --metadata enable-oslogin=TRUE \
      --tags kvcached-ci \
      --labels "kvcached-ci=true,run=${GITHUB_RUN_ID:-manual}" 2>"$err"; then
    echo "zone=$zone" >>"$output"
    exit 0
  fi
  cat "$err" >&2
  if grep -qE "ZONE_RESOURCE_POOL_EXHAUSTED|does not have enough resources|QUOTA_EXCEEDED|stockout" "$err"; then
    echo "No capacity in $zone; trying the next zone"
    continue
  fi
  exit 1
done

echo "No zone in '$zones' had capacity for $machine_type" >&2
exit 1
