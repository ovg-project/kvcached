#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

# Decide what the GPU Tests workflow should do with this trigger.
#
# Reads the trigger through the environment so the decision can be exercised
# on a hosted runner without GitHub actually firing the workflow:
#
#   EVENT             schedule | workflow_dispatch | pull_request
#   LABELS            space-separated pull request label names
#   DISPATCH_PROFILE  the workflow_dispatch profile input
#   DISPATCH_DEVICES  the workflow_dispatch devices input
#   SINGLE_DEVICES    device selection for the one-GPU profiles
#   DUAL_DEVICES      device selection for nixl and all
#   MIN_MERGES        commits that must land before a scheduled slot is spent
#   GH_TOKEN          token for the workflow-runs query (schedule only)
#
# Writes `run`, and on a run also `profile` and `devices`, to GITHUB_OUTPUT.

set -euo pipefail

EVENT="${EVENT:-}"
LABELS="${LABELS:-}"
DISPATCH_PROFILE="${DISPATCH_PROFILE:-}"
DISPATCH_DEVICES="${DISPATCH_DEVICES:-}"
SINGLE_DEVICES="${SINGLE_DEVICES:-}"
DUAL_DEVICES="${DUAL_DEVICES:-}"
MIN_MERGES="${MIN_MERGES:-5}"
GITHUB_OUTPUT="${GITHUB_OUTPUT:-/dev/null}"

emit() { printf '%s\n' "$@" >>"${GITHUB_OUTPUT}"; }

decline() {
  echo "$1"
  emit "run=false"
  exit 0
}

has_label() {
  case " ${LABELS} " in (*" $1 "*) return 0 ;; (*) return 1 ;; esac
}

case "${EVENT}" in
  schedule)
    profile=all
    ;;
  workflow_dispatch)
    profile="${DISPATCH_PROFILE:-core}"
    ;;
  pull_request)
    # The label is the only gate, and it is deliberately the only one: a
    # fork's code runs on the registered host once someone with write or
    # triage permission asks for it. GitHub withholds secrets and issues a
    # read-only token for a fork's pull_request, so the exposure is code
    # execution on that host, nothing else. Reviewing a contributor's change
    # before merging it is what the GPU run is for, and the contributors here
    # work from forks, so a same-repository check would rule out the case
    # that matters.
    #
    # This assumes the owner of the registered host has accepted that policy,
    # because the person applying the label is not necessarily that owner.
    # Whoever registers the runner should agree to it first, and can add a
    # second gate with Settings > Actions > "Require approval for all outside
    # collaborators".
    #
    # Specific labels win over the plain gpu-ci one, so carrying both does
    # not silently downgrade the run to core.
    if   has_label gpu-ci-all;     then profile=all
    elif has_label gpu-ci-nixl;    then profile=nixl
    elif has_label gpu-ci-engines; then profile=engines
    elif has_label gpu-ci-sglang;  then profile=sglang
    elif has_label gpu-ci-vllm;    then profile=vllm
    elif has_label gpu-ci;         then profile=core
    else
      decline "no gpu-ci label on this pull request"
    fi
    ;;
  *)
    echo "unsupported event: '${EVENT}'" >&2
    exit 2
    ;;
esac

# The scheduled slot is a budget, not an obligation: only spend the GPU once
# enough pull requests have squash-merged since the last successful pass.
if [[ "${EVENT}" == "schedule" ]]; then
  last="$(gh api \
    "repos/${GITHUB_REPOSITORY}/actions/workflows/gpu-tests.yml/runs?event=schedule&status=success&per_page=1" \
    --jq '.workflow_runs[0].head_sha // empty')"
  if [[ -z "${last}" ]] || ! git cat-file -e "${last}^{commit}" 2>/dev/null; then
    echo "no usable previous scheduled run; treating this slot as due"
    merged="${MIN_MERGES}"
  else
    merged="$(git rev-list --count "${last}..HEAD")"
  fi
  echo "${merged} commit(s) since ${last:-the beginning}, threshold ${MIN_MERGES}"
  if (( merged < MIN_MERGES )); then
    decline "not enough has landed to spend the GPU on this slot"
  fi
fi

# nixl and all drive a two-GPU prefill/decode transfer.
case "${profile}" in
  nixl|all) devices="${DISPATCH_DEVICES:-${DUAL_DEVICES}}" ;;
  *)        devices="${DISPATCH_DEVICES:-${SINGLE_DEVICES}}" ;;
esac
if [[ -z "${devices}" ]]; then
  echo "no GPU device selection is configured for profile ${profile}" >&2
  echo "set KVCACHED_GPU_VISIBLE_DEVICES, or KVCACHED_GPU_DUAL_DEVICES for nixl and all" >&2
  exit 1
fi

echo "profile=${profile}"
echo "devices=${devices}"
emit "run=true" "profile=${profile}" "devices=${devices}"
