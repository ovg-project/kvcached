#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0
#
# The GPU runner runs this before every job it is given
# (ACTIONS_RUNNER_HOOK_JOB_STARTED, set by setup.sh); when it fails, the job
# fails before any of its steps runs. A pull request can bring its own
# workflow files, so the runner, not the workflows, decides which jobs it
# takes: the GPU CI workflows of this repository's main branch, started on
# schedule, by hand, or by a maintainer's /ci run comment (gpu-pr.yml).
# GitHub sets the GITHUB_* variables; a workflow cannot change them.

set -euo pipefail

repo=ovg-project/kvcached
workflows=$repo/.github/workflows

allowed=no
if [ "${GITHUB_REPOSITORY:-}" = "$repo" ] && [ "${GITHUB_REF:-}" = refs/heads/main ]; then
  case "${GITHUB_EVENT_NAME:-}:${GITHUB_WORKFLOW_REF:-}" in
    "schedule:$workflows/gpu-correctness.yml@refs/heads/main" | \
      "workflow_dispatch:$workflows/gpu-correctness.yml@refs/heads/main" | \
      "issue_comment:$workflows/gpu-pr.yml@refs/heads/main")
      allowed=yes
      ;;
  esac
fi

if [ "$allowed" != yes ]; then
  echo "This runner only takes the GPU CI jobs of $repo's main branch; refusing the" \
    "${GITHUB_EVENT_NAME:-?} job of ${GITHUB_WORKFLOW_REF:-?} (${GITHUB_REF:-?})." >&2
  exit 1
fi
