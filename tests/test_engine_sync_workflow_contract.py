# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

ROOT = Path(__file__).parents[1]
WORKFLOW = ROOT / ".github" / "workflows" / "engine-upstream-sync.yml"


def test_push_step_keeps_cross_repository_token_available():
    source = WORKFLOW.read_text(encoding="utf-8")

    sync_step = source.split(
        "- name: Merge upstream and run compatibility checks", maxsplit=1
    )[1].split("- name: Open compatibility pull request", maxsplit=1)[0]

    assert "GH_TOKEN:" in sync_step
    assert "secrets.OVG_SYNC_TOKEN" in sync_step
    assert "python tools/sync_engine_upstream.py" in sync_step


def test_fork_token_is_not_exposed_to_dry_runs():
    source = WORKFLOW.read_text(encoding="utf-8")

    assert "inputs.dry_run != true && secrets.OVG_SYNC_TOKEN || ''" in source


def test_scheduled_sync_rebases_the_ovg_patch_stack():
    source = WORKFLOW.read_text(encoding="utf-8")

    assert "--strategy rebase" in source


def test_sync_targets_a_separate_kvcached_integration_branch():
    source = WORKFLOW.read_text(encoding="utf-8")

    assert 'base="${VLLM_BASE:-kvcached-main}"' in source
    assert 'base="${SGLANG_BASE:-kvcached-main}"' in source
    assert '--base-branch "${{ steps.config.outputs.base }}"' in source
    assert '--base "${BASE}"' in source


def test_unconfigured_engine_repository_is_skipped_without_failing_schedule():
    source = WORKFLOW.read_text(encoding="utf-8")

    missing_target = source.split('if [[ -z "${target}" ]]', maxsplit=1)[1].split(
        "\n          fi", maxsplit=1
    )[0]

    assert "::notice::No target configured" in missing_target
    assert 'echo "skip=true" >> "${GITHUB_OUTPUT}"' in missing_target
    assert "exit 0" in missing_target
    assert "::error::" not in missing_target
