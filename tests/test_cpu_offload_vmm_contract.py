# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import yaml

ROOT = Path(__file__).parents[1]


def test_page_allocator_exposes_stable_id_offload_lifecycle():
    header = (ROOT / "csrc" / "inc" / "page_allocator.hpp").read_text()
    implementation = (ROOT / "csrc" / "page_allocator.cpp").read_text()
    bindings = (ROOT / "csrc" / "torch_bindings.cpp").read_text()

    for method in (
        "offload_page",
        "restore_page",
        "get_num_offloaded_pages",
        "is_page_offloaded",
    ):
        assert method in header
        assert method in implementation
        assert method in bindings

    assert "offloaded_page_ids_" in header
    assert "transitioning_page_ids_" in header
    assert "Cannot resize while pages are CPU-offloaded" in implementation


def test_offload_bindings_release_the_gil_while_mapping_pages():
    bindings = (ROOT / "csrc" / "torch_bindings.cpp").read_text()

    for method in ("offload_page", "restore_page"):
        binding = bindings.split(f'.def("{method}"', 1)[1].split(
            "\n      .def(", 1
        )[0]
        assert "gil_scoped_release" in binding


def test_offload_unmaps_and_restore_maps_the_same_page_id():
    implementation = (ROOT / "csrc" / "page_allocator.cpp").read_text()

    offload_body = implementation.split(
        "void PageAllocator::offload_page", 1
    )[1].split("void PageAllocator::restore_page", 1)[0]
    restore_body = implementation.split(
        "void PageAllocator::restore_page", 1
    )[1].split("bool PageAllocator::resize", 1)[0]

    assert "unmap_pages({page_id})" in offload_body
    assert "map_pages({page_id})" in restore_body


def test_gpu_workflow_targets_the_persistent_runner():
    workflow = yaml.safe_load(
        (ROOT / ".github" / "workflows" / "cpu-offload-gpu.yml").read_text()
    )
    job = workflow["jobs"]["vmm-roundtrip"]

    assert job["runs-on"] == ["self-hosted", "linux", "x64", "gpu", "kvcached"]
    run_step = next(step for step in job["steps"] if "run" in step)
    assert run_step["run"] == "bash tools/run_cpu_offload_h20_validation.sh"


def test_actionlint_knows_cpu_offload_runner_labels():
    config = yaml.safe_load(
        (ROOT / ".github" / "actionlint.yaml").read_text(encoding="utf-8")
    )

    labels = config["self-hosted-runner"]["labels"]
    assert "gpu" in labels
    assert "kvcached" in labels


def test_two_gpu_campaign_preserves_results_and_compares_vllm_baseline():
    script = (ROOT / "tools" / "run_two_gpu_cpu_offload_campaign.sh").read_text()

    assert 'ARTIFACT_ROOT="${ARTIFACT_ROOT:-}"' in script
    assert "trap finalize EXIT" in script
    assert "core-gpu0" in script
    assert "core-gpu1" in script
    assert "dual-gpu-isolation" in script
    assert "vllm-baseline-vs-offload" in script
    assert "nixl-regression" in script
    assert "targeted-model-matrix" in script
    assert 'BASELINE_URL="http://127.0.0.1:${BASELINE_PORT}"' in script
    assert "MANIFEST.sha256" in script
