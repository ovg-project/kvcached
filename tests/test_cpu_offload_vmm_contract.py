# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path


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
