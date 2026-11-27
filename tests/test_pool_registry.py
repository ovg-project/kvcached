# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import multiprocessing
import sys
import threading
import types

import pytest

if "torch" not in sys.modules and importlib.util.find_spec("torch") is None:
    sys.modules.setdefault("torch", types.ModuleType("torch"))

from kvcached import pool_registry


class Pool:
    pass


def _pool_count_in_child(result):
    result.put(len(pool_registry.get_registered_kv_cache_pools(integration="test")))


@pytest.fixture(autouse=True)
def clear_registry():
    pool_registry.clear_registered_kv_cache_pools()
    yield
    pool_registry.clear_registered_kv_cache_pools()


@pytest.mark.skipif("fork" not in multiprocessing.get_all_start_methods(),
                    reason="requires fork")
def test_forked_process_starts_with_an_empty_pool_registry():
    pool = Pool()
    pool_registry.register_kv_cache_pool(pool, integration="test")
    context = multiprocessing.get_context("fork")
    result = context.Queue()
    process = context.Process(target=_pool_count_in_child, args=(result,))
    process.start()
    try:
        process.join(timeout=5)
        assert process.exitcode == 0
        assert result.get(timeout=2) == 0
    finally:
        if process.is_alive():
            process.kill()
            process.join()
        process.close()


@pytest.mark.skipif("fork" not in multiprocessing.get_all_start_methods(),
                    reason="requires fork")
def test_fork_replaces_a_registry_lock_held_by_another_thread():
    pool = Pool()
    pool_registry.register_kv_cache_pool(pool, integration="test")
    locked, release = threading.Event(), threading.Event()

    def hold_registry_lock():
        with pool_registry._registered_pools_lock:
            locked.set()
            release.wait()

    thread = threading.Thread(target=hold_registry_lock)
    thread.start()
    try:
        assert locked.wait(timeout=5)
        context = multiprocessing.get_context("fork")
        result = context.Queue()
        process = context.Process(target=_pool_count_in_child, args=(result,))
        process.start()
        try:
            process.join(timeout=5)
            assert process.exitcode == 0
            assert result.get(timeout=2) == 0
        finally:
            if process.is_alive():
                process.kill()
                process.join()
            process.close()
    finally:
        release.set()
        thread.join()
