# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0
"""Rollback unpublished native prefill pages, rows and admission pins.

An existing prefix tail is borrowed, so out_cache_loc is not a rollback list.
The backing adapter journals the native allocator's newly selected page IDs.
"""

from contextlib import ExitStack
from functools import wraps
from threading import get_ident

from .rank_plan import request_slot


class PrefillAllocationMiss(RuntimeError):
    """An allocator explicitly returned None before forward."""


class PrefillPrepareDeferred(RuntimeError):
    """Every rank returned its unpublished allocations; restore admission."""


_MISSING = object()


def _snapshot(obj, names):
    return {name: getattr(obj, name, _MISSING) for name in names}


def _restore(obj, values):
    for name, value in values.items():
        if value is _MISSING:
            if hasattr(obj, name):
                delattr(obj, name)
        else:
            setattr(obj, name, value)


class PrefillAllocationTransaction:
    @classmethod
    def create(cls, batch):
        allocator = getattr(batch, "token_to_kv_pool_allocator", None)
        pool = getattr(batch, "req_to_token_pool", None)
        begin = getattr(allocator, "_kvcached_prepare_transaction", None)
        if begin is None or pool is None:
            return None
        if getattr(pool, "_aux_cache", None) is not None or hasattr(pool, "mamba_allocator"):
            return None
        return cls(batch, begin())

    def __init__(self, batch, pages):
        self.batch, self.pages = batch, pages
        self.stack = ExitStack()
        self.owner = get_ident()
        self.requests = [
            (
                req,
                request_slot(req),
                _snapshot(
                    req,
                    (
                        "extend_batch_idx",
                        "cached_tokens",
                        "cached_tokens_device",
                        "cached_tokens_host",
                        "cached_tokens_storage",
                        "already_computed",
                        "is_retracted",
                        "_cache_breakdown_computed",
                        "kv",
                    ),
                ),
                _snapshot(
                    req.kv,
                    (
                        "req_pool_idx",
                        "kv_allocated_len",
                        "kv_committed_len",
                    ),
                ),
            )
            for req in batch.reqs
        ]

    def __enter__(self):
        try:
            self.stack.enter_context(self.pages)
            for obj, name in (
                (self.batch.token_to_kv_pool_allocator, "alloc_extend"),
                (self.batch.req_to_token_pool, "alloc"),
            ):
                original = getattr(obj, name)
                previous = _snapshot(obj, (name,)) if name in vars(obj) else {name: _MISSING}

                @wraps(original)
                def checked(*args, _original=original, _name=name, **kwargs):
                    result = _original(*args, **kwargs)
                    if result is None and get_ident() == self.owner:
                        raise PrefillAllocationMiss(f"{_name} returned None")
                    return result

                setattr(obj, name, checked)
                self.stack.callback(_restore, obj, previous)
        except BaseException:
            self.stack.close()
            raise
        return self

    def __exit__(self, *exc):
        self.stack.close()

    def rollback(self):
        self.pages.rollback()
        pool = self.batch.req_to_token_pool
        for req, old_slot, req_state, kv_state in self.requests:
            if old_slot is None and request_slot(req) is not None:
                pool.free(req)
            _restore(req, req_state)
            _restore(req.kv, kv_state)
        # Eviction and cache loading are real. Keep their state; only entries
        # beyond the restored committed cursor are discarded and overwritten.


class PrefillScheduleSnapshot:
    """Restore queue/chunk membership and release only this pass's pins."""

    def __init__(self, scheduler):
        self.scheduler = scheduler
        self.waiting = list(scheduler.waiting_queue)
        self.chunk = scheduler.chunked_req
        reqs = self.waiting + ([self.chunk] if self.chunk is not None else [])
        self.requests = [
            (
                req,
                _snapshot(
                    req,
                    (
                        "extend_range",
                        "inflight_middle_chunks",
                        "lock_receipt",
                        "swa_uuid_for_lock",
                        "skip_lock_node_ids",
                    ),
                ),
            )
            for req in reqs
        ]
        self.admitted = []

    def record_admission(self, req):
        self.admitted.append((req.last_node, req.lock_receipt))

    def rollback(self, running_batch):
        for node, receipt in reversed(self.admitted):
            if receipt is None:
                self.scheduler.tree_cache.dec_lock_ref(node)
            else:
                self.scheduler.tree_cache.dec_lock_ref(node, receipt)
        self.scheduler.waiting_queue = self.waiting
        self.scheduler.chunked_req = self.chunk
        for req, state in self.requests:
            _restore(req, state)
        running_batch.batch_is_full = False
