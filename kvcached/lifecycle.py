# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

"""Lifecycle readiness state for one kvcached KV pool (issue #375, item 5).

State is the surface: a level-triggered phase per pool that consumers poll
through ``KVCacheManager.lifecycle_phase`` (also exported as the
``lifecycle_phase`` field of ``KVCachePoolSnapshot``) or gate on with
``KVCacheManager.wait_ready()``. Failures are recorded facts, not deliveries:
there are no callbacks, so nothing here re-enters consumer code from an
allocator thread (issue #371).

This module depends on neither torch nor the compiled extension, so a
control plane can import the phase vocabulary cheaply.
"""

from __future__ import annotations

import threading
from enum import Enum
from typing import Optional, Tuple

from kvcached.utils import get_kvcached_logger

logger = get_kvcached_logger()


class LifecyclePhase(str, Enum):
    """Level-triggered phase of one KV pool."""

    INITIALIZING = "initializing"
    READY = "ready"
    # Still serving, but accounting is suspect: an unmap broadcast failed
    # and its outcome across TP/PP ranks is unknown.
    DEGRADED = "degraded"
    # Cannot serve correctly. wait_ready() raises the recorded error.
    FAILED = "failed"


class LifecycleState:
    """Thread-safe holder for one pool's lifecycle phase.

    Transitions:

    * INITIALIZING -> READY on post-init success, -> FAILED on post-init error.
    * READY -> DEGRADED when an unmap broadcast fails. Phase 1 has no
      per-rank delivery tracking and no group-wide reconciliation, so every
      unmap failure, a timeout included, is an unknown cross-rank outcome and
      degrades the pool. Map broadcast failures do not transition: they are
      routinely the expected co-tenancy capacity miss that ``alloc_page()``
      rolls back and ``_alloc()`` reports as a scheduling miss (#453), and
      phase 1 cannot tell that recoverable miss from an unknown outcome
      until #373's structured per-rank results (phase 2).
    * READY -> INITIALIZING -> READY around ``KVCacheManager.clear()``, or
      -> FAILED if ``clear()`` raises.
    * DEGRADED -> INITIALIZING around ``clear()`` likewise: the pool is torn
      down and rebuilt, so the readiness gate must hold for the window. The
      cause is carried into the pending slot and ``mark_ready()`` settles the
      pool back to DEGRADED, not READY. Stickiness preserves the cause across
      reinitialization; it does not suppress the transient INITIALIZING phase.
    * DEGRADED and FAILED are sticky: ``mark_ready()`` does not leave a
      settled DEGRADED or FAILED phase and the first cause is kept. Recovery
      is a control-plane decision.

    A degradation observed while INITIALIZING is applied when that window
    ends, so a settled phase always means the init or clear work finished.
    """

    def __init__(self, name: str):
        self.name = name
        self._cond = threading.Condition(threading.Lock())
        self._phase = LifecyclePhase.INITIALIZING
        self._reason = ""
        self._error: Optional[BaseException] = None
        self._pending_degraded: Optional[Tuple[str, Optional[BaseException]]] = None

    @property
    def phase(self) -> LifecyclePhase:
        with self._cond:
            return self._phase

    @property
    def error(self) -> Optional[BaseException]:
        """The error behind a DEGRADED or FAILED phase, if any."""
        with self._cond:
            return self._error

    @property
    def reason(self) -> str:
        with self._cond:
            return self._reason

    def _set(
        self,
        phase: LifecyclePhase,
        reason: str,
        error: Optional[BaseException],
    ) -> LifecyclePhase:
        """Caller holds ``_cond``. Returns the previous phase."""
        old = self._phase
        self._phase = phase
        self._reason = reason
        self._error = error
        self._cond.notify_all()
        return old

    def _log(self, old: LifecyclePhase, new: LifecyclePhase, reason: str) -> None:
        """Every transition is logged; the level follows the phase entered."""
        if new is LifecyclePhase.FAILED:
            log = logger.error
        elif new is LifecyclePhase.DEGRADED:
            log = logger.warning
        else:
            log = logger.info
        log("kvcached pool %s lifecycle: %s -> %s (%s)",
            self.name, old.value, new.value, reason)

    def mark_ready(self) -> None:
        """INITIALIZING -> READY, or -> DEGRADED if a degradation is pending."""
        with self._cond:
            if self._phase is not LifecyclePhase.INITIALIZING:
                return
            pending, self._pending_degraded = self._pending_degraded, None
            if pending is None:
                reason = "initialization complete"
                old = self._set(LifecyclePhase.READY, "", None)
                new = LifecyclePhase.READY
            else:
                reason, error = pending
                old = self._set(LifecyclePhase.DEGRADED, reason, error)
                new = LifecyclePhase.DEGRADED
        self._log(old, new, reason)

    def mark_degraded(self, reason: str, error: Optional[BaseException] = None) -> None:
        """READY -> DEGRADED. Deferred while INITIALIZING; no-op once
        DEGRADED or FAILED."""
        with self._cond:
            if self._phase is LifecyclePhase.INITIALIZING:
                if self._pending_degraded is None:
                    self._pending_degraded = (reason, error)
                return
            if self._phase is not LifecyclePhase.READY:
                return
            old = self._set(LifecyclePhase.DEGRADED, reason, error)
        self._log(old, LifecyclePhase.DEGRADED, reason)

    def mark_failed(self, reason: str, error: Optional[BaseException] = None) -> None:
        """Any phase -> FAILED. The first failure is kept."""
        with self._cond:
            if self._phase is LifecyclePhase.FAILED:
                return
            old = self._set(LifecyclePhase.FAILED, reason, error)
        self._log(old, LifecyclePhase.FAILED, reason)

    def begin_reinit(self) -> None:
        """READY or DEGRADED -> INITIALIZING for the ``clear()`` window.

        A DEGRADED pool is torn down and rebuilt like a READY one, so
        consumers must not pass the readiness gate meanwhile: its cause is
        carried into the pending slot and ``mark_ready()`` settles the pool
        back to DEGRADED. No-op while INITIALIZING or FAILED.
        """
        reason = "clear() in progress"
        with self._cond:
            if self._phase not in (LifecyclePhase.READY, LifecyclePhase.DEGRADED):
                return
            if self._phase is LifecyclePhase.DEGRADED:
                self._pending_degraded = (self._reason, self._error)
            old = self._set(LifecyclePhase.INITIALIZING, reason, None)
        self._log(old, LifecyclePhase.INITIALIZING, reason)

    def wait_settled(self, timeout: Optional[float] = None) -> bool:
        """Block until the phase is not INITIALIZING. False on timeout."""
        with self._cond:
            return self._cond.wait_for(
                lambda: self._phase is not LifecyclePhase.INITIALIZING, timeout)

    def raise_if_failed(self) -> None:
        """Re-raise the recorded error if the pool is FAILED."""
        with self._cond:
            phase, error, reason = self._phase, self._error, self._reason
        if phase is not LifecyclePhase.FAILED:
            return
        if error is not None:
            raise error
        raise RuntimeError(f"kvcached pool {self.name} failed: {reason}")

    def record_broadcast_failure(self, op: str, exc: BaseException) -> None:
        """Apply the #375 rule to a failed broadcast.

        The outcome across ranks is unknown (a rank may have applied the
        command and failed to answer, which is what a timeout looks like), so
        the pool degrades. Phase 1 does not distinguish a command that
        provably reached no rank; that needs per-rank delivery tracking, so
        the manager calls this for unmap broadcasts only (see the class
        docstring for why map failures are excluded).
        """
        self.mark_degraded(
            f"{op} broadcast failed, outcome across ranks unknown: {exc}", exc)
