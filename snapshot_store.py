"""Process-wide holder for the transactions snapshot and its background sync.

One instance serves every dashboard session. ``current()`` hands out the
loaded frame and never waits on the DLD API; ``request_sync()`` starts at
most one background worker that adopts a newer stored snapshot and tops the
frame up with the days it is missing. A synced frame replaces the served one
for all sessions at once, so new rows can never be shown to one rerun and
lost on the next. No Streamlit imports — the behaviour is tested offline.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Callable

LOGGER = logging.getLogger(__name__)

Snapshot = tuple[Any, dict]  # (frame, metadata)


@dataclass(frozen=True)
class TopUp:
    """One top-up round: the frame to serve next and what it added."""

    frame: Any
    meta: dict
    added_rows: int = 0
    more_pending: bool = False  # still behind — run another round right away


@dataclass(frozen=True)
class SyncStatus:
    """Point-in-time view of the store for the UI."""

    running: bool
    version: int                  # bumps whenever the served frame changes
    started_at: datetime | None   # start of the running (or last) sync
    checked_at: datetime | None   # end of the last finished sync
    error: str | None             # why the last finished sync failed, if it did
    added_rows: int               # rows the last finished sync added


class SnapshotStore:
    """Serve one shared snapshot; keep it current from a single worker.

    ``load`` returns the stored snapshot, ``remote_token`` cheaply identifies
    the stored version (compared with ``meta["snapshot_token"]``), and
    ``top_up`` fetches what the served frame is missing — returning ``None``
    when there is nothing new. All three run on the worker thread except the
    very first ``load``, which the first caller of ``current()`` waits for.
    """

    def __init__(
        self,
        load: Callable[[], Snapshot],
        top_up: Callable[[Any, dict], TopUp | None],
        remote_token: Callable[[], str | None] = lambda: None,
        *,
        recheck_seconds: float = 300.0,
        retry_seconds: float = 900.0,
        max_rounds: int = 12,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._load = load
        self._top_up = top_up
        self._remote_token = remote_token
        self._recheck_seconds = recheck_seconds
        self._retry_seconds = retry_seconds
        self._max_rounds = max_rounds
        self._clock = clock

        self._load_lock = threading.Lock()
        self._lock = threading.Lock()
        self._snapshot: Snapshot | None = None
        self._version = 0
        self._running = False
        self._worker: threading.Thread | None = None
        self._next_check = 0.0
        self._next_top_up = 0.0
        self._started_at: datetime | None = None
        self._checked_at: datetime | None = None
        self._error: str | None = None
        self._added_rows = 0

    @property
    def loaded(self) -> bool:
        return self._snapshot is not None

    def current(self) -> Snapshot:
        """The served (frame, meta). Only the first call in a process loads."""
        with self._load_lock:
            if self._snapshot is None:
                self._install(self._load())
                # The snapshot was just read, so the stored copy needs no
                # recheck yet; a top-up may still be due immediately.
                self._next_check = self._clock() + self._recheck_seconds
        with self._lock:
            return self._snapshot

    def status(self) -> SyncStatus:
        with self._lock:
            return SyncStatus(
                running=self._running,
                version=self._version,
                started_at=self._started_at,
                checked_at=self._checked_at,
                error=self._error,
                added_rows=self._added_rows,
            )

    def request_sync(self, top_up: bool = True) -> bool:
        """Start a background sync if one is due. True when a worker started.

        Cheap enough to call on every rerun: it returns immediately while a
        sync is running or cooling down. ``top_up=False`` only looks for a
        newer stored snapshot (the served frame already covers today).
        """
        now = self._clock()
        with self._lock:
            if self._running or self._snapshot is None:
                return False
            check_due = now >= self._next_check
            top_up_due = top_up and now >= self._next_top_up
            if not (check_due or top_up_due):
                return False
            self._running = True
            self._started_at = datetime.now(UTC)
            self._worker = threading.Thread(
                target=self._sync,
                args=(check_due, top_up_due),
                name="snapshot-sync",
                daemon=True,
            )
            worker = self._worker
        worker.start()
        return True

    def wait(self, timeout: float | None = None) -> bool:
        """Block until the running sync ends (tests). True when idle."""
        worker = self._worker
        if worker is not None:
            worker.join(timeout)
        return not self.status().running

    def _install(self, snapshot: Snapshot) -> None:
        with self._lock:
            self._snapshot = snapshot
            self._version += 1

    def _adopt_stored_snapshot(self) -> None:
        token = self._remote_token()
        with self._lock:
            served = self._snapshot[1].get("snapshot_token")
        if token is not None and token != served:
            self._install(self._load())

    def _sync(self, check: bool, top_up: bool) -> None:
        error: str | None = None
        added = 0
        try:
            if check:
                self._adopt_stored_snapshot()
            if top_up:
                for _ in range(self._max_rounds):
                    with self._lock:
                        frame, meta = self._snapshot
                    result = self._top_up(frame, meta)
                    if result is None:
                        break
                    self._install((result.frame, result.meta))
                    added += result.added_rows
                    if not result.more_pending:
                        break
        except BaseException as exc:  # polars' PanicException is a BaseException
            error = f"{type(exc).__name__}: {exc}"[:300]
            LOGGER.exception("Snapshot sync failed: %s", exc)
        finally:
            now = self._clock()
            with self._lock:
                self._running = False
                self._checked_at = datetime.now(UTC)
                self._error = error
                self._added_rows = added
                if check:
                    self._next_check = now + self._recheck_seconds
                if top_up:
                    self._next_top_up = now + self._retry_seconds
