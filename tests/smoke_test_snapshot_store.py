"""Offline check of the shared snapshot holder behind the dashboard.

Drives `SnapshotStore` with in-memory fakes and a fake clock: the first load
happens once, a top-up is served to every later caller (never lost), only
one sync runs however many sessions ask, failures are reported without
leaving the store stuck, the cooldown holds, a newer stored snapshot is
adopted, and a long catch-up proceeds in rounds. No Streamlit, no network.
"""

import logging
import sys
import threading

from snapshot_store import SnapshotStore, TopUp


def check(condition: bool, label: str) -> bool:
    print(f"{'[ok]  ' if condition else '[FAIL]'} {label}")
    return condition


class Harness:
    """A store wired to counters and a controllable clock."""

    def __init__(self, top_up, remote_token=lambda: None):
        self.now = 1000.0
        self.loads = 0
        self.stored = (["day1"], {"snapshot_token": "g1"})
        self.store = SnapshotStore(
            load=self._load,
            top_up=top_up,
            remote_token=remote_token,
            recheck_seconds=300,
            retry_seconds=900,
            clock=lambda: self.now,
        )

    def _load(self):
        self.loads += 1
        return self.stored


def check_load_once() -> bool:
    harness = Harness(top_up=lambda frame, meta: None)
    first = harness.store.current()
    second = harness.store.current()
    return check(
        harness.loads == 1 and first is second and harness.store.status().version == 1,
        "the snapshot is loaded once and shared",
    )


def check_top_up_is_durable() -> bool:
    calls = []

    def top_up(frame, meta):
        calls.append(list(frame))
        return TopUp(frame + ["day2"], {"snapshot_token": "g2"}, added_rows=7)

    harness = Harness(top_up)
    harness.store.current()
    started = harness.store.request_sync()
    idle = harness.store.wait(5)
    frames = [harness.store.current()[0] for _ in range(3)]
    status = harness.store.status()
    return check(
        started and idle and calls == [["day1"]]
        and all(frame == ["day1", "day2"] for frame in frames)
        and harness.loads == 1
        and status.version == 2 and status.added_rows == 7 and status.error is None,
        "a top-up is served to every later rerun and session",
    )


def check_single_flight() -> bool:
    gate = threading.Event()
    calls = []

    def top_up(frame, meta):
        calls.append(1)
        gate.wait(5)
        return TopUp(frame + ["day2"], dict(meta), added_rows=1)

    harness = Harness(top_up)
    harness.store.current()
    answers = []
    visitors = [
        threading.Thread(target=lambda: answers.append(harness.store.request_sync()))
        for _ in range(5)
    ]
    for visitor in visitors:
        visitor.start()
    for visitor in visitors:
        visitor.join()
    running = harness.store.status().running
    gate.set()
    harness.store.wait(5)
    return check(
        answers.count(True) == 1 and len(calls) == 1 and running,
        "five simultaneous visitors start exactly one fetch",
    )


def check_cooldown() -> bool:
    calls = []

    def top_up(frame, meta):
        calls.append(1)
        return None

    harness = Harness(top_up)
    harness.store.current()
    harness.store.request_sync()
    harness.store.wait(5)
    immediately = harness.store.request_sync()
    harness.now += 899
    harness.store.request_sync()  # may recheck the stored copy; must not fetch
    harness.store.wait(5)
    fetches_inside_cooldown = len(calls)
    harness.now += 2
    harness.store.request_sync()
    harness.store.wait(5)
    return check(
        not immediately and fetches_inside_cooldown == 1 and len(calls) == 2,
        "the DLD API is asked again only after the cooldown",
    )


def check_failures_are_reported() -> bool:
    class Panic(BaseException):
        """Stands in for polars' PanicException (not an Exception)."""

    outcomes = iter([RuntimeError("gateway 502"), Panic("rust panic")])

    def top_up(frame, meta):
        raise next(outcomes)

    harness = Harness(top_up)
    before = harness.store.current()
    store_log = logging.getLogger("snapshot_store")
    store_log.disabled = True  # the two failures below are expected
    try:
        harness.store.request_sync()
        harness.store.wait(5)
        first = harness.store.status()
        harness.now += 901
        harness.store.request_sync()
        harness.store.wait(5)
        second = harness.store.status()
    finally:
        store_log.disabled = False
    return check(
        not first.running and "gateway 502" in (first.error or "")
        and not second.running and "rust panic" in (second.error or "")
        and harness.store.current() is before and second.version == 1,
        "a failed sync is reported, keeps the old data and never sticks",
    )


def check_adopts_stored_snapshot() -> bool:
    token = {"value": "g1"}
    harness = Harness(top_up=lambda frame, meta: None, remote_token=lambda: token["value"])
    harness.store.current()
    early = harness.store.request_sync(top_up=False)
    harness.now += 301
    unchanged = harness.store.request_sync(top_up=False)
    harness.store.wait(5)
    loads_when_unchanged = harness.loads
    token["value"] = "g2"
    harness.stored = (["day1", "day2", "day3"], {"snapshot_token": "g2"})
    harness.now += 301
    changed = harness.store.request_sync(top_up=False)
    harness.store.wait(5)
    return check(
        not early and unchanged and loads_when_unchanged == 1
        and changed and harness.loads == 2
        and harness.store.current()[0] == ["day1", "day2", "day3"],
        "a snapshot written elsewhere is adopted, an unchanged one is not reloaded",
    )


def check_catch_up_rounds() -> bool:
    served = []

    def top_up(frame, meta):
        served.append(len(frame))
        if len(frame) >= 4:
            return None
        return TopUp(frame + ["more"], dict(meta), added_rows=10, more_pending=len(frame) < 3)

    harness = Harness(top_up)
    harness.store.current()
    harness.store.request_sync()
    harness.store.wait(5)
    status = harness.store.status()
    return check(
        served == [1, 2, 3] and len(harness.store.current()[0]) == 4
        and status.added_rows == 30 and status.version == 4,
        "a long catch-up proceeds in rounds within one sync",
    )


def check_no_sync_before_load() -> bool:
    harness = Harness(top_up=lambda frame, meta: None)
    return check(
        harness.store.request_sync() is False and not harness.store.loaded,
        "nothing is fetched before the snapshot has been loaded",
    )


def main() -> int:
    ok = check_load_once()
    ok &= check_top_up_is_durable()
    ok &= check_single_flight()
    ok &= check_cooldown()
    ok &= check_failures_are_reported()
    ok &= check_adopts_stored_snapshot()
    ok &= check_catch_up_rounds()
    ok &= check_no_sync_before_load()
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
