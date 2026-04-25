"""
detectors/reset.py — DetectorResetCoordinator

Purpose
-------
When the AIF rolls back from MLIN to MLIO (because the active model has
been poisoned, or simply performs worse), the four detectors (CDD, DDD,
DPD, CPD) hold internal state that was accumulated while the poisoned
MLIN was producing predictions:

* ``CDD``'s Page-Hinkley cumulative sum tracks residuals against the
  poisoned model's predictions — restoring MLIO without resetting PH
  leaves those residuals driving the detector into an immediate
  re-fire the moment a single check interval elapses.
* ``DDD``'s KS reference window keeps pointing at rows the poisoned
  model was producing predictions for (and whose LOB entries were
  therefore poisoned).
* ``DPD``'s IsolationForest was fit on a window contaminated by the
  poisoned model's influence on downstream state.
* ``CPD``'s shadow model was possibly refit *after* the poisoned MLIN
  went ACTIVE, bound to either the post-attack buffers or (in the
  hardened path) a trusted snapshot. In the former case the shadow
  is useless; in the latter it's fine — but the output-distribution
  and correlation references are still stamped with post-poisoning
  statistics.

The evidence from ``dashboard_live.log`` (79 rollbacks in 30 s, CPD
re-fires within 2-5 s of every rollback) is the live symptom of this
bug: the detector state carries enough poison residue to immediately
re-trigger on the restored-and-innocent MLIO.

Strategy
--------
Per-slot snapshot-and-restore. Whenever ``RTP.notify_model_updated``
installs a new MLIN, the RTP asks this coordinator to capture a
snapshot of all four detectors keyed by the new slot's ``slot_id``.
When ``AIF.rollback()`` re-activates an older slot, the coordinator
looks up the snapshot previously stamped against that slot's id and
restores each detector to the state it held at the moment that older
slot was last the ACTIVE one.

If no snapshot exists for the slot being re-activated (edge case:
rollback fires before any successful deploy has completed) the
coordinator emits a ``DETECTOR_RESET_FAILED`` marker via the event
callback and leaves the detectors untouched. The RTP then records
that a refit is owed on the next available clean window — the
current behaviour degrades gracefully rather than crashing.

Cache semantics
---------------
The snapshot cache is bounded (default N=3) so a long-running process
that repeatedly deploys new models does not grow unbounded. The cache
is FIFO on slot_id, so the OLDEST snapshot is evicted first — that is
what you want: a rollback always targets the MOST-RECENT past slot,
and anything older is lost but still correctly handled via the
``DETECTOR_RESET_FAILED`` path.
"""
from __future__ import annotations

import logging
from collections import OrderedDict
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# DetectorResetCoordinator
# ---------------------------------------------------------------------------

class DetectorResetCoordinator:
    """
    Owns the per-slot detector snapshot cache and dispatches snapshot /
    restore across the four detectors.

    Parameters
    ----------
    ddd, dpd, cdd, cpd
        The live detector instances RTP constructed. The coordinator
        keeps references to them and calls ``snapshot_state`` /
        ``restore_state`` on each in turn.
    cache_size : int
        Maximum number of snapshots to retain. Default 3: covers the
        "typical" rollback pattern (deploy → poison → rollback →
        deploy → poison → rollback) without retaining unbounded
        history on a long-running process.
    event_cb : callable, optional
        Invoked as ``event_cb(event_type: str, payload: dict)`` to
        surface coordinator-level events (``DETECTOR_RESET`` for
        success, ``DETECTOR_RESET_FAILED`` for the missing-snapshot
        case) back to the RTP's event log.
    """

    # Detector names surfaced in event payloads — keep stable so
    # auditors can filter on them.
    DETECTOR_NAMES = ("cdd", "ddd", "dpd", "cpd")

    def __init__(
        self,
        *,
        ddd: Any,
        dpd: Any,
        cdd: Any,
        cpd: Any,
        cache_size: int = 3,
        event_cb: Optional[Callable[[str, dict], None]] = None,
    ) -> None:
        if cache_size < 1:
            raise ValueError(
                f"DetectorResetCoordinator: cache_size must be >= 1 "
                f"(got {cache_size})"
            )
        self._detectors = {
            "ddd": ddd,
            "dpd": dpd,
            "cdd": cdd,
            "cpd": cpd,
        }
        self._cache_size = int(cache_size)
        # OrderedDict keyed by slot_id so we can evict the oldest entry
        # via ``popitem(last=False)`` when the cache is full.
        self._snapshots: "OrderedDict[int, dict]" = OrderedDict()
        self._event_cb = event_cb

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def capture(self, slot_id: int) -> None:
        """
        Take a snapshot of all four detectors and cache it under
        ``slot_id``. Call this from ``RTP.notify_model_updated`` right
        after the new model has been promoted to ACTIVE and the
        detectors have been re-baselined against it.

        If ``slot_id`` already has a snapshot in the cache it is
        overwritten (re-deploy of the same slot is unusual but
        behaviourally correct: the latest baseline is what a later
        rollback should restore to).
        """
        snap: dict[str, dict] = {}
        for name, det in self._detectors.items():
            try:
                snap[name] = det.snapshot_state()
            except Exception as exc:  # pragma: no cover - defensive
                logger.error(
                    "DetectorResetCoordinator.capture: %s.snapshot_state() "
                    "raised %r; using empty state for this detector.",
                    name, exc,
                )
                snap[name] = {}

        # Refresh LRU ordering by removing then re-inserting at the end.
        if slot_id in self._snapshots:
            del self._snapshots[slot_id]
        self._snapshots[slot_id] = snap

        # Bounded cache — evict from the front (oldest entry).
        while len(self._snapshots) > self._cache_size:
            evicted_id, _ = self._snapshots.popitem(last=False)
            logger.debug(
                "DetectorResetCoordinator: evicted snapshot for slot_id=%d "
                "(cache_size=%d).",
                evicted_id, self._cache_size,
            )

        logger.info(
            "DetectorResetCoordinator: captured snapshot for slot_id=%d "
            "(cached=%d/%d).",
            slot_id, len(self._snapshots), self._cache_size,
        )

    def restore(self, slot_id: int, source: str = "rollback") -> bool:
        """
        Restore all four detectors to the state previously captured for
        ``slot_id``.

        Returns ``True`` when the restore succeeded, ``False`` when no
        snapshot exists for the requested slot. In the failure case the
        caller is expected to fall back to a full detector re-fit on the
        next available clean window.

        ``source`` is logged in the emitted event; callers typically
        pass ``"rollback"`` but alternative paths (e.g. explicit
        operator request) can label their reset differently.
        """
        snap = self._snapshots.get(slot_id)
        if snap is None:
            logger.warning(
                "DetectorResetCoordinator.restore: no snapshot for "
                "slot_id=%d (have %s). Detector state will NOT be "
                "rolled back — caller must refit.",
                slot_id, sorted(self._snapshots.keys()),
            )
            if self._event_cb is not None:
                try:
                    self._event_cb("DETECTOR_RESET_FAILED", {
                        "slot_id": int(slot_id),
                        "source": source,
                        "available_snapshots": sorted(self._snapshots.keys()),
                        "reason": "no_snapshot_for_slot",
                    })
                except Exception as exc:  # pragma: no cover - defensive
                    logger.error(
                        "DetectorResetCoordinator: event_cb raised %r on "
                        "DETECTOR_RESET_FAILED.",
                        exc,
                    )
            return False

        restored: list[str] = []
        for name, det in self._detectors.items():
            state = snap.get(name, {})
            try:
                det.restore_state(state)
                restored.append(name)
            except Exception as exc:  # pragma: no cover - defensive
                logger.error(
                    "DetectorResetCoordinator.restore: %s.restore_state() "
                    "raised %r; detector left in current state.",
                    name, exc,
                )

        logger.warning(
            "DetectorResetCoordinator: restored %d/%d detectors for "
            "slot_id=%d (source=%s).",
            len(restored), len(self._detectors), slot_id, source,
        )
        if self._event_cb is not None:
            try:
                self._event_cb("DETECTOR_RESET", {
                    "slot_id": int(slot_id),
                    "source": source,
                    "detectors_restored": restored,
                })
            except Exception as exc:  # pragma: no cover - defensive
                logger.error(
                    "DetectorResetCoordinator: event_cb raised %r on "
                    "DETECTOR_RESET.",
                    exc,
                )
        return True

    # ------------------------------------------------------------------
    # Introspection (mostly for tests)
    # ------------------------------------------------------------------

    def has_snapshot(self, slot_id: int) -> bool:
        return slot_id in self._snapshots

    @property
    def cache_size(self) -> int:
        return self._cache_size

    @property
    def cached_slot_ids(self) -> list[int]:
        """Most-recent-last list of slot ids currently cached."""
        return list(self._snapshots.keys())

    def __repr__(self) -> str:
        return (
            f"DetectorResetCoordinator(cached={len(self._snapshots)}"
            f"/{self._cache_size}, slots={list(self._snapshots.keys())})"
        )
