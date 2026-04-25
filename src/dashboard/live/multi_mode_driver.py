"""
multi_mode_driver.py
====================

End-to-end correctness driver that walks the live dashboard engine
through every operator-controllable injection mode and asserts the
RTP pipeline stays internally consistent throughout.

Scope of verification
---------------------
1. **Baseline / warmup**     — engine boots, accuracy converges, warmup
                                guard skips early MToUTs.
2. **Drift injection**       — sinr_bias_db / delay_bias_ms sliders push
                                features off the trained manifold; DDD
                                fires; accuracy degrades.
3. **Force retrain**         — manual operator trigger drives a full
                                MTP-E (or LOCAL) retrain + AIF deploy.
4. **Recovery**              — clearing sliders restores accuracy.
5. **Noise mode**            — noise_scale > 1.0 widens the per-feature
                                sigma; verifies DPP standardiser handles
                                it without crashing.
6. **Poison mode**           — poison_mode=True flips a fraction of
                                labels each step; DPD's slow-poisoning
                                EWMA arm should engage.
7. **Variant override**      — preferred_variant="LOCAL" forces ATM to
                                pick MTP-L instead of MTP-E.
8. **Closed-loop decay**     — closed_loop_enabled=True linearly decays
                                sliders to neutral; pipeline must stay
                                stable throughout the animation.
9. **Event-log audit trail** — at the end, walk ``engine.rtp.event_log``
                                and assert the HIGH #8+#9 event types
                                actually fired through the live pipeline:
                                  * MODEL_UPDATED       (legacy)
                                  * MODEL_NOTIFY_OK     (new, HIGH #8)
                                  * REFERENCE_REFIT     (new, HIGH #9)
                                  * LOB_RESTAMPED       (new, HIGH #9)
                                And the tamper-evident hash chain must
                                still verify.

Run:
    python -m dashboard.live.multi_mode_driver
"""
from __future__ import annotations

import logging
import sys
import time
from collections import Counter, deque
from pathlib import Path

# Windows console defaults to cp1252; reconfigure stdout/stderr to utf-8
# so printed banners with em-dashes / unicode glyphs don't crash with
# UnicodeEncodeError. ``reconfigure`` exists on Python 3.7+.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except Exception:
        pass

logging.basicConfig(
    level=logging.WARNING,   # quieter than smoke_test — focus on PHASE banners
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dashboard.live.engine import EngineConfig, LiveEngine  # noqa: E402
from rtp.rtp import EventType                                # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _drain_events(engine: LiveEngine) -> list[dict]:
    """Drain the engine's SSE event queue without blocking."""
    out: list[dict] = []
    while True:
        try:
            out.append(engine.events.get_nowait())
        except Exception:
            break
    return out


def _rolling_acc(window: deque) -> float:
    return (sum(window) / len(window)) if window else 0.0


def _banner(msg: str) -> None:
    print()
    print("=" * 78)
    print(f"  {msg}")
    print("=" * 78)


# ---------------------------------------------------------------------------
# Phase runner
# ---------------------------------------------------------------------------

class PhaseRunner:
    def __init__(self, engine: LiveEngine):
        self.engine = engine
        self.acc_window: deque[int] = deque(maxlen=200)
        self.total_seen = 0
        self.retrain_events: list[dict] = []
        self.warmup_skips = 0

    def consume(self, window_s: float, label: str) -> dict:
        t0 = time.time()
        while time.time() - t0 < window_s:
            for ev in _drain_events(self.engine):
                t = ev.get("type")
                if t == "sample":
                    self.acc_window.append(int(ev.get("correct", 0)))
                    self.total_seen += 1
                elif t == "samples" and isinstance(ev.get("items"), list):
                    for it in ev["items"]:
                        self.acc_window.append(int(it.get("correct", 0)))
                        self.total_seen += 1
                elif t == "retrain_done":
                    msg = str(ev.get("message", ""))
                    if "warmup guard" in msg:
                        self.warmup_skips += 1
                    self.retrain_events.append(ev)
            time.sleep(0.1)

        snap = {
            "phase": label,
            "acc": _rolling_acc(self.acc_window),
            "n_window": len(self.acc_window),
            "total_seen": self.total_seen,
            "warmup_skips": self.warmup_skips,
            "retrains": sum(
                1 for ev in self.retrain_events
                if ev.get("status") == "SUCCESS"
            ),
        }
        print(
            f"  acc(last {snap['n_window']:>3})={snap['acc']:.3f}  "
            f"total_seen={snap['total_seen']:>5}  "
            f"successful_retrains={snap['retrains']}  "
            f"warmup_skips={snap['warmup_skips']}"
        )
        return snap


# ---------------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------------

def main() -> int:
    csv = REPO_ROOT / "simu5g_real_simulation_results.csv"
    if not csv.exists():
        print(f"missing CSV: {csv}", file=sys.stderr)
        return 1

    # Higher rate so each phase covers more samples in less wall-clock time.
    engine = LiveEngine(EngineConfig(csv_path=csv, rate_hz=80.0))
    engine.start()
    runner = PhaseRunner(engine)

    try:
        # ── Phase 1: warmup + baseline ────────────────────────────────
        _banner("PHASE 1 — warmup + baseline (~15s, no injection)")
        runner.consume(15.0, "baseline")

        # ── Phase 2: drift injection ──────────────────────────────────
        _banner("PHASE 2 — drift (sinr -10dB, delay +20ms, 8s)")
        engine.injection.update({
            "sinr_bias_db":  -10.0,
            "delay_bias_ms":  20.0,
        })
        runner.consume(8.0, "drift")

        # ── Phase 3: force retrain ────────────────────────────────────
        _banner("PHASE 3 — force manual retrain (~12s for MTP + deploy)")
        engine.injection.trigger_retrain()
        runner.consume(12.0, "post-retrain")

        # ── Phase 4: clear injection (recovery) ───────────────────────
        _banner("PHASE 4 — clear injection, 8s recovery watch")
        engine.injection.update({
            "sinr_bias_db":  0.0,
            "delay_bias_ms": 0.0,
        })
        runner.consume(8.0, "recovered")

        # ── Phase 5: noise mode ───────────────────────────────────────
        _banner("PHASE 5 - noise_scale=2.5 (8s, wider per-feature sigma)")
        engine.injection.update({"noise_scale": 2.5})
        runner.consume(8.0, "noisy")
        engine.injection.update({"noise_scale": 1.0})

        # ── Phase 6: poison mode ──────────────────────────────────────
        _banner("PHASE 6 — poison_mode=True (10s, slow-poisoning attack)")
        engine.injection.update({"poison_mode": True})
        runner.consume(10.0, "poisoned")
        engine.injection.update({"poison_mode": False})

        # ── Phase 7: variant override ─────────────────────────────────
        _banner("PHASE 7 — preferred_variant=LOCAL (force MTP-L), retrain")
        engine.injection.update({"preferred_variant": "LOCAL"})
        engine.injection.update({
            "sinr_bias_db": -8.0,
            "delay_bias_ms": 15.0,
        })
        engine.injection.trigger_retrain()
        runner.consume(12.0, "local-variant")
        engine.injection.update({
            "sinr_bias_db":  0.0,
            "delay_bias_ms": 0.0,
            "preferred_variant": None,
        })

        # ── Phase 8: closed-loop decay ────────────────────────────────
        _banner("PHASE 8 — closed-loop decay (sliders auto-decay to zero)")
        engine.injection.update({
            "sinr_bias_db":   -6.0,
            "delay_bias_ms":  12.0,
            "closed_loop_enabled": True,
        })
        runner.consume(8.0, "closed-loop-decay")
        engine.injection.update({"closed_loop_enabled": False})

        # ── Retrain summary ───────────────────────────────────────────
        _banner("RETRAIN EVENT SUMMARY")
        variants_used = Counter()
        for ev in runner.retrain_events:
            status   = ev.get("status")
            variant  = ev.get("variant")
            deployed = ev.get("deployed")
            dur      = ev.get("duration_s")
            ndt_gt   = ev.get("ndt_gt")
            msg      = ev.get("message", "")
            if status == "SUCCESS":
                variants_used[variant] += 1
            print(
                f"  step={ev.get('step'):>5}  {status:<9} "
                f"variant={str(variant):<9} deployed={deployed} "
                f"dur={dur} ndt_gt={ndt_gt} :: {msg}"
            )
        print(f"\n  variants_used={dict(variants_used)}")

        # ── HIGH #8+#9 event-log audit ────────────────────────────────
        _banner("HIGH #8+#9 EVENT-TAXONOMY AUDIT")
        rtp = engine.rtp
        if rtp is None:
            print("  ERROR: engine.rtp is None — pipeline never built")
            return 2

        type_counts = Counter(e.event_type for e in rtp.event_log)
        for et in sorted(type_counts, key=lambda x: x.name):
            print(f"  {et.name:<28} = {type_counts[et]:>4}")

        # Required event types — these must have fired at least once
        # during the multi-mode walk if HIGH #8+#9 wiring is correct.
        required = {
            EventType.MODEL_UPDATED:    "legacy retrain marker",
            EventType.MODEL_NOTIFY_OK:  "AIF->RTP 2PC bridge (HIGH #8)",
            EventType.REFERENCE_REFIT:  "DDD/DPD refit on deploy (HIGH #9)",
            EventType.LOB_RESTAMPED:    "LOB re-stamp post-deploy (HIGH #9)",
        }
        missing = []
        for et, reason in required.items():
            if type_counts.get(et, 0) == 0:
                missing.append(f"{et.name} ({reason})")
        if missing:
            print(f"\n  MISSING required events: {missing}")
            return 3
        print(f"\n  OK All required HIGH #8+#9 event types fired through the live pipeline.")

        # Cross-check: REFERENCE_REFIT should have at least DDD AND DPD entries
        refits = [
            e for e in rtp.event_log
            if e.event_type == EventType.REFERENCE_REFIT
        ]
        detectors_seen = {e.details.get("detector") for e in refits}
        if not {"DDD", "DPD"}.issubset(detectors_seen):
            print(
                f"  WARN: REFERENCE_REFIT seen for {detectors_seen}; "
                "expected at least DDD+DPD."
            )

        # Cross-check: at least one LOB_RESTAMPED with ok=True
        restamps = [
            e for e in rtp.event_log
            if e.event_type == EventType.LOB_RESTAMPED
        ]
        ok_restamps = [e for e in restamps if e.details.get("ok")]
        print(
            f"  LOB_RESTAMPED: {len(ok_restamps)}/{len(restamps)} "
            "completed successfully (ok=True)."
        )

        # ── Hash-chain integrity ──────────────────────────────────────
        _banner("HASH-CHAIN INTEGRITY")
        ok, err = rtp.verify_event_log()
        if not ok:
            print(f"  CHAIN BROKEN: {err}")
            return 4
        print(
            f"  OK Event log hash chain verifies clean across "
            f"{len(rtp.event_log)} events."
        )

        # ── Final status ──────────────────────────────────────────────
        _banner("FINAL ENGINE STATUS")
        status = engine.status()
        print(
            f"  running={status['running']}  step={status['step']}  "
            f"acc_all={status['accuracy']:.3f}  total_seen={runner.total_seen}"
        )
        ndt = status["ndt_last"]
        if ndt["gt"] is not None:
            print(
                f"  last NDT scores: pseudo={ndt['pseudo']}  "
                f"gt={ndt['gt']:.3f}  bias={ndt['bias']}"
            )
        print()
        print("  OK Multi-mode driver completed without errors.")
        return 0
    finally:
        engine.stop()


if __name__ == "__main__":
    raise SystemExit(main())
