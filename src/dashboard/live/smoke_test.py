"""
smoke_test.py
=============

Quick end-to-end verification of the dashboard engine.  Not a pytest —
just a runnable script that starts the engine, waits through warmup,
pokes the injection knobs, and prints per-phase rolling accuracy so we
can see whether the pseudo-label loop is actually closed.

Run:
    python -m dashboard.live.smoke_test
"""
from __future__ import annotations

import logging
import sys
import time
from collections import deque
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dashboard.live.engine import EngineConfig, LiveEngine  # noqa: E402


def _drain_events(engine: LiveEngine) -> list[dict]:
    out: list[dict] = []
    while True:
        try:
            out.append(engine.events.get_nowait())
        except Exception:
            break
    return out


def _rolling_accuracy(samples: deque) -> float:
    if not samples:
        return 0.0
    return sum(samples) / len(samples)


def _banner(msg: str) -> None:
    print()
    print("=" * 72)
    print(f"  {msg}")
    print("=" * 72)


def main() -> int:
    csv = REPO_ROOT / "simu5g_real_simulation_results.csv"
    if not csv.exists():
        print(f"missing CSV: {csv}", file=sys.stderr)
        return 1

    engine = LiveEngine(EngineConfig(csv_path=csv, rate_hz=60.0))
    engine.start()

    # rolling window of 'correct' flags for the accuracy metric shown live
    acc_window: deque[int] = deque(maxlen=200)
    retrain_events: list[dict] = []
    skipped_warmup = 0
    total_seen = 0

    def _consume(window_s: float, label: str) -> dict:
        nonlocal skipped_warmup, total_seen
        t0 = time.time()
        snapshots = []
        while time.time() - t0 < window_s:
            for ev in _drain_events(engine):
                t = ev.get("type")
                if t == "sample":
                    acc_window.append(int(ev.get("correct", 0)))
                    total_seen += 1
                elif t == "retrain_done":
                    msg = ev.get("message", "")
                    if "warmup guard" in str(msg):
                        skipped_warmup += 1
                    retrain_events.append(ev)
                elif t == "samples" and isinstance(ev.get("items"), list):
                    for it in ev["items"]:
                        acc_window.append(int(it.get("correct", 0)))
                        total_seen += 1
            time.sleep(0.1)
        snap = {
            "phase":       label,
            "acc":         _rolling_accuracy(acc_window),
            "n_window":    len(acc_window),
            "total_seen":  total_seen,
            "warmup_skips": skipped_warmup,
        }
        snapshots.append(snap)
        return snap

    try:
        _banner("PHASE 1 — warmup + baseline (~20s, no injection)")
        snap = _consume(20.0, "baseline")
        print(f"  acc(last 200)={snap['acc']:.3f}  total_seen={snap['total_seen']}  "
              f"warmup_retrains_skipped={snap['warmup_skips']}")

        _banner("PHASE 2 — inject drift (sinr -10dB, delay +20ms, 10s)")
        engine.injection.update({
            "sinr_bias_db":  -10.0,
            "delay_bias_ms":  20.0,
        })
        snap = _consume(10.0, "drift")
        print(f"  acc(last 200)={snap['acc']:.3f}  total_seen={snap['total_seen']}  "
              f"warmup_retrains_skipped={snap['warmup_skips']}")

        _banner("PHASE 3 — force manual retrain, wait 15s for MTP + deploy")
        engine.injection.trigger_retrain()
        snap = _consume(15.0, "post-retrain")
        print(f"  acc(last 200)={snap['acc']:.3f}  total_seen={snap['total_seen']}  "
              f"warmup_retrains_skipped={snap['warmup_skips']}")

        _banner("PHASE 4 — clear injection, 10s recovery watch")
        engine.injection.update({
            "sinr_bias_db":  0.0,
            "delay_bias_ms": 0.0,
        })
        snap = _consume(10.0, "recovered")
        print(f"  acc(last 200)={snap['acc']:.3f}  total_seen={snap['total_seen']}  "
              f"warmup_retrains_skipped={snap['warmup_skips']}")

        _banner("RETRAIN EVENTS")
        for ev in retrain_events:
            status   = ev.get("status")
            variant  = ev.get("variant")
            deployed = ev.get("deployed")
            msg      = ev.get("message", "")
            ndt_gt   = ev.get("ndt_gt")
            dur      = ev.get("duration_s")
            print(f"  step={ev.get('step'):>5}  {status:<9} "
                  f"variant={str(variant):<9} deployed={deployed} "
                  f"ndt_gt={ndt_gt} dur={dur} :: {msg}")

        status = engine.status()
        print(f"\nfinal engine status:")
        print(f"  running={status['running']} step={status['step']} "
              f"acc_all={status['accuracy']:.3f}")
        return 0
    finally:
        engine.stop()


if __name__ == "__main__":
    raise SystemExit(main())
