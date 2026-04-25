"""
web_smoke_test.py
=================

End-to-end verification of the dashboard **through the web seam**: REST
endpoints + WebSocket fan-out, using FastAPI's ``TestClient`` so we exercise
the exact same handlers a browser would hit without binding a port.

This complements ``smoke_test.py`` (which talks to ``LiveEngine`` in-process
and therefore can't catch bugs in ``server.py``, the WS event shapes, the
fan-out batching, or the pydantic request models).

Run:
    python -m dashboard.live.web_smoke_test
"""
from __future__ import annotations

import json
import logging
import sys
import threading
import time
from collections import Counter, deque
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("dashboard.live.web_smoke_test")

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fastapi.testclient import TestClient  # noqa: E402

from dashboard.live import server as srv  # noqa: E402


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _banner(msg: str) -> None:
    print()
    print("=" * 72)
    print(f"  {msg}")
    print("=" * 72)


class WSListener:
    """Background WS reader. starlette's TestClient websocket session uses
    blocking ``receive_text()`` (no timeout kwarg), so we drain it from a
    thread and stash frames in a deque that the main thread polls at its
    own cadence. ``stop()`` closes the websocket, which unblocks the final
    ``receive_text`` with a disconnect."""

    def __init__(self, client: TestClient):
        self.client = client
        self.frames: deque[dict] = deque()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._err: Exception | None = None
        self._ws = None
        self._ready = threading.Event()

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        # Wait until the WS session is actually open before returning
        self._ready.wait(timeout=5.0)

    def _run(self) -> None:
        try:
            with self.client.websocket_connect("/ws") as ws:
                self._ws = ws
                self._ready.set()
                while True:
                    try:
                        text = ws.receive_text()  # blocks until a frame or disconnect
                    except Exception:
                        break
                    try:
                        frame = json.loads(text)
                    except json.JSONDecodeError:
                        continue
                    self.frames.append(frame)
                    if self._stop.is_set():
                        break
        except Exception as exc:
            self._err = exc
        finally:
            self._ready.set()

    def stop(self) -> None:
        self._stop.set()
        try:
            if self._ws is not None:
                self._ws.close()
        except Exception:
            pass
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if self._err is not None:
            log.warning("WSListener error: %r", self._err)

    def drain(self) -> list[dict]:
        out = []
        while self.frames:
            out.append(self.frames.popleft())
        return out


def _rolling_accuracy(samples: deque) -> float:
    return sum(samples) / len(samples) if samples else 0.0


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> int:
    csv = REPO_ROOT / "simu5g_real_simulation_results.csv"
    if not csv.exists():
        print(f"missing CSV: {csv}", file=sys.stderr)
        return 1

    # Fresh process state — wipe any stale engine from a previous import
    srv.engine = None

    # TestClient's context manager fires FastAPI startup/shutdown, which
    # builds the engine, launches the fan-out task, and tears them down.
    with TestClient(srv.app) as client:
        # Sanity: status endpoint reachable before start
        r = client.get("/api/status")
        assert r.status_code == 200, f"status pre-start: {r.status_code}"
        log.info("GET /api/status pre-start: running=%s", r.json().get("running"))

        # Open WS *before* start so we don't miss early warmup-guard frames
        ws = WSListener(client)
        ws.start()
        time.sleep(0.2)

        # Start the engine
        r = client.post("/api/control", json={"action": "start"})
        assert r.status_code == 200, f"start: {r.status_code} {r.text}"
        assert r.json()["ok"] is True

        # Rate sanity (doesn't affect correctness, just exercises the path)
        r = client.post("/api/rate", json={"rate_hz": 60.0})
        assert r.status_code == 200, f"rate: {r.status_code}"
        assert abs(r.json()["rate_hz"] - 60.0) < 1e-6

        # Mode endpoint exercise — AUTO variant + golden NDT on
        r = client.post("/api/mode", json={
            "preferred_variant":   "AUTO",
            "closed_loop_enabled": False,
            "use_golden_ndt":      True,
        })
        assert r.status_code == 200, f"mode: {r.status_code} {r.text}"
        mode_snap = r.json()["injection"]
        assert mode_snap["preferred_variant"] is None, f"AUTO should map to None, got {mode_snap}"
        assert mode_snap["use_golden_ndt"] is True

        # Negative path: invalid variant must be rejected
        r = client.post("/api/mode", json={"preferred_variant": "BOGUS"})
        assert r.status_code == 400, f"invalid variant should 400, got {r.status_code}"

        # rolling accuracy window + event counters
        acc_window: deque[int] = deque(maxlen=200)
        event_counts: Counter = Counter()
        retrain_events: list[dict] = []
        warmup_skips = 0
        total_samples = 0

        def consume(window_s: float, label: str) -> dict:
            nonlocal warmup_skips, total_samples
            t0 = time.time()
            while time.time() - t0 < window_s:
                for frame in ws.drain():
                    t = frame.get("type")
                    event_counts[t] += 1
                    if t == "samples":
                        for it in frame.get("items", []) or []:
                            acc_window.append(int(it.get("correct", 0)))
                            total_samples += 1
                    elif t == "sample":   # shouldn't fire (server coalesces), but be safe
                        acc_window.append(int(frame.get("correct", 0)))
                        total_samples += 1
                    elif t == "retrain_done":
                        retrain_events.append(frame)
                        msg = str(frame.get("message", ""))
                        if "warmup guard" in msg:
                            warmup_skips += 1
                time.sleep(0.1)
            return {
                "phase":        label,
                "acc":          _rolling_accuracy(acc_window),
                "n_window":     len(acc_window),
                "total":        total_samples,
                "warmup_skips": warmup_skips,
            }

        try:
            # ── PHASE 1 — warmup + baseline ─────────────────────────────
            _banner("PHASE 1 — warmup + baseline (~20s, no injection)")
            s = consume(20.0, "baseline")
            print(f"  acc(last 200)={s['acc']:.3f}  total_seen={s['total']}  "
                  f"warmup_retrains_skipped={s['warmup_skips']}")
            assert s["total"] >= 400, f"expected ≥400 samples in 20s, got {s['total']}"
            assert s["acc"] >= 0.80, f"baseline acc too low: {s['acc']:.3f}"

            # ── PHASE 2 — drift injection via /api/inject ────────────────
            _banner("PHASE 2 — inject drift (sinr -10dB, delay +20ms, 10s)")
            r = client.post("/api/inject", json={
                "sinr_bias_db":  -10.0,
                "delay_bias_ms":  20.0,
            })
            assert r.status_code == 200, f"inject: {r.status_code} {r.text}"
            inj_snap = r.json()["injection"]
            assert inj_snap["sinr_bias_db"] == -10.0
            assert inj_snap["delay_bias_ms"] == 20.0
            s = consume(10.0, "drift")
            print(f"  acc(last 200)={s['acc']:.3f}  total_seen={s['total']}  "
                  f"warmup_retrains_skipped={s['warmup_skips']}")

            # ── PHASE 3 — manual retrain via /api/control ────────────────
            _banner("PHASE 3 — force manual retrain, wait 15s for MTP + deploy")
            r = client.post("/api/control", json={"action": "force_retrain"})
            assert r.status_code == 200, f"force_retrain: {r.status_code}"
            s = consume(15.0, "post-retrain")
            print(f"  acc(last 200)={s['acc']:.3f}  total_seen={s['total']}  "
                  f"warmup_retrains_skipped={s['warmup_skips']}")

            # ── PHASE 4 — clear injection ────────────────────────────────
            _banner("PHASE 4 — clear injection, 10s recovery watch")
            r = client.post("/api/inject", json={
                "sinr_bias_db":  0.0,
                "delay_bias_ms": 0.0,
            })
            assert r.status_code == 200, f"clear inject: {r.status_code}"
            s = consume(10.0, "recovered")
            print(f"  acc(last 200)={s['acc']:.3f}  total_seen={s['total']}  "
                  f"warmup_retrains_skipped={s['warmup_skips']}")

            # ── Final status via REST ────────────────────────────────────
            r = client.get("/api/status")
            assert r.status_code == 200
            final = r.json()
            print("\nfinal engine status (via /api/status):")
            print(f"  running={final['running']} step={final['step']} "
                  f"acc_all={final['accuracy']:.3f}")

            # ── Event summary ────────────────────────────────────────────
            _banner("WS EVENT SUMMARY")
            for k, v in sorted(event_counts.items(), key=lambda kv: -kv[1]):
                print(f"  {k:<24} {v:>6}")

            _banner("RETRAIN EVENTS")
            for ev in retrain_events:
                print(f"  step={ev.get('step'):>5}  "
                      f"{str(ev.get('status')):<9} "
                      f"variant={str(ev.get('variant')):<9} "
                      f"deployed={ev.get('deployed')} "
                      f"ndt_gt={ev.get('ndt_gt')} "
                      f"dur={ev.get('duration_s')} :: "
                      f"{ev.get('message', '')}")

            # ── Assertions on the web seam ───────────────────────────────
            # (1) Sample frames actually flow through WS (coalesced "samples"
            #     frames, never raw "sample" frames since the fanout batches).
            assert event_counts["samples"] > 0, "no 'samples' frames received on WS"
            assert event_counts.get("sample", 0) == 0, \
                f"server should coalesce 'sample' into 'samples', saw {event_counts.get('sample')}"
            # (2) At least one retrain cycle completed (the manual one)
            assert len(retrain_events) >= 1, "no retrain_done frames received"
            # (3) The manual retrain should have deployed (non-warmup, gt>>base)
            deployed = [ev for ev in retrain_events if ev.get("deployed")]
            assert deployed, "manual retrain never deployed via web path"
            # (4) Final accuracy sane
            assert final["accuracy"] >= 0.80, f"final acc too low: {final['accuracy']:.3f}"
            # (5) Status frame greeted us on WS open (server.py:237)
            assert event_counts.get("status", 0) >= 1, "no 'status' greeting on WS connect"

            # ── Closed-loop decay-cancel wire test ───────────────────────
            # (cheap check: flip closed_loop on, ensure /api/mode round-trips)
            r = client.post("/api/mode", json={"closed_loop_enabled": True})
            assert r.status_code == 200
            assert r.json()["injection"]["closed_loop_enabled"] is True

            print("\n" + "=" * 72)
            print("  ALL WEB-SEAM ASSERTIONS PASSED")
            print("=" * 72)
            return 0

        finally:
            # Best-effort teardown
            ws.stop()
            try:
                client.post("/api/control", json={"action": "stop"})
            except Exception:
                pass


if __name__ == "__main__":
    raise SystemExit(main())
