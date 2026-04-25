"""
sinr_drift_demo.py
==================

End-to-end demo of the AIMP self-healing loop.  Drives the live
dashboard's HTTP + WS API.

What you will see
-----------------
After we shock the network state:

    1. Detectors fire (DDD / CPD / CDD start triggering).
    2. MToUT is raised, ATM kicks off a retrain cycle.
    3. A new model deploys.
    4. Detectors GO QUIET against the new SINR distribution -- the
       point of self-healing is that the post-deploy regime IS the
       new normal.
    5. Accuracy stays high (the model adapts to the shifted KPIs).

Pure -26.5 dB SINR shift, on its own, does NOT make the model fail:
``derive_label_scalar`` uses ``sinr < 40 dB`` as the label rule, so
when the SINR drops every sample's label flips to 1 in lockstep with
the model's own prediction -- accuracy stays at 1.0 trivially and
nothing observable happens.  The only way to see the loop work is to
also push the joint feature distribution somewhere the trained model
hasn't seen, so DDD/CPD's KS / shadow tests light up.  We pair the
user's headline -26.5 dB SINR shock with a moderate noise_scale bump
(2.0) so KS-divergence on the joint distribution is unambiguous AND
the model has something it actually has to relearn.

Run it
------
Start a FRESH server (after the 4 QA fixes landed)::

    .venv\\Scripts\\python.exe -m dashboard.live --host 127.0.0.1 --port 8765

Then in a second shell::

    .venv\\Scripts\\python.exe dashboard/live/sinr_drift_demo.py
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
import urllib.error
import urllib.request
from collections import Counter
from typing import Any

import websockets


BASE = "http://127.0.0.1:8765"
WS = "ws://127.0.0.1:8765/ws"


# ---------------------------------------------------------------------------
# Async-friendly HTTP helpers (urllib offloaded to a thread so the asyncio
# loop stays free for the WS tap)
# ---------------------------------------------------------------------------

def _http_sync(method: str, path: str, body: dict | None,
               timeout: float) -> dict:
    data = None
    headers = {}
    if body is not None:
        data = json.dumps(body).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(
        BASE + path, data=data, method=method, headers=headers,
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


async def _http(method: str, path: str, body: dict | None = None,
                timeout: float = 60.0) -> dict:
    """
    Run urllib in a worker thread so the parent event loop can keep
    pumping WebSocket frames while we wait for the dashboard's reply.
    The first /api/status on a freshly booted server takes ~25 s while
    LiveEngine constructs (CSV load + initial model fit), hence the
    generous default timeout.
    """
    return await asyncio.to_thread(_http_sync, method, path, body, timeout)


async def status() -> dict:
    return await _http("GET", "/api/status")


async def control(action: str) -> dict:
    return await _http("POST", "/api/control", {"action": action})


async def inject(**patch) -> dict:
    return await _http("POST", "/api/inject", patch)


async def set_rate(hz: float) -> dict:
    return await _http("POST", "/api/rate", {"rate_hz": hz})


# ---------------------------------------------------------------------------
# Pretty printers
# ---------------------------------------------------------------------------

def _hdr(msg: str) -> None:
    print()
    print("=" * 78)
    print(f"  {msg}")
    print("=" * 78)


def _row(t_rel: float, step: int, acc: float, tag: str = "") -> None:
    print(f"  t={t_rel:6.1f}s  step={step:>6}  acc={acc:6.3f}   {tag}")


# ---------------------------------------------------------------------------
# Detector-frame helpers
# ---------------------------------------------------------------------------

def _detector_fired(frame: dict) -> tuple[bool, list[str]]:
    """Return (any_fired, [name_of_each_fired_detector])."""
    fired: list[str] = []
    for name in ("ddd", "dpd", "cdd", "cpd"):
        sub = frame.get(name) or {}
        if isinstance(sub, dict) and bool(sub.get("triggered")):
            fired.append(name.upper())
    return (len(fired) > 0, fired)


# ---------------------------------------------------------------------------
# WS event collector -- non-blocking, survives quiet periods
# ---------------------------------------------------------------------------

class EventTap:
    """
    Background coroutine that connects to /ws and stores every non-sample
    frame.  Designed to survive long quiet periods (ATM cycles take 5-15 s
    during which no detector frames are emitted) by treating recv timeouts
    as "no news" rather than "stop listening".
    """

    def __init__(self, t0: float) -> None:
        self.t0 = t0
        self.frames: list[tuple[float, dict]] = []
        self._stop = False
        self._connected = False

    async def run(self) -> None:
        try:
            async with websockets.connect(WS, max_size=2 ** 24) as ws:
                self._connected = True
                while not self._stop:
                    try:
                        raw = await asyncio.wait_for(ws.recv(), timeout=2.0)
                    except asyncio.TimeoutError:
                        # Quiet period -- keep listening rather than giving
                        # up.  The previous version exited the entire loop
                        # on the first timeout, which killed the tap as
                        # soon as the producer paused for 1.5 s.
                        continue
                    f = json.loads(raw)
                    t = f.get("type", "?")
                    # Drop the high-volume sample batches.
                    if t == "samples":
                        continue
                    self.frames.append((time.monotonic() - self.t0, f))
        except Exception as exc:    # pragma: no cover - diagnostic only
            print(f"  WS tap error: {exc!r}", file=sys.stderr)

    def stop(self) -> None:
        self._stop = True


# ---------------------------------------------------------------------------
# Main demo
# ---------------------------------------------------------------------------

async def amain(
    sinr_drop_db: float,
    noise_scale: float,
    delay_bump_ms: float,
    baseline_s: float,
    drift_s: float,
) -> int:
    # -- 0. Pre-flight ---------------------------------------------------
    try:
        _ = await status()
    except urllib.error.URLError as exc:
        print(
            f"ERROR: cannot reach {BASE} -- start the server first.\n"
            f"       .venv\\Scripts\\python.exe -m dashboard.live "
            f"--host 127.0.0.1 --port 8765\n"
            f"detail: {exc}",
            file=sys.stderr,
        )
        return 2

    _hdr("STEP 1 -- Reset to a clean baseline")
    await inject(sinr_bias_db=0.0, rsrp_bias_db=0.0, delay_bias_ms=0.0,
                 tput_scale=1.0, noise_scale=1.0, poison_mode=False)
    await set_rate(60.0)
    if not (await status()).get("running"):
        await control("start")
        await asyncio.sleep(0.5)
    s0 = await status()
    print(f"  injection cleared, rate=60 Hz, running={s0['running']}, "
          f"step={s0['step']}, acc={s0['accuracy']:.3f}")

    # -- 1. Spin up the WS tap ------------------------------------------
    t0 = time.monotonic()
    tap = EventTap(t0)
    tap_task = asyncio.create_task(tap.run())
    # Give the tap a moment to actually connect before we start counting.
    for _ in range(20):
        if tap._connected:
            break
        await asyncio.sleep(0.05)

    # -- 2. Baseline ----------------------------------------------------
    _hdr(f"STEP 2 -- Capture baseline ({baseline_s:.0f}s, no injection)")
    baseline_acc: list[float] = []
    while time.monotonic() - t0 < baseline_s:
        st = await status()
        baseline_acc.append(float(st.get("accuracy", 0.0)))
        _row(time.monotonic() - t0, st["step"], st["accuracy"], "baseline")
        await asyncio.sleep(2.0)
    baseline_mean = sum(baseline_acc) / max(len(baseline_acc), 1)
    print(f"  baseline mean accuracy = {baseline_mean:.3f}")

    # -- 3. Inject the KPI shock ---------------------------------------
    _hdr(
        f"STEP 3 -- Inject KPI shock: "
        f"sinr_bias_db={sinr_drop_db:+.1f}  "
        f"noise_scale={noise_scale}  delay_bias_ms={delay_bump_ms:+.1f}"
    )
    inj_t = time.monotonic() - t0
    r = await inject(
        sinr_bias_db=sinr_drop_db,
        noise_scale=noise_scale,
        delay_bias_ms=delay_bump_ms,
    )
    print(f"  echo: sinr={r['injection']['sinr_bias_db']:+.2f} dB  "
          f"noise={r['injection']['noise_scale']}  "
          f"delay={r['injection']['delay_bias_ms']:+.1f} ms")
    print(f"  t={inj_t:.1f}s  <- shock applied; stays on for the whole run")

    # -- 4. Watch the loop work ----------------------------------------
    _hdr(f"STEP 4 -- Observe drift -> detect -> retrain -> deploy ({drift_s:.0f}s)")
    drift_acc: list[tuple[float, int, float]] = []
    deploy_t0_rel: float | None = None       # first post-injection deploy
    deploys_seen = 0
    end_t = time.monotonic() + drift_s
    while time.monotonic() < end_t:
        st = await status()
        rel = time.monotonic() - t0
        drift_acc.append((rel, int(st["step"]), float(st["accuracy"])))

        # Tally deploys / mtouts / detector fires from the WS tap so we
        # have something to show in real time without depending on a
        # status payload that doesn't expose ATM counters.
        # The engine's 'retrain_done' frame carries the outcome inside the
        # ``message`` string ("ATM: model deployed via MTP-E ... ndt=pass"
        # vs "ATM: NDT validation failed for candidate model. Keeping
        # current MLIN.").  No structured top-level ``ndt`` field exists,
        # so we scan the message text rather than a dedicated field.
        deploys_now = sum(
            1 for ts, f in tap.frames
            if ts >= inj_t and f.get("type") == "retrain_done"
            and "model deployed" in str(f.get("message", "")).lower()
        )
        if deploys_now > deploys_seen:
            if deploy_t0_rel is None:
                deploy_t0_rel = rel
            deploys_seen = deploys_now

        mtouts = sum(
            1 for ts, f in tap.frames
            if ts >= inj_t and f.get("type") == "mtout"
        )
        det_fires = sum(
            1 for ts, f in tap.frames
            if ts >= inj_t and f.get("type") == "detector"
            and _detector_fired(f)[0]
        )
        tag = (
            f"detfires={det_fires:4d}  mtouts={mtouts:2d}  "
            f"deploys={deploys_seen}"
        )
        if deploy_t0_rel is not None and abs(rel - deploy_t0_rel) < 2.5:
            tag += "  ** NEW DEPLOY **"
        _row(rel, st["step"], st["accuracy"], tag)
        await asyncio.sleep(2.0)

    # -- 5. Stop WS tap -------------------------------------------------
    tap.stop()
    try:
        await asyncio.wait_for(tap_task, timeout=3.0)
    except asyncio.TimeoutError:
        tap_task.cancel()

    # -- 6. Build the pre/post-deploy detector-fire breakdown ----------
    pre_inj = post_inj_pre_dep = post_dep = 0
    pre_inj_kinds: Counter[str] = Counter()
    post_inj_pre_dep_kinds: Counter[str] = Counter()
    post_dep_kinds: Counter[str] = Counter()
    for ts, f in tap.frames:
        if f.get("type") != "detector":
            continue
        any_fired, kinds = _detector_fired(f)
        if not any_fired:
            continue
        if ts < inj_t:
            pre_inj += 1
            for k in kinds: pre_inj_kinds[k] += 1
        elif deploy_t0_rel is not None and ts >= deploy_t0_rel:
            post_dep += 1
            for k in kinds: post_dep_kinds[k] += 1
        else:
            post_inj_pre_dep += 1
            for k in kinds: post_inj_pre_dep_kinds[k] += 1

    # -- 7. Print the trimmed event timeline ---------------------------
    _hdr("STEP 5 -- State-transition timeline (interesting frames only)")
    counts: Counter[str] = Counter()
    timeline_types = {"mtout", "retrain_done", "reference_refit",
                      "retrain_start", "started", "stopped", "error",
                      "closed_loop_start", "closed_loop_end"}
    for ts, f in tap.frames:
        t = f.get("type", "?")
        counts[t] += 1
        if t not in timeline_types:
            continue
        payload_bits: list[str] = []
        for key in ("step", "variant", "ndt", "result", "ok",
                    "reasons", "detail", "message", "version"):
            if key in f:
                v = f[key]
                if isinstance(v, (dict, list)):
                    v = json.dumps(v, default=str)
                    if len(v) > 60:
                        v = v[:57] + "..."
                payload_bits.append(f"{key}={v}")
        print(f"  t={ts:6.1f}s  [{t:18s}] {'  '.join(payload_bits)}")

    # -- 8. Final report ------------------------------------------------
    _hdr("STEP 6 -- Summary")
    print(f"  Baseline mean accuracy        : {baseline_mean:.3f}")
    if drift_acc:
        lowest = min(a for _, _, a in drift_acc)
        idx_low = min(range(len(drift_acc)), key=lambda i: drift_acc[i][2])
        recovered = max((a for _, _, a in drift_acc[idx_low:]),
                        default=lowest)
        print(f"  Lowest accuracy after shock   : {lowest:.3f}  "
              f"(at t={drift_acc[idx_low][0]:.1f}s)")
        print(f"  Recovered accuracy (max)      : {recovered:.3f}")
        print(f"  Recovery gain                 : {recovered - lowest:+.3f}")

    print()
    print(f"  --- DETECTOR FIRES PER WINDOW (success = post-deploy << pre-deploy) ---")
    print(f"    (a) pre-injection          : {pre_inj:4d}  {dict(pre_inj_kinds)}")
    print(f"    (b) post-inj, pre-deploy   : {post_inj_pre_dep:4d}  "
          f"{dict(post_inj_pre_dep_kinds)}")
    if deploy_t0_rel is not None:
        print(f"    (c) post-deploy            : {post_dep:4d}  "
              f"{dict(post_dep_kinds)}")
        print(f"        first deploy landed at t={deploy_t0_rel:.1f}s")
    else:
        print(f"    (c) post-deploy            : N/A (no deploy occurred)")

    print()
    print(f"  WS frame counts               : {dict(counts)}")
    print(f"  Total WS frames captured      : {len(tap.frames)}")
    fin = await status()
    print(f"  Final step                    : {fin.get('step')}")
    print(f"  Final accuracy                : {fin.get('accuracy'):.3f}")
    print()
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sinr-drop", type=float, default=-26.5,
                    help="SINR bias in dB (default: -26.5)")
    ap.add_argument("--noise-scale", type=float, default=2.0,
                    help="Noise scale (default: 2.0 - decorrelates "
                         "predictions from labels so the model has "
                         "something to actually relearn)")
    ap.add_argument("--delay-bump", type=float, default=15.0,
                    help="Delay bias ms (default: 15.0)")
    ap.add_argument("--baseline", type=float, default=10.0,
                    help="Baseline window seconds (default: 10)")
    ap.add_argument("--drift", type=float, default=120.0,
                    help="Drift+recovery observation seconds (default: 120)")
    args = ap.parse_args()
    return asyncio.run(amain(
        args.sinr_drop, args.noise_scale, args.delay_bump,
        args.baseline, args.drift,
    ))


if __name__ == "__main__":
    raise SystemExit(main())
