"""
server.py
=========

FastAPI app that owns one ``LiveEngine`` and exposes:

  * ``GET  /``              → SPA (index.html)
  * ``GET  /api/status``    → engine status snapshot
  * ``POST /api/control``   → {action: "start"|"pause"|"resume"|"stop"|"reset"|"force_retrain"}
  * ``POST /api/rate``      → {rate_hz: float}
  * ``POST /api/inject``    → partial update of InjectionState
  * ``WS   /ws``            → engine-event stream (samples, detectors, MToUT, retrainings)

A single background task drains the engine's event queue and fan-outs to all
connected WebSockets. Messages are JSON frames.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import threading
from pathlib import Path
from typing import Set

# orjson is 3-5x faster than stdlib json for numeric-heavy dicts and
# returns bytes directly (so the WebSocket layer can send_bytes without
# a UTF-8 encode pass).  Fall back to stdlib json if it's not installed
# so the dashboard still boots in a barebones venv.
try:
    import orjson  # type: ignore[import-not-found]

    def _dumps(obj: dict) -> bytes:
        return orjson.dumps(obj, default=str)

    _SEND_BYTES = True
except ImportError:  # pragma: no cover — orjson is in pyproject; this is belt+braces
    def _dumps(obj: dict) -> bytes:
        return json.dumps(obj, default=str).encode("utf-8")

    _SEND_BYTES = True

# Ensure project root is importable before the engine is loaded
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from dashboard.live.engine import EngineConfig, LiveEngine  # noqa: E402

log = logging.getLogger("dashboard.live.server")

STATIC_DIR = Path(__file__).resolve().parent / "static"
DEFAULT_CSV = REPO_ROOT / "simu5g_real_simulation_results.csv"

# ---------------------------------------------------------------------------
# App & global state
# ---------------------------------------------------------------------------

app = FastAPI(title="AIMP × Simu5G Live Dashboard")

engine: LiveEngine | None = None
clients: Set[WebSocket] = set()
_fanout_task: asyncio.Task | None = None
# Set by ``/api/source`` after a rebuild so the fanout loop refreshes its
# cached ``eng`` reference exactly once instead of doing a global lookup
# every 33 ms (see bottleneck audit).
_engine_rebuilt: asyncio.Event = asyncio.Event()


def get_engine() -> LiveEngine:
    global engine
    if engine is None:
        csv_env = os.environ.get("AIMP_DASH_CSV")
        csv_path = Path(csv_env) if csv_env else DEFAULT_CSV
        if not csv_path.exists():
            raise FileNotFoundError(
                f"Simu5G CSV not found: {csv_path}. "
                "Run run_simu5g_parallel.py first, or set AIMP_DASH_CSV."
            )
        # Default boot mode: Live RAN (RANSimulator + closed-loop actuator).
        # Override with ``AIMP_DASH_MODE=csv`` for the legacy OMNeT-CSV replay.
        # The CSV is still loaded — RANLiveSource generates its own corpus from
        # the simulator at boot, but MTPC also needs the historical archive on
        # disk and its path is taken from cfg.csv_path.
        mode_env = (os.environ.get("AIMP_DASH_MODE") or "live").lower()
        if mode_env not in {"live", "csv"}:
            log.warning(
                "AIMP_DASH_MODE=%r unrecognised; falling back to 'live'", mode_env,
            )
            mode_env = "live"
        live_mode = (mode_env == "live")
        log.info(
            "engine boot mode = %s (set AIMP_DASH_MODE=csv to revert)",
            "LIVE RAN simulator" if live_mode else "CSV replay",
        )
        engine = LiveEngine(EngineConfig(
            csv_path=csv_path,
            live_mode=live_mode,
            actuator_enabled=live_mode,   # closed-loop actuator on iff live
        ))
    return engine


# ---------------------------------------------------------------------------
# Event fan-out: drain engine.events → broadcast JSON to all clients
# ---------------------------------------------------------------------------

async def _broadcast(msg: dict) -> None:
    if not clients:
        return
    # Serialize once, broadcast to all clients.  ``send_bytes`` avoids
    # FastAPI's per-message UTF-8 encode pass that ``send_text`` does.
    # Browser WebSocket clients see the bytes as a binary frame and can
    # decode them with ``new TextDecoder().decode(event.data)`` — already
    # what the dashboard JS does.
    payload = _dumps(msg)
    dead: list[WebSocket] = []
    for ws in list(clients):
        try:
            await ws.send_bytes(payload)
        except Exception:
            dead.append(ws)
    for ws in dead:
        clients.discard(ws)


async def _fanout_loop() -> None:
    loop = asyncio.get_running_loop()
    batch_timeout = 0.033  # ~30 Hz client refresh

    # Cache the engine handle locally; refresh only when /api/source signals
    # a rebuild (set on _engine_rebuilt).  Avoids a global-dict lookup per
    # 33 ms iteration on the hot fanout path.
    eng: LiveEngine | None = None
    while True:
        try:
            if eng is None or _engine_rebuilt.is_set():
                _engine_rebuilt.clear()
                try:
                    eng = get_engine()
                except Exception:
                    eng = None
                    await asyncio.sleep(0.5)
                    continue

            assert eng is not None  # for type-checkers

            def _drain(_eng=eng) -> list[dict]:
                out = []
                while True:
                    try:
                        out.append(_eng.events.get_nowait())
                        if len(out) >= 200:  # cap per batch
                            break
                    except Exception:
                        break
                return out

            drained = await loop.run_in_executor(None, _drain)

            if drained:
                # Coalesce sample events to lower WS chatter under heavy rates
                samples: list[dict] = []
                others:  list[dict] = []
                for ev in drained:
                    if ev.get("type") == "sample":
                        samples.append(ev)
                    else:
                        others.append(ev)
                if samples:
                    await _broadcast({"type": "samples", "items": samples})
                for ev in others:
                    await _broadcast(ev)
            await asyncio.sleep(batch_timeout)
        except asyncio.CancelledError:
            break
        except Exception as exc:
            log.exception("fanout loop error: %s", exc)
            await asyncio.sleep(0.25)


# ---------------------------------------------------------------------------
# REST endpoints
# ---------------------------------------------------------------------------

class ControlReq(BaseModel):
    action: str   # start | pause | resume | stop | reset | force_retrain


class RateReq(BaseModel):
    rate_hz: float


class InjectReq(BaseModel):
    sinr_bias_db:  float | None = None
    rsrp_bias_db:  float | None = None
    delay_bias_ms: float | None = None
    tput_scale:    float | None = None
    poison_mode:   bool  | None = None
    noise_scale:   float | None = None


class ModeReq(BaseModel):
    # ``preferred_variant`` accepts "AUTO" | "LOCAL" | "EXTERNAL" | "CLOUD"
    preferred_variant:   str  | None = None
    closed_loop_enabled: bool | None = None
    use_golden_ndt:      bool | None = None
    # Level-2 closed-loop RAN — toggle the actuator without rebuilding
    actuator_enabled:    bool | None = None


class SourceReq(BaseModel):
    """Switch the engine's sample source (CSV replay vs live RAN simulator)."""
    mode:             str   # "csv" | "live"
    actuator_enabled: bool | None = None


@app.get("/api/status")
def api_status() -> JSONResponse:
    return JSONResponse(get_engine().status())


@app.post("/api/control")
async def api_control(req: ControlReq) -> JSONResponse:
    """
    Control the engine without blocking the event loop.

    ``start`` is the heavy one — on the very first call it triggers
    ``_build_pipeline`` (CSV load + sklearn fit + MTPCloud spin-up + MLflow
    init, ~1-2s on a warm cache).  Even though FastAPI runs sync handlers in
    a threadpool, ``await``-ing it via ``run_in_executor`` lets the loop keep
    pumping WebSocket frames so the UI stays responsive.  The pre-warm thread
    spawned in ``on_startup`` usually finishes before the user clicks Start,
    in which case ``eng.start()`` is essentially instant.
    """
    eng = get_engine()
    a = req.action.lower()
    loop = asyncio.get_running_loop()
    if   a == "start":         await loop.run_in_executor(None, eng.start)
    elif a == "pause":         eng.pause()
    elif a == "resume":        eng.resume()
    elif a == "stop":          await loop.run_in_executor(None, eng.stop)
    elif a == "reset":         eng.injection.trigger_reset()
    elif a == "force_retrain": eng.injection.trigger_retrain()
    else:
        return JSONResponse({"error": f"unknown action: {a}"}, status_code=400)
    return JSONResponse({"ok": True, "status": eng.status()})


@app.post("/api/rate")
def api_rate(req: RateReq) -> JSONResponse:
    eng = get_engine()
    eng.set_rate(req.rate_hz)
    return JSONResponse({"ok": True, "rate_hz": eng._rate_hz})


@app.post("/api/inject")
def api_inject(req: InjectReq) -> JSONResponse:
    eng = get_engine()
    patch = {k: v for k, v in req.model_dump().items() if v is not None}
    eng.injection.update(patch)
    return JSONResponse({"ok": True, "injection": eng.injection.snapshot()})


@app.post("/api/mode")
def api_mode(req: ModeReq) -> JSONResponse:
    """
    Set dashboard-level modes that are *not* per-sample KPI injections:
      * ``preferred_variant``  — which MTP variant to prefer when retraining
                                 ("AUTO" lets MTPComposer choose).
      * ``closed_loop_enabled``— if True, injection sliders decay to zero
                                 after a successful retrain (simulated
                                 policy-action effect).
      * ``use_golden_ndt``     — if True, NDT validates the candidate on the
                                 disjoint golden-holdout runs (ground-truth
                                 labels) instead of pseudo-labels from the
                                 current MLIN.
    """
    eng = get_engine()
    patch: dict = {}
    if req.preferred_variant is not None:
        val = req.preferred_variant.upper()
        if val not in {"AUTO", "LOCAL", "EXTERNAL", "CLOUD"}:
            return JSONResponse(
                {"error": f"invalid preferred_variant: {val}"},
                status_code=400,
            )
        patch["preferred_variant"] = val
    if req.closed_loop_enabled is not None:
        patch["closed_loop_enabled"] = bool(req.closed_loop_enabled)
    if req.use_golden_ndt is not None:
        patch["use_golden_ndt"] = bool(req.use_golden_ndt)
    if patch:
        eng.injection.update(patch)
    # actuator toggle — applies live (no rebuild needed)
    if req.actuator_enabled is not None and eng.actuator is not None:
        eng.actuator.set_enabled(bool(req.actuator_enabled))
    return JSONResponse({
        "ok": True,
        "injection": eng.injection.snapshot(),
        "actuator":  eng.actuator.snapshot() if eng.actuator else None,
    })


@app.post("/api/source")
def api_source(req: SourceReq) -> JSONResponse:
    """
    Switch the engine's sample source between the offline Simu5G CSV
    replay (``mode="csv"``) and the live, controller-mutable RAN
    simulator (``mode="live"``).

    The engine is stopped, torn down, and rebuilt with the new config —
    on the next ``/api/control start`` the closed-loop RAN comes online
    (or the CSV replay resumes, depending on ``mode``).
    """
    global engine
    mode = (req.mode or "").lower()
    if mode not in {"csv", "live"}:
        return JSONResponse(
            {"error": f"unknown source mode: {mode!r} (use 'csv' or 'live')"},
            status_code=400,
        )

    csv_env = os.environ.get("AIMP_DASH_CSV")
    csv_path = Path(csv_env) if csv_env else DEFAULT_CSV
    if not csv_path.exists():
        return JSONResponse(
            {"error": f"Simu5G CSV not found: {csv_path}"}, status_code=400,
        )

    # Tear down existing engine (if any) — the fanout loop will simply
    # see no events until the new engine.start() repopulates the queue.
    if engine is not None:
        try:
            engine.stop()
        except Exception as exc:
            log.warning("engine.stop during /api/source: %s", exc)
        engine = None

    cfg = EngineConfig(
        csv_path=csv_path,
        live_mode=(mode == "live"),
        actuator_enabled=(
            bool(req.actuator_enabled) if req.actuator_enabled is not None
            else (mode == "live")
        ),
    )
    engine = LiveEngine(cfg)
    # Notify the fanout loop to refresh its cached engine handle.
    _engine_rebuilt.set()
    log.info("engine source switched to mode=%s", mode)
    return JSONResponse({"ok": True, "status": engine.status()})


# ---------------------------------------------------------------------------
# WebSocket
# ---------------------------------------------------------------------------

@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket) -> None:
    await ws.accept()
    clients.add(ws)
    try:
        # Greet with current status so the UI paints immediately
        # Match the broadcast wire format: bytes via _dumps (orjson if installed).
        await ws.send_bytes(_dumps({"type": "status", **get_engine().status()}))
        while True:
            # We don't expect client→server messages, but consume to keep alive
            await ws.receive_text()
    except WebSocketDisconnect:
        pass
    except Exception as exc:
        log.debug("ws closed: %s", exc)
    finally:
        clients.discard(ws)


# ---------------------------------------------------------------------------
# Static SPA
# ---------------------------------------------------------------------------

@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/favicon.ico")
def favicon():
    """
    Inline 16×16 SVG bolt — gets rid of the browser-driven 404 in the
    uvicorn access log when the SPA loads.  Returned as
    ``image/svg+xml`` so we don't need to ship a binary file alongside
    the static bundle; modern browsers accept SVG favicons.
    """
    from fastapi.responses import Response
    svg = (
        b'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 16 16">'
        b'<path d="M9 1 L3 9 H7 L6 15 L13 7 H9 Z" fill="#f7c51b"/>'
        b'</svg>'
    )
    return Response(content=svg, media_type="image/svg+xml")


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------

@app.on_event("startup")
async def on_startup() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    try:
        eng = get_engine()  # fail-fast if CSV missing
    except FileNotFoundError as exc:
        log.error(str(exc))
        eng = None

    # Pre-warm the heavy pipeline (CSV load → sklearn fit → MTPCloud →
    # MLflow init → label calibration) on a daemon thread so the user's
    # first Start click hits an already-built engine instead of blocking
    # for ~2 s.  We deliberately don't `await` this — the HTTP server
    # should accept connections immediately; the SPA is harmless until
    # the user clicks Start, by which time _pipeline_ready is usually set.
    # `_build_pipeline` is a no-op on subsequent calls (guards inside).
    if eng is not None:
        threading.Thread(
            target=eng._build_pipeline,
            name="pipeline-prewarm",
            daemon=True,
        ).start()
        log.info("pipeline pre-warm started in background thread")

    global _fanout_task
    _fanout_task = asyncio.create_task(_fanout_loop())
    log.info("dashboard started — http://127.0.0.1:8765")


@app.on_event("shutdown")
async def on_shutdown() -> None:
    global _fanout_task
    if _fanout_task:
        _fanout_task.cancel()
    if engine is not None:
        engine.stop()
