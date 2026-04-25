"""
safety.py
=========

Boot-time safety preflight for the AIMP × Simu5G dashboard.

Runs **before** the FastAPI/uvicorn server begins accepting connections so
that misconfigurations (missing CSV, broken imports, missing static assets,
port already in use, wrong Python venv, …) surface as a clean error log
instead of a half-booted dashboard that returns 500s 30 seconds later.

Wired into:

  * ``dashboard/live/__main__.py`` – called from ``main()`` before
    ``uvicorn.run`` so launching ``python -m dashboard.live`` (the path
    used by ``.claude/launch.json``) gates on the preflight.

  * Optional escape hatch: ``AIMP_DASH_SKIP_SAFETY=1`` skips all checks.
    Intended for CI / offline debugging only — never set this in normal use.

Exit codes (when invoked stand-alone):
    0  – all checks passed (or skipped via env var)
    2  – at least one critical check failed
"""
from __future__ import annotations

import importlib
import logging
import os
import socket
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Tuple

log = logging.getLogger("dashboard.live.safety")

# ---------------------------------------------------------------------------
# Repo paths (re-derived so ``safety`` can be imported without server.py)
# ---------------------------------------------------------------------------

_THIS_DIR  = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parents[1]
_STATIC    = _THIS_DIR / "static"
_DEFAULT_CSV = _REPO_ROOT / "simu5g_real_simulation_results.csv"

# Required column subset we must see in the OMNeT-derived CSV.  Anything
# missing means MTPC / golden-NDT will silently degrade later, so we'd
# rather refuse to boot.  The raw CSV is wide-format per-tick KPIs from
# Simu5G — labels are derived downstream by the engine from ``config`` /
# ``phase`` (e.g. "ho_traffic" → handover-traffic regime), so we don't
# require a literal "label" column here.
_REQUIRED_CSV_COLS = (
    "config",
    "phase",
    "run",
    "ue",
    "t",
    "rsrp_dbm",
    "sinr_db",
    "throughput_mbps",
    "delay_ms",
)

# Static assets we must serve for the SPA to render.  Missing any of these
# means the user opens the dashboard and sees a 404 / blank page.
_REQUIRED_STATIC = ("index.html", "app.js", "style.css")

# Python modules that must import cleanly.  We don't construct anything —
# just verify the import graph is intact, which catches half-installed
# venvs, missing third-party deps, and renamed siblings.
_REQUIRED_IMPORTS = (
    "fastapi",
    "uvicorn",
    "pandas",
    "numpy",
    "dashboard.live.engine",
    "dashboard.live.ran_simulator",
    "dashboard.live.ran_actuator",
)

# Soft imports — warn but don't fail.  The dashboard has fallbacks for
# these (server.py falls back to stdlib json if orjson is missing).
_SOFT_IMPORTS = ("orjson",)


# ---------------------------------------------------------------------------
# Result aggregation
# ---------------------------------------------------------------------------

@dataclass
class CheckResult:
    name:    str
    ok:      bool
    detail:  str = ""
    fatal:   bool = True   # if False → warn, don't abort


@dataclass
class PreflightSummary:
    checks: List[CheckResult] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return all(c.ok or not c.fatal for c in self.checks)

    @property
    def fatal_failures(self) -> List[CheckResult]:
        return [c for c in self.checks if not c.ok and c.fatal]

    @property
    def warnings(self) -> List[CheckResult]:
        return [c for c in self.checks if not c.ok and not c.fatal]

    def add(self, name: str, ok: bool, detail: str = "", *, fatal: bool = True) -> None:
        self.checks.append(CheckResult(name=name, ok=ok, detail=detail, fatal=fatal))

    def render(self) -> str:
        lines: List[str] = []
        lines.append("=" * 72)
        lines.append("Dashboard preflight summary")
        lines.append("-" * 72)
        for c in self.checks:
            badge = "[ OK ]" if c.ok else ("[FAIL]" if c.fatal else "[WARN]")
            row = f"  {badge}  {c.name}"
            if c.detail:
                row += f"   — {c.detail}"
            lines.append(row)
        lines.append("-" * 72)
        if self.ok and not self.warnings:
            lines.append("Result: ALL CHECKS PASSED — safe to start the server.")
        elif self.ok:
            lines.append(
                f"Result: PASSED with {len(self.warnings)} warning(s) — "
                "server will start.",
            )
        else:
            lines.append(
                f"Result: FAILED — {len(self.fatal_failures)} blocking check(s). "
                "Refusing to start the server.",
            )
        lines.append("=" * 72)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------

def _check_python(s: PreflightSummary) -> None:
    v = sys.version_info
    actual = f"{v.major}.{v.minor}.{v.micro}"
    # Project targets 3.13 but we accept 3.11+ so the dashboard still
    # boots if the venv was rebuilt with a newer interpreter.
    ok = (v.major, v.minor) >= (3, 11)
    s.add(
        "python interpreter",
        ok,
        f"{actual} ({sys.executable})" if ok
        else f"{actual} — need >= 3.11",
    )


def _check_repo_layout(s: PreflightSummary) -> None:
    must_exist = (
        _REPO_ROOT / "dashboard" / "live" / "engine.py",
        _REPO_ROOT / "dashboard" / "live" / "ran_simulator.py",
        _REPO_ROOT / "dashboard" / "live" / "ran_actuator.py",
    )
    missing = [p for p in must_exist if not p.exists()]
    if missing:
        rels = ", ".join(p.relative_to(_REPO_ROOT).as_posix() for p in missing)
        s.add("repo layout", False, f"missing: {rels}")
    else:
        s.add("repo layout", True, _REPO_ROOT.as_posix())


def _check_static(s: PreflightSummary) -> None:
    if not _STATIC.is_dir():
        s.add("static assets", False, f"directory not found: {_STATIC}")
        return
    missing = [name for name in _REQUIRED_STATIC if not (_STATIC / name).exists()]
    if missing:
        s.add("static assets", False, f"missing: {', '.join(missing)}")
    else:
        sizes = ", ".join(
            f"{n}={(_STATIC / n).stat().st_size}B" for n in _REQUIRED_STATIC
        )
        s.add("static assets", True, sizes)


def _check_csv(s: PreflightSummary) -> None:
    csv_env = os.environ.get("AIMP_DASH_CSV")
    csv_path = Path(csv_env) if csv_env else _DEFAULT_CSV
    if not csv_path.exists():
        s.add(
            "Simu5G CSV",
            False,
            f"not found: {csv_path}  (set AIMP_DASH_CSV or run "
            "run_simu5g_parallel.py)",
        )
        return
    size = csv_path.stat().st_size
    if size < 1024:
        s.add("Simu5G CSV", False, f"{csv_path.name} is suspiciously small ({size}B)")
        return
    # Peek at the header to confirm required columns are present.
    try:
        with csv_path.open("r", encoding="utf-8") as fh:
            header = fh.readline().strip()
    except Exception as exc:
        s.add("Simu5G CSV", False, f"could not read header: {exc!r}")
        return
    cols = {c.strip() for c in header.split(",")}
    missing = [c for c in _REQUIRED_CSV_COLS if c not in cols]
    if missing:
        s.add(
            "Simu5G CSV",
            False,
            f"{csv_path.name} missing columns: {', '.join(missing)}",
        )
    else:
        s.add(
            "Simu5G CSV",
            True,
            f"{csv_path.name} ({size // 1024} KB, header has all required cols)",
        )


def _check_imports(s: PreflightSummary, modules: Iterable[str], *, fatal: bool) -> None:
    failed: List[Tuple[str, str]] = []
    for mod in modules:
        try:
            importlib.import_module(mod)
        except Exception as exc:                                # noqa: BLE001
            failed.append((mod, repr(exc)))
    label = "required imports" if fatal else "optional imports"
    if failed:
        s.add(
            label,
            False,
            "; ".join(f"{m}: {err}" for m, err in failed),
            fatal=fatal,
        )
    else:
        s.add(label, True, f"{len(list(modules))} module(s) loaded", fatal=fatal)


def _check_engine_config(s: PreflightSummary) -> None:
    """Confirm EngineConfig accepts the resolved boot mode without raising."""
    try:
        from dashboard.live.engine import EngineConfig  # noqa: WPS433
    except Exception as exc:                                # noqa: BLE001
        s.add("engine config", False, f"import EngineConfig failed: {exc!r}")
        return
    mode_env = (os.environ.get("AIMP_DASH_MODE") or "live").lower()
    if mode_env not in {"live", "csv"}:
        s.add(
            "engine config",
            False,
            f"AIMP_DASH_MODE={mode_env!r} not in (live, csv)",
            fatal=False,   # server.py also normalises with a warning
        )
        mode_env = "live"
    live_mode = mode_env == "live"
    csv_env = os.environ.get("AIMP_DASH_CSV")
    csv_path = Path(csv_env) if csv_env else _DEFAULT_CSV
    try:
        cfg = EngineConfig(
            csv_path=csv_path,
            live_mode=live_mode,
            actuator_enabled=live_mode,
        )
    except Exception as exc:                                # noqa: BLE001
        s.add("engine config", False, f"construction failed: {exc!r}")
        return
    s.add(
        "engine config",
        True,
        f"mode={'LIVE' if cfg.live_mode else 'CSV'}, "
        f"actuator={'on' if cfg.actuator_enabled else 'off'}, "
        f"rate_hz={cfg.rate_hz}",
    )


def _check_port(s: PreflightSummary, host: str, port: int) -> None:
    """Verify ``host:port`` is bindable, with hints on what's holding it.

    Fatal: if the port is already in use, uvicorn will fail with a much less
    actionable WinError 10048 / EADDRINUSE message *after* the engine has
    started initialising. Catching it during preflight lets the user kill
    the stale process cleanly and re-run.
    """
    fam = socket.AF_INET6 if ":" in host else socket.AF_INET
    sock = socket.socket(fam, socket.SOCK_STREAM)
    try:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((host, port))
    except OSError as exc:
        hint = _holder_hint(port)
        detail = f"{host}:{port} already in use ({exc.errno}: {exc.strerror})"
        if hint:
            detail += f" — {hint}"
        s.add("port available", False, detail)
        return
    finally:
        sock.close()
    s.add("port available", True, f"{host}:{port} is free")


def _holder_hint(port: int) -> str:
    """Best-effort: identify the PID/exe holding ``port`` so the user can kill it.

    Windows-only via ``netstat``/``tasklist``; silently returns "" elsewhere
    or if the lookup fails.  Adds an actionable ``Stop-Process -Id N -Force``
    suggestion to the failure detail.
    """
    if sys.platform != "win32":
        return ""
    try:
        import subprocess  # noqa: WPS433
        ns = subprocess.run(
            ["netstat", "-ano", "-p", "TCP"],
            capture_output=True, text=True, timeout=3, check=False,
        )
        if ns.returncode != 0:
            return ""
        target = f":{port}"
        pid: str | None = None
        for line in ns.stdout.splitlines():
            parts = line.split()
            # Windows netstat columns: Proto  LocalAddr  RemoteAddr  State  PID
            if len(parts) >= 5 and parts[3].upper() == "LISTENING" and target in parts[1]:
                pid = parts[-1]
                break
        if not pid:
            return ""
        tl = subprocess.run(
            ["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"],
            capture_output=True, text=True, timeout=3, check=False,
        )
        exe = ""
        if tl.returncode == 0 and tl.stdout.strip():
            first_field = tl.stdout.strip().split('","', 1)[0].lstrip('"')
            exe = first_field
        suffix = f" ({exe})" if exe else ""
        return f"held by PID {pid}{suffix}; stop it with:  Stop-Process -Id {pid} -Force"
    except Exception:                                       # noqa: BLE001
        return ""


def _check_skip_flag(s: PreflightSummary) -> bool:
    """Return True if the user explicitly skipped safety."""
    if os.environ.get("AIMP_DASH_SKIP_SAFETY", "").strip().lower() in {"1", "true", "yes"}:
        s.add(
            "safety preflight",
            True,
            "SKIPPED via AIMP_DASH_SKIP_SAFETY (use only for offline debugging)",
            fatal=False,
        )
        return True
    return False


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def preflight(*, host: str = "127.0.0.1", port: int = 8765) -> PreflightSummary:
    """Run all preflight checks and return a summary.

    Caller decides what to do with ``summary.ok`` — ``__main__`` aborts the
    launch on False, but importers may want to continue and just log.
    """
    s = PreflightSummary()
    if _check_skip_flag(s):
        return s
    _check_python(s)
    _check_repo_layout(s)
    _check_static(s)
    _check_csv(s)
    _check_imports(s, _REQUIRED_IMPORTS, fatal=True)
    _check_imports(s, _SOFT_IMPORTS, fatal=False)
    _check_engine_config(s)
    _check_port(s, host, port)
    return s


def run_or_exit(*, host: str = "127.0.0.1", port: int = 8765) -> None:
    """Run preflight; emit summary to log + stderr; exit(2) on fatal failure."""
    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%H:%M:%S",
        )
    summary = preflight(host=host, port=port)
    rendered = summary.render()
    # Always print to stderr so it's visible even when uvicorn captures stdout.
    print(rendered, file=sys.stderr, flush=True)
    log.info("preflight finished: ok=%s warnings=%d failures=%d",
             summary.ok, len(summary.warnings), len(summary.fatal_failures))
    if not summary.ok:
        log.critical("preflight refused to start the dashboard server")
        sys.exit(2)


if __name__ == "__main__":
    # Allow ``py -3.13 -m dashboard.live.safety`` for ad-hoc verification.
    run_or_exit()
