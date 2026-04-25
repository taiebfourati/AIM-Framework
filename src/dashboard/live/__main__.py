r"""
Entry point:
    py -3.13 -m dashboard.live            # http://127.0.0.1:8765
    py -3.13 -m dashboard.live --host 0.0.0.0 --port 9000
    AIMP_DASH_CSV=path\to\other.csv  py -3.13 -m dashboard.live

Boot order:
    1. Run ``dashboard.live.safety.preflight`` — verify the CSV is on disk,
       static assets exist, every required module imports cleanly,
       ``EngineConfig`` accepts the resolved boot mode, and the listening
       port is free.  A fatal failure aborts with exit code 2 *before*
       uvicorn binds the socket so the dashboard never reaches a half-booted
       state.
    2. Only then hand control to uvicorn.

Skip the preflight (offline debugging only) with ``--skip-safety`` or
``AIMP_DASH_SKIP_SAFETY=1``.
"""
from __future__ import annotations

import argparse
import os
import sys

import uvicorn

from dashboard.live.safety import run_or_exit


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--reload", action="store_true")
    p.add_argument(
        "--skip-safety",
        action="store_true",
        help="Skip the boot-time safety preflight (offline debugging only).",
    )
    args = p.parse_args()

    if args.skip_safety:
        # Forward the CLI flag to the env so any module that re-runs
        # the preflight (e.g. server-side guard) sees the same intent.
        os.environ["AIMP_DASH_SKIP_SAFETY"] = "1"
        print(
            "[safety] preflight SKIPPED via --skip-safety — server will boot "
            "without verifying CSV / imports / port. Do not use in production.",
            file=sys.stderr,
            flush=True,
        )
    else:
        # Aborts with exit code 2 on fatal failure — uvicorn is never reached.
        run_or_exit(host=args.host, port=args.port)

    uvicorn.run(
        "dashboard.live.server:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info",
    )


if __name__ == "__main__":
    main()
