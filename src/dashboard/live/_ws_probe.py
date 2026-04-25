"""
Tiny stand-alone WS probe used by manual_walkthrough.ps1.

Connects to ws://127.0.0.1:8765/ws, reads frames for ~8s, then prints a
single-line JSON summary of the frame-type counts to stdout so the
PowerShell driver can ConvertFrom-Json it.
"""
from __future__ import annotations

import asyncio
import json
import sys

import websockets


async def main() -> None:
    seen = {
        "samples": 0,
        "status": 0,
        "detectors": 0,
        "retrain_done": 0,
        "mtout": 0,
        "other": 0,
    }
    async with websockets.connect(
        "ws://127.0.0.1:8765/ws", max_size=2 ** 24
    ) as ws:
        try:
            while True:
                msg = await asyncio.wait_for(ws.recv(), timeout=8.0)
                f = json.loads(msg)
                t = f.get("type", "?")
                seen[t if t in seen else "other"] += 1
        except asyncio.TimeoutError:
            pass
    print(json.dumps(seen))
    sys.stdout.flush()


if __name__ == "__main__":
    asyncio.run(main())
