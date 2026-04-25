"""
Ensure the project's ``src/`` directory is on sys.path so
``from detectors...``, ``from aif...`` etc. resolve when pytest runs from
anywhere -- even without an editable install (``pip install -e .``).

Also reconfigure stdout/stderr so stray Unicode in log messages (e.g. Greek
letters in detector diagnostics) never crash the cp1252 console on Windows.
UnicodeEncodeErrors in ``logging.StreamHandler.emit`` can generate multi-MB
tracebacks per message and deadlock pytest's capture buffer.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

# Prefer the src/ layout, fall back to the project root for backward
# compatibility with anyone running tests against an older checkout.
for _candidate in (SRC, ROOT):
    if _candidate.exists() and str(_candidate) not in sys.path:
        sys.path.insert(0, str(_candidate))

# Make stdout/stderr tolerant of non-cp1252 characters (\u0394 etc.) so
# ``logger.warning("... Δ ...")`` never crashes the handler.
for _stream_name in ("stdout", "stderr"):
    _stream = getattr(sys, _stream_name, None)
    if _stream is not None and hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8", errors="backslashreplace")
        except Exception:
            pass
