"""
Helpers shared by the Tier 1 detector tests.

Each detector's ``check()`` takes a populated LIB (and LOB, for CPD).
These helpers build a buffer pre-filled with a deterministic stream so
every test can focus on the *stimulus* rather than plumbing.
"""
from __future__ import annotations

import numpy as np

from aif.buffers import LIB, LOB


def fill_lib(X: np.ndarray, capacity: int | None = None) -> LIB:
    """Push every row of X into a fresh LIB and return it.

    capacity defaults to ``max(len(X), 10)`` so the LIB never evicts
    during the fill — the tests want the full series in buffer.
    """
    X = np.atleast_2d(np.asarray(X, dtype=float))
    cap = capacity if capacity is not None else max(len(X), 10)
    lib = LIB(maxlen=cap)
    for row in X:
        lib.push(row)
    return lib


def fill_lob(y: np.ndarray, capacity: int | None = None) -> LOB:
    """Push every entry of y into a fresh LOB."""
    y = np.atleast_1d(np.asarray(y, dtype=float)).ravel()
    cap = capacity if capacity is not None else max(len(y), 10)
    lob = LOB(maxlen=cap)
    for v in y:
        lob.push(np.atleast_1d(v))
    return lob


def iid_gaussian(
    n: int,
    d: int,
    *,
    mean: float | np.ndarray = 0.0,
    std: float | np.ndarray = 1.0,
    seed: int = 0,
) -> np.ndarray:
    """Deterministic IID gaussian batch of shape (n, d)."""
    rng = np.random.default_rng(seed)
    mean_arr = np.broadcast_to(np.asarray(mean, dtype=float), (d,))
    std_arr  = np.broadcast_to(np.asarray(std,  dtype=float), (d,))
    return rng.normal(mean_arr, std_arr, size=(n, d))
