"""
baselines/spectral_signatures.py — Tran et al. spectral-signature defence.

Reference: Tran, Li, Madry, "Spectral Signatures in Backdoor Attacks",
NeurIPS 2018.  Given a batch of suspected examples
``X in R^{N x d}``, the defender:

  1. centres the batch:  X_c = X - mean(X)
  2. takes the top right singular vector ``v`` of ``X_c``
  3. projects each row onto ``v``:  s_i = (X_c[i] @ v) ** 2
  4. flags as poisoned the ``epsilon * N`` rows with the largest s_i

The intuition is that backdoor examples sit far from the class centroid
along the principal direction of the perturbation, so their squared
projection on the top singular vector is anomalously large compared to
clean examples.

For tabular data without a learned latent representation we apply the
defence directly in the input feature space (the paper uses the
penultimate layer of a CNN; we use the raw 4-D feature vector).  This
is the standard adaptation when the model is non-differentiable
(e.g. random forests in our scenario).

Public surface
--------------
``SpectralSignatures(epsilon=0.10, min_window=80).update(x) -> bool``

Each call appends one row to the rolling window; once ``min_window``
samples are buffered, the SVD-and-flag procedure is run on every call.
``update`` returns True when at least one row in the *current* window
was flagged as anomalous on the most recent SVD pass.  In practice the
defender invokes the test at every "check interval" boundary rather
than every sample — see :func:`check_now` for the explicit form.
"""
from __future__ import annotations

from collections import deque
from typing import Deque, List, Optional

import numpy as np


class SpectralSignatures:
    """Spectral-signature poisoning detector (Tran et al., 2018)."""

    def __init__(
        self,
        epsilon: float = 0.10,
        min_window: int = 80,
        max_window: int = 300,
    ) -> None:
        if not 0.0 < epsilon < 1.0:
            raise ValueError("epsilon must be in (0, 1)")
        self.epsilon = float(epsilon)
        self.min_window = int(min_window)
        self.max_window = int(max_window)
        self._window: Deque[np.ndarray] = deque(maxlen=max_window)

    def reset(self) -> None:
        self._window.clear()

    def update(self, x: np.ndarray) -> bool:
        """Append one row to the window. Returns True iff the most
        recent SVD-and-flag pass marked at least one row as anomalous.
        Cheap: O(1) append, O(N d) per check."""
        self._window.append(np.asarray(x, dtype=float).ravel())
        if len(self._window) < self.min_window:
            return False
        flagged = self.check_now()
        return len(flagged) > 0

    def check_now(self) -> List[int]:
        """Run the spectral-signature test on the current window and
        return the indices (within the window) flagged as anomalous.

        Returns an empty list before ``min_window`` is reached."""
        if len(self._window) < self.min_window:
            return []
        X = np.vstack(self._window)
        N = X.shape[0]
        # Centre the batch.
        mu = X.mean(axis=0, keepdims=True)
        X_c = X - mu
        # Top right singular vector.  We use the truncated path via
        # ``np.linalg.svd(full_matrices=False)`` since d is small (=4).
        try:
            _, _, vt = np.linalg.svd(X_c, full_matrices=False)
        except np.linalg.LinAlgError:
            return []
        v = vt[0]  # top right singular vector (length d)
        # Squared projection on the top singular direction.
        s = (X_c @ v) ** 2
        # Flag the top ``epsilon * N`` rows.
        n_flag = max(1, int(np.ceil(self.epsilon * N)))
        threshold = np.partition(s, -n_flag)[-n_flag]
        flagged = np.flatnonzero(s >= threshold).tolist()
        return flagged

    def __repr__(self) -> str:
        return (
            f"SpectralSignatures(eps={self.epsilon}, "
            f"|W|={len(self._window)})"
        )
