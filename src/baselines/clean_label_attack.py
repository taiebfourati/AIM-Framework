"""
baselines/clean_label_attack.py — distribution-preserving clean-label adversary.

This is the adversary used in the bias-floor analysis of
Section~\\ref{sec:eval:bias_floor}.  The threat model is the third NDT
failure regime documented in Section~\\ref{sec:disc:rqs} (RQ3
qualification): an attacker who poisons the LOB replay buffer with
examples whose \\emph{marginal} feature distribution is preserved
(``clean-label'' in the sense of Shafahi et al., NeurIPS 2018) and
whose labels are individually plausible (``distribution-preserving''
in the sense of Wang et al., arXiv:1905.13409).

Concretely, the attack does NOT inject out-of-distribution feature
spikes (which would be visible to DDD/DPD) and does NOT shift the
class prior (which would be visible to the CUSUM / class-prior
monitors).  Instead it picks a small attacker-controlled subset of the
training stream sampled from the SAME generator and KEEPS each
sample's true label, but biases the training ratio so that the learned
decision boundary tilts toward the attacker's preferred half-space.

The strength of the attack is governed by ``attacker_fraction``
(the fraction of the candidate's training stream the attacker
controls) and ``bias_band`` (how close to the true boundary the
attacker concentrates its over-sampling).  Higher attacker_fraction
means a more aggressive attack and lower expected golden-corpus
accuracy for the resulting candidate model.

This file is intentionally a small reference implementation; nothing
here calls into the production RTP / AIF / NDT code paths.  The
``scripts/bias_floor_sweep.py`` driver wires the candidate produced
here into the actual NDT gate.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np


def make_clean_data(rng: np.random.Generator, n: int, noise: float = 0.05
                    ) -> Tuple[np.ndarray, np.ndarray]:
    """The same data generator the production simulation uses.

    Sampling X ~ N(0, I_4); ground-truth label y = 1{X[:,0]+X[:,1] > 0}
    with i.i.d. label noise ``noise``.  Reused here so the candidate
    sees the SAME marginal as the operator's golden corpus.
    """
    X = rng.normal(0.0, 1.0, size=(n, 4))
    y = ((X[:, 0] + X[:, 1]) > 0).astype(int)
    flip = rng.random(n) < noise
    y[flip] = 1 - y[flip]
    return X, y


def make_clean_label_poisoned_buffer(
    rng: np.random.Generator,
    n_clean: int = 600,
    attacker_fraction: float = 0.20,
    bias_band: float = 0.6,
    bias_target: int = 0,
    noise: float = 0.05,
) -> Tuple[np.ndarray, np.ndarray]:
    """Synthesise a candidate-training buffer that is ``attacker_fraction``
    poisoned without a single label being individually wrong and
    without any feature being out-of-distribution.

    Construction:
      1. Sample ``n_clean`` clean samples from the operator's
         distribution (same generator as the golden corpus).
      2. Sample additional ``n_attack = attacker_fraction * n_clean /
         (1 - attacker_fraction)`` examples from the SAME generator,
         but accept only those whose feature-sum lies in the
         attacker's bias band ``|X[:,0]+X[:,1]| <= bias_band``
         (near the true decision boundary) AND whose true label
         equals ``bias_target``.  These two filters together select
         examples that are (a) individually correct under f^* and
         (b) maximally informative about the boundary's location ---
         oversampling them tilts the learned boundary into the
         OPPOSITE class's territory while preserving every sample's
         in-distribution status.
      3. Concatenate and shuffle.

    Mechanism.  Each accepted attacker sample is a true-class
    ``bias_target`` example sitting near the decision boundary.  In a
    clean buffer such examples make up roughly
    ``noise + 0.5 * P(|X[:,0]+X[:,1]| <= bias_band)`` of all samples;
    after the attack they are over-represented by a factor proportional
    to ``attacker_fraction``.  The classifier therefore sees an
    inflated density of ``y == bias_target`` evidence in the boundary
    zone, which shifts its decision boundary AWAY from the
    attacker-preferred class --- the operator's golden-corpus accuracy
    drops because formerly-correct opposite-class predictions near the
    boundary now flip to ``bias_target``.

    The key property: every (X, y) row in the returned buffer is
    \\emph{individually clean} (X is in-distribution, y matches
    f^*(X) up to the same noise level the operator's data has), but
    the batch as a whole is biased.  This is exactly the failure
    regime the dual-score NDT gate is admitted not to defend against.
    """
    if not 0.0 <= attacker_fraction < 1.0:
        raise ValueError("attacker_fraction must be in [0, 1)")

    X_clean, y_clean = make_clean_data(rng, n_clean, noise=noise)

    n_attack = int(round(attacker_fraction * n_clean / (1.0 - attacker_fraction)))
    if n_attack == 0:
        return X_clean, y_clean

    # Over-sample candidates from the same generator, keep only those
    # whose feature-sum lies INSIDE the attacker's near-boundary band
    # AND whose noisy label equals the attacker's target class.
    attack_pool_X: list[np.ndarray] = []
    attack_pool_y: list[int] = []
    rejected = 0
    while len(attack_pool_X) < n_attack:
        X_pool, y_pool = make_clean_data(rng, n_attack * 8, noise=noise)
        feat_sum = X_pool[:, 0] + X_pool[:, 1]
        for i in range(len(X_pool)):
            if len(attack_pool_X) >= n_attack:
                break
            in_band = abs(feat_sum[i]) <= bias_band
            matches_target = int(y_pool[i]) == int(bias_target)
            keep = in_band and matches_target
            if keep:
                attack_pool_X.append(X_pool[i])
                attack_pool_y.append(int(y_pool[i]))
            else:
                rejected += 1
        # Safety: prevent infinite loop if the band is empty
        if rejected > n_attack * 1000:
            break

    if attack_pool_X:
        X_attack = np.vstack(attack_pool_X)
        y_attack = np.asarray(attack_pool_y, dtype=int)
        X_full = np.vstack([X_clean, X_attack])
        y_full = np.concatenate([y_clean, y_attack])
    else:
        X_full, y_full = X_clean, y_clean

    idx = rng.permutation(len(X_full))
    return X_full[idx], y_full[idx]
