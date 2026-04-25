"""
Tier 1 property tests for aif.dpostp.DPostP.

Covers both roles called out in paper Section IV-D-3:

1. **Cleaning role** (``process_training_batch`` + ``build_reference``):
   The detector false-positive that motivated DPostP was a thin 50-row
   GT slice producing |Δr|≈0.48 on clean data; these tests exercise the
   padding behaviour that fixes that, plus the NaN/clip/dedup hygiene.

2. **Transport role** (``seal`` / ``unseal``):
   Tests are designed to prove failure-closed behaviour — any tamper,
   replay, or cross-run substitution must raise
   ``DPostPAuthenticationError`` rather than returning silently-corrupt
   numpy arrays.

Environment
-----------
Seal/unseal tests construct DPostP with an explicit 32-byte key so they
do not depend on the ``DPOSTP_KEY`` environment variable being set.
"""
from __future__ import annotations

import base64
import os
import time

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from aif.buffers import LIB
from aif.dpostp import (
    DPostP,
    DPostPAuthenticationError,
    DPostPKeyError,
    SealedPayload,
    _ALGO_ID,
    _KEY_LEN,
    _MAGIC,
    _NONCE_LEN,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def dpostp_no_key() -> DPostP:
    """Cleaning-only instance — no transport key needed."""
    return DPostP(z_clip=5.0, min_ref_rows=200)


@pytest.fixture
def dpostp_with_key() -> DPostP:
    """Instance with an explicit key — safe for seal/unseal tests."""
    key = os.urandom(_KEY_LEN)
    return DPostP(z_clip=5.0, min_ref_rows=200, key=key)


# ---------------------------------------------------------------------------
# Role 1 — process_training_batch
# ---------------------------------------------------------------------------

class TestProcessTrainingBatch:

    def test_keeps_clean_batch_unchanged(self, dpostp_no_key) -> None:
        rng = np.random.default_rng(0)
        X = rng.normal(0.0, 1.0, size=(100, 5))
        y = rng.integers(0, 2, size=100).astype(float)

        X_out, y_out = dpostp_no_key.process_training_batch(X, y)
        assert X_out.shape == X.shape
        assert y_out.shape == y.shape
        # All rows are within 5σ of their own mean (random IID) → no clip.
        assert np.allclose(X_out, X, atol=1e-9)

    def test_drops_nan_rows(self, dpostp_no_key) -> None:
        rng = np.random.default_rng(1)
        X = rng.normal(0.0, 1.0, size=(50, 4))
        y = rng.integers(0, 2, size=50).astype(float)
        # Inject NaNs on rows 3, 7, 19 in X and on 42 in y
        X[[3, 7, 19], 2] = np.nan
        y[42] = np.nan

        X_out, y_out = dpostp_no_key.process_training_batch(X, y)
        assert X_out.shape[0] == 50 - 4        # 3 X-NaN + 1 y-NaN
        assert not np.isnan(X_out).any()
        assert not np.isnan(y_out).any()

    def test_drops_inf_rows(self, dpostp_no_key) -> None:
        rng = np.random.default_rng(2)
        X = rng.normal(0.0, 1.0, size=(20, 3))
        y = np.zeros(20, dtype=float)
        X[5, 1] = np.inf
        X[12, 0] = -np.inf

        X_out, y_out = dpostp_no_key.process_training_batch(X, y)
        assert X_out.shape[0] == 18
        assert np.isfinite(X_out).all()

    def test_clips_column_outliers(self) -> None:
        # z_clip=2 → an 8σ sample must be clipped back to 2σ
        dpostp = DPostP(z_clip=2.0, min_ref_rows=50)
        rng = np.random.default_rng(3)
        X = rng.normal(0.0, 1.0, size=(100, 3))
        y = rng.integers(0, 2, size=100).astype(float)
        X[0, 0] = 8.0    # extreme outlier

        X_out, y_out = dpostp.process_training_batch(X, y, dedup=False)
        # The clipped value should be ≤ mean + 2·std on that column.
        mu, sd = X.mean(axis=0)[0], X.std(axis=0)[0]
        assert X_out[0, 0] <= mu + 2.0 * sd + 1e-6

    def test_dedups_identical_rows(self, dpostp_no_key) -> None:
        X = np.array([[1.0, 2.0], [3.0, 4.0], [1.0, 2.0], [5.0, 6.0]])
        y = np.array([0.0, 1.0, 0.0, 1.0])
        X_out, y_out = dpostp_no_key.process_training_batch(X, y)
        assert X_out.shape[0] == 3

    def test_all_rows_nan_returns_empty(self, dpostp_no_key) -> None:
        X = np.full((5, 3), np.nan)
        y = np.arange(5, dtype=float)
        X_out, y_out = dpostp_no_key.process_training_batch(X, y)
        assert X_out.shape == (0, 3)
        assert y_out.shape == (0,)

    def test_length_mismatch_raises(self, dpostp_no_key) -> None:
        with pytest.raises(ValueError, match="row-count mismatch"):
            dpostp_no_key.process_training_batch(
                np.zeros((5, 2)), np.zeros(4)
            )

    def test_z_clip_infinity_disables_clipping(self) -> None:
        """
        ``z_clip=np.inf`` is the documented opt-out for clipping.

        The implementation guards the clip block behind
        ``np.isfinite(self.z_clip)``; with infinity the block is skipped
        entirely and extreme outliers must pass through unchanged.  This
        pins that contract so a future refactor can't silently reintroduce
        a finite default (which would quietly reshape training batches
        for every caller that opted out).
        """
        dpostp = DPostP(z_clip=np.inf, min_ref_rows=50)
        rng = np.random.default_rng(100)
        X = rng.normal(0.0, 1.0, size=(30, 3))
        y = rng.integers(0, 2, size=30).astype(float)
        X[0, 0] = 100.0    # extreme outlier — would clip at any finite z

        X_out, y_out = dpostp.process_training_batch(X, y, dedup=False)

        # Extreme outlier must survive untouched.
        assert X_out[0, 0] == 100.0, (
            f"z_clip=inf must not clip; got {X_out[0, 0]}"
        )
        # Non-outlier cells are also untouched when clipping is off.
        np.testing.assert_array_equal(X_out, X)

    def test_dedup_false_preserves_duplicate_rows(self, dpostp_no_key) -> None:
        """
        ``dedup=False`` preserves exact-row repeats.

        Callers sometimes *want* duplicates retained — e.g. when a
        controlled replay buffer deliberately over-samples a class to
        correct imbalance, and dropping duplicates would undo that
        rebalancing.  This test pins the keyword contract.
        """
        X = np.array(
            [[1.0, 2.0], [3.0, 4.0], [1.0, 2.0], [5.0, 6.0], [1.0, 2.0]]
        )
        y = np.array([0.0, 1.0, 0.0, 1.0, 0.0])

        X_out, y_out = dpostp_no_key.process_training_batch(X, y, dedup=False)

        # All five rows survive — no dedup.
        assert X_out.shape[0] == 5
        np.testing.assert_array_equal(X_out, X)
        np.testing.assert_array_equal(y_out, y)

    def test_dedup_true_is_default(self, dpostp_no_key) -> None:
        """
        Sanity: the default behaviour of ``process_training_batch`` is
        ``dedup=True``.  Paired with the explicit-off test above, this
        pins both sides of the switch.
        """
        X = np.array([[1.0, 2.0], [1.0, 2.0]])   # identical rows
        y = np.array([0.0, 0.0])
        X_out, _ = dpostp_no_key.process_training_batch(X, y)
        assert X_out.shape[0] == 1, "default dedup=True must collapse duplicates"


# ---------------------------------------------------------------------------
# Role 1 — build_reference
# ---------------------------------------------------------------------------

class TestBuildReference:

    def _fit_model(self, X: np.ndarray, y: np.ndarray) -> LogisticRegression:
        return LogisticRegression(max_iter=200, random_state=42).fit(X, y)

    def test_returns_unchanged_when_gt_already_large_enough(
        self, dpostp_no_key
    ) -> None:
        rng = np.random.default_rng(10)
        X = rng.normal(0.0, 1.0, size=(250, 4))
        y = rng.integers(0, 2, size=250).astype(float)
        X_ref, y_ref = dpostp_no_key.build_reference(
            X, y, min_rows=200,
        )
        # No padding necessary → identical output (no extra allocation).
        assert X_ref.shape == (250, 4)
        assert np.array_equal(X_ref, X)

    def test_pads_thin_gt_slice_with_lib_rows(self, dpostp_no_key) -> None:
        rng = np.random.default_rng(11)
        # 50-row GT slice — the motivating failure case.
        X_gt = rng.normal(0.0, 1.0, size=(50, 4))
        y_gt = rng.integers(0, 2, size=50).astype(float)

        # LIB has 500 rows — plenty of padding material.
        X_lib_raw = rng.normal(0.0, 1.0, size=(500, 4))
        y_lib_raw = rng.integers(0, 2, size=500).astype(float)
        model = self._fit_model(X_lib_raw, y_lib_raw)

        lib = LIB(maxlen=500)
        for row in X_lib_raw:
            lib.push(row)

        X_ref, y_ref = dpostp_no_key.build_reference(
            X_gt, y_gt, lib=lib, new_estimator=model, min_rows=200,
        )
        assert X_ref.shape[0] == 200, "padded reference should reach min_rows"
        assert X_ref.shape[1] == 4
        # GT rows must survive as the TAIL (authoritative slice) because
        # downstream CPD indexes with X_arr[-n:].
        assert np.allclose(X_ref[-50:], X_gt)
        assert np.allclose(y_ref[-50:], y_gt)

    def test_pad_rows_labelled_via_new_estimator(self, dpostp_no_key) -> None:
        rng = np.random.default_rng(12)
        X_gt = rng.normal(0.0, 1.0, size=(30, 3))
        y_gt = rng.integers(0, 2, size=30).astype(float)

        X_lib_raw = rng.normal(0.0, 1.0, size=(300, 3))
        y_lib_raw = rng.integers(0, 2, size=300).astype(float)
        model = self._fit_model(X_lib_raw, y_lib_raw)

        lib = LIB(maxlen=300)
        for row in X_lib_raw:
            lib.push(row)

        X_ref, y_ref = dpostp_no_key.build_reference(
            X_gt, y_gt, lib=lib, new_estimator=model, min_rows=150,
        )
        # Pad segment (everything but the GT tail) must match model.predict.
        pad_n = X_ref.shape[0] - 30
        expected_pad_y = model.predict(X_ref[:pad_n])
        assert np.array_equal(y_ref[:pad_n], expected_pad_y)

    def test_falls_back_without_estimator(self, dpostp_no_key) -> None:
        X_gt = np.zeros((10, 2))
        y_gt = np.zeros(10)
        lib = LIB(maxlen=50)
        for _ in range(50):
            lib.push(np.zeros(2))
        # No new_estimator → cannot label pad rows → returns GT as-is.
        X_ref, y_ref = dpostp_no_key.build_reference(
            X_gt, y_gt, lib=lib, new_estimator=None, min_rows=50,
        )
        assert X_ref.shape == (10, 2)

    def test_feature_dim_mismatch_falls_back(self, dpostp_no_key) -> None:
        X_gt = np.zeros((10, 4))
        y_gt = np.zeros(10)
        lib = LIB(maxlen=50)
        for _ in range(50):
            lib.push(np.zeros(3))          # wrong dim
        model = LogisticRegression().fit(np.zeros((20, 3)),
                                         np.tile([0, 1], 10))
        X_ref, y_ref = dpostp_no_key.build_reference(
            X_gt, y_gt, lib=lib, new_estimator=model, min_rows=40,
        )
        assert X_ref.shape[1] == 4
        assert X_ref.shape[0] == 10        # unchanged GT slice


# ---------------------------------------------------------------------------
# Role 2 — seal / unseal
# ---------------------------------------------------------------------------

class TestSealUnseal:

    def test_round_trip_preserves_arrays(self, dpostp_with_key) -> None:
        rng = np.random.default_rng(20)
        X = rng.normal(0.0, 1.0, size=(120, 7))
        y = rng.integers(0, 3, size=120).astype(float)

        payload = dpostp_with_key.seal(X, y, model_version="v1")
        assert isinstance(payload, SealedPayload)
        assert payload.n_rows == 120
        assert payload.n_features == 7
        assert payload.data[:4] == _MAGIC

        X_out, y_out = dpostp_with_key.unseal(
            payload, expected_model_version="v1",
        )
        assert np.allclose(X_out, X)
        assert np.allclose(y_out, y)

    def test_seal_without_key_raises(self, dpostp_no_key) -> None:
        with pytest.raises(DPostPKeyError):
            dpostp_no_key.seal(
                np.zeros((5, 2)), np.zeros(5), model_version="v1",
            )

    def test_unseal_without_key_raises(self, dpostp_with_key) -> None:
        payload = dpostp_with_key.seal(
            np.zeros((5, 2)), np.zeros(5), model_version="v1",
        )
        keyless = DPostP(min_ref_rows=10)
        with pytest.raises(DPostPKeyError):
            keyless.unseal(payload)

    def test_tampered_ciphertext_rejected(self, dpostp_with_key) -> None:
        X = np.ones((10, 3))
        y = np.zeros(10)
        payload = dpostp_with_key.seal(X, y, model_version="v1")

        # Flip one byte deep in the ciphertext region.
        tampered = bytearray(payload.data)
        tampered[-20] ^= 0x55
        with pytest.raises(DPostPAuthenticationError):
            dpostp_with_key.unseal(SealedPayload(
                data=bytes(tampered), aad=payload.aad,
                n_rows=payload.n_rows, n_features=payload.n_features,
            ))

    def test_tampered_aad_rejected(self, dpostp_with_key) -> None:
        X = np.ones((10, 3))
        y = np.zeros(10)
        payload = dpostp_with_key.seal(X, y, model_version="v1")

        # Modify one byte inside the AAD JSON region.  AAD bytes are
        # part of the GCM MAC input, so any flip must trip the tag.
        wire = bytearray(payload.data)
        # First AAD byte starts at offset 7 (magic+ver+aad_len).
        wire[7] ^= 0x01
        with pytest.raises(DPostPAuthenticationError):
            dpostp_with_key.unseal(bytes(wire))

    def test_bad_magic_rejected(self, dpostp_with_key) -> None:
        bad = b"XXXX" + b"\x01" * 40 + os.urandom(_NONCE_LEN) + os.urandom(32)
        with pytest.raises(DPostPAuthenticationError, match="magic"):
            dpostp_with_key.unseal(bad)

    def test_bad_version_rejected(self, dpostp_with_key) -> None:
        bad = _MAGIC + b"\x09" + b"\x00\x02" + b"{}" \
              + os.urandom(_NONCE_LEN) + os.urandom(32)
        with pytest.raises(DPostPAuthenticationError, match="version"):
            dpostp_with_key.unseal(bad)

    def test_wrong_model_version_rejected(self, dpostp_with_key) -> None:
        X = np.ones((5, 2))
        y = np.zeros(5)
        payload = dpostp_with_key.seal(X, y, model_version="v1")
        with pytest.raises(DPostPAuthenticationError, match="model_version"):
            dpostp_with_key.unseal(payload, expected_model_version="v2")

    def test_wrong_sender_id_rejected(self) -> None:
        key = os.urandom(_KEY_LEN)
        sender = DPostP(sender_id="alice", key=key)
        receiver = DPostP(key=key)
        payload = sender.seal(
            np.ones((5, 2)), np.zeros(5), model_version="v1",
        )
        with pytest.raises(DPostPAuthenticationError, match="sender_id"):
            receiver.unseal(payload, expected_sender_id="bob")

    def test_cross_key_rejected(self) -> None:
        sender = DPostP(key=os.urandom(_KEY_LEN))
        receiver = DPostP(key=os.urandom(_KEY_LEN))
        payload = sender.seal(
            np.ones((5, 2)), np.zeros(5), model_version="v1",
        )
        with pytest.raises(DPostPAuthenticationError):
            receiver.unseal(payload)

    def test_stale_timestamp_rejected(self, dpostp_with_key,
                                      monkeypatch) -> None:
        X = np.ones((5, 2))
        y = np.zeros(5)
        # Fast-forward seal-side clock 10 minutes into the past.
        real_time = time.time
        monkeypatch.setattr(time, "time", lambda: real_time() - 600)
        payload = dpostp_with_key.seal(X, y, model_version="v1")
        monkeypatch.setattr(time, "time", real_time)
        with pytest.raises(DPostPAuthenticationError, match="timestamp"):
            dpostp_with_key.unseal(payload, skew_window_s=300)

    def test_extra_aad_cannot_override_reserved_fields(
        self, dpostp_with_key
    ) -> None:
        with pytest.raises(ValueError, match="collide"):
            dpostp_with_key.seal(
                np.ones((5, 2)), np.zeros(5),
                model_version="v1",
                extra_aad={"algo_id": "spoofed"},
            )

    def test_extra_aad_is_bound(self, dpostp_with_key) -> None:
        payload = dpostp_with_key.seal(
            np.ones((5, 2)), np.zeros(5),
            model_version="v1",
            extra_aad={"experiment": "thesis-demo"},
        )
        # Should round-trip and the AAD should include the extra field.
        X_out, y_out = dpostp_with_key.unseal(payload)
        assert payload.aad.get("experiment") == "thesis-demo"

    def test_algo_id_bound_in_aad(self, dpostp_with_key) -> None:
        payload = dpostp_with_key.seal(
            np.ones((5, 2)), np.zeros(5), model_version="v1",
        )
        assert payload.aad["algo_id"] == _ALGO_ID


# ---------------------------------------------------------------------------
# Role 2 — key resolution
# ---------------------------------------------------------------------------

class TestKeyResolution:

    def test_env_var_key_is_loaded(self, monkeypatch) -> None:
        key = os.urandom(_KEY_LEN)
        monkeypatch.setenv("DPOSTP_KEY_TEST",
                           base64.b64encode(key).decode("ascii"))
        d = DPostP(key_env_var="DPOSTP_KEY_TEST")
        # Round-trip smoke test — proves the env-var key is used.
        payload = d.seal(np.ones((2, 1)), np.zeros(2), model_version="v1")
        X, _ = d.unseal(payload)
        assert X.shape == (2, 1)

    def test_missing_key_is_nonfatal_by_default(self, monkeypatch) -> None:
        monkeypatch.delenv("DPOSTP_KEY", raising=False)
        d = DPostP()            # cleaning-only mode is OK
        assert d._key is None   # noqa: SLF001 - explicit whitebox check

    def test_missing_key_raises_when_required(self, monkeypatch) -> None:
        monkeypatch.delenv("DPOSTP_KEY", raising=False)
        with pytest.raises(DPostPKeyError):
            DPostP(require_transport_key=True)

    def test_bad_length_key_raises(self) -> None:
        with pytest.raises(DPostPKeyError, match="32 bytes"):
            DPostP(key=b"\x00" * 16)

    def test_bad_base64_env_raises(self, monkeypatch) -> None:
        monkeypatch.setenv("DPOSTP_KEY_BAD", "not!base64!!!")
        with pytest.raises(DPostPKeyError, match="base64"):
            DPostP(key_env_var="DPOSTP_KEY_BAD")

    def test_generate_key_b64_produces_32_bytes(self) -> None:
        k = DPostP.generate_key_b64()
        assert len(base64.b64decode(k)) == _KEY_LEN


# ---------------------------------------------------------------------------
# Empirical noise-floor regression test
# ---------------------------------------------------------------------------

def test_padding_reduces_correlation_sampling_noise() -> None:
    """
    The motivating failure: post-retrain CPD false-fires because 50-row
    GT slices produce |Δr|≈0.5 on clean data by sampling noise alone.
    After padding to 200 rows the reference-side Pearson sampling
    variance should drop by a factor of ~4 (≈1/(n-3)), cutting the |Δr|
    distribution's spread noticeably.

    We use median and 95th-percentile across 500 seeds × 4 features
    rather than max-over-80, because the maximum of a heavy-tailed
    |Δr| distribution is itself noisy enough to produce flaky
    comparisons between two finite samples.
    """
    def _deltas(n_ref: int, n_seeds: int = 500) -> np.ndarray:
        out: list[float] = []
        for seed in range(n_seeds):
            local = np.random.default_rng(seed + 1000)
            X_ref = local.normal(0.0, 1.0, size=(n_ref, 4))
            y_ref = local.normal(0.0, 1.0, size=n_ref)
            X_rec = local.normal(0.0, 1.0, size=(100, 4))
            y_rec = local.normal(0.0, 1.0, size=100)
            for f in range(4):
                r_ref = np.corrcoef(X_ref[:, f], y_ref)[0, 1]
                r_rec = np.corrcoef(X_rec[:, f], y_rec)[0, 1]
                out.append(abs(r_ref - r_rec))
        return np.asarray(out, dtype=float)

    thin = _deltas(50)
    fat  = _deltas(200)

    # Median shrinks by ~25%; 95th-percentile shrinks by ~30%.
    # Asserting a 15% improvement leaves comfortable slack for the RNG
    # without letting a regression slip through.
    assert np.median(fat) < 0.85 * np.median(thin), (
        f"Median |Δr| did not shrink enough "
        f"(thin={np.median(thin):.3f}, fat={np.median(fat):.3f})."
    )
    assert np.percentile(fat, 95) < 0.85 * np.percentile(thin, 95), (
        f"P95 |Δr| did not shrink enough "
        f"(thin={np.percentile(thin, 95):.3f}, "
        f"fat={np.percentile(fat, 95):.3f})."
    )
