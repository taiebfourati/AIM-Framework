"""
simu5g/generator.py — Simu5G-calibrated synthetic data generator.

Generates 5G NR KPI data calibrated against Simu5G's channel model and
3GPP propagation parameters, enabling framework testing WITHOUT requiring
a full OMNeT++ / Simu5G installation.

The generator models the Simu5G radio environment:
  - 3GPP 38.901 Urban Macro channel (same as Simu5G's URBAN_MACROCELL)
  - Path loss: FSPL + log-distance with shadowing (σ=8dB)
  - Fast fading: Rayleigh distributed amplitude
  - SINR computed from received power - noise - interference
  - Throughput via Shannon capacity with AMC mapping
  - Latency from HARQ round-trip + transport delay

KPI ranges match Simu5G output:
  SS-RSRP:     [-156, -31] dBm   (3GPP TS 38.133)
  SS-SINR:     [-23, 40] dB      (3GPP TS 38.133)
  Throughput:  [0, 1000] Mbps    (NR SA, 100MHz bandwidth)
  Latency:     [1, 100] ms       (end-to-end)

Reference:
  G. Nardini et al., "Simu5G — An OMNeT++ Library for End-to-End
  Performance Evaluation of 5G Networks," IEEE Access, 2020.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

log = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Simu5G-calibrated radio parameters
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class Simu5GRadioConfig:
    """
    Radio environment parameters calibrated against Simu5G defaults.

    These values match the Simu5G URBAN_MACROCELL scenario with:
      - gNB height: 25m, UE height: 1.5m
      - Carrier: 3.5 GHz (NR band n78)
      - Bandwidth: 100 MHz, numerology μ=1 (30 kHz SCS)
      - ISD: 500m (3 gNBs, triangular layout)
    """
    # gNB parameters
    # Note: RSRP is measured per-RE (resource element), not per-beam.
    # Simu5G applies beamforming gain internally but RSRP is the
    # per-RE measurement. Effective isotropic gain for RSRP is ~0 dBi.
    gnb_tx_power_dbm: float = 46.0      # Typical macro gNB Tx power
    gnb_height_m: float = 25.0
    gnb_antenna_gain_dbi: float = 0.0   # Per-RE RSRP (no beam gain)
    num_gnbs: int = 3

    # UE parameters
    ue_height_m: float = 1.5
    ue_noise_figure_db: float = 7.0
    ue_antenna_gain_dbi: float = 0.0

    # Channel parameters (3GPP 38.901 UMa NLOS)
    carrier_freq_ghz: float = 3.5
    bandwidth_mhz: float = 100.0
    num_prbs: int = 273                  # 100 MHz @ 30 kHz SCS
    shadowing_std_db: float = 8.0        # Log-normal shadow fading σ
    thermal_noise_dbm: float = -174.0    # dBm/Hz at 290K

    # HARQ / scheduling
    harq_max_retx: int = 4
    tti_ms: float = 0.5                  # NR slot duration (μ=1)
    scheduling_overhead: float = 0.85    # Control channel overhead

    # Deployment
    isd_m: float = 500.0
    cell_radius_m: float = 400.0         # Effective coverage radius

    @property
    def noise_power_dbm(self) -> float:
        """Thermal noise power in dBm for the full bandwidth."""
        bw_hz = self.bandwidth_mhz * 1e6
        return self.thermal_noise_dbm + 10 * np.log10(bw_hz) + self.ue_noise_figure_db


# ──────────────────────────────────────────────────────────────────────────────
# Channel model (3GPP 38.901 simplified)
# ──────────────────────────────────────────────────────────────────────────────

def _path_loss_uma_nlos(d_m: float, fc_ghz: float, h_bs: float, h_ue: float) -> float:
    """
    3GPP TR 38.901 Urban Macro NLOS path loss (simplified).

    PL_UMa_NLOS = 13.54 + 39.08*log10(d3D) + 20*log10(fc) - 0.6*(h_UT-1.5)

    This matches Simu5G's channelModel implementation for URBAN_MACROCELL.

    Uses ``math.*`` (not ``numpy``) for scalar inputs — about 30x faster
    on a single float thanks to no array dispatch overhead.  This
    function is called once per RAN simulator tick (~200 Hz in the live
    dashboard); ``np.log10(scalar)`` and ``np.sqrt(scalar)`` were the
    dominant arithmetic cost in ``RANSimulator.step`` before this swap.
    """
    d3d = math.sqrt(d_m * d_m + (h_bs - h_ue) ** 2)
    if d3d < 10.0:
        d3d = 10.0  # Minimum distance

    pl = 13.54 + 39.08 * math.log10(d3d) + 20 * math.log10(fc_ghz) - 0.6 * (h_ue - 1.5)
    return pl


def _compute_sinr(
    rsrp_dbm: float,
    noise_power_dbm: float,
    interference_dbm: float = -105.0,
) -> float:
    """Compute SINR from RSRP, noise, and interference (all in dBm).

    Scalar-only math: ``math.log10`` is ~30x faster than ``np.log10`` on
    a single float, and avoids numpy's array-dispatch overhead per tick.
    """
    # x ** y where y is float beats 10 ** (rsrp_dbm / 10) only marginally,
    # but explicit math.pow is the cleanest way to make the intent clear.
    signal_mw = math.pow(10.0, rsrp_dbm * 0.1)
    noise_mw  = math.pow(10.0, noise_power_dbm * 0.1)
    interf_mw = math.pow(10.0, interference_dbm * 0.1)
    sinr_linear = signal_mw / (noise_mw + interf_mw)
    if sinr_linear < 1e-10:
        sinr_linear = 1e-10
    return 10.0 * math.log10(sinr_linear)


def _sinr_to_throughput(
    sinr_db: float,
    bandwidth_mhz: float,
    num_prbs: int,
    overhead: float,
) -> float:
    """
    Map SINR to achievable throughput using Shannon bound with AMC clipping.

    Matches Simu5G's AMC module which maps CQI → MCS → transport block size.
    Scalar-only math (``math.log2`` instead of ``np.log2``) for the same
    per-tick perf reason as the path-loss function.
    """
    sinr_lin = math.pow(10.0, sinr_db * 0.1)
    # Shannon capacity with practical AMC efficiency (~0.65)
    amc_efficiency = 0.65
    capacity_bps  = bandwidth_mhz * 1e6 * math.log2(1.0 + sinr_lin) * amc_efficiency * overhead
    capacity_mbps = capacity_bps / 1e6

    # CQI clipping: CQI 0 → 0 Mbps, CQI 15 → ~1000 Mbps (64QAM, 948/1024)
    if   capacity_mbps < 0.0:    return 0.0
    elif capacity_mbps > 1000.0: return 1000.0
    return capacity_mbps


def _compute_latency(
    sinr_db: float,
    distance_m: float,
    rng: np.random.Generator,
    harq_max: int = 4,
    tti_ms: float = 0.5,
) -> float:
    """
    Compute end-to-end latency from radio + transport components.

    Components:
      1. UE processing: 1-2 ms
      2. Transmission: 0.5 ms (1 TTI at μ=1)
      3. HARQ retransmissions: 0-4 rounds × 8 ms each (depends on SINR)
      4. Core network / transport: 2-10 ms
      5. Queuing delay: variable
    """
    # BLER from SINR (simplified sigmoid, calibrated to Simu5G BLER curves)
    # Higher sensitivity: BLER rises steeply below SINR=8 dB.
    # ``math.exp`` is ~30x faster than ``np.exp`` on scalars.
    bler = 1.0 / (1.0 + math.exp(0.6 * (sinr_db - 8.0)))

    # Number of HARQ transmissions (geometric distribution)
    n_harq = 0
    for _ in range(harq_max):
        if rng.random() < bler:
            n_harq += 1
        else:
            break

    # Component delays — more realistic model:
    #   - UE processing scales with SINR (lower SINR → more decoding time)
    #   - Core network delay includes UPF + transport (higher at cell edge)
    #   - Queuing delay increases exponentially with distance (more users
    #     share the same cell resources at the edge)
    ue_processing = rng.uniform(1.0, 3.0)
    transmission = tti_ms
    harq_delay = n_harq * 8.0  # 8 ms HARQ RTT per retransmission

    # Core network: 3-15ms base, higher at cell edge due to X2/Xn forwarding
    core_base = 3.0 + 12.0 * (distance_m / 600.0)
    core_network = rng.uniform(core_base * 0.8, core_base * 1.2)

    # Queuing: exponential with rate proportional to load (cell-edge = high load)
    queue_rate = max(1.0, 8.0 * (distance_m / 400.0))
    queuing = rng.exponential(queue_rate)

    total_ms = ue_processing + transmission + harq_delay + core_network + queuing
    # Inline scalar clamp — avoids np.clip dispatch overhead per tick.
    if   total_ms < 1.0:   return 1.0
    elif total_ms > 100.0: return 100.0
    return total_ms


# ──────────────────────────────────────────────────────────────────────────────
# Main generator
# ──────────────────────────────────────────────────────────────────────────────

class Simu5GDataGenerator:
    """
    Generate Simu5G-calibrated 5G NR KPI data for framework testing.

    Produces time-series data that statistically matches Simu5G output
    for the HandoverNR scenario, without requiring OMNeT++.

    Phase structure (aligned with handover_nr.ini):
      Phase 1 (t=0-40s):   Stable — UEs near serving gNB
      Phase 2 (t=40-60s):  Drift — UEs moving, RSRP degrades
      Phase 3a (t=60-65s): Subtle interference (background cells)
      Phase 3b (t=65-70s): Aggressive interference (jamming)
      Phase 4 (t=70-90s):  Recovery — interference cleared

    Usage:
        gen = Simu5GDataGenerator()
        X, y, timestamps = gen.generate(seed=42)
    """

    def __init__(self, config: Optional[Simu5GRadioConfig] = None) -> None:
        self.config = config or Simu5GRadioConfig()

    def generate(
        self,
        seed: int = 42,
        samples_per_second: float = 10.0,
        num_ues: int = 1,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate a full 90-second Simu5G-calibrated KPI trace.

        Parameters
        ----------
        seed : int
            Random seed for reproducibility
        samples_per_second : float
            Sampling rate (10 = one sample per 100ms TTI)
        num_ues : int
            Number of UEs to simulate (features are per-UE)

        Returns
        -------
        X : np.ndarray, shape (n_samples, 4)
            [rsrp, sinr, throughput, latency]
        y : np.ndarray, shape (n_samples,)
            Ground-truth handover labels (1=handover needed, 0=stay)
        timestamps : np.ndarray, shape (n_samples,)
            Simulation time in seconds
        """
        rng = np.random.default_rng(seed)
        cfg = self.config

        # Time vector
        total_time = 90.0  # seconds
        dt = 1.0 / samples_per_second
        timestamps = np.arange(0, total_time, dt)
        n = len(timestamps)

        # UE position trajectory (distance from serving gNB in meters)
        distance = self._ue_trajectory(timestamps, rng)

        # Per-sample KPI computation
        rsrp_arr = np.zeros(n)
        sinr_arr = np.zeros(n)
        tput_arr = np.zeros(n)
        lat_arr = np.zeros(n)

        for i, (t, d) in enumerate(zip(timestamps, distance)):
            phase = self._get_phase(t)

            # Path loss
            pl = _path_loss_uma_nlos(d, cfg.carrier_freq_ghz, cfg.gnb_height_m, cfg.ue_height_m)

            # Shadow fading (correlated across time, decorrelation distance ~50m)
            shadow = rng.normal(0, cfg.shadowing_std_db)

            # RSRP = Tx power + antenna gains - path loss - shadow fading
            rsrp = (cfg.gnb_tx_power_dbm + cfg.gnb_antenna_gain_dbi
                     + cfg.ue_antenna_gain_dbi - pl - shadow)

            # Fast fading (Rayleigh)
            fast_fade = 20 * np.log10(max(rng.rayleigh(1.0), 0.01))
            rsrp += fast_fade

            # Phase-specific interference
            interf_dbm = self._interference_level(t, rng)

            # SINR
            sinr = _compute_sinr(rsrp, cfg.noise_power_dbm, interf_dbm)

            # Apply phase-specific perturbations
            # The distance-based RSRP already shifts across phases due to the
            # UE trajectory. These additional perturbations model environmental
            # changes beyond just distance (interference, congestion, jamming).
            if phase == "drift":
                # Concept drift: increased interference from neighbouring cells
                # as UE moves to cell edge + additional building penetration
                rsrp -= rng.uniform(5, 15)
                sinr = _compute_sinr(rsrp, cfg.noise_power_dbm, interf_dbm)
            elif phase == "subtle_poison":
                # Subtle anomaly: moderate interference + sporadic jamming
                rsrp -= rng.uniform(8, 18)
                sinr -= rng.uniform(5, 12)
            elif phase == "aggressive_poison":
                # Aggressive anomaly: strong jammer + severe channel degradation
                rsrp -= rng.uniform(15, 30)
                sinr -= rng.uniform(12, 25)

            # Clip to 3GPP ranges
            rsrp = np.clip(rsrp, -156.0, -31.0)
            sinr = np.clip(sinr, -23.0, 40.0)

            # Throughput
            tput = _sinr_to_throughput(sinr, cfg.bandwidth_mhz, cfg.num_prbs, cfg.scheduling_overhead)

            # Latency
            lat = _compute_latency(sinr, d, rng, cfg.harq_max_retx, cfg.tti_ms)

            rsrp_arr[i] = rsrp
            sinr_arr[i] = sinr
            tput_arr[i] = tput
            lat_arr[i] = lat

        X = np.column_stack([rsrp_arr, sinr_arr, tput_arr, lat_arr])

        # Ground-truth handover labels (A3 event threshold)
        y = self._handover_label(rsrp_arr, sinr_arr, lat_arr)

        log.info(
            "Generated %d samples (%.1fs, %d UE). "
            "RSRP=[%.1f, %.1f], SINR=[%.1f, %.1f], "
            "Tput=[%.1f, %.1f] Mbps, Lat=[%.1f, %.1f] ms, "
            "Handover rate=%.1f%%",
            n, total_time, num_ues,
            rsrp_arr.min(), rsrp_arr.max(),
            sinr_arr.min(), sinr_arr.max(),
            tput_arr.min(), tput_arr.max(),
            lat_arr.min(), lat_arr.max(),
            100 * y.mean(),
        )

        return X, y, timestamps

    def _ue_trajectory(
        self,
        timestamps: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """
        Generate UE distance trajectory (meters from serving gNB).

        Phase 1: Mixed positions (100-350m) — both near-gNB and mid-cell,
                 creating a realistic mix of handover=0 and handover=1 labels.
        Phase 2: Moving towards cell edge (200→500m) — concept drift.
        Phase 3: At/beyond cell edge (400-600m) — poisoning phases.
        Phase 4: Moving back towards gNB (500→200m) — recovery.

        The distances are calibrated so that with gnb_tx=46dBm and 0dBi
        antenna gain, the A3 threshold (RSRP < -100, SINR < 5) triggers
        at approximately 300-400m distance.
        """
        n = len(timestamps)
        distance = np.zeros(n)

        for i, t in enumerate(timestamps):
            if t <= 40.0:
                # Stable: realistic mix — some near gNB, some at mid-cell
                # ~30% of samples at 280-400m (near cell edge, may trigger HO)
                if rng.random() < 0.30:
                    distance[i] = rng.uniform(280, 400)
                else:
                    distance[i] = rng.uniform(80, 250)
                distance[i] += rng.normal(0, 20)
            elif t <= 60.0:
                # Drift: UE moving towards and beyond cell edge
                progress = (t - 40.0) / 20.0
                distance[i] = 200 + progress * 300 + rng.normal(0, 30)
            elif t <= 70.0:
                # Poisoning: at/beyond cell edge
                distance[i] = rng.uniform(400, 650) + rng.normal(0, 25)
            else:
                # Recovery: moving back
                progress = (t - 70.0) / 20.0
                distance[i] = 500 - progress * 300 + rng.normal(0, 30)

            distance[i] = max(distance[i], 10.0)  # Minimum distance

        return distance

    def _interference_level(
        self,
        t: float,
        rng: np.random.Generator,
    ) -> float:
        """
        Background interference level (dBm) varying by phase.

        Models Simu5G's BackgroundCell interference:
          Phase 1/4: Low interference (distant cells, -110 dBm)
          Phase 2:   Moderate (neighbouring cell power, -100 dBm)
          Phase 3a:  Elevated (background cells power up, -90 dBm)
          Phase 3b:  High (near-jamming interference, -75 dBm)
        """
        phase = self._get_phase(t)
        # Without the 15 dBi beam gain, interference now dominates sooner.
        # These levels model Simu5G BackgroundCell emissions at 3.5 GHz.
        base_levels = {
            "stable": -100.0,        # normal inter-cell interference
            "drift": -90.0,          # increased load on neighbours
            "subtle_poison": -82.0,  # background cells power up
            "aggressive_poison": -70.0,  # near-jamming conditions
            "recovery": -98.0,       # subsiding interference
        }
        base = base_levels.get(phase, -100.0)
        return base + rng.normal(0, 3.0)

    @staticmethod
    def _get_phase(t: float) -> str:
        """Map simulation time to phase name."""
        if t <= 40.0:
            return "stable"
        elif t <= 60.0:
            return "drift"
        elif t <= 65.0:
            return "subtle_poison"
        elif t <= 70.0:
            return "aggressive_poison"
        else:
            return "recovery"

    @staticmethod
    def _handover_label(
        rsrp: np.ndarray,
        sinr: np.ndarray,
        latency: np.ndarray,
    ) -> np.ndarray:
        """
        A3-event handover decision boundary (same as enhanced_simulation.py).

        Handover = 1 when:
          (RSRP < -100 dBm AND SINR < 5 dB) OR latency > 50 ms
        """
        weak = (rsrp < -100.0) & (sinr < 5.0)
        high_lat = latency > 50.0
        return (weak | high_lat).astype(int)
