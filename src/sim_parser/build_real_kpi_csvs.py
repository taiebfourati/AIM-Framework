"""
sim_parser/build_real_kpi_csvs.py
================================

Fast vectorised driver that converts opp_scavetool vector CSVs exported
from Simu5G into the two consolidated files the AIMP integration test
(and the thesis pipeline) consume:

  1. simu5g_real_simulation_results.csv — long-format, one row per
     (config, run, ue, time_bin), columns:
         config, phase, run, ue, t,
         rsrp_dbm, sinr_db, cqi,
         throughput_mbps, delay_ms,
         serving_cell, handover_flag

  2. simu5g_real_summary.csv — compact cross-run summary, one row per
     (config, phase, ue), columns:
         config, phase, ue, n_samples,
         rsrp_mean, rsrp_std, sinr_mean, sinr_std,
         tput_mean, tput_std, delay_mean, delay_std,
         handovers

Design notes
------------
*  Input files (`results_rtp/RTP_{Stable,Drift,Poisoning}_vectors.csv`) are
   ~80–150 MB each.  We use chunked read + vectorised `str.split` +
   `DataFrame.explode` rather than `iterrows()` — roughly 30× faster.
*  Simu5G does NOT emit RSRP as a vector in the default `.ini`, so we
   derive it from the per-UE `distance` vector using 3GPP TR 38.901
   Urban-Macro path loss @ 2 GHz:
       PL(dB)    = 28.0 + 22·log10(d_m) + 20·log10(f_GHz)
       RSRP(dBm) = eNodeBTxPower − PL      (eNodeBTxPower = 46 dBm)
*  Handover is computed as `servingCell.diff() != 0` per UE.
*  Each config (RTP_Stable / RTP_Drift / RTP_Poisoning) is treated as a
   separate "phase" of the 4-phase AIMP evaluation.
"""

from __future__ import annotations

import argparse
import logging
import math
import re
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

log = logging.getLogger("build_real_kpi_csvs")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DEFAULT_RESULTS_DIR = Path(
    r"C:\Users\taieb\Downloads\Simu5G-1.2.2\simulations\NR"
    r"\standalone_multicell\results_rtp"
)

# Map scavetool config name  ->  logical AIMP phase label
PHASE_MAP = {
    "RTP_Stable":     "stable",
    "RTP_Drift":      "drift",
    "RTP_Poisoning":  "poison",
}

# KPI channels we care about — module-pattern + signal-name pairs.
# Regex on module field, exact match on name field.
KPI_CHANNELS: dict[str, tuple[str, str]] = {
    "sinr_db":          (r"ue\[(\d+)\]\.cellularNic\.nrChannelModel\[0\]",
                         "measuredSinrDl:vector"),
    "cqi":              (r"ue\[(\d+)\]\.cellularNic\.nrPhy",
                         "averageCqiDl:vector"),
    "throughput_bps":   (r"ue\[(\d+)\]\.cellularNic\.nrRlc\.um",
                         "rlcThroughputDl:vector"),
    "delay_s":          (r"ue\[(\d+)\]\.app\[0\]",
                         "cbrFrameDelay:vector"),
    "serving_cell":     (r"ue\[(\d+)\]\.cellularNic\.nrPhy",
                         "servingCell:vector"),
    "distance_m":       (r"ue\[(\d+)\]\.cellularNic\.nrChannelModel\[0\]",
                         "distance:vector"),
}

# Path-loss & TX power constants (from rtp_observer.ini)
F_GHZ   = 2.0
TX_DBM  = 46.0      # eNodeBTxPower
PL_BASE = 28.0 + 20.0 * math.log10(F_GHZ)  # constant term


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ue_regex_for(kpi: str) -> re.Pattern:
    return re.compile(KPI_CHANNELS[kpi][0])


def _parse_space_floats(s: str) -> np.ndarray:
    """Parse a space-separated numeric string → float32 ndarray."""
    if not isinstance(s, str) or not s:
        return np.empty(0, dtype=np.float32)
    return np.fromstring(s, sep=" ", dtype=np.float32)


def _extract_vector_rows(
    df: pd.DataFrame,
    kpi: str,
) -> pd.DataFrame:
    """
    Filter `df` (already restricted to type=='vector') to rows matching
    the given KPI channel, parse vectime/vecvalue, and return a tidy
    long-format DataFrame with columns [run, ue, t, value].
    """
    module_re = _ue_regex_for(kpi)
    target_name = KPI_CHANNELS[kpi][1]

    # Name filter first (cheapest)
    sub = df[df["name"] == target_name]
    if sub.empty:
        return pd.DataFrame(columns=["run", "ue", "t", "value"])

    # Module filter + UE extraction (vectorised via str.extract)
    ue_series = sub["module"].str.extract(module_re, expand=False)
    mask = ue_series.notna()
    if not mask.any():
        return pd.DataFrame(columns=["run", "ue", "t", "value"])

    sub = sub.loc[mask].copy()
    sub["ue"] = ue_series[mask].astype(int)

    # Parse vectime / vecvalue into arrays row-wise
    times_list  = sub["vectime"].map(_parse_space_floats).to_list()
    values_list = sub["vecvalue"].map(_parse_space_floats).to_list()

    # Build the long-format frame via explode — but keep only matching-length pairs
    records: list[pd.DataFrame] = []
    runs = sub["run"].to_numpy()
    ues  = sub["ue"].to_numpy()
    for run_id, ue_id, t_arr, v_arr in zip(runs, ues, times_list, values_list):
        n = min(len(t_arr), len(v_arr))
        if n == 0:
            continue
        records.append(pd.DataFrame({
            "run":   run_id,
            "ue":    ue_id,
            "t":     t_arr[:n],
            "value": v_arr[:n],
        }))

    if not records:
        return pd.DataFrame(columns=["run", "ue", "t", "value"])

    return pd.concat(records, ignore_index=True)


def _distance_to_rsrp_dbm(d_m: np.ndarray) -> np.ndarray:
    """
    3GPP TR 38.901 UMa LOS path loss (simplified) at 2 GHz:
        PL_dB   = 28.0 + 22·log10(d) + 20·log10(f_GHz)
        RSRP_dBm = TX_dBm − PL_dB
    Values clipped to 3GPP TS 38.133 RSRP range [−156, −31] dBm.
    """
    d = np.maximum(d_m.astype(np.float64), 1.0)  # avoid log10(0)
    pl = PL_BASE + 22.0 * np.log10(d)
    rsrp = TX_DBM - pl
    return np.clip(rsrp, -156.0, -31.0)


def _resample_to_grid(
    df: pd.DataFrame,
    bin_ms: float = 100.0,
) -> pd.DataFrame:
    """
    Resample irregular per-UE samples onto a uniform time grid.
    Takes the MEAN over each (run, ue, t_bin) tuple.
    """
    if df.empty:
        return df
    df = df.copy()
    df["t_bin"] = (df["t"] / (bin_ms / 1000.0)).astype(np.int64)
    df["t"] = df["t_bin"] * (bin_ms / 1000.0)
    agg = df.groupby(["run", "ue", "t"], as_index=False)["value"].mean()
    return agg


# ---------------------------------------------------------------------------
# Main per-config processing
# ---------------------------------------------------------------------------

def process_config(
    config: str,
    vectors_csv: Path,
    bin_ms: float = 100.0,
) -> pd.DataFrame:
    """
    Read one config's scavetool vectors CSV and return a merged long-format
    DataFrame keyed by (config, phase, run, ue, t).
    """
    log.info("[%s] loading %s (%.1f MB)",
             config, vectors_csv.name, vectors_csv.stat().st_size / 1e6)

    # Only keep vector rows — drops ~half the file.  pandas read_csv is
    # fast enough for these sizes on modern hardware; if memory is tight
    # switch to chunksize.
    df = pd.read_csv(
        vectors_csv,
        dtype={"run": "string",
               "type": "string",
               "module": "string",
               "name": "string",
               "vectime": "string",
               "vecvalue": "string"},
        low_memory=False,
    )
    df = df[df["type"] == "vector"].reset_index(drop=True)
    log.info("[%s]   %d vector rows total", config, len(df))

    per_kpi_frames: dict[str, pd.DataFrame] = {}
    for kpi in KPI_CHANNELS:
        tidy = _extract_vector_rows(df, kpi)
        if tidy.empty:
            log.warning("[%s]   KPI %s: no rows matched", config, kpi)
            continue
        tidy = _resample_to_grid(tidy, bin_ms=bin_ms)
        tidy = tidy.rename(columns={"value": kpi})
        per_kpi_frames[kpi] = tidy
        log.info("[%s]   KPI %s: %d samples after resample",
                 config, kpi, len(tidy))

    if not per_kpi_frames:
        log.error("[%s] no KPIs could be extracted", config)
        return pd.DataFrame()

    # Outer-join all KPIs on (run, ue, t)
    merged = None
    for kpi, frame in per_kpi_frames.items():
        merged = frame if merged is None else merged.merge(
            frame, on=["run", "ue", "t"], how="outer")

    merged = merged.sort_values(["run", "ue", "t"]).reset_index(drop=True)

    # Forward/backward-fill within each (run, ue) group to align sparse signals
    kpi_cols = [c for c in merged.columns if c not in ("run", "ue", "t")]
    merged[kpi_cols] = (
        merged.groupby(["run", "ue"])[kpi_cols]
        .transform(lambda s: s.ffill().bfill())
    )

    # Derive RSRP from distance
    if "distance_m" in merged.columns:
        merged["rsrp_dbm"] = _distance_to_rsrp_dbm(merged["distance_m"].to_numpy())
    else:
        merged["rsrp_dbm"] = np.nan

    # Convert throughput bps → Mbps, delay s → ms
    if "throughput_bps" in merged.columns:
        merged["throughput_mbps"] = merged["throughput_bps"] / 1e6
    if "delay_s" in merged.columns:
        merged["delay_ms"] = merged["delay_s"] * 1e3

    # Handover flag = serving-cell change per (run, ue)
    if "serving_cell" in merged.columns:
        merged["handover_flag"] = (
            merged.groupby(["run", "ue"])["serving_cell"]
            .transform(lambda s: s.diff().fillna(0).ne(0).astype(np.int8))
        )
    else:
        merged["handover_flag"] = 0

    merged.insert(0, "config", config)
    merged.insert(1, "phase", PHASE_MAP.get(config, config))

    keep = [
        "config", "phase", "run", "ue", "t",
        "rsrp_dbm", "sinr_db", "cqi",
        "throughput_mbps", "delay_ms",
        "serving_cell", "handover_flag",
    ]
    for col in keep:
        if col not in merged.columns:
            merged[col] = np.nan
    return merged[keep]


def build_summary(results: pd.DataFrame) -> pd.DataFrame:
    """Compact per (config, phase, ue) summary across runs."""
    if results.empty:
        return pd.DataFrame()

    def _agg(g: pd.DataFrame) -> pd.Series:
        return pd.Series({
            "n_samples":   len(g),
            "rsrp_mean":   g["rsrp_dbm"].mean(),
            "rsrp_std":    g["rsrp_dbm"].std(),
            "sinr_mean":   g["sinr_db"].mean(),
            "sinr_std":    g["sinr_db"].std(),
            "cqi_mean":    g["cqi"].mean(),
            "tput_mean":   g["throughput_mbps"].mean(),
            "tput_std":    g["throughput_mbps"].std(),
            "delay_mean":  g["delay_ms"].mean(),
            "delay_std":   g["delay_ms"].std(),
            "handovers":   int(g["handover_flag"].sum()),
        })

    summary = (
        results.groupby(["config", "phase", "ue"], as_index=False)
        .apply(_agg, include_groups=False)
    )
    return summary.reset_index(drop=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help="Directory holding RTP_*_vectors.csv files.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent,
        help="Directory to write the consolidated CSVs into.",
    )
    parser.add_argument(
        "--bin-ms",
        type=float,
        default=100.0,
        help="Time-bin width (ms) for resampling. Default 100 ms.",
    )
    parser.add_argument(
        "--configs",
        nargs="+",
        default=list(PHASE_MAP.keys()),
        help="Which scavetool configs to process.",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)

    frames: list[pd.DataFrame] = []
    for cfg in args.configs:
        csv = args.results_dir / f"{cfg}_vectors.csv"
        if not csv.exists():
            log.warning("skip %s — %s not found", cfg, csv)
            continue
        frames.append(process_config(cfg, csv, bin_ms=args.bin_ms))

    if not frames:
        log.error("no configs processed — aborting")
        return 1

    results = pd.concat(frames, ignore_index=True)
    summary = build_summary(results)

    out_results = args.out_dir / "simu5g_real_simulation_results.csv"
    out_summary = args.out_dir / "simu5g_real_summary.csv"

    results.to_csv(out_results, index=False)
    summary.to_csv(out_summary, index=False)

    log.info("wrote %s  (%d rows, %.1f MB)",
             out_results, len(results),
             out_results.stat().st_size / 1e6)
    log.info("wrote %s  (%d rows)", out_summary, len(summary))

    # Quick sanity preview
    log.info("\nSummary preview:\n%s",
             summary.head(12).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
