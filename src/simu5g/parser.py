"""
simu5g/parser.py — Parse Simu5G OMNeT++ result files into pandas DataFrames.

Supports three Simu5G output formats:
  1. CSV exported via `opp_scavetool export -o results.csv`
  2. Native .vec (vector) files — time-series per signal
  3. Native .sca (scalar) files — aggregate statistics

The parser extracts the KPIs needed by the RTP observer framework:
  - SS-RSRP (dBm) from rcvdSinrDl / measuredRsrp signals
  - SS-SINR (dB) from rcvdSinrDl signal
  - DL throughput (Mbps) from RLC-layer throughput signals
  - End-to-end latency (ms) from application-layer delay signals
  - Handover events from servingCell changes

Reference: OMNeT++ Result File Formats
  https://doc.omnetpp.org/omnetpp/manual/#sec:ana:result-file-formats

Usage:
    from simu5g.parser import Simu5GParser
    parser = Simu5GParser()

    # From CSV (recommended)
    df = parser.from_csv("results/campaign.csv")

    # From .vec files
    df = parser.from_vec("results/run0.vec")

    # Get RTP-ready KPI matrix
    X, timestamps = parser.to_kpi_matrix(df)
"""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Signal name mappings: Simu5G OMNeT++ signal names → RTP KPI columns
# ──────────────────────────────────────────────────────────────────────────────

# Simu5G records these signals via @statistic declarations in NED modules.
# The exact names depend on Simu5G version; we match common patterns.

RSRP_PATTERNS = [
    r"measuredRsrp",
    r"measuredSinrDl",      # Simu5G often bundles RSRP info here
    r"servingCellRsrp",
    r"rsrp",
]

SINR_PATTERNS = [
    r"rcvdSinrDl",
    r"measuredSinrDl",
    r"averageSinrDl",
    r"sinrDl",
]

THROUGHPUT_PATTERNS = [
    r"rlcThroughputDl",
    r"throughputDl",
    r"rcvdThroughput",
    r"avgThroughput",
    r".*[Tt]hroughput.*[Dd]l",
]

LATENCY_PATTERNS = [
    r"rlcDelayDl",
    r"e2eDelay",
    r"endToEndDelay",
    r"voIPReceivedDelay",
    r".*[Dd]elay.*",
]

HANDOVER_PATTERNS = [
    r"servingCell",
    r"handoverLatency",
    r"handover.*",
]


def _match_signal(name: str, patterns: list[str]) -> bool:
    """Check if a signal name matches any of the given regex patterns."""
    for pat in patterns:
        if re.search(pat, name, re.IGNORECASE):
            return True
    return False


# ──────────────────────────────────────────────────────────────────────────────
# Vector file parser
# ──────────────────────────────────────────────────────────────────────────────

class Simu5GParser:
    """
    Parse Simu5G / OMNeT++ result files into structured DataFrames.

    The parser normalises all KPIs to the ranges expected by the RTP framework:
      RSRP:       [-156, -31] dBm   (3GPP TS 38.133 Table 10.1.6.1-1)
      SINR:       [-23,  40] dB     (3GPP TS 38.133 Table 10.1.16.1-1)
      Throughput:  [0, 1000] Mbps
      Latency:     [1, 100]  ms
    """

    # 3GPP-compliant ranges (same as enhanced_simulation.py)
    RSRP_RANGE       = (-156.0, -31.0)
    SINR_RANGE       = (-23.0, 40.0)
    THROUGHPUT_RANGE  = (0.0, 1000.0)
    LATENCY_RANGE     = (1.0, 100.0)

    def __init__(self) -> None:
        self._vec_declarations: dict[int, dict] = {}
        self._raw_signals: dict[str, pd.DataFrame] = {}

    # ── CSV parsing (opp_scavetool export) ────────────────────────────────

    def from_csv(self, path: str | Path) -> pd.DataFrame:
        """
        Parse a CSV file exported by `opp_scavetool export`.

        opp_scavetool CSV columns:
          run, type, module, name, attrname, attrvalue, value,
          count, sumweights, mean, min, max, stddev,
          binedges, binvalues, vectime, vecvalue

        Returns a unified DataFrame with columns:
          [timestamp, module, ue_id, rsrp, sinr, throughput_mbps, latency_ms]
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"CSV file not found: {path}")

        log.info("Parsing Simu5G CSV: %s", path)
        df = pd.read_csv(path)

        # Separate vector rows (time-series) from scalar rows
        if "type" in df.columns:
            vectors = df[df["type"] == "vector"].copy()
            scalars = df[df["type"] == "scalar"].copy()
        else:
            # Assume all rows are data if no type column
            vectors = df.copy()
            scalars = pd.DataFrame()

        return self._process_vectors_csv(vectors)

    def _process_vectors_csv(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract time-series vectors from opp_scavetool CSV format."""
        records = []

        for _, row in df.iterrows():
            name = str(row.get("name", ""))
            module = str(row.get("module", ""))
            vectime = row.get("vectime", "")
            vecvalue = row.get("vecvalue", "")

            # Parse space-separated time and value arrays
            if pd.isna(vectime) or pd.isna(vecvalue):
                continue

            try:
                times = np.array([float(t) for t in str(vectime).split()])
                values = np.array([float(v) for v in str(vecvalue).split()])
            except (ValueError, AttributeError):
                continue

            if len(times) != len(values) or len(times) == 0:
                continue

            # Extract UE ID from module path (e.g., "HandoverNR.ue[3].app[0]")
            ue_match = re.search(r'ue\[(\d+)\]', module)
            ue_id = int(ue_match.group(1)) if ue_match else 0

            # Classify signal type
            signal_type = self._classify_signal(name)
            if signal_type is None:
                continue

            for t, v in zip(times, values):
                records.append({
                    "timestamp": t,
                    "module": module,
                    "ue_id": ue_id,
                    "signal_name": name,
                    "signal_type": signal_type,
                    "value": v,
                })

        if not records:
            log.warning("No vector data found in CSV")
            return pd.DataFrame()

        signals_df = pd.DataFrame(records)

        # Pivot: one row per (timestamp, ue_id), columns = KPIs
        return self._pivot_to_kpi(signals_df)

    # ── .vec file parsing (native OMNeT++) ────────────────────────────────

    def from_vec(self, path: str | Path) -> pd.DataFrame:
        """
        Parse a native OMNeT++ .vec file.

        .vec format:
          - Declaration lines: "vector <id> <module> <name> [ETV|TV]"
          - Data lines: "<id> <eventNumber> <simtime> <value>"
                    or: "<id> <simtime> <value>" (TV mode)

        Returns unified KPI DataFrame.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f".vec file not found: {path}")

        log.info("Parsing Simu5G .vec: %s", path)

        declarations = {}
        data_rows = []

        with open(path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or line.startswith("version"):
                    continue

                # Vector declaration
                if line.startswith("vector "):
                    parts = line.split()
                    if len(parts) >= 4:
                        vec_id = int(parts[1])
                        module = parts[2]
                        name = parts[3]
                        declarations[vec_id] = {
                            "module": module,
                            "name": name,
                        }
                    continue

                # Attribute lines
                if line.startswith("attr ") or line.startswith("param "):
                    continue

                # Data line: id eventNum time value  OR  id time value
                parts = line.split()
                if len(parts) < 3:
                    continue

                try:
                    vec_id = int(parts[0])
                    if len(parts) == 4:
                        # ETV format: id event time value
                        simtime = float(parts[2])
                        value = float(parts[3])
                    elif len(parts) == 3:
                        # TV format: id time value
                        simtime = float(parts[1])
                        value = float(parts[2])
                    else:
                        continue

                    if vec_id in declarations:
                        decl = declarations[vec_id]
                        ue_match = re.search(r'ue\[(\d+)\]', decl["module"])
                        ue_id = int(ue_match.group(1)) if ue_match else 0

                        signal_type = self._classify_signal(decl["name"])
                        if signal_type:
                            data_rows.append({
                                "timestamp": simtime,
                                "module": decl["module"],
                                "ue_id": ue_id,
                                "signal_name": decl["name"],
                                "signal_type": signal_type,
                                "value": value,
                            })
                except (ValueError, IndexError):
                    continue

        if not data_rows:
            log.warning("No recognisable KPI data in .vec file")
            return pd.DataFrame()

        signals_df = pd.DataFrame(data_rows)
        return self._pivot_to_kpi(signals_df)

    # ── .sca file parsing ─────────────────────────────────────────────────

    def from_sca(self, path: str | Path) -> pd.DataFrame:
        """
        Parse a native OMNeT++ .sca file (aggregate per-run statistics).

        Returns a flat DataFrame with one row per (module, statistic) pair.
        Useful for aggregate analysis but NOT for time-series RTP feeding.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f".sca file not found: {path}")

        records = []
        current_run = ""

        with open(path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if line.startswith("run "):
                    current_run = line.split(maxsplit=1)[1]
                elif line.startswith("scalar "):
                    parts = line.split()
                    if len(parts) >= 4:
                        records.append({
                            "run": current_run,
                            "module": parts[1],
                            "name": parts[2],
                            "value": float(parts[3]),
                        })
                elif line.startswith("statistic "):
                    parts = line.split()
                    if len(parts) >= 3:
                        records.append({
                            "run": current_run,
                            "module": parts[1],
                            "name": parts[2],
                            "value": np.nan,
                        })

        return pd.DataFrame(records) if records else pd.DataFrame()

    # ── Signal classification ─────────────────────────────────────────────

    @staticmethod
    def _classify_signal(name: str) -> Optional[str]:
        """Map an OMNeT++/Simu5G signal name to a KPI type."""
        if _match_signal(name, RSRP_PATTERNS):
            return "rsrp"
        if _match_signal(name, SINR_PATTERNS):
            return "sinr"
        if _match_signal(name, THROUGHPUT_PATTERNS):
            return "throughput"
        if _match_signal(name, LATENCY_PATTERNS):
            return "latency"
        if _match_signal(name, HANDOVER_PATTERNS):
            return "handover"
        return None

    # ── Pivot time-series into KPI matrix ─────────────────────────────────

    def _pivot_to_kpi(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert long-format signal data into a wide KPI matrix.

        Groups by (timestamp rounded to TTI, ue_id) and takes the last
        value per signal type within each time bin.
        """
        if df.empty:
            return pd.DataFrame()

        # Bin timestamps to TTI granularity (1ms for NR numerology 1)
        df = df.copy()
        df["time_bin"] = (df["timestamp"] * 1000).round(0) / 1000  # 1ms bins

        # Pivot: one row per (time_bin, ue_id)
        kpi_types = ["rsrp", "sinr", "throughput", "latency"]
        pivot_frames = []

        for kpi in kpi_types:
            subset = df[df["signal_type"] == kpi]
            if subset.empty:
                continue
            grouped = (
                subset.groupby(["time_bin", "ue_id"])["value"]
                .last()
                .reset_index()
                .rename(columns={"value": kpi})
            )
            pivot_frames.append(grouped)

        if not pivot_frames:
            return pd.DataFrame()

        # Merge all KPIs on (time_bin, ue_id)
        result = pivot_frames[0]
        for pf in pivot_frames[1:]:
            result = result.merge(pf, on=["time_bin", "ue_id"], how="outer")

        result = result.sort_values(["ue_id", "time_bin"]).reset_index(drop=True)

        # Rename time_bin to timestamp
        result = result.rename(columns={"time_bin": "timestamp"})

        return result

    # ── Convert to RTP-ready feature matrix ───────────────────────────────

    def to_kpi_matrix(
        self,
        df: pd.DataFrame,
        ue_id: Optional[int] = None,
        interpolate: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Convert a KPI DataFrame into a numpy feature matrix ready for RTP.

        Parameters
        ----------
        df : pd.DataFrame
            Output from from_csv() / from_vec() with columns:
            [timestamp, ue_id, rsrp, sinr, throughput, latency]
        ue_id : int, optional
            Filter to a specific UE. If None, interleaves all UEs.
        interpolate : bool
            Forward-fill missing KPI values (common when signals
            are recorded at different rates).

        Returns
        -------
        X : np.ndarray, shape (n_samples, 4)
            Feature matrix: [rsrp, sinr, throughput_mbps, latency_ms]
        timestamps : np.ndarray, shape (n_samples,)
            Simulation timestamps for each sample.
        """
        if df.empty:
            return np.empty((0, 4)), np.empty(0)

        if ue_id is not None:
            df = df[df["ue_id"] == ue_id].copy()

        df = df.sort_values("timestamp").reset_index(drop=True)

        kpi_cols = ["rsrp", "sinr", "throughput", "latency"]

        # Ensure all KPI columns exist
        for col in kpi_cols:
            if col not in df.columns:
                df[col] = np.nan

        if interpolate:
            df[kpi_cols] = df[kpi_cols].ffill().bfill()

        # Drop rows where all KPIs are still NaN
        df = df.dropna(subset=kpi_cols, how="all")

        # Fill remaining NaN with column medians
        for col in kpi_cols:
            if df[col].isna().any():
                df[col] = df[col].fillna(df[col].median())

        # Unit conversion (Simu5G reports some signals in linear / different units)
        # RSRP: Simu5G may report in dBm (no conversion) or linear power
        # SINR: Simu5G reports in dB (no conversion needed)
        # Throughput: Simu5G reports in bps → convert to Mbps
        # Latency: Simu5G reports in seconds → convert to ms

        # Detect if throughput is in bps (values > 10000 suggest bps)
        if df["throughput"].median() > 10000:
            df["throughput"] = df["throughput"] / 1e6  # bps → Mbps

        # Detect if latency is in seconds (values < 1 suggest seconds)
        if df["latency"].median() < 1.0:
            df["latency"] = df["latency"] * 1000  # s → ms

        # Clip to 3GPP-compliant ranges
        df["rsrp"] = np.clip(df["rsrp"], self.RSRP_RANGE[0], self.RSRP_RANGE[1])
        df["sinr"] = np.clip(df["sinr"], self.SINR_RANGE[0], self.SINR_RANGE[1])
        df["throughput"] = np.clip(df["throughput"], self.THROUGHPUT_RANGE[0], self.THROUGHPUT_RANGE[1])
        df["latency"] = np.clip(df["latency"], self.LATENCY_RANGE[0], self.LATENCY_RANGE[1])

        X = df[kpi_cols].values.astype(np.float64)
        timestamps = df["timestamp"].values.astype(np.float64)

        return X, timestamps

    # ── Handover label extraction ─────────────────────────────────────────

    def extract_handover_labels(
        self,
        df: pd.DataFrame,
        ue_id: Optional[int] = None,
    ) -> np.ndarray:
        """
        Extract ground-truth handover labels from Simu5G serving cell changes.

        A handover event (label=1) is detected when the serving cell ID changes
        between consecutive time steps. This provides TRUE labels (not synthetic)
        for validation.

        If no servingCell signal is available, falls back to the A3-event
        threshold decision boundary.
        """
        if ue_id is not None:
            df = df[df["ue_id"] == ue_id].copy()

        df = df.sort_values("timestamp").reset_index(drop=True)

        # Try to use actual serving cell changes
        if "handover" in df.columns and not df["handover"].isna().all():
            serving = df["handover"].ffill().fillna(0)
            labels = (serving.diff().abs() > 0).astype(int).values
            labels[0] = 0  # First sample has no prior cell
            return labels

        # Fallback: use A3-event threshold (same as enhanced_simulation.py)
        log.info("No servingCell signal found; using A3-event threshold labels")
        rsrp = df.get("rsrp", pd.Series(dtype=float)).values
        sinr = df.get("sinr", pd.Series(dtype=float)).values
        latency = df.get("latency", pd.Series(dtype=float)).values

        weak = (rsrp < -100.0) & (sinr < 5.0)
        high_lat = latency > 50.0
        return (weak | high_lat).astype(int)

    # ── Multi-run batch loading ───────────────────────────────────────────

    def load_campaign(
        self,
        result_dir: str | Path,
        pattern: str = "*.vec",
    ) -> dict[int, pd.DataFrame]:
        """
        Load all result files from a Simu5G campaign directory.

        Returns a dict mapping run_index → KPI DataFrame.
        """
        result_dir = Path(result_dir)
        files = sorted(result_dir.glob(pattern))

        if not files:
            log.warning("No files matching '%s' in %s", pattern, result_dir)
            return {}

        campaign: dict[int, pd.DataFrame] = {}
        for i, f in enumerate(files):
            log.info("Loading run %d: %s", i, f.name)
            if f.suffix == ".csv":
                campaign[i] = self.from_csv(f)
            elif f.suffix == ".vec":
                campaign[i] = self.from_vec(f)
            else:
                log.warning("Skipping unsupported file: %s", f)

        return campaign
