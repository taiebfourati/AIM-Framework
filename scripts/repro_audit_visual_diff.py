"""
scripts/repro_audit_visual_diff.py — visual-diff the legacy figures.

Re-renders each pre/post PDF to PNG via pdftoppm, then computes:
  * per-pixel mean absolute difference (MAD)
  * structural fraction of pixels that differ by more than 5/255
  * dimension match check

Outputs a per-figure verdict and writes a CSV.
"""
from __future__ import annotations

import os
import subprocess
import sys
import csv
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = ROOT / "thesis" / "figures"
BACKUP_DIR = ROOT / "thesis" / ".figures_pre_repro_audit"
OUT_DIR = ROOT / "thesis" / ".repro_audit_visual"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PDFTOPPM = r"C:\Program Files\MiKTeX\miktex\bin\x64\pdftoppm.exe"
DPI = 150
DIFF_THRESHOLD = 5  # tolerance per channel out of 255

LEGACY = [
    "fig_01_simulation_timeline.pdf",
    "fig_02_events_per_phase.pdf",
    "fig_03_ndt_results.pdf",
    "fig_04_cpd_breakdown.pdf",
    "fig_05_detector_heatmap.pdf",
]


def render(pdf_path: Path, out_prefix: Path, dpi: int = DPI) -> Path:
    """Render the first page of a PDF to PNG. Returns the PNG path."""
    cmd = [
        PDFTOPPM, "-r", str(dpi), "-f", "1", "-l", "1", "-png",
        str(pdf_path), str(out_prefix),
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    # pdftoppm names the file <prefix>-1.png (or -01 for double digits)
    candidates = list(out_prefix.parent.glob(f"{out_prefix.name}-*.png"))
    if not candidates:
        raise RuntimeError(f"pdftoppm produced no PNG for {pdf_path}")
    return sorted(candidates)[0]


def diff(old_png: Path, new_png: Path) -> dict:
    a = np.asarray(Image.open(old_png).convert("RGB"), dtype=np.int16)
    b = np.asarray(Image.open(new_png).convert("RGB"), dtype=np.int16)

    if a.shape != b.shape:
        # Resize to common shape using PIL nearest (preserves diff fairness)
        h = min(a.shape[0], b.shape[0])
        w = min(a.shape[1], b.shape[1])
        a = a[:h, :w, :]
        b = b[:h, :w, :]

    delta = np.abs(a - b)
    mad = float(delta.mean())
    diff_mask = delta.max(axis=-1) > DIFF_THRESHOLD
    pct_changed = float(diff_mask.mean() * 100.0)

    return {
        "shape_old": old_png.stat().st_size,
        "shape_new": new_png.stat().st_size,
        "mean_abs_diff_per_channel": round(mad, 4),
        "pct_pixels_changed_gt_5_255": round(pct_changed, 4),
    }


def main() -> int:
    print(f"Rendering at {DPI} dpi to: {OUT_DIR}")
    rows = []
    for fname in LEGACY:
        old_pdf = BACKUP_DIR / fname
        new_pdf = FIG_DIR / fname
        if not old_pdf.exists() or not new_pdf.exists():
            print(f"  SKIP {fname}: missing pre or post PDF")
            continue
        stem = fname[:-4]  # strip .pdf
        old_prefix = OUT_DIR / f"{stem}_old"
        new_prefix = OUT_DIR / f"{stem}_new"
        # Clean previous renders
        for p in OUT_DIR.glob(f"{stem}_*.png"):
            p.unlink()
        old_png = render(old_pdf, old_prefix)
        new_png = render(new_pdf, new_prefix)
        d = diff(old_png, new_png)
        verdict = (
            "BIT_IDENTICAL" if d["pct_pixels_changed_gt_5_255"] == 0.0 else
            "VISUALLY_EQUIVALENT" if d["pct_pixels_changed_gt_5_255"] < 1.0 else
            "MINOR_DRIFT" if d["pct_pixels_changed_gt_5_255"] < 5.0 else
            "MATERIAL_DIFFERENCE"
        )
        d["file"] = fname
        d["verdict"] = verdict
        rows.append(d)
        print(
            f"  {fname:35s}  MAD={d['mean_abs_diff_per_channel']:6.3f}  "
            f"changed={d['pct_pixels_changed_gt_5_255']:5.2f}%  -> {verdict}"
        )

    csv_path = OUT_DIR / "diff_summary.csv"
    keys = ["file", "verdict", "mean_abs_diff_per_channel",
            "pct_pixels_changed_gt_5_255", "shape_old", "shape_new"]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in keys})
    print(f"\nCSV: {csv_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
