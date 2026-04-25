"""
run_simu5g_parallel.py
======================

End-to-end "simulate-then-observe" driver that:

  1. Launches opp_run for RTP_Stable / RTP_Drift / RTP_Poisoning **in
     parallel** as subprocesses.
  2. Waits for all three to finish, streaming each one's stdout to its
     own log file.
  3. Exports each config's .vec files to CSV via opp_scavetool.
  4. Runs sim_parser/build_real_kpi_csvs.py to produce
     simu5g_real_simulation_results.csv + simu5g_real_summary.csv.
  5. (Optional, `--aimp`) invokes test_aimp_real_simu5g.py against the
     freshly-produced CSV so the observer sees the new run.

Rationale
---------
OMNeT++ writes .vec files with an index that is only finalised at run
completion, so a true "streaming observer that consumes KPIs as they
happen inside the simulator" would require a custom C++ module.  In
lieu of that, running the three multi-phase configs **concurrently**
minimises wall-clock time, and the observer is invoked the instant the
parser finishes — which is the closest thing to parallel observation
the OMNeT++ toolchain supports out of the box.

Usage
-----
    py -3.13 run_simu5g_parallel.py                 # sim + parse
    py -3.13 run_simu5g_parallel.py --aimp          # sim + parse + AIMP
    py -3.13 run_simu5g_parallel.py --skip-sim      # just parse + AIMP
    py -3.13 run_simu5g_parallel.py -c RTP_Stable   # single config
"""

from __future__ import annotations

import argparse
import logging
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

log = logging.getLogger("simu5g_parallel")

# ---------------------------------------------------------------------------
# Paths (Windows-native, MSYS2/MinGW toolchain)
# ---------------------------------------------------------------------------

OMNETPP_ROOT = Path(
    r"C:\Users\taieb\Downloads\omnetpp-6.0.3-windows-x86_64\omnetpp-6.0.3"
)
INET_ROOT    = Path(r"C:\Users\taieb\Downloads\inet-4.5.4-src\inet4.5")
SIMU5G_ROOT  = Path(r"C:\Users\taieb\Downloads\Simu5G-1.2.2")

SIM_DIR      = SIMU5G_ROOT / "simulations" / "NR" / "standalone_multicell"
RESULTS_DIR  = SIM_DIR / "results_rtp"
INI_FILE     = "rtp_observer.ini"

OPP_RUN       = OMNETPP_ROOT / "bin" / "opp_run.exe"
OPP_SCAVETOOL = OMNETPP_ROOT / "bin" / "opp_scavetool.exe"

# MSYS2 bash shipped with OMNeT++.  The OMNeT++/INET setenv scripts rely on
# MSYS2 path-manipulation primitives and environment variables that are
# awkward to replicate in a pure-Python os.environ copy, so we delegate to
# bash which is the supported launch path.
MSYS_BASH = OMNETPP_ROOT / "tools" / "win32.x86_64" / "usr" / "bin" / "bash.exe"

DEFAULT_CONFIGS = ("RTP_Stable", "RTP_Drift", "RTP_Poisoning")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _win2msys(p: Path) -> str:
    """Convert `C:\\foo\\bar` to `/c/foo/bar` for MSYS2 bash."""
    s = str(p).replace("\\", "/")
    if len(s) > 1 and s[1] == ":":
        s = f"/{s[0].lower()}{s[2:]}"
    return s


def _bash_env() -> dict[str, str]:
    """Inherit minimal env; bash setenv does the heavy lifting."""
    env = os.environ.copy()
    env["MSYSTEM"] = "MINGW64"
    env["CHERE_INVOKING"] = "1"
    return env


def _opp_run_bash_cmd(config: str, log_path: Path) -> str:
    """Build the bash `-c` command string that sources setenv and runs opp_run."""
    ned_path = ":".join([
        _win2msys(SIMU5G_ROOT / "simulations"),
        _win2msys(SIMU5G_ROOT / "src"),
        _win2msys(INET_ROOT / "src"),
    ])
    # libINET.dll and libsimu5g.dll live in their respective src/ dirs.
    # OMNeT++'s LoadLibrary needs those dirs on PATH to resolve the DLL's
    # own dependencies (neither setenv adds them by default).
    dll_path_extra = ":".join([
        _win2msys(INET_ROOT / "src"),
        _win2msys(SIMU5G_ROOT / "src"),
    ])
    return (
        f"source {shlex.quote(_win2msys(OMNETPP_ROOT))}/setenv -q && "
        f"source {shlex.quote(_win2msys(INET_ROOT))}/setenv -q && "
        f"export PATH={shlex.quote(dll_path_extra)}:$PATH && "
        f"cd {shlex.quote(_win2msys(SIM_DIR))} && "
        f"opp_run -u Cmdenv -f {INI_FILE} -c {config} "
        f"-n {shlex.quote(ned_path)} "
        f"-l {shlex.quote(_win2msys(SIMU5G_ROOT / 'src' / 'simu5g'))} "
        f"-l {shlex.quote(_win2msys(INET_ROOT / 'src' / 'INET'))} "
        f"--result-dir={shlex.quote(_win2msys(RESULTS_DIR))} "
        f"> {shlex.quote(_win2msys(log_path))} 2>&1"
    )


def launch_config(config: str) -> tuple[subprocess.Popen, Path]:
    """Launch opp_run for one config via MSYS2 bash, return (process, log)."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    log_path = RESULTS_DIR / f"{config}_log.txt"
    bash_cmd = _opp_run_bash_cmd(config, log_path)
    cmd = [str(MSYS_BASH), "-lc", bash_cmd]
    log.info("[%s] launching via bash: opp_run -c %s", config, config)
    log.debug("[%s]   bash cmd: %s", config, bash_cmd)
    proc = subprocess.Popen(
        cmd,
        env=_bash_env(),
    )
    return proc, log_path


def export_csvs(config: str) -> tuple[Path | None, Path | None]:
    """Run opp_scavetool to export .vec/.sca → CSV."""
    config_dir = RESULTS_DIR / config
    if not config_dir.is_dir():
        log.warning("[%s] no result dir — skipping CSV export", config)
        return None, None

    vecs = sorted(config_dir.glob("*.vec"))
    scas = sorted(config_dir.glob("*.sca"))
    if not vecs:
        log.warning("[%s] no .vec files — skipping CSV export", config)
        return None, None

    vec_csv = RESULTS_DIR / f"{config}_vectors.csv"
    sca_csv = RESULTS_DIR / f"{config}_scalars.csv"

    # opp_scavetool 6.0.3 quirks:
    #   * exporter selector is `-F CSV-R` (NOT `-f csv`; lowercase `-f` is the
    #     filter-expression flag).
    #   * Simu5G emits result files whose names start with `-` (e.g. `-0.vec`)
    #     when the INI leaves result-file-name-base unset. A bare glob on the
    #     command line then makes scavetool parse them as CLI options. Working
    #     inside the config_dir and passing `./filename` sidesteps that.
    cfg_msys = _win2msys(config_dir)

    def _scave_bash(out: Path, files: list[Path]) -> str:
        file_args = " ".join(shlex.quote(f"./{f.name}") for f in files)
        return (
            f"source {shlex.quote(_win2msys(OMNETPP_ROOT))}/setenv -q && "
            f"cd {shlex.quote(cfg_msys)} && "
            f"opp_scavetool export -F CSV-R "
            f"-o {shlex.quote(_win2msys(out))} {file_args}"
        )

    log.info("[%s] exporting %d .vec files → %s", config, len(vecs), vec_csv)
    subprocess.run(
        [str(MSYS_BASH), "-lc", _scave_bash(vec_csv, vecs)],
        check=True, env=_bash_env(),
    )
    vec_size = vec_csv.stat().st_size if vec_csv.exists() else 0
    if vec_size < 1024:
        # Guard against the "Exported empty data set" failure silently
        # clobbering a good prior CSV. Remove the stub so downstream tooling
        # (and any preserved backup) can be handled explicitly.
        log.error("[%s] vector export produced only %d bytes — "
                  "treating as failure (not overwriting parser inputs)",
                  config, vec_size)
        try:
            vec_csv.unlink()
        except OSError:
            pass
        return None, None

    if scas:
        log.info("[%s] exporting %d .sca files → %s",
                 config, len(scas), sca_csv)
        subprocess.run(
            [str(MSYS_BASH), "-lc", _scave_bash(sca_csv, scas)],
            check=True, env=_bash_env(),
        )
    return vec_csv, sca_csv


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------

def run_parallel(configs: list[str]) -> dict[str, int]:
    """Launch all configs concurrently, return {config: returncode}."""
    started: dict[str, tuple[subprocess.Popen, Path, float]] = {}
    for cfg in configs:
        proc, log_path = launch_config(cfg)
        started[cfg] = (proc, log_path, time.time())

    rcs: dict[str, int] = {}
    while started:
        for cfg in list(started.keys()):
            proc, log_path, t0 = started[cfg]
            rc = proc.poll()
            if rc is None:
                continue
            elapsed = time.time() - t0
            log.info("[%s] finished rc=%d in %.1fs (log: %s)",
                     cfg, rc, elapsed, log_path.name)
            rcs[cfg] = rc
            del started[cfg]
        if started:
            time.sleep(2)
    return rcs


def run_python(script: Path, extra_args: list[str] | None = None) -> int:
    extra_args = extra_args or []
    cmd = [sys.executable, str(script), *extra_args]
    log.info("running: %s", " ".join(shlex.quote(c) for c in cmd))
    return subprocess.call(cmd)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("-c", "--config", action="append", default=[],
                   choices=list(DEFAULT_CONFIGS),
                   help="Run only these configs (repeatable). Default: all three.")
    p.add_argument("--skip-sim", action="store_true",
                   help="Skip opp_run; go straight to CSV export + parse.")
    p.add_argument("--skip-export", action="store_true",
                   help="Skip opp_scavetool export; assume CSVs already exist.")
    p.add_argument("--aimp", action="store_true",
                   help="After parsing, also run test_aimp_real_simu5g.py.")
    p.add_argument("--max-per-phase", type=int, default=None,
                   help="Passed through to test_aimp_real_simu5g.py when --aimp.")
    args = p.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    configs = args.config or list(DEFAULT_CONFIGS)
    log.info("pipeline starting. configs=%s, skip_sim=%s, skip_export=%s, aimp=%s",
             configs, args.skip_sim, args.skip_export, args.aimp)

    project_root = Path(__file__).resolve().parent

    # ── 1. Run Simu5G in parallel ─────────────────────────────────────
    if not args.skip_sim:
        if not OPP_RUN.exists():
            log.error("opp_run not found at %s", OPP_RUN)
            return 2
        if not (SIM_DIR / INI_FILE).exists():
            log.error("INI not found: %s", SIM_DIR / INI_FILE)
            return 2

        t0 = time.time()
        rcs = run_parallel(configs)
        log.info("all opp_run processes done in %.1fs: %s",
                 time.time() - t0, rcs)
        failed = [c for c, rc in rcs.items() if rc != 0]
        if failed:
            log.error("opp_run failed for: %s (check per-config _log.txt files)",
                      failed)
            return 3

    # ── 2. Export CSVs ────────────────────────────────────────────────
    if not args.skip_export:
        for cfg in configs:
            export_csvs(cfg)

    # ── 3. Run the parser driver ──────────────────────────────────────
    parser_rc = run_python(
        project_root / "sim_parser" / "build_real_kpi_csvs.py",
        ["--results-dir", str(RESULTS_DIR),
         "--out-dir",     str(project_root),
         "--configs",     *configs],
    )
    if parser_rc != 0:
        log.error("parser driver returned %d", parser_rc)
        return parser_rc

    # ── 4. Optionally kick off the AIMP observer ──────────────────────
    if args.aimp:
        aimp_args: list[str] = []
        if args.max_per_phase is not None:
            aimp_args += ["--max-per-phase", str(args.max_per_phase)]
        aimp_rc = run_python(
            project_root / "test_aimp_real_simu5g.py", aimp_args
        )
        if aimp_rc != 0:
            log.error("AIMP test returned %d", aimp_rc)
            return aimp_rc

    log.info("pipeline complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
