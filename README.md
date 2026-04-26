# RTP-Observer — AI Management Framework for AI-Native 5G/6G Networks

Reference implementation of the AI Management Platform (AIMP) described in the
Bachelor's thesis *"Implementation of an AI Management Framework for AI-Native
5G/6G Networks"* (Taieb Fourati, Faculty of Electronics and Information
Technology, Warsaw University of Technology, 2026).

The framework operates a closed-loop **Real-Time Processor (RTP) ->
Adaptive-Training Manager (ATM) -> Network Digital Twin (NDT) -> RAN actuator**
pipeline that detects four classes of input drift, retrains the production model
out-of-band, validates the candidate against a digital twin, and only then
deploys it to the live RAN.

---

## Repository layout

The project follows the standard Python **src layout**: all importable
packages live under `src/`, while top-level entry-point scripts, tests, and
figure-generation utilities stay at the repository root.

```
.
├── src/                  # All importable Python packages (installed via `pip install -e .`)
│   ├── aif/              #   Adversarial-Input Filter: golden corpus, audit chain, DPostP/AEAD
│   ├── aimp/             #   Top-level AIMP orchestrator + MTP-C contract test
│   ├── atm/              #   Adaptive Training Manager (MTP-L / MTP-E / MTP-C variants)
│   ├── baselines/        #   Comparison baselines (ADWIN, CUSUM, spectral, clean-label attack)
│   ├── dashboard/        #   Streamlit dashboard + FastAPI live engine
│   │   └── live/         #     Live RAN simulator, RAN actuator, WebSocket server
│   ├── detectors/        #   Drift detectors: DDD (KS+MMD), CDD (Page-Hinkley), DPD (IF), CPD
│   ├── ndt/              #   Network Digital Twin validation gate
│   ├── rtp/              #   Real-Time Processor (sample-loop core, MToUT signal)
│   ├── sim_parser/       #   Parser for OMNeT++ Simu5G .vec result files -> KPI CSVs
│   └── simu5g/           #   Simu5G adapter, generator, scenarios (.ini/.ned)
│
├── scripts/              # Ablation, baseline comparison, bias-floor sweep, overhead measurement
├── tests/                # Unit tests (per-detector) + integration tests (E2E scenarios)
├── thesis_figures/       # Scripts that produce the figures used in the thesis (gen_arch,
│                         #   generate_eval_figures, regenerate_evaluation)
│
├── main.py                          # AIMP entry point (single-run simulation)
├── enhanced_simulation.py           # Full multi-phase synthetic simulation
├── simu5g_simulation.py             # Run AIMP against real Simu5G traces
├── run_simu5g_parallel.py           # Parallel sweep launcher
├── generate_figures.py              # Master figure / experiment driver
├── test_aimp_integration.py         # Top-level smoke test: AIMP on synthetic data
├── test_aimp_real_simu5g.py         # Top-level smoke test: AIMP on real Simu5G data
├── live_ran_corpus.csv              # Golden corpus seed for the AIF gate
├── dashboard.bat                    # One-click dashboard launcher (Windows)
├── pyproject.toml                   # Build / install configuration (src layout)
├── requirements.txt                 # All Python dependencies
└── requirements_dashboard.txt       # Subset for the dashboard only
```

---

## Quick start

```bash
# 1. Create and activate a virtualenv
python -m venv .venv
.\.venv\Scripts\activate            # Windows
# source .venv/bin/activate         # Linux/macOS

# 2. Install the project in editable mode
#    This installs all runtime dependencies AND makes the packages under
#    src/ (aif, aimp, atm, ...) importable from anywhere.
pip install -e .

#    To also install the dashboard / FastAPI live engine:
pip install -e ".[dashboard]"

#    To install dev / test extras as well:
pip install -e ".[dev]"

# 3. Run a synthetic end-to-end simulation
python main.py

# 4. Run AIMP against pre-recorded Simu5G traces
python simu5g_simulation.py

# 5. Launch the live dashboard
streamlit run src/dashboard/app.py
# or on Windows:
dashboard.bat

---

## Tests

```bash
# All unit tests (per-detector, per-component)
pytest tests/unit -v

# Integration scenarios (drift, poisoning, retraining, golden-corpus tampering, ...)
pytest tests/integration -v

# Top-level smoke tests
pytest test_aimp_integration.py test_aimp_real_simu5g.py -v
```

---

## Reproducing thesis figures

The scripts in `thesis_figures/` regenerate every PDF figure included in the
thesis manuscript:

```bash
# Architecture / block-diagram figures (TikZ-compatible PDF output)
python thesis_figures/gen_arch.py

# All evaluation figures (headline boxplot, ROC/PR, calibration,
#  significance matrix, sweep Pareto, latency CDF, ablation, ...)
python thesis_figures/generate_eval_figures.py

# Re-run the full evaluation pipeline (writes to results/ then plots)
python thesis_figures/regenerate_evaluation.py
```


---

## Simu5G reference

The radio-layer traces consumed by `simu5g_simulation.py` are produced by
**Simu5G**, an open-source OMNeT++ / INET-based 5G New Radio simulator developed
at the University of Pisa.

- Project page: <https://simu5g.org/>
- GitHub: <https://github.com/Unipisa/Simu5G>
- Paper: G. Nardini, D. Sabella, G. Stea, P. Thakkar, A. Virdis,
  *"Simu5G — An OMNeT++ Library for End-to-End Performance Evaluation of 5G
  Networks"*, IEEE Access, vol. 8, pp. 181176-181191, 2020,
  doi:10.1109/ACCESS.2020.3028550.

The OMNeT++ scenarios used in this thesis (`simu5g/scenarios/handover_nr.ini`
and `handover_nr.ned`) target Simu5G ≥ 1.2 with INET 4.5. After running the
scenarios in OMNeT++ the resulting `.vec` files are parsed into the KPI CSVs
under `simu5g/results/` by `sim_parser/build_real_kpi_csvs.py`. The `.vec`
files themselves are *not* committed to the repository (they are large and
fully regenerable from the included scenarios).

---

## Citation

If you use this code, please cite the thesis:

```bibtex
@mastersthesis{fourati2026aimp,
  author  = {Taieb Fourati},
  title   = {Implementation of an AI Management Framework for AI-Native 5G/6G Networks},
  school  = {Warsaw University of Technology, Faculty of Electronics and Information Technology},
  year    = {2026},
  type    = {Bachelor's thesis}
}
```

---

## License

See `LICENSE` (to be added before public release).
