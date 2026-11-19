# Early Forecasting of Network Attack Onset from Flow Telemetry under False-Alarm Constraints

M.Tech research project on CIC-IDS2017. Each minute the system asks *"will an attack start in the
next H minutes?"* using only whole-network flow statistics (no IPs, ports, IDs or clock time), and
raises alarms under a false-alarm budget (fixed threshold, static conformal, adaptive conformal).

Design documents: [PROJECT_SPECIFICATION.md](PROJECT_SPECIFICATION.md) ·
[SYSTEM_ARCHITECTURE.md](SYSTEM_ARCHITECTURE.md) · [DATASET_INSPECTION_REPORT.md](DATASET_INSPECTION_REPORT.md) ·
[IMPLEMENTATION_TRACKER.md](IMPLEMENTATION_TRACKER.md). Results: [results/REPORT.md](results/REPORT.md).

## Quick start (Windows PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe -m streamlit run app/streamlit_app.py
```
The repository already contains the processed minute table, trained models and results, so the
dashboard works **without** the raw dataset and without retraining.
(macOS/Linux: use `.venv/bin/python` instead of `.\.venv\Scripts\python.exe`.)

## Full retraining from the raw data

1. Download CIC-IDS2017 `GeneratedLabelledFlows.zip` and extract it. You need the folder
   `TrafficLabelling/` with the 8 CSV files (≈1.1 GB; it is **not** in git).
2. Run the pipeline (≈5 min for steps 1–4, ≈20 min for step 5 on a laptop CPU):

```powershell
$py = ".\.venv\Scripts\python.exe"
& $py scripts/01_prepare.py --raw-dir "C:\path\to\TrafficLabelling"   # minute table
& $py scripts/02_build.py        # labels, features, power gate
& $py scripts/03_train.py        # LogReg / RF / XGBoost bundles -> models/
& $py scripts/04_evaluate.py     # alerting + evaluation -> results/
& $py scripts/05_experiments.py  # sensitivity, ablation, drift, leave-one-day-out
& $py scripts/05b_detection_check.py  # diagnostic: can the same features detect running attacks?
& $py scripts/06_report.py       # results/REPORT.md
& $py -m pytest -q               # tests (causality, leakage, replay == evaluation, app pages)
```
Instead of `--raw-dir` you can set `$env:NAF_RAW_DIR`. All parameters are in `configs/config.yaml`.
Changing a label/feature/model/alerting parameter changes the config hash; old model bundles then refuse
to load until `03_train.py` is re-run.

**Google Colab (optional):** upload the repo and the `TrafficLabelling` folder to Drive, then
`!pip install -r requirements-dev.txt` and run the same scripts with `!python scripts/01_prepare.py --raw-dir /content/drive/...`.
Training is small enough that a CPU runtime is sufficient.

## Deploy on Streamlit Community Cloud

1. Push this repository to GitHub (the committed `data/processed`, `models`, `results` are small).
2. On share.streamlit.io → *New app* → choose the repo, branch, main file `app/streamlit_app.py`.
3. In *Advanced settings* choose Python **3.10** (the pinned versions were verified on 3.10.1).
The hosted app only loads saved models; it never trains and never needs the raw CSVs.

## Repository layout

```
configs/config.yaml     all parameters (fixed before evaluation)
src/naf/                io_raw, clean, windows, labels, features, folds, models, registry,
                        alerting, pipeline, summary, stats, replay
scripts/01..06          pipeline steps
app/streamlit_app.py    dashboard (Plotly)
data/processed/         minute table, dataset, onsets, tiny bursts (committed)
models/F0|F1|F2/...     model bundles: model.joblib + meta.json (committed)
results/                all experiment outputs + REPORT.md (committed)
tests/                  pytest suite
```

## Honesty rules
Every number in the dashboard and report comes from `results/` produced by the scripts, or is computed
live by a saved model on recorded data. Negative or weak results are reported as they are.
