# SYSTEM ARCHITECTURE  (v0.3 — as implemented)

One pipeline, one config, fixed data contracts. Heavy work (training) writes artefacts; the hosted app only reads them.

```
RAW CSV (outside git)
  │ 01_prepare     clean, repair timestamps, availability minute = ceil(start+duration)+1
  ▼
data/processed/minute_table.parquet            CONTRACT A  (committed, 2,454 rows)
  │ 02_build       onset/tiny-burst/campaign labels, features, POWER GATE report
  ▼
data/processed/dataset.parquet (+ onsets.csv, tiny_bursts.csv, feature_names.json)   CONTRACT B
  │ 03_train       LogReg (1 run), RF, XGB (seeds 0-4) for folds F0/F1/F2 -> models/
  │ 04_evaluate    load SAVED bundles, alert thresholds from calibration negatives,
  │                event metrics, baselines, pooled intervals -> results/
  │ 05_experiments sensitivity (G,H), no-Monday ablation, drift, leave-one-day-out diagnostic
  │ 06_report      results/REPORT.md generated from result files
  ▼
models/  results/                              CONTRACTS C, D
  ▼
app/streamlit_app.py   (replay.py calls pipeline.evaluate_scores, the same function as 04_evaluate)
```

## Layout (as implemented)
```
configs/config.yaml   all parameters + feature allow-list + forbidden columns/tokens
src/naf/
  config.py     YAML loading, repo-relative paths, config hash
  io_raw.py     CSV reading (latin-1, stripped names, empty rows dropped), timestamp repair
  clean.py      inf->NaN, negative durations dropped, label normalisation
  windows.py    availability minute, per-minute aggregates (CONTRACT A)
  labels.py     episodes, tiny bursts, campaigns, onsets, status, y
  features.py   aggregates + causal history (prev-L mean, z-score, 1-step delta); forbidden check
  folds.py      forward-chaining folds, calibration block, purge; leave-one-day-out diagnostic
  models.py     LogReg / RF / XGBoost behind one interface
  registry.py   bundle save/load with feature-list and config-hash checks
  alerting.py   fixed, static conformal, adaptive conformal (delayed feedback), refractory
  pipeline.py   dataset build/load, training, scoring, baselines, evaluate_scores (shared path)
  stats.py      Clopper-Pearson, t-interval, cluster bootstrap
  summary.py    pooling over folds/seeds with intervals; per-onset table
  replay.py     replay of a test day with a saved bundle
scripts/01..06  thin CLIs          app/streamlit_app.py   tests/ (pytest)
```
(The v0.2 plan showed sub-packages; they were flattened into single modules to keep the code small.
Dataset access stays isolated in `io_raw.py` so a second dataset adapter can be added later.)

## Data contracts
**A. minute_table** — one row per observed minute: metadata (`minute`, `day`, `n_flows`, `n_attack_flows`, `attack_label_counts`) kept **separate** from model columns `f__*`. `f__*` are whole-network aggregates only: **no port-, IP-, ID- or time-derived column can exist here** (checked at build time).
**B. dataset** — `X` (columns exactly `feature_names.json`, from `f__*` + causal history), `y_H{H}`, `status` ∈ {eligible, ambiguous, ongoing, purged}, `onset_id`, `onset_family` (analysis only), `fold`/`block` ∈ {train, calib, test}. Row `minute`/`day` are never passed to a model.
**C. model bundle** — `model file` + `meta.json` {model, seed, fold, H, G, L, feature_names, scaler/imputer params, train/calib time ranges, config hash, library versions}. `registry.load()` rejects mismatched feature list or config hash.
**D. results** — `events.csv` (per test onset: warned, lead_min, alerter, model, seed, fold), `alarms.csv` (with class: hit / false / ambiguous / ongoing), `metrics.json` (with intervals), `calibration.json`, `power_gate.json`.

## Leakage-prevention rules (each backed by a unit test)
1. **Causal features:** feature(t) uses only flows available ≤ t (ceil(start+duration)+1 min). Test: mutating all data after t leaves feature(t) unchanged.
2. **Forbidden features:** `feature_names ∩ forbidden = ∅`; shuffling `minute`/`day`/IP/port columns changes nothing.
3. **Purge:** gap H+L+G between train→calib→test; no label horizon, history window or campaign crosses a boundary (test asserts).
4. **Fit-on-train only:** scaler, imputer, hyper-parameters (calibration block), thresholds and conformal quantiles (calibration negatives). The test fold never influences any of them.
5. **Delayed feedback:** adaptive conformal may use label of minute t' only at t'+H or later (test with synthetic stream).
6. **Single touch:** test results are written once per configuration; primary config fixed in `config.yaml` before the first test run (config hash stored in results).
7. **Same code path** for replay and evaluation.

## Power gate (end of Phase 3, before training)
Prints per fold/block: minutes, positives, negatives, onsets, families. If training onsets are too few, the project stops for your decision.

## Why these choices (beginner-friendly)
- *Remove IPs/ports/time:* in this dataset one attacker address and fixed victim ports identify attacks, and attack times are fixed; a model using them would "cheat" and fail in real life.
- *Forward-chaining:* train on the past, test on the following day, like reality. A random split would leak near-identical neighbouring minutes.
- *Calibration block:* the alarm threshold must be set on data the model hasn't memorised; otherwise promised false-alarm rates are fiction.
- *Conformal:* gives threshold rules with a stated false-alarm target; the static version needs no feedback, the adaptive version re-tunes itself as labels arrive.
- *Event-level intervals:* with ~10 test onsets, window-level scores look impressive but hide how few independent events there are.

## Stack
pandas, numpy, pyarrow, scikit-learn, xgboost, scipy, PyYAML, streamlit, plotly, pytest — versions pinned from the working project venv at Phase 1 (nothing guessed). No PyTorch.

## Risks
| Risk | Handling |
|---|---|
| Very few events (14; 10 in test) | power gate, interval methods, optional 2nd dataset, no significance claims |
| Whole-network aggregates may carry no onset signal | baselines; report null results per family |
| Day identity confound | benign-score comparison, no-Monday ablation |
| Timestamp truncation to minute | +1 minute availability lag; no sub-minute claims |
| Lab data, single attacker | stated limitation |
