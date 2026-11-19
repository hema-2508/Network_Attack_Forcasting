# CLAUDE.md — project rules (v0.3)

M.Tech project: *Early Forecasting of Network Attack Onset from Flow Telemetry under False-Alarm Constraints* (CIC-IDS2017). Read PROJECT_SPECIFICATION.md, SYSTEM_ARCHITECTURE.md, DATASET_INSPECTION_REPORT.md, IMPLEMENTATION_TRACKER.md first. Run order and commands: README.md.

## Non-negotiable rules
1. **Never fabricate or guarantee.** Every number comes from a script into `results/`. No hard-coded predictions or scripted fake attacks. Never promise model performance; report null/weak results plainly with intervals.
2. **Forbidden model inputs:** Flow ID, Source/Destination IP, **Source/Destination Port and anything derived from ports**, Timestamp and every clock/calendar/position feature (hour, minute-of-day, weekday, day/row index, elapsed time). Only whole-network aggregates of the allow-listed numeric flow fields, protocol shares, and *causal* history (previous L minutes, relative only). Enforced by the config forbidden list and tests.
3. **Causality & no leakage:** feature(t) uses flows available ≤ t (ceil(start + duration) + 1 min). Forward-chaining folds only; purge gap H+L+G; scaler/thresholds/conformal quantiles fitted on training or calibration blocks only; test fold touched once; primary config fixed before first test run (config hash stored with models and results). Adaptive conformal uses only feedback for minutes ≥ H+1 old.
4. **Definitions are fixed in config** (gap_merge=2, min_active_minutes=3, G=10, H=5, L=10); tiny bursts are masked (neither onset nor negative). Do not retune them on test results; sensitivity analysis is reported separately.
5. **Models:** Logistic Regression, Random Forest, XGBoost only, fixed hyper-parameters. No deep learning unless the user re-approves.
6. **Honest statistics:** 10 primary test onsets → report counts, exact/bootstrap event-level intervals, no significance claims; 5 seeds only for stochastic scorers (RF, XGB, random baseline).
7. **One integrated project:** obey the data contracts; any contract change updates docs, consumers and tests together. Replay and evaluation share `pipeline.evaluate_scores`.
8. **Config-driven, portable:** parameters in `configs/config.yaml`; `pathlib`, relative paths, Windows-compatible, no Docker; raw path via `--raw-dir`/`NAF_RAW_DIR`. Pinned requirements from the verified venv.
9. **Hosted app never trains**, needs no raw CSVs and no PyTorch.
10. **Keep it simple;** explain important decisions in beginner-friendly language.

## Dataset gotchas (verified)
`encoding="latin-1"`; strip column names; drop all-NaN rows (Thursday web-attacks: 288,602); parse `%d/%m/%Y`; 12-hour clock without AM/PM (hour<8 → +12); 1-minute resolution; sort by time; inf→NaN; drop negative `Flow Duration` (115 rows); `Init_Win_bytes_* = -1` is a sentinel (own feature).

## Workflow
Update IMPLEMENTATION_TRACKER.md after each phase. Tests (`pytest -q`) cover causality, forbidden features, purge gaps, onset labelling, delayed feedback, save/load, replay == evaluation, and every dashboard page. Use project `.venv`, not global Python. No commits unless asked; never commit raw data/venv/secrets. On Windows PowerShell 5.1 never rewrite UTF-8 files with Get-Content/Set-Content (it corrupts non-ASCII); use the editor tools or Python.
