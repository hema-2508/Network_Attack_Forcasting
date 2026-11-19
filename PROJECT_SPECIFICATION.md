# PROJECT SPECIFICATION  (v0.3 — approved v0.2 + implementation corrections, see tracker)

**Title:** Early Forecasting of Network Attack Onset from Flow Telemetry under False-Alarm Constraints
Changes from v0.1 are listed in IMPLEMENTATION_TRACKER.md → "Change log". Items marked **[D#]** need your approval before coding.

## 1. Problem (plain language)
Every minute, the system looks at *summaries of recent network traffic* and answers: "Will an attack **start** within the next H minutes?" It raises an **early warning** only if confident, and we measure (a) how early, (b) how many false alarms per hour. **No guarantee is made that any model performs well; the study measures what is achievable.**

## 2. Research questions
- RQ1. How much lead time can whole-network flow statistics give before an attack onset, at a fixed false-alarm budget — and is it better than trivial baselines?
- RQ2. Do Logistic Regression, Random Forest and XGBoost differ?
- RQ3. Do conformal alerting rules hold the false-alarm rate closer to its target than a fixed threshold?
- RQ4. How do results change on later days (temporal drift) and for attack families not seen in training?

## 3. Data and suitability (see DATASET_INSPECTION_REPORT.md §8)
CIC-IDS2017 GeneratedLabelledFlows (2,830,743 real flows, 5 days, 1-minute timestamps). **Kept as the primary dataset** (decision [D8]); it is limited to **14 onset events** under the primary definition. A second dataset (CSE-CIC-IDS2018) is an *optional* replication if the Phase-3 power gate fails or the supervisor wants stronger evidence. Raw CSVs stay outside git; path from `configs/config.yaml` / env `NAF_RAW_DIR`.

## 4. Definitions (fixed in config *before* looking at any test result)
| Term | Definition |
|---|---|
| Step | 1 minute (native resolution). |
| Observable time of a flow | `ceil_minute(start_minute + Flow Duration) + 1 minute` (v0.3 correction: the v0.2 `floor(...)+1` could still be up to 1 minute early because start times are truncated to the minute; `ceil(...)+1` is never earlier than the true end — unit-tested) **[D1]**. |
| Attack flow / active minute | flow with Label ≠ BENIGN (by start time); minute with ≥1 attack flow. |
| Episode | maximal run of active minutes with gaps ≤ `gap_merge`=2 min. |
| **Tiny burst** | episode with < `min_active_minutes`=3 active minutes. Tiny bursts are **not onsets** and are **not negatives**: minutes from (burst start − H) to burst end are marked `ambiguous` and excluded from training and scoring; alarms there are reported separately, never hidden **[D4]**. |
| **Campaign** | valid (non-tiny) episodes merged when separated by < `G`=10 quiet minutes. Overlapping families (e.g. Wednesday's four DoS tools) form **one** campaign. |
| **Onset** | first active minute of a campaign. `onset_family` = the dominant label in its first 3 minutes (used only for analysis, never as a model input). |
| Ongoing / cool-down | from campaign start to campaign end + G minutes: marked `ongoing`, excluded from forecasting samples (we forecast *new* onsets). |
| Horizon H | 5 min primary; 3 and 10 as sensitivity **[D3]**. |
| Label `y_t` | 1 iff an onset occurs in minutes (t, t+H]; features at t use only flows observable ≤ t. Eligible sample = not ambiguous, not ongoing. |

**Measured under the primary definition: 14 onsets** — Tue 2 (FTP-Patator, SSH-Patator), Wed 2 (DoS campaign 09:48, Heartbleed), Thu 4 (Web brute-force, XSS, 2×Infiltration), Fri 6 (Bot, 4×PortScan, DDoS). Sensitivity: G=5 → 18, G=15 → 12; min_active_minutes=2 → 16, 5 → 12. 16 tiny bursts are masked.

## 5. Features — strictly whole-network flow statistics
**Allowed:** per-minute aggregates (count, mean, std, p50/p95/max) of an explicit allow-list of numeric CICFlowMeter fields (duration, packet/byte counts, packet-length stats, inter-arrival-time stats, flow rates, TCP flag counts, window sizes, active/idle times); TCP flag ratios; protocol shares (TCP/UDP/other) **[D2]**; and **causal history** of those aggregates over the previous L=10 minutes (lags, rolling mean/std/slope, deviation from a rolling past baseline).
**Forbidden (enforced by a config `forbidden` list + unit test):** Flow ID, Source/Destination IP, **Source/Destination Port and anything derived from ports (port groups, port entropy, unique ports)**, and **all clock/calendar/position features** (timestamp, hour, minute-of-day, weekday, day index, row index, elapsed time since start). History features are allowed because forecasting needs recent context; they use *relative* past minutes only, never absolute time.
Imputation/scaling statistics are fitted on the training block only.
**Honest consequence:** with ports removed, attacks that add only a handful of flows to ~800 background flows/min (e.g. Patators, web attacks, Heartbleed, Infiltration, Bot) may leave no trace in network-wide aggregates. "No predictable signal" is an expected, reportable outcome for such families.

## 6. Models (simple on purpose) 
Logistic Regression (class-weighted, L2), Random Forest, XGBoost; common interface `fit / predict_proba / save / load`. Implemented with fixed hyper-parameters from config (no grid search, v0.3): calibration blocks contain only 1–2 onsets, so tuning on them would fit noise. GRU and any deep model are **out of scope** (listed as future work) — ~1,500 training minutes and ≤4 training events per fold cannot support them. Hyper-parameters: small fixed grids chosen on the calibration block only. Seeds 0–4 for RF/XGBoost (logistic regression is deterministic → run once, not "5 seeds").

## 7. Chronological protocol (improved) **[D5]**
Only **forward-chaining** (train on the past, test on the next day); the test day is touched once.
For every test day D:
1. *Pre-test period* = all days before D (Monday is always included: attack-free).
2. Last 20% of that period (chronological block) = **calibration block**; the rest = **training block**. A **purge gap** of H+L+G minutes is removed at each boundary (train→calib, calib→test) so no label horizon, history window or campaign straddles a boundary.
3. Fit scaler, models on training block; choose hyper-parameters and **thresholds/conformal quantiles on the calibration block (negatives only needed)**; evaluate on D.

| Fold | Test day | Train | Calibration | Test onsets | Role |
|---|---|---|---|---|---|
| F1 | Thu | Mon–Wed (first 80%) | last 20% of Mon–Wed | 4 | primary |
| F2 | Fri | Mon–Thu (first 80%) | last 20% of Mon–Thu | 6 | primary |
| F0 | Wed | Mon–Tue (first 80%) | last 20% | 2 | exploratory, not pooled (≈2 training events) |
Tue cannot be a test day (training would contain no attacks). **Headline results pool F1+F2 = 10 test onsets** and are also shown per fold.
**Phase-3 power gate:** before any model is trained, print positives/negatives and onset counts per block. If the training block holds too few events to be meaningful, **stop and ask you** (options: report as an exploratory event study, or add CSE-CIC-IDS2018).
Note: because attack families cluster by day, nearly every test family is **unseen in training** — drift and family novelty are confounded in this protocol. This is stated, not hidden.
**Diagnostic (clearly labelled non-chronological): leave-one-family-out** — train on all other days' data with the held-out campaign's neighbourhood removed (±(H+L+G) min), test on that campaign; shows family generalisation separately from drift. Never used for headline numbers.
**Day-identity control:** benign traffic differs by day, so a model could learn "which day" instead of "onset". Check: score distribution of calibration-block benign minutes vs training benign, and an ablation without Monday.

## 8. Alerting (all causal)
Alarm at minute t when risk score ≥ threshold, followed by a refractory period (5 min) so one episode = one alarm.
1. **Fixed threshold:** τ = the score quantile giving the target false-alarm rate (e.g. 1/hour) on **calibration-block negatives only**.
2. **Static split-conformal:** τ from the conformal quantile of calibration negatives (finite-sample corrected).
3. **Adaptive conformal (ACI-style):** online threshold from a rolling window of **past, label-resolved** negative scores plus `α_{t+1}=α_t+γ(α*−err_t)`. Minute t' is only usable once its label is resolved (t'+H). The assumption that such feedback exists in deployment is stated; variant 2 needs no feedback at all.
Leakage rules: no threshold/quantile/γ is ever tuned on test data; τ never sees test scores; the rolling window never includes unresolved or future minutes.

## 9. Evaluation
Event level: **detection** = alarm in [onset−H, onset−1]; **lead time** = onset − first such alarm (≥1 min); **false alarm** = alarm in an eligible quiet minute with no onset within H; alarms in `ambiguous`/`ongoing` minutes reported separately; **false alarms per hour** over eligible quiet minutes. Window level: PR-AUC, Brier (sanity only).
Baselines that any model must beat: random alarms at the same budget (5 seeds) and a "current-anomaly" score (max |z| of flow count, packet rate and SYN share vs the previous 10 minutes, no learning). (v0.3: the periodic baseline was dropped — with quantile thresholds its tied scores make it fragile, and the random baseline already measures chance at the same alarm rate.)
**Small-sample statistics:** report counts first (e.g. "4/6 onsets warned"); detection-rate intervals by exact Clopper–Pearson; lead-time and false-alarm/hour intervals by **cluster bootstrap over onset events / quiet blocks**; seeds averaged inside each bootstrap draw; per-family table shown individually. No significance claims; with 10 test events only large effects are visible. Sensitivity grid G∈{5,10,15}, H∈{3,5,10} is reported in full but **the primary setting is fixed above and was not chosen from results**.

**Detection diagnostic (v0.3, not a headline result):** separate models on the same features and folds classify "attack flow started in the previous 3 minutes" vs quiet minutes. Good detection with poor forecasting means the features see attacks but carry little pre-onset signal; poor detection means whole-network aggregates are blind to those attacks.

## 10. Replay
The recorded minute table of a chosen day is streamed through the saved model and chosen alerter by the same function used in evaluation; the dashboard shows the genuine risk curve, alarms, true onset and lead time. Nothing hard-coded.

## 11. Dashboard and deployment
Streamlit + Plotly; pages: summary & limitations · dataset facts · replay · model comparison with intervals · drift / unseen family · alert-policy explorer. Hosted app reads only committed `data/processed/`, `models/`, `results/` (all small); never retrains; needs no PyTorch. Windows-compatible, no Docker, relative paths, pinned requirements generated from the verified environment. Training runs locally (minutes on CPU); a Colab notebook is optional.

## 12. Decisions for you
| ID | Decision | Recommendation |
|---|---|---|
| D1 | Availability = flow end minute + 1 | yes |
| D2 | Protocol shares allowed as features (not identifiers) | yes (switchable off in config) |
| D3 | H=5 primary; 3, 10 sensitivity | yes |
| D4 | Tiny-burst rule: <3 active minutes → masked, not onset, not negative | yes |
| D5 | Forward-chaining F1/F2 primary, F0 exploratory, 20% calibration block, purge H+L+G | yes |
| D6 | Event-level exact/bootstrap intervals, no significance claims | yes |
| D7 | Python 3.10 dev, 3.10/3.11 host | yes |
| D8 | Keep CIC-IDS2017; optional CSE-CIC-IDS2018 replication | yes |
| D9 | Interpret "time-based features" as clock/calendar/position; keep causal lag/rolling history | confirm |
