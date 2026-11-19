"""Shared code path: dataset building, training, scoring, alerting and event evaluation.

Evaluation scripts, the replay engine and the dashboard all call `evaluate_scores`, so a
replayed day is exactly the evaluated day.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from . import alerting, folds, models
from .config import config_hash, repo_path
from .features import build_features
from .labels import label_minutes

META_COLS = ["minute", "day", "n_flows", "n_attack_flows", "status", "y", "onset_id"]
BASELINES = ["baseline_random", "baseline_anomaly"]


# ---------------------------------------------------------------- dataset
def build_dataset(mt: pd.DataFrame, cfg: dict):
    mt = mt.sort_values("minute").reset_index(drop=True)
    lab, onsets = label_minutes(mt, cfg)
    X = build_features(mt, cfg)
    meta = pd.concat([mt[["minute", "day", "n_flows", "n_attack_flows"]], lab], axis=1)
    return meta, X, onsets


def save_dataset(meta, X, onsets, cfg) -> None:
    out = repo_path(cfg, "processed_dir")
    pd.concat([meta, X], axis=1).to_parquet(out / "dataset.parquet", index=False)
    onsets.to_csv(out / "onsets.csv", index=False)
    onsets.attrs["tiny_bursts"].to_csv(out / "tiny_bursts.csv", index=False)
    (out / "feature_names.json").write_text(json.dumps(list(X.columns), indent=0), encoding="utf-8")


def load_dataset(cfg: dict):
    d = repo_path(cfg, "processed_dir")
    df = pd.read_parquet(d / "dataset.parquet")
    names = json.loads((d / "feature_names.json").read_text(encoding="utf-8"))
    onsets = pd.read_csv(d / "onsets.csv", parse_dates=["onset", "end"])
    return df[META_COLS].copy(), df[names].copy(), onsets


def fold_block(meta: pd.DataFrame, fold: str, cfg: dict, variant: str = "forward") -> pd.Series:
    if variant == "lodo":
        return folds.leave_one_day_out(meta, fold, cfg)
    day = cfg["folds"]["test_days"][fold]
    return folds.forward_fold(meta, day, cfg, drop_monday=(variant == "no_monday"))


def train_rows(meta, block):
    return (block == "train") & (meta["status"] == "eligible")


def calib_negative_rows(meta, block):
    return (block == "calib") & (meta["status"] == "eligible") & (meta["y"] == 0)


# ---------------------------------------------------------------- scoring
def train_model(name, seed, meta, X, block, cfg):
    m = train_rows(meta, block)
    return models.fit_model(name, cfg, seed, X[m].to_numpy(), meta.loc[m, "y"].to_numpy())


def score_rows(kind: str, seed: int, model, meta: pd.DataFrame, X: pd.DataFrame,
               rows: pd.Series, cfg: dict) -> pd.Series:
    """Risk score for the selected rows (calibration + test)."""
    idx = meta.index[rows]
    if kind in models.MODEL_NAMES:
        s = models.predict_risk(model, X.loc[idx].to_numpy())
    elif kind == "baseline_random":
        # chance baseline: i.i.d. uniform scores, reproducible per (seed, minute)
        keys = (meta.loc[idx, "minute"].astype("int64") // 60_000_000_000).to_numpy()
        s = np.array([np.random.default_rng([seed, int(k)]).random() for k in keys])
    elif kind == "baseline_anomaly":
        cols = [f"f__{c}__z" for c in cfg["evaluation"]["anomaly_baseline_features"]]
        s = X.loc[idx, cols].abs().fillna(0).max(axis=1).to_numpy()
    else:
        raise ValueError(kind)
    return pd.Series(s, index=idx, dtype=float)


# ---------------------------------------------------------------- evaluation
def evaluate_scores(scores: pd.Series, meta: pd.DataFrame, block: pd.Series, onsets: pd.DataFrame,
                    cfg: dict, alerter: str):
    """Apply an alerter to the test block and compute event-level results.

    scores: risk for every calib and test row. Thresholds only ever see calibration negatives.
    Returns (minutes, events, metrics).
    """
    cal_neg = scores[calib_negative_rows(meta, block)].to_numpy()
    test = meta[block == "test"].sort_values("minute")
    H = cfg["labels"]["H"]
    rows = []
    for day, g in test.groupby("day", sort=True):
        s = scores.loc[g.index].to_numpy()
        res = alerting.apply_alerter(alerter, s, (g["n_attack_flows"] > 0).to_numpy(), cal_neg, cfg)
        rows.append(g.assign(score=s, exceed=res["exceed"], alarm=res["alarm"], tau=res["tau"],
                             alpha_t=res["alpha_t"]))
    mins = pd.concat(rows)
    elig = mins["status"] == "eligible"
    mins["alarm_class"] = np.where(~mins["alarm"], "", np.where(
        elig & (mins["y"] == 1), "hit", np.where(elig, "false", mins["status"])))

    ev = []
    for _, o in onsets[onsets["day"].isin(test["day"].unique())].iterrows():
        w = mins[(mins["onset_id"] == o["onset_id"])]
        hit = w[w["exceed"]]
        lead = (o["onset"] - hit["minute"].min()) / pd.Timedelta(minutes=1) if len(hit) else np.nan
        ev.append({"onset_id": o["onset_id"], "day": o["day"], "family": o["family"],
                   "onset": o["onset"], "window_minutes": len(w), "warned": bool(len(hit)),
                   "lead_min": lead})
    events = pd.DataFrame(ev, columns=["onset_id", "day", "family", "onset", "window_minutes",
                                       "warned", "lead_min"])

    quiet = elig & (mins["y"] == 0)
    m = {"n_onsets": len(events), "n_warned": int(events["warned"].sum()),
         "mean_lead_min": float(events["lead_min"].mean()) if events["warned"].any() else np.nan,
         "n_false_alarms": int((mins["alarm_class"] == "false").sum()),
         "quiet_minutes": int(quiet.sum()),
         "n_ambiguous_alarms": int((mins["alarm_class"] == "ambiguous").sum()),
         "n_ongoing_alarms": int((mins["alarm_class"] == "ongoing").sum()),
         "quiet_exceed_rate": float(mins.loc[quiet, "exceed"].mean()) if quiet.any() else np.nan,
         "n_calib_negatives": len(cal_neg)}
    m["fa_per_hour"] = m["n_false_alarms"] / (m["quiet_minutes"] / 60) if m["quiet_minutes"] else np.nan
    yy, ss = mins.loc[elig, "y"].to_numpy(), mins.loc[elig, "score"].to_numpy()
    both = 0 < yy.sum() < len(yy)
    m["pr_auc"] = float(average_precision_score(yy, ss)) if both else np.nan
    m["roc_auc"] = float(roc_auc_score(yy, ss)) if both else np.nan
    m["positive_rate"] = float(yy.mean()) if len(yy) else np.nan
    in01 = len(ss) and ss.min() >= 0 and ss.max() <= 1
    m["brier"] = float(brier_score_loss(yy, ss)) if (len(yy) and in01) else np.nan
    return mins, events, m


def run_all_scorers(meta, X, onsets, block, cfg, fold_name, variant, model_names=None,
                    baselines=True, load_dir: Path | None = None, alerters=None):
    """Train (or load) every model/seed for one fold and evaluate under every alerter."""
    from .registry import bundle_dir, load_bundle
    model_names = models.MODEL_NAMES if model_names is None else model_names
    alerters = alerting.ALERTERS if alerters is None else alerters
    rows = (block == "calib") | (block == "test")
    jobs = [(n, s) for n in model_names for s in models.seeds_for(n, cfg)]
    if baselines:
        jobs += [("baseline_random", s) for s in cfg["models"]["seeds"]] + [("baseline_anomaly", 0)]
    all_min, all_ev, all_runs = [], [], []
    for kind, seed in jobs:
        model = None
        if kind in models.MODEL_NAMES:
            if load_dir is not None:
                model, _ = load_bundle(bundle_dir(load_dir, fold_name, kind, seed),
                                       feature_names=list(X.columns), config_hash=config_hash(cfg))
            else:
                model = train_model(kind, seed, meta, X, block, cfg)
        scores = score_rows(kind, seed, model, meta, X, rows, cfg)
        for al in alerters:
            mins, ev, m = evaluate_scores(scores, meta, block, onsets, cfg, al)
            tag = {"variant": variant, "fold": fold_name, "model": kind, "seed": seed, "alerter": al}
            all_min.append(mins.assign(**tag))
            all_ev.append(ev.assign(**tag))
            all_runs.append({**tag, **m})
    return pd.concat(all_min), pd.concat(all_ev), pd.DataFrame(all_runs)
