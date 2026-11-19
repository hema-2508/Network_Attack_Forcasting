"""Pool per-run results over folds and compute intervals (PROJECT_SPECIFICATION.md §9).

- detection rate: event-cluster bootstrap (onsets resampled, seeds averaged inside each draw);
  exact Clopper-Pearson for single-seed (deterministic) scorers.
- false alarms / hour: bootstrap over 60-minute blocks of quiet minutes (seeds pooled).
- lead time: mean over warned (onset, seed) pairs; bootstrap over onsets.
- seed spread: t-interval of per-seed pooled detection rate (stochastic scorers only).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .stats import bootstrap_ratio, clopper_pearson, t_interval

KEYS = ["variant", "model", "alerter"]


def summarize(runs: pd.DataFrame, events: pd.DataFrame, minutes: pd.DataFrame, fold_list, cfg) -> pd.DataFrame:
    B, blk = cfg["evaluation"]["bootstrap"], cfg["evaluation"]["fa_block_minutes"]
    runs = runs[runs["fold"].isin(fold_list)]
    events = events[events["fold"].isin(fold_list)]
    minutes = minutes[minutes["fold"].isin(fold_list)]
    out = []
    for key, r in runs.groupby(KEYS):
        e = events.set_index(KEYS).loc[[key]].reset_index() if len(events) else events
        mi = minutes.set_index(KEYS).loc[[key]].reset_index()
        seeds = sorted(r["seed"].unique())
        per_seed = r.groupby("seed").agg(w=("n_warned", "sum"), n=("n_onsets", "sum"),
                                         fa=("n_false_alarms", "sum"), q=("quiet_minutes", "sum"))
        det_seed = per_seed["w"] / per_seed["n"]
        # event bootstrap: per-onset warned fraction across seeds
        po = e.groupby("onset_id").agg(w=("warned", "sum"), k=("warned", "size"),
                                       lead_sum=("lead_min", "sum"))
        det_ci = bootstrap_ratio(po["w"].to_numpy(), po["k"].to_numpy(), B)
        lead_ci = bootstrap_ratio(po["lead_sum"].to_numpy(), po["w"].to_numpy(), B)
        # false-alarm blocks: quiet minutes grouped in 60-minute blocks within each fold-day
        q = mi[(mi["status"] == "eligible") & (mi["y"] == 0)].copy()
        q["blk"] = q["fold"] + "_" + (q.groupby(["fold", "seed"]).cumcount() // blk).astype(str)
        fb = q.groupby("blk").agg(fa=("alarm_class", lambda s: (s == "false").sum()), m=("alarm", "size"))
        fa_ci = bootstrap_ratio(fb["fa"].to_numpy(), fb["m"].to_numpy(), B, scale=60.0)
        n_on = int(per_seed["n"].iloc[0])
        row = dict(zip(KEYS, key))
        row.update({
            "folds": "+".join(fold_list), "n_seeds": len(seeds), "n_onsets": n_on,
            "warned_mean": float(per_seed["w"].mean()),
            "detection": float(det_seed.mean()), "det_boot_lo": det_ci[0], "det_boot_hi": det_ci[1],
            "det_seed_lo": t_interval(det_seed)[0] if len(seeds) > 1 else np.nan,
            "det_seed_hi": t_interval(det_seed)[1] if len(seeds) > 1 else np.nan,
            "mean_lead_min": float(e.loc[e["warned"], "lead_min"].mean()) if e["warned"].any() else np.nan,
            "lead_lo": lead_ci[0], "lead_hi": lead_ci[1],
            "fa_per_hour": float((per_seed["fa"] / (per_seed["q"] / 60)).mean()),
            "fa_lo": fa_ci[0], "fa_hi": fa_ci[1],
            "quiet_hours": float(per_seed["q"].iloc[0] / 60),
            "ambiguous_alarms_mean": float(r.groupby("seed")["n_ambiguous_alarms"].sum().mean()),
            "ongoing_alarms_mean": float(r.groupby("seed")["n_ongoing_alarms"].sum().mean()),
            "pr_auc_mean": float(r["pr_auc"].mean()), "roc_auc_mean": float(r["roc_auc"].mean()),
            "positive_rate": float(r["positive_rate"].mean()),
        })
        if len(seeds) == 1:
            row["det_cp_lo"], row["det_cp_hi"] = clopper_pearson(int(per_seed["w"].iloc[0]), n_on)
        else:
            row["det_cp_lo"] = row["det_cp_hi"] = np.nan
        out.append(row)
    return pd.DataFrame(out)


def family_table(events: pd.DataFrame, fold_list) -> pd.DataFrame:
    e = events[events["fold"].isin(fold_list)]
    return (e.groupby(["variant", "model", "alerter", "fold", "onset_id", "family"])
            .agg(warned_frac=("warned", "mean"), mean_lead_min=("lead_min", "mean"),
                 window_minutes=("window_minutes", "first")).reset_index())
