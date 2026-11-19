"""Diagnostic (not a headline result): can the SAME features *detect* attacks that are already
running? This separates two explanations of a null forecasting result:
  (a) features see attacks, but there is no precursor before onset  -> detection works, forecasting fails
  (b) whole-network aggregates (no ports/IPs) cannot see these attacks at all -> detection fails too.

Detection target d_t = 1 if any attack flow started in minutes [t-3, t-1] (so the flows can already be
available at t); negatives = eligible quiet minutes with y = 0 and no attack in [t-3, t-1].
Same forward folds, same features, same models, same calibration-block threshold (static conformal).
Per onset we report the delay from onset to the first exceedance inside its campaign (minutes).
"""
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

import _common  # noqa: F401

from naf import alerting, models, pipeline
from naf.config import load_config, repo_path


def detection_labels(meta: pd.DataFrame, lag: int = 3) -> pd.Series:
    out = pd.Series(np.nan, index=meta.index)
    for _, idx in meta.groupby("day").indices.items():
        idx = np.sort(idx)
        act = (meta["n_attack_flows"].to_numpy()[idx] > 0).astype(int)
        c = np.concatenate([[0], np.cumsum(act)])
        n = len(idx)
        lo, hi = np.clip(np.arange(n) - lag, 0, n), np.arange(n)
        recent = (c[hi] - c[lo]) > 0
        st, y = meta["status"].to_numpy()[idx], meta["y"].to_numpy()[idx]
        d = np.where(recent, 1.0, np.where((st == "eligible") & (y == 0), 0.0, np.nan))
        out.iloc[idx] = d
    return out


def main():
    cfg = load_config()
    meta, X, onsets = pipeline.load_dataset(cfg)
    d = detection_labels(meta)
    alpha = alerting.alpha_from_cfg(cfg)
    rows, ev = [], []
    for fold in cfg["folds"]["test_days"]:
        block = pipeline.fold_block(meta, fold, cfg)
        tr = (block == "train") & d.notna()
        cal_neg = (block == "calib") & (d == 0)
        te = (block == "test") & d.notna()
        for name in models.MODEL_NAMES:
            for seed in models.seeds_for(name, cfg):
                m = models.fit_model(name, cfg, seed, X[tr].to_numpy(), d[tr].to_numpy().astype(int))
                tau = alerting.threshold_conformal(models.predict_risk(m, X[cal_neg].to_numpy()), alpha)
                s_te = models.predict_risk(m, X[te].to_numpy())
                yt = d[te].to_numpy()
                rows.append({"fold": fold, "model": name, "seed": seed, "train_attack_minutes": int(d[tr].sum()),
                             "test_attack_minutes": int(yt.sum()), "test_quiet_minutes": int((yt == 0).sum()),
                             "roc_auc": roc_auc_score(yt, s_te), "pr_auc": average_precision_score(yt, s_te),
                             "attack_rate": float(yt.mean()),
                             "recall_at_budget": float((s_te[yt == 1] > tau).mean()),
                             "quiet_exceed_rate": float((s_te[yt == 0] > tau).mean())})
                day = cfg["folds"]["test_days"][fold]
                dm = meta[(meta.day == day)].sort_values("minute")
                sd = pd.Series(models.predict_risk(m, X.loc[dm.index].to_numpy()), index=dm.index)
                for _, o in onsets[onsets.day == day].iterrows():
                    w = dm[(dm.minute > o.onset) & (dm.minute <= o.end + pd.Timedelta(minutes=3))]
                    hit = w[sd[w.index] > tau]
                    ev.append({"fold": fold, "model": name, "seed": seed, "onset_id": o.onset_id, "family": o.family,
                               "detected": bool(len(hit)),
                               "delay_min": (hit.minute.min() - o.onset) / pd.Timedelta(minutes=1) if len(hit) else np.nan})
        print(f"{fold} done", flush=True)
    res = repo_path(cfg, "results_dir")
    r = pd.DataFrame(rows)
    r.to_csv(res / "detection_check_runs.csv", index=False)
    e = pd.DataFrame(ev)
    e.to_csv(res / "detection_check_events.csv", index=False)
    print(r.groupby(["fold", "model"])[["roc_auc", "pr_auc", "attack_rate", "recall_at_budget", "quiet_exceed_rate"]]
          .mean().round(3).to_string())
    print(e.groupby(["onset_id", "family", "model"]).agg(detected=("detected", "mean"), delay=("delay_min", "mean"))
          .unstack("model").round(2).to_string())


if __name__ == "__main__":
    main()
