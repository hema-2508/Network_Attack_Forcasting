"""Phases 5-6: load SAVED bundles, calibrate alerters on calibration blocks, evaluate each test
day once, add baselines, pool primary folds with intervals. Writes everything to results/."""
import json

import pandas as pd

import _common  # noqa: F401

from naf import pipeline
from naf.config import config_hash, load_config, repo_path
from naf.summary import family_table, summarize


def main():
    cfg = load_config()
    meta, X, onsets = pipeline.load_dataset(cfg)
    res = repo_path(cfg, "results_dir")
    mdir = repo_path(cfg, "models_dir")
    M, E, R = [], [], []
    for fold in cfg["folds"]["test_days"]:
        block = pipeline.fold_block(meta, fold, cfg)
        mi, ev, ru = pipeline.run_all_scorers(meta, X, onsets, block, cfg, fold, "forward", load_dir=mdir)
        M.append(mi); E.append(ev); R.append(ru)
        print(f"{fold}: evaluated {len(ru)} runs", flush=True)
    minutes, events, runs = pd.concat(M), pd.concat(E), pd.concat(R)
    keep = ["variant", "fold", "model", "seed", "alerter", "minute", "day", "status", "y", "onset_id",
            "n_attack_flows", "score", "exceed", "alarm", "alarm_class", "tau", "alpha_t"]
    minutes[keep].to_parquet(res / "minutes.parquet", index=False)
    events.to_csv(res / "events.csv", index=False)
    runs.to_csv(res / "runs.csv", index=False)
    prim, expl = cfg["folds"]["primary"], cfg["folds"]["exploratory"]
    s_prim = summarize(runs, events, minutes, prim, cfg)
    s_fold = pd.concat([summarize(runs, events, minutes, [f], cfg) for f in cfg["folds"]["test_days"]])
    s_prim.to_csv(res / "summary_primary.csv", index=False)
    s_fold.to_csv(res / "summary_per_fold.csv", index=False)
    family_table(events, prim + expl).to_csv(res / "family_events.csv", index=False)
    (res / "run_info.json").write_text(json.dumps({
        "config_hash": config_hash(cfg), "primary_folds": prim, "exploratory_folds": expl,
        "H": cfg["labels"]["H"], "G": cfg["labels"]["G"], "fa_target_per_hour": cfg["alerting"]["fa_target_per_hour"],
        "note": "Each test day evaluated once with the fixed primary configuration."}, indent=1), encoding="utf-8")
    cols = ["model", "alerter", "n_seeds", "n_onsets", "detection", "det_boot_lo", "det_boot_hi",
            "mean_lead_min", "fa_per_hour", "fa_lo", "fa_hi", "pr_auc_mean"]
    with pd.option_context("display.width", 200, "display.precision", 3):
        print(s_prim[cols].to_string(index=False))


if __name__ == "__main__":
    main()
