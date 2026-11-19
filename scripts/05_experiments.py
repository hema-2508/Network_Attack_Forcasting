"""Phase 7: sensitivity grid (G, H), no-Monday ablation, drift diagnostics and the
NON-CHRONOLOGICAL leave-one-day-out (unseen family) diagnostic. Models are trained in memory;
none of these results feed the primary headline numbers."""
import itertools

import numpy as np
import pandas as pd

import _common  # noqa: F401

from naf import pipeline
from naf.config import load_config, repo_path, with_overrides
from naf.summary import family_table, summarize


def run_variant(mt, cfg, variant, fold_names, alerters=None, baselines=True):
    meta, X, onsets = pipeline.build_dataset(mt, cfg)
    M, E, R = [], [], []
    for f in fold_names:
        block = pipeline.fold_block(meta, f, cfg, variant=variant)
        mi, ev, ru = pipeline.run_all_scorers(meta, X, onsets, block, cfg, f, variant,
                                              alerters=alerters, baselines=baselines)
        M.append(mi); E.append(ev); R.append(ru)
    return pd.concat(M), pd.concat(E), pd.concat(R)


def main():
    cfg = load_config()
    res = repo_path(cfg, "results_dir")
    mt = pd.read_parquet(repo_path(cfg, "processed_dir") / "minute_table.parquet")
    prim = cfg["folds"]["primary"]

    # 1. sensitivity grid over the onset gap G and horizon H (primary folds)
    rows = []
    sens = cfg["evaluation"]["sensitivity"]
    for G, H in itertools.product(sens["G"], sens["H"]):
        c = with_overrides(cfg, G=G, H=H)
        mi, ev, ru = run_variant(mt, c, "forward", prim, alerters=["conformal", "aci"])
        s = summarize(ru, ev, mi, prim, c).assign(G=G, H=H)
        rows.append(s)
        print(f"sensitivity G={G} H={H}: {len(ev[ev.model == 'logreg'].onset_id.unique())} test onsets", flush=True)
    pd.concat(rows).to_csv(res / "sensitivity.csv", index=False)

    # 2. day-identity control: training without attack-free Monday
    mi, ev, ru = run_variant(mt, cfg, "no_monday", prim)
    summarize(ru, ev, mi, prim, cfg).to_csv(res / "ablation_no_monday.csv", index=False)
    print("no-Monday ablation done", flush=True)

    # 3. drift: calibration-negative vs test-negative score shift and realised exceedance
    meta, X, onsets = pipeline.build_dataset(mt, cfg)
    stored = pd.read_parquet(res / "minutes.parquet")
    drift = []
    for f in cfg["folds"]["test_days"]:
        block = pipeline.fold_block(meta, f, cfg)
        for (model, seed, al), g in stored[(stored.fold == f)].groupby(["model", "seed", "alerter"]):
            q = g[(g.status == "eligible") & (g.y == 0)]
            drift.append({"fold": f, "model": model, "seed": seed, "alerter": al,
                          "target_exceed_rate": cfg["alerting"]["fa_target_per_hour"] / 60,
                          "test_quiet_exceed_rate": q["exceed"].mean(),
                          "test_quiet_score_median": q["score"].median()})
    pd.DataFrame(drift).to_csv(res / "drift.csv", index=False)

    # calibration-block score medians for the same saved models (score-shift check)
    from naf.registry import bundle_dir, load_bundle
    from naf.config import config_hash
    shift = []
    for f in cfg["folds"]["test_days"]:
        block = pipeline.fold_block(meta, f, cfg)
        for model in ["logreg", "rf", "xgb"]:
            m, _ = load_bundle(bundle_dir(repo_path(cfg, "models_dir"), f, model, 0), list(X.columns), config_hash(cfg))
            for b in ["train", "calib", "test"]:
                r = (block == b) & (meta.status == "eligible") & (meta.y == 0)
                s = pipeline.score_rows(model, 0, m, meta, X, r, cfg)
                shift.append({"fold": f, "model": model, "block": b, "negatives": int(r.sum()),
                              "score_median": float(s.median()), "score_p95": float(s.quantile(0.95))})
    pd.DataFrame(shift).to_csv(res / "score_shift.csv", index=False)
    print("drift diagnostics done", flush=True)

    # 4. leave-one-day-out (NON-CHRONOLOGICAL): each attack day tested with all other attack days
    #    in training and Monday as calibration -> family generalisation separate from time order
    days = [d for d in sorted(meta.day.unique()) if d != cfg["folds"]["monday"]]
    M, E, R = [], [], []
    for d in days:
        block = pipeline.fold_block(meta, d, cfg, variant="lodo")
        mi, ev, ru = pipeline.run_all_scorers(meta, X, onsets, block, cfg, d, "lodo")
        M.append(mi); E.append(ev); R.append(ru)
    mi, ev, ru = pd.concat(M), pd.concat(E), pd.concat(R)
    ru.to_csv(res / "lodo_runs.csv", index=False)
    ev.to_csv(res / "lodo_events.csv", index=False)
    summarize(ru, ev, mi, days, cfg).to_csv(res / "lodo_summary.csv", index=False)
    family_table(ev, days).to_csv(res / "lodo_family_events.csv", index=False)
    print("leave-one-day-out diagnostic done", flush=True)


if __name__ == "__main__":
    main()
