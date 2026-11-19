"""Phase 3: minute table -> labels, features, dataset.parquet and the POWER GATE report."""
import json

import pandas as pd

import _common  # noqa: F401

from naf import pipeline
from naf.config import load_config, repo_path


def power_gate(meta: pd.DataFrame, onsets: pd.DataFrame, cfg: dict) -> dict:
    rep = {}
    for fold, day in cfg["folds"]["test_days"].items():
        block = pipeline.fold_block(meta, fold, cfg)
        r = {"test_day": day, "role": "primary" if fold in cfg["folds"]["primary"] else "exploratory"}
        for b in ["train", "calib", "test", "purged"]:
            m = block == b
            e = m & (meta["status"] == "eligible")
            ids = set(meta.loc[e & (meta["y"] == 1), "onset_id"])
            r[b] = {"minutes": int(m.sum()), "eligible": int(e.sum()),
                    "positives": int((e & (meta["y"] == 1)).sum()),
                    "negatives": int((e & (meta["y"] == 0)).sum()),
                    "onsets_with_positive_minutes": len(ids),
                    "families": sorted(onsets.loc[onsets["onset_id"].isin(ids), "family"].unique().tolist())}
        rep[fold] = r
    return rep


def main():
    cfg = load_config()
    mt = pd.read_parquet(repo_path(cfg, "processed_dir") / "minute_table.parquet")
    meta, X, onsets = pipeline.build_dataset(mt, cfg)
    pipeline.save_dataset(meta, X, onsets, cfg)
    print(f"dataset: {len(meta)} minutes x {X.shape[1]} features")
    print(f"onsets: {len(onsets)}, tiny bursts: {len(onsets.attrs['tiny_bursts'])}")
    print(onsets[["onset_id", "family", "duration_min", "attack_flows"]].to_string(index=False))
    print("status counts:", meta["status"].value_counts().to_dict(), " positives:", int(meta["y"].sum()))
    gate = power_gate(meta, onsets, cfg)
    (repo_path(cfg, "results_dir") / "power_gate.json").write_text(json.dumps(gate, indent=1), encoding="utf-8")
    for f, r in gate.items():
        print(f"{f} ({r['role']}, test {r['test_day']}): " + " | ".join(
            f"{b}: {r[b]['minutes']} min, {r[b]['positives']} pos, {r[b]['onsets_with_positive_minutes']} onsets"
            for b in ["train", "calib", "test"]))


if __name__ == "__main__":
    main()
