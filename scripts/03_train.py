"""Phase 4: train LogReg / RF / XGBoost for every forward fold (and seed) and save bundles."""
import shutil
import time

import _common  # noqa: F401

from naf import models, pipeline
from naf.config import config_hash, load_config, repo_path
from naf.registry import bundle_dir, save_bundle


def main():
    cfg = load_config()
    meta, X, _ = pipeline.load_dataset(cfg)
    mdir = repo_path(cfg, "models_dir")
    for old in mdir.glob("F*"):
        shutil.rmtree(old)
    h = config_hash(cfg)
    for fold in cfg["folds"]["test_days"]:
        block = pipeline.fold_block(meta, fold, cfg)
        tr = pipeline.train_rows(meta, block)
        for name in models.MODEL_NAMES:
            for seed in models.seeds_for(name, cfg):
                t0 = time.time()
                m = pipeline.train_model(name, seed, meta, X, block, cfg)
                save_bundle(bundle_dir(mdir, fold, name, seed), m, {
                    "model": name, "seed": seed, "fold": fold, "test_day": cfg["folds"]["test_days"][fold],
                    "H": cfg["labels"]["H"], "G": cfg["labels"]["G"], "L": cfg["features"]["L"],
                    "feature_names": list(X.columns), "config_hash": h,
                    "train_minutes": int(tr.sum()), "train_positives": int(meta.loc[tr, "y"].sum()),
                    "train_range": [str(meta.loc[tr, "minute"].min()), str(meta.loc[tr, "minute"].max())],
                    "calib_range": [str(meta.loc[block == "calib", "minute"].min()),
                                    str(meta.loc[block == "calib", "minute"].max())]})
                print(f"{fold} {name} seed{seed}: {int(tr.sum())} train rows, "
                      f"{int(meta.loc[tr, 'y'].sum())} positives, {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
