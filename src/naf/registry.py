"""Save / load model bundles (CONTRACT C): model.joblib + meta.json."""
from __future__ import annotations

import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import joblib
import sklearn
import xgboost


def bundle_dir(models_dir: Path, fold: str, model: str, seed: int) -> Path:
    return Path(models_dir) / fold / model / f"seed{seed}"


def save_bundle(path: Path, model, meta: dict) -> None:
    path.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path / "model.joblib", compress=3)
    meta = dict(meta)
    meta.update({"created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                 "versions": {"python": platform.python_version(), "sklearn": sklearn.__version__,
                              "xgboost": xgboost.__version__}})
    (path / "meta.json").write_text(json.dumps(meta, indent=1, default=str), encoding="utf-8")


def load_bundle(path: Path, feature_names: list[str] | None = None, config_hash: str | None = None):
    meta = json.loads((Path(path) / "meta.json").read_text(encoding="utf-8"))
    if feature_names is not None and meta["feature_names"] != list(feature_names):
        raise ValueError(f"{path}: feature list does not match the dataset")
    if config_hash is not None and meta["config_hash"] != config_hash:
        raise ValueError(f"{path}: trained with config {meta['config_hash']}, current {config_hash}")
    return joblib.load(Path(path) / "model.joblib"), meta
