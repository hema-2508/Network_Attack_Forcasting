"""Load the YAML config and resolve repo-relative paths (Windows/Linux safe)."""
from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs" / "config.yaml"


def load_config(path: str | Path | None = None) -> dict:
    with open(path or DEFAULT_CONFIG, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def with_overrides(cfg: dict, **labels) -> dict:
    """Copy of cfg with label parameters (e.g. H=3, G=15) replaced; used for sensitivity runs."""
    new = copy.deepcopy(cfg)
    new["labels"].update(labels)
    return new


def repo_path(cfg: dict, key: str) -> Path:
    p = ROOT / cfg["paths"][key]
    p.mkdir(parents=True, exist_ok=True)
    return p


def raw_dir(cfg: dict, override: str | None = None) -> Path:
    if override:
        return Path(override)
    if os.environ.get("NAF_RAW_DIR"):
        return Path(os.environ["NAF_RAW_DIR"])
    return ROOT / cfg["paths"]["raw_dir"]


def config_hash(cfg: dict) -> str:
    """Hash of every setting that changes data, labels, models or alerts."""
    keys = ["data", "labels", "features", "folds", "models", "alerting"]
    blob = json.dumps({k: cfg[k] for k in keys}, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:12]
