"""Replay a recorded test day through a SAVED model and an alerter.

Uses exactly the same functions as the evaluation script (pipeline.evaluate_scores), so the
replayed risk curve and alarms are the genuine evaluated outputs, not a re-enactment.
"""
from __future__ import annotations

from . import pipeline
from .config import config_hash, repo_path
from .registry import bundle_dir, load_bundle


def replay(cfg: dict, fold: str, model: str, seed: int, alerter: str, data=None):
    meta, X, onsets = data if data is not None else pipeline.load_dataset(cfg)
    block = pipeline.fold_block(meta, fold, cfg)
    rows = (block == "calib") | (block == "test")
    m = None
    if model in pipeline.models.MODEL_NAMES:
        m, _ = load_bundle(bundle_dir(repo_path(cfg, "models_dir"), fold, model, seed),
                           feature_names=list(X.columns), config_hash=config_hash(cfg))
    scores = pipeline.score_rows(model, seed, m, meta, X, rows, cfg)
    return pipeline.evaluate_scores(scores, meta, block, onsets, cfg, alerter)
