"""Checks against the committed processed data (skipped if it has not been built)."""
import numpy as np
import pandas as pd
import pytest

from naf import pipeline
from naf.config import ROOT, repo_path
from naf.features import check_forbidden

PROC = ROOT / "data" / "processed"
pytestmark = pytest.mark.skipif(not (PROC / "dataset.parquet").exists(), reason="dataset not built")


def test_counts_match_inspection(cfg):
    meta, X, onsets = pipeline.load_dataset(cfg)
    assert len(meta) == 2454
    assert len(onsets) == 14
    assert onsets.groupby("day").size().tolist() == [2, 2, 4, 6]
    check_forbidden(list(X.columns), cfg)


def test_no_label_or_time_column_in_features(cfg):
    _, X, _ = pipeline.load_dataset(cfg)
    assert not set(X.columns) & set(pipeline.META_COLS)


def test_replay_equals_stored_evaluation(cfg):
    res = repo_path(cfg, "results_dir") / "minutes.parquet"
    if not res.exists():
        pytest.skip("evaluation not run")
    from naf.replay import replay
    stored = pd.read_parquet(res)
    for fold, model, seed, al in [("F2", "xgb", 0, "aci"), ("F1", "logreg", 0, "conformal")]:
        mins, _, _ = replay(cfg, fold, model, seed, al)
        s = stored[(stored.variant == "forward") & (stored.fold == fold) & (stored.model == model)
                   & (stored.seed == seed) & (stored.alerter == al)].sort_values("minute")
        assert np.allclose(mins["score"].to_numpy(), s["score"].to_numpy())
        assert np.array_equal(mins["alarm"].to_numpy(), s["alarm"].to_numpy())
