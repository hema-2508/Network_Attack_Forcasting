import numpy as np
import pytest

from naf import alerting
from naf.models import MODEL_NAMES, fit_model, predict_risk
from naf.registry import load_bundle, save_bundle


def test_conformal_threshold_finite_sample():
    cal = np.arange(1, 100, dtype=float)          # 99 negatives
    tau = alerting.threshold_conformal(cal, 0.1)  # ceil(100*0.9)=90th smallest
    assert tau == 90.0
    assert alerting.threshold_conformal(cal, 0.001) == float("inf")


def test_fixed_threshold_controls_rate_on_same_distribution():
    rng = np.random.default_rng(0)
    cal, test = rng.random(20000), rng.random(20000)
    tau = alerting.threshold_fixed(cal, 1 / 60)
    assert abs((test > tau).mean() - 1 / 60) < 0.005


def test_aci_uses_only_resolved_past(cfg):
    rng = np.random.default_rng(1)
    n, t = 300, 150
    scores, active = rng.random(n), np.zeros(n, bool)
    active[200:210] = True
    cal = rng.random(500)
    a = cfg["alerting"]["aci"]
    args = (1 / 60, cfg["labels"]["H"], cfg["labels"]["G"], a["gamma"], a["window"], a["alpha_min"], a["alpha_max"])
    e1, tau1, _ = alerting.run_aci(scores, active, cal, *args)
    s2, act2 = scores.copy(), active.copy()
    s2[t + 1:] = rng.random(n - t - 1)          # change the future
    act2[t:] = ~act2[t:]
    e2, tau2, _ = alerting.run_aci(s2, act2, cal, *args)
    assert np.array_equal(e1[:t + 1], e2[:t + 1]) and np.allclose(tau1[:t + 1], tau2[:t + 1])


def test_refractory():
    ex = np.array([1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1], bool)
    assert list(np.flatnonzero(alerting.refractory(ex, 5))) == [0, 5, 11]


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_bundle_round_trip(cfg, tmp_path, name):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 6))
    X[rng.random(X.shape) < 0.05] = np.nan
    y = (rng.random(200) < 0.1).astype(int)
    m = fit_model(name, cfg, 0, X, y)
    meta = {"feature_names": [f"f__a{i}" for i in range(6)], "config_hash": "abc"}
    save_bundle(tmp_path / name, m, meta)
    m2, _ = load_bundle(tmp_path / name, feature_names=meta["feature_names"], config_hash="abc")
    assert np.array_equal(predict_risk(m, X), predict_risk(m2, X))
    with pytest.raises(ValueError):
        load_bundle(tmp_path / name, feature_names=["f__other"])
