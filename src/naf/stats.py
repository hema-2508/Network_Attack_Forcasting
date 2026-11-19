"""Small-sample statistics: exact binomial and cluster-bootstrap intervals."""
from __future__ import annotations

import numpy as np
from scipy import stats


def clopper_pearson(k: int, n: int, level: float = 0.95) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    a = 1 - level
    lo = 0.0 if k == 0 else stats.beta.ppf(a / 2, k, n - k + 1)
    hi = 1.0 if k == n else stats.beta.ppf(1 - a / 2, k + 1, n - k)
    return float(lo), float(hi)


def t_interval(values, level: float = 0.95) -> tuple[float, float]:
    v = np.asarray([x for x in values if np.isfinite(x)], float)
    if len(v) < 2:
        return (float("nan"), float("nan"))
    h = stats.t.ppf(0.5 + level / 2, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v))
    return float(v.mean() - h), float(v.mean() + h)


def bootstrap_ratio(num: np.ndarray, den: np.ndarray, B: int, seed: int = 12345,
                    level: float = 0.95, scale: float = 1.0) -> tuple[float, float]:
    """Cluster bootstrap of sum(num)/sum(den): resample clusters (rows) with replacement."""
    num, den = np.asarray(num, float), np.asarray(den, float)
    if len(num) == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(num), size=(B, len(num)))
    d = den[idx].sum(1)
    r = np.where(d > 0, num[idx].sum(1) / np.where(d > 0, d, 1), np.nan) * scale
    r = r[np.isfinite(r)]
    if len(r) == 0:
        return (float("nan"), float("nan"))
    a = (1 - level) / 2
    return float(np.quantile(r, a)), float(np.quantile(r, 1 - a))
