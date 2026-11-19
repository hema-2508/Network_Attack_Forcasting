"""Model features (CONTRACT B): whole-network aggregates + causal history.

History uses only the previous L minutes of the same day (shift(1) before rolling), never
absolute time. Every feature name is checked against the forbidden-token list.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

EPS = 1e-6


def check_forbidden(names: list[str], cfg: dict) -> None:
    bad_tokens = set(cfg["features"]["forbidden_tokens"])
    for n in names:
        if not n.startswith("f__"):
            raise ValueError(f"feature '{n}' is not an aggregate column (must start with f__)")
        tokens = set(re.split(r"[_\W]+", n.lower())) - {"", "f"}
        hit = tokens & bad_tokens
        if hit:
            raise ValueError(f"forbidden token(s) {hit} in feature '{n}'")


def base_columns(mt: pd.DataFrame) -> list[str]:
    return [c for c in mt.columns if c.startswith("f__")]


def build_features(mt: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    L, mp = cfg["features"]["L"], cfg["features"]["min_history"]
    base = base_columns(mt)
    check_forbidden(base, cfg)
    X = mt[base].astype(float)
    parts = [X]
    by_day = X.groupby(mt["day"].values, sort=False)
    past = by_day.shift(1)                                    # strictly previous minutes
    pg = past.groupby(mt["day"].values, sort=False)
    pmean = pg.rolling(L, min_periods=mp).mean().reset_index(level=0, drop=True).sort_index()
    pstd = pg.rolling(L, min_periods=mp).std().reset_index(level=0, drop=True).sort_index()
    parts.append(pmean.add_suffix("__pmean"))
    parts.append(((X - pmean) / (pstd + EPS)).clip(-50, 50).add_suffix("__z"))
    parts.append((X - past).add_suffix("__d1"))
    out = pd.concat(parts, axis=1)
    check_forbidden(list(out.columns), cfg)
    return out
