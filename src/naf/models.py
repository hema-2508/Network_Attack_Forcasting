"""The three approved models behind one interface. Hyper-parameters are fixed in config
(calibration blocks hold only 1-4 onsets, so tuning on them would mostly fit noise)."""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

MODEL_NAMES = ["logreg", "rf", "xgb"]
STOCHASTIC = {"rf", "xgb"}


def make_model(name: str, cfg: dict, seed: int, y_train: np.ndarray):
    p = cfg["models"][name]
    if name == "logreg":
        clf = LogisticRegression(C=p["C"], max_iter=p["max_iter"], class_weight="balanced")
        return Pipeline([("impute", SimpleImputer(strategy="median")),
                         ("scale", StandardScaler()), ("clf", clf)])
    if name == "rf":
        clf = RandomForestClassifier(n_estimators=p["n_estimators"], min_samples_leaf=p["min_samples_leaf"],
                                     max_features=p["max_features"], class_weight="balanced_subsample",
                                     random_state=seed, n_jobs=1)  # n_jobs=1: bit-exact replay
        return Pipeline([("impute", SimpleImputer(strategy="median")), ("clf", clf)])
    if name == "xgb":
        pos = max(int(y_train.sum()), 1)
        neg = max(int(len(y_train) - y_train.sum()), 1)
        return XGBClassifier(n_estimators=p["n_estimators"], max_depth=p["max_depth"],
                             learning_rate=p["learning_rate"], subsample=p["subsample"],
                             colsample_bytree=p["colsample_bytree"], scale_pos_weight=neg / pos,
                             random_state=seed, n_jobs=4, tree_method="hist", eval_metric="logloss")
    raise ValueError(name)


def seeds_for(name: str, cfg: dict) -> list[int]:
    return list(cfg["models"]["seeds"]) if name in STOCHASTIC else [0]


def fit_model(name: str, cfg: dict, seed: int, X: np.ndarray, y: np.ndarray):
    m = make_model(name, cfg, seed, y)
    m.fit(X, y)
    return m


def predict_risk(model, X: np.ndarray) -> np.ndarray:
    return model.predict_proba(X)[:, 1]
