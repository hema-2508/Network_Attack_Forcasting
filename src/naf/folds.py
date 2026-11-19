"""Chronological fold assignment (PROJECT_SPECIFICATION.md §7).

Forward-chaining: for test day D, the pre-test period is every earlier day. Its last
`calib_fraction` of minutes is the calibration block, the rest is training. Training minutes
within P = H + L + G minutes before the calibration start (same day) are purged, so no label
horizon, history window or campaign cool-down straddles the boundary. Days are separate
captures, so nothing crosses midnight.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def purge_minutes(cfg: dict) -> int:
    return cfg["labels"]["H"] + cfg["features"]["L"] + cfg["labels"]["G"]


def forward_fold(meta: pd.DataFrame, test_day: str, cfg: dict, drop_monday: bool = False) -> pd.Series:
    """Block per row: train / calib / test / purged / unused. meta has 'minute', 'day'."""
    block = pd.Series("unused", index=meta.index, dtype=object)
    pre = meta.index[meta["day"] < test_day]
    pre = meta.loc[pre].sort_values("minute").index
    n_cal = int(round(len(pre) * cfg["folds"]["calib_fraction"]))
    cal_idx, tr_idx = pre[len(pre) - n_cal:], pre[:len(pre) - n_cal]
    block[tr_idx] = "train"
    block[cal_idx] = "calib"
    block[meta.index[meta["day"] == test_day]] = "test"
    cal_start = meta.loc[cal_idx, "minute"].min()
    P = pd.Timedelta(minutes=purge_minutes(cfg))
    purge = (block == "train") & (meta["day"] == str(cal_start.date())) & \
            (meta["minute"] >= cal_start - P)
    block[purge] = "purged"
    if drop_monday:
        block[(block == "train") & (meta["day"] == cfg["folds"]["monday"])] = "unused"
    return block


def leave_one_day_out(meta: pd.DataFrame, test_day: str, cfg: dict) -> pd.Series:
    """NON-CHRONOLOGICAL diagnostic: train on all other attack days (incl. future ones),
    calibrate on attack-free Monday (negatives only), test on `test_day`."""
    block = pd.Series("unused", index=meta.index, dtype=object)
    monday = cfg["folds"]["monday"]
    block[(meta["day"] != test_day) & (meta["day"] != monday)] = "train"
    block[meta["day"] == monday] = "calib"
    block[meta["day"] == test_day] = "test"
    return block
