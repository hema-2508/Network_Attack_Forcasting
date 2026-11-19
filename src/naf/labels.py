"""Onset labelling (PROJECT_SPECIFICATION.md §4).

episode  : run of attack-active minutes whose consecutive differences are <= gap_merge
tiny     : episode with < min_active_minutes active minutes -> not an onset, masked 'ambiguous'
campaign : valid episodes separated by < G quiet minutes are merged
onset    : first minute of a campaign
status   : 'ongoing'  for [campaign start, campaign end + G]
           'ambiguous' for [tiny start - H, tiny end]           (ongoing wins over ambiguous)
           'eligible' otherwise
y_t      : 1 iff an onset lies in (t, t+H]
"""
from __future__ import annotations

import json
from collections import Counter

import numpy as np
import pandas as pd


def find_episodes(active_idx: np.ndarray, gap_merge: int) -> list[tuple[int, int, int]]:
    """(start, end, n_active) for sorted integer minute indices of active minutes."""
    if len(active_idx) == 0:
        return []
    eps, st, pv, n = [], active_idx[0], active_idx[0], 1
    for m in active_idx[1:]:
        if m - pv > gap_merge:
            eps.append((st, pv, n))
            st, n = m, 0
        pv, n = m, n + 1
    eps.append((st, pv, n))
    return eps


def merge_campaigns(valid: list[tuple[int, int, int]], G: int) -> list[list[int]]:
    camps: list[list[int]] = []
    for st, en, _ in valid:
        if camps and st - camps[-1][1] < G:
            camps[-1][1] = en
        else:
            camps.append([st, en])
    return camps


def _family(label_json: pd.Series, st: int, window: int) -> str:
    c: Counter = Counter()
    for js in label_json.iloc[st:st + window]:
        c.update(json.loads(js))
    return c.most_common(1)[0][0] if c else "unknown"


def label_minutes(mt: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (per-minute labels aligned with mt, onset table). mt must be sorted by minute."""
    lab = cfg["labels"]
    H, G = lab["H"], lab["G"]
    status = np.array(["eligible"] * len(mt), dtype=object)
    y = np.zeros(len(mt), dtype=int)
    onset_id = np.array([""] * len(mt), dtype=object)       # onset that makes y_t = 1
    onsets, tinies = [], []
    for day, idx in mt.groupby("day").indices.items():
        idx = np.sort(idx)
        d = mt.iloc[idx]
        n = len(d)
        active = np.flatnonzero(d["n_attack_flows"].to_numpy() > 0)
        eps = find_episodes(active, lab["gap_merge"])
        valid = [e for e in eps if e[2] >= lab["min_active_minutes"]]
        tiny = [e for e in eps if e[2] < lab["min_active_minutes"]]
        st_local = np.array(["eligible"] * n, dtype=object)
        for s, e, k in tiny:
            st_local[max(0, s - H):e + 1] = "ambiguous"
            tinies.append({"day": day, "start": d["minute"].iloc[s], "end": d["minute"].iloc[e],
                           "active_minutes": k, "attack_flows": int(d["n_attack_flows"].iloc[s:e + 1].sum())})
        camps = merge_campaigns(valid, G)
        for s, e in camps:
            st_local[s:min(n, e + G + 1)] = "ongoing"
        y_local = np.zeros(n, dtype=int)
        oid_local = np.array([""] * n, dtype=object)
        for s, e in camps:
            oid = f"{day}_{d['minute'].iloc[s]:%H%M}"
            onsets.append({"onset_id": oid, "day": day, "onset": d["minute"].iloc[s],
                           "end": d["minute"].iloc[e], "duration_min": e - s + 1,
                           "attack_flows": int(d["n_attack_flows"].iloc[s:e + 1].sum()),
                           "family": _family(d["attack_label_counts"], s, lab["family_window"])})
            lo = max(0, s - H)
            y_local[lo:s] = 1                       # t in [s-H, s-1]  <=>  s in (t, t+H]
            oid_local[lo:s] = oid
        status[idx], y[idx], onset_id[idx] = st_local, y_local, oid_local
    out = pd.DataFrame({"status": status, "y": y, "onset_id": onset_id}, index=mt.index)
    out.loc[out["status"] != "eligible", "y"] = 0          # only eligible minutes are samples
    out.loc[out["status"] != "eligible", "onset_id"] = ""
    on = pd.DataFrame(onsets)
    on.attrs["tiny_bursts"] = pd.DataFrame(tinies)
    return out, on
