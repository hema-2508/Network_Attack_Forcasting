"""Turn cleaned flows into the per-minute table (CONTRACT A).

Causality rule (beginner explanation): a flow can only be used once it has *finished*.
Start times are truncated to the minute, so the true end can be up to 60 s after
start_minute + duration. We therefore make a flow available at
    ceil_minute(start_minute + duration) + 1 minute,
and the feature row for minute t (decision made at the start of minute t) only contains
flows available at or before t. Labels (attack activity) use the flow START minute.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

ONE_MIN = pd.Timedelta(minutes=1)


def available_minute(start: pd.Series, duration_us: pd.Series) -> pd.Series:
    start_min = start.dt.floor("min")
    return (start_min + pd.to_timedelta(duration_us, unit="us")).dt.ceil("min") + ONE_MIN


def minute_grid(start_min: pd.Series) -> pd.DataFrame:
    """Every minute from the first to the last flow start of each day."""
    rows = []
    for day, g in start_min.groupby(start_min.dt.date):
        rng = pd.date_range(g.min(), g.max(), freq="min")
        rows.append(pd.DataFrame({"minute": rng, "day": str(day)}))
    return pd.concat(rows, ignore_index=True)


def aggregate_features(df: pd.DataFrame, key: pd.Series, cfg: dict) -> pd.DataFrame:
    f = cfg["features"]
    out = {}
    g_count = df.groupby(key).size()
    out["f__log_n_flows"] = np.log1p(g_count)
    vals = pd.DataFrame({short: np.log1p(df[raw].clip(lower=0)) for raw, short in f["fields"].items()})
    grp = vals.groupby(key.values)
    mean, p95 = grp.mean(), grp.quantile(0.95)
    for short in f["fields"].values():
        out[f"f__{short}_mean"] = mean[short]
        out[f"f__{short}_p95"] = p95[short]
    for raw in f["sentinel_fields"]:
        out[f"f__{f['fields'][raw]}_missing_frac"] = (df[raw] == -1).groupby(key.values).mean()
    for raw, short in f["flags"].items():
        out[f"f__{short}_frac"] = (df[raw] > 0).groupby(key.values).mean()
    out["f__frac_fwd_single_pkt"] = (df["Total Fwd Packets"] <= 1).groupby(key.values).mean()
    out["f__frac_no_bwd_pkt"] = (df["Total Backward Packets"] == 0).groupby(key.values).mean()
    out["f__frac_short_flow"] = (df["Flow Duration"] < 1000).groupby(key.values).mean()
    if f["include_protocol_shares"]:
        proto = df["Protocol"]
        out["f__proto_tcp_frac"] = (proto == 6).groupby(key.values).mean()
        out["f__proto_udp_frac"] = (proto == 17).groupby(key.values).mean()
        out["f__proto_other_frac"] = (~proto.isin([6, 17])).groupby(key.values).mean()
    res = pd.DataFrame(out)
    res.index.name = "minute"
    return res


def build_minute_table(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    start_min = df["ts"].dt.floor("min")
    grid = minute_grid(start_min)
    avail = available_minute(df["ts"], df["Flow Duration"])
    feats = aggregate_features(df, avail, cfg)

    attack = df["Label"] != "BENIGN"
    n_attack = attack.groupby(start_min).sum().rename("n_attack_flows")
    counts = (df[attack].groupby([start_min[attack], df.loc[attack, "Label"]]).size()
              .rename("n").reset_index())
    counts.columns = ["minute", "Label", "n"]
    label_json = pd.Series(
        {m: json.dumps(dict(zip(g["Label"], g["n"].astype(int).tolist())))
         for m, g in counts.groupby("minute")}, dtype=object, name="attack_label_counts")
    n_avail = avail.groupby(avail).size().rename("n_flows")

    mt = grid.set_index("minute")
    mt = mt.join(n_avail).join(n_attack).join(label_json).join(feats)
    mt["n_flows"] = mt["n_flows"].fillna(0).astype(int)
    mt["n_attack_flows"] = mt["n_attack_flows"].fillna(0).astype(int)
    mt["attack_label_counts"] = mt["attack_label_counts"].fillna("{}")
    mt["f__log_n_flows"] = mt["f__log_n_flows"].fillna(0.0)
    return mt.reset_index()
