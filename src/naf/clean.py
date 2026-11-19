"""Row-level cleaning of flows. Every dropped row is counted in the report."""
from __future__ import annotations

import numpy as np
import pandas as pd


def normalise_label(s: pd.Series) -> pd.Series:
    # latin-1 decoding turns the en dash of "Web Attack – XSS" into '\x96'
    return (s.astype(str).str.replace("\x96", "-", regex=False)
            .str.replace("�", "-", regex=False).str.replace(r"\s+", " ", regex=True).str.strip())


def clean_flows(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    rep = {"rows_in": len(df)}
    num = df.select_dtypes(include=[np.number]).columns
    rep["inf_values_to_nan"] = int(np.isinf(df[num].to_numpy()).sum())
    df[num] = df[num].replace([np.inf, -np.inf], np.nan)
    df["Label"] = normalise_label(df["Label"])
    bad_ts = df["ts"].isna()
    bad_dur = df["Flow Duration"].isna() | (df["Flow Duration"] < 0)
    rep["dropped_bad_timestamp"] = int(bad_ts.sum())
    rep["dropped_negative_or_missing_duration"] = int((bad_dur & ~bad_ts).sum())
    df = df[~bad_ts & ~bad_dur].reset_index(drop=True)
    rep["rows_out"] = len(df)
    rep["labels"] = df["Label"].value_counts().to_dict()
    return df, rep
