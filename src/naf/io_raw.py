"""Read the CIC-IDS2017 GeneratedLabelledFlows CSVs and repair their timestamps.

Verified quirks (DATASET_INSPECTION_REPORT.md): latin-1 encoding, leading spaces in column
names, 288,602 empty rows in the Thursday web-attacks file, d/m/Y dates, 12-hour clock with
no AM/PM, minute resolution (Monday has seconds), unsorted rows.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

TS_PATTERN = r"^(\d+)/(\d+)/(\d+) (\d+):(\d+)(?::(\d+))?$"


def parse_timestamps(raw: pd.Series, pm_hour_below: int = 8) -> pd.Series:
    """Parse 'd/m/Y H:MM[:SS]' strings written with a 12-hour clock and no AM/PM.

    Hours below `pm_hour_below` (1..7) are afternoon, so 12 is added. 12 stays 12 (noon).
    Unparseable strings become NaT.
    """
    p = raw.astype(str).str.strip().str.extract(TS_PATTERN).astype(float)
    hour = p[3].where(p[3] >= pm_hour_below, p[3] + 12)
    parts = pd.DataFrame({"year": p[2], "month": p[1], "day": p[0], "hour": hour,
                          "minute": p[4], "second": p[5].fillna(0)})
    ok = parts.notna().all(axis=1)
    out = pd.Series(pd.NaT, index=raw.index, dtype="datetime64[ns]")
    if ok.any():
        out[ok] = pd.to_datetime(parts[ok].astype(int))
    return out


def needed_columns(cfg: dict) -> list[str]:
    f = cfg["features"]
    cols = ["Timestamp", "Label", "Protocol", "Flow Duration"]
    cols += [c for c in list(f["fields"]) + list(f["flags"]) if c not in cols]
    # Forbidden identifier columns are never even loaded (except Timestamp, used for binning/labels).
    bad = set(cols) & (set(f["forbidden_raw_columns"]) - {"Timestamp"})
    assert not bad, f"forbidden columns requested: {bad}"
    return cols


def read_file(path: Path, cfg: dict) -> pd.DataFrame:
    want = set(needed_columns(cfg))
    df = pd.read_csv(path, encoding=cfg["data"]["encoding"], low_memory=False,
                     usecols=lambda c: c.strip() in want)
    df.columns = [c.strip() for c in df.columns]
    n_raw = len(df)
    df = df.dropna(how="all")
    df.attrs["n_raw_rows"] = n_raw
    df.attrs["n_empty_rows"] = n_raw - len(df)
    df["ts"] = parse_timestamps(df.pop("Timestamp"), cfg["data"]["pm_hour_below"])
    df["source_file"] = path.name
    return df


def load_all(raw_dir: Path, cfg: dict) -> tuple[pd.DataFrame, dict]:
    frames, report = [], {}
    for name in cfg["data"]["files"]:
        path = Path(raw_dir) / name
        if not path.exists():
            raise FileNotFoundError(f"{path} not found. Set NAF_RAW_DIR or --raw-dir.")
        df = read_file(path, cfg)
        report[name] = {"raw_rows": df.attrs["n_raw_rows"], "empty_rows": df.attrs["n_empty_rows"],
                        "rows": len(df), "ts_min": str(df.ts.min()), "ts_max": str(df.ts.max()),
                        "bad_timestamps": int(df.ts.isna().sum())}
        frames.append(df)
        print(f"  read {name}: {len(df):,} rows", flush=True)
    out = pd.concat(frames, ignore_index=True)
    num = [c for c in out.columns if c not in ("Label", "ts", "source_file")]
    out[num] = out[num].apply(pd.to_numeric, errors="coerce").astype(np.float64)
    out = out.sort_values("ts", kind="stable").reset_index(drop=True)
    return out, report
