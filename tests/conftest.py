import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from naf.config import load_config  # noqa: E402


@pytest.fixture
def cfg():
    return load_config()


def synthetic_flows(cfg, n_minutes=60, day="2017-07-04", attack_minutes=(), seed=0):
    """Small fake flow table with every allow-listed column (for unit tests only)."""
    rng = np.random.default_rng(seed)
    start = pd.Timestamp(f"{day} 09:00")
    rows = []
    for m in range(n_minutes):
        for _ in range(20):
            rows.append({"ts": start + pd.Timedelta(minutes=m), "Label": "BENIGN"})
        if m in attack_minutes:
            for _ in range(5):
                rows.append({"ts": start + pd.Timedelta(minutes=m), "Label": "PortScan"})
    df = pd.DataFrame(rows)
    f = cfg["features"]
    for col in list(f["fields"]) + list(f["flags"]):
        df[col] = rng.random(len(df)) * 100
    df["Flow Duration"] = rng.integers(0, 90_000_000, len(df)).astype(float)
    df["Protocol"] = rng.choice([6, 17, 0], len(df)).astype(float)
    return df
