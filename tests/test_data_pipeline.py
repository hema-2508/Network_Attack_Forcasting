import numpy as np
import pandas as pd
import pytest
from conftest import synthetic_flows

from naf.features import build_features, check_forbidden
from naf.io_raw import parse_timestamps
from naf.windows import available_minute, build_minute_table


def test_timestamps_d_m_y_and_12h_clock():
    s = pd.Series(["7/7/2017 3:30", "6/7/2017 8:59", "03/07/2017 08:55:58", "4/7/2017 12:05", "bad"])
    out = parse_timestamps(s)
    assert out[0] == pd.Timestamp("2017-07-07 15:30")     # afternoon, no AM/PM in file
    assert out[1] == pd.Timestamp("2017-07-06 08:59")     # day/month order
    assert out[2] == pd.Timestamp("2017-07-03 08:55:58")
    assert out[3] == pd.Timestamp("2017-07-04 12:05")     # noon stays noon
    assert pd.isna(out[4])


def test_available_minute_is_never_before_true_end():
    # true start can be up to 59 s after the truncated minute; availability must cover that
    start = pd.Series(pd.to_datetime(["2017-07-04 10:04:00"] * 3))
    dur_us = pd.Series([0.0, 40e6, 60e6])
    avail = available_minute(start, dur_us)
    worst_true_end = start + pd.Timedelta(seconds=59.999) + pd.to_timedelta(dur_us, unit="us")
    assert (avail >= worst_true_end).all()
    assert avail[0] == pd.Timestamp("2017-07-04 10:05")


def test_minute_table_is_causal(cfg):
    """Changing flows that become available after minute t must not change rows <= t."""
    df = synthetic_flows(cfg, n_minutes=40)
    mt1 = build_minute_table(df, cfg)
    t = pd.Timestamp("2017-07-04 09:20")
    avail = available_minute(df["ts"], df["Flow Duration"])
    df2 = df.copy()
    late = avail > t
    df2.loc[late, "Flow Bytes/s"] = 1e9
    df2.loc[late, "SYN Flag Count"] = 0
    mt2 = build_minute_table(df2, cfg)
    cols = [c for c in mt1.columns if c.startswith("f__")]
    a = mt1.loc[mt1["minute"] <= t, cols].to_numpy()
    b = mt2.loc[mt2["minute"] <= t, cols].to_numpy()
    assert np.allclose(a, b, equal_nan=True)
    assert not np.allclose(mt1[cols].to_numpy(), mt2[cols].to_numpy(), equal_nan=True)


def test_features_are_causal_and_time_free(cfg):
    df = synthetic_flows(cfg, n_minutes=40)
    mt = build_minute_table(df, cfg)
    X1 = build_features(mt, cfg)
    mt2 = mt.copy()
    k = 25
    mt2.loc[k:, [c for c in mt.columns if c.startswith("f__")]] *= 3.0
    X2 = build_features(mt2, cfg)
    assert np.allclose(X1.iloc[:k].to_numpy(), X2.iloc[:k].to_numpy(), equal_nan=True)
    # shifting the clock changes nothing: no absolute-time feature exists
    mt3 = mt.copy()
    mt3["minute"] = mt3["minute"] + pd.Timedelta(hours=3)
    assert np.allclose(X1.to_numpy(), build_features(mt3, cfg).to_numpy(), equal_nan=True)


@pytest.mark.parametrize("name", ["f__dst_port_entropy", "f__src_ip_count", "f__hour_of_day",
                                  "f__minute", "f__flow_id_x", "f__day_index", "n_flows"])
def test_forbidden_feature_names_rejected(cfg, name):
    with pytest.raises(ValueError):
        check_forbidden([name], cfg)


def test_allow_list_contains_no_identifier_columns(cfg):
    f = cfg["features"]
    allowed = set(f["fields"]) | set(f["flags"])
    assert not allowed & set(f["forbidden_raw_columns"])
