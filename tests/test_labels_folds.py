import json

import numpy as np
import pandas as pd

from naf.folds import forward_fold, purge_minutes
from naf.labels import find_episodes, label_minutes


def make_mt(active, day="2017-07-04", n=120):
    minutes = pd.date_range(f"{day} 09:00", periods=n, freq="min")
    att = np.zeros(n, int)
    att[list(active)] = 3
    return pd.DataFrame({"minute": minutes, "day": day, "n_attack_flows": att,
                         "attack_label_counts": [json.dumps({"PortScan": 3}) if a else "{}" for a in att]})


def test_episode_merging():
    assert find_episodes(np.array([1, 2, 4, 10]), 2) == [(1, 4, 3), (10, 10, 1)]


def test_onset_tiny_burst_and_campaign_rules(cfg):
    H, G = cfg["labels"]["H"], cfg["labels"]["G"]
    # tiny burst at 20 (1 minute); campaign A 40-45; episode 50-53 (gap 5 < G -> same campaign);
    # campaign B 80-84 (gap 27 >= G -> new onset)
    active = [20] + list(range(40, 46)) + list(range(50, 54)) + list(range(80, 85))
    lab, on = label_minutes(make_mt(active), cfg)
    assert list(on["onset"].dt.minute) == [40, 20]                 # 09:40 and 10:20
    assert len(on.attrs["tiny_bursts"]) == 1
    assert (lab.loc[20 - H:20, "status"] == "ambiguous").all()     # tiny burst masked, not negative
    assert lab.loc[40 - H:39, "y"].eq(1).all() and lab.loc[40 - H - 1, "y"] == 0
    assert (lab.loc[40:53 + G, "status"] == "ongoing").all()       # campaign + cool-down excluded
    assert lab.loc[80 - H:79, "y"].eq(1).all()
    assert lab.loc[lab["status"] != "eligible", "y"].eq(0).all()


def test_forward_fold_purge_and_no_future(cfg):
    days = ["2017-07-03", "2017-07-04", "2017-07-05"]
    meta = pd.concat([pd.DataFrame({"minute": pd.date_range(f"{d} 09:00", periods=100, freq="min"), "day": d})
                      for d in days], ignore_index=True)
    block = forward_fold(meta, "2017-07-05", cfg)
    assert set(block[meta["day"] == "2017-07-05"]) == {"test"}
    assert not (block[meta["day"] < "2017-07-05"] == "test").any()
    cal_start = meta.loc[block == "calib", "minute"].min()
    train_end = meta.loc[block == "train", "minute"].max()
    assert cal_start - train_end > pd.Timedelta(minutes=purge_minutes(cfg))
    assert meta.loc[block == "train", "minute"].max() < cal_start
