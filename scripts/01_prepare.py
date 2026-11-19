"""Phase 2: raw CSVs -> data/processed/minute_table.parquet (+ prepare_report.json).

Usage:  python scripts/01_prepare.py --raw-dir "C:/path/to/TrafficLabelling"
"""
import argparse
import json
import time

import _common  # noqa: F401

from naf.clean import clean_flows
from naf.config import load_config, raw_dir, repo_path
from naf.io_raw import load_all
from naf.windows import build_minute_table


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default=None)
    args = ap.parse_args()
    cfg = load_config()
    t0 = time.time()
    src = raw_dir(cfg, args.raw_dir)
    print(f"Reading raw CSVs from {src}")
    flows, file_rep = load_all(src, cfg)
    flows, clean_rep = clean_flows(flows)
    print(f"Clean flows: {len(flows):,}")
    mt = build_minute_table(flows, cfg)
    out = repo_path(cfg, "processed_dir")
    mt.to_parquet(out / "minute_table.parquet", index=False)
    rep = {"files": file_rep, "cleaning": clean_rep,
           "total_real_rows": sum(v["rows"] for v in file_rep.values()),
           "minutes": len(mt), "minutes_per_day": mt.groupby("day").size().to_dict(),
           "minutes_with_attack_flows": int((mt["n_attack_flows"] > 0).sum()),
           "minutes_without_available_flows": int((mt["n_flows"] == 0).sum()),
           "flows_binned_into_grid": int(mt["n_flows"].sum()),
           "n_base_features": sum(c.startswith("f__") for c in mt.columns),
           "seconds": round(time.time() - t0, 1)}
    (out / "prepare_report.json").write_text(json.dumps(rep, indent=1, default=str), encoding="utf-8")
    print(json.dumps({k: v for k, v in rep.items() if k not in ("files",)}, indent=1, default=str))


if __name__ == "__main__":
    main()
