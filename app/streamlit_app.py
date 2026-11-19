"""Streamlit dashboard. Reads only committed artefacts (data/processed, models, results).
It never trains. Every number shown is either read from results/ (written by the scripts)
or computed live by a saved model on recorded data."""
from __future__ import annotations

import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from naf import pipeline  # noqa: E402
from naf.config import load_config, repo_path  # noqa: E402
from naf.replay import replay  # noqa: E402

st.set_page_config(page_title="Attack-Onset Forecasting", layout="wide")

# Reference palette (dataviz skill): fixed slot per entity, never cycled.
COLOR = {"logreg": "#2a78d6", "rf": "#eb6834", "xgb": "#1baf7a",
         "baseline_random": "#898781", "baseline_anomaly": "#52514e"}
DASH = {"baseline_random": "dot", "baseline_anomaly": "dash"}
NAMES = {"logreg": "Logistic Regression", "rf": "Random Forest", "xgb": "XGBoost",
         "baseline_random": "Baseline: random alarms", "baseline_anomaly": "Baseline: current anomaly"}
ALERTER_NAMES = {"fixed": "Fixed threshold", "conformal": "Static conformal", "aci": "Adaptive conformal (ACI)"}
STATUS = {"hit": ("#0ca30c", "circle", "Early warning (hit)"),
          "false": ("#d03b3b", "x", "False alarm"),
          "ambiguous": ("#eda100", "diamond", "Alarm near tiny burst"),
          "ongoing": ("#6250d6", "triangle-up", "Alarm during ongoing attack")}


# ---------------------------------------------------------------- data access
@st.cache_resource
def cfg():
    return load_config()


@st.cache_data
def dataset():
    return pipeline.load_dataset(cfg())


@st.cache_data
def results(name: str):
    p = repo_path(cfg(), "results_dir") / name
    if not p.exists():
        return None
    if p.suffix == ".csv":
        return pd.read_csv(p)
    if p.suffix == ".json":
        return json.loads(p.read_text(encoding="utf-8"))
    return pd.read_parquet(p)


@st.cache_data
def processed_json(name: str):
    return json.loads((repo_path(cfg(), "processed_dir") / name).read_text(encoding="utf-8"))


@st.cache_data
def run_replay(fold, model, seed, alerter):
    mins, events, metrics = replay(cfg(), fold, model, seed, alerter, data=dataset())
    return mins, events, metrics


def fmt_ci(v, lo, hi, pct=False, nd=2):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    f = (lambda x: f"{100 * x:.0f}%") if pct else (lambda x: f"{x:.{nd}f}")
    if lo is None or np.isnan(lo):
        return f(v)
    return f"{f(v)} [{f(lo)}, {f(hi)}]"


def summary_table(s: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame({
        "Model": s["model"].map(NAMES), "Alerter": s["alerter"].map(ALERTER_NAMES),
        "Seeds": s["n_seeds"], "Test onsets": s["n_onsets"],
        "Onsets warned (mean over seeds)": s["warned_mean"].round(1),
        "Detection [95% bootstrap]": [fmt_ci(a, b, c, pct=True) for a, b, c in
                                      zip(s["detection"], s["det_boot_lo"], s["det_boot_hi"])],
        "Mean lead, min [95%]": [fmt_ci(a, b, c, nd=1) for a, b, c in
                                 zip(s["mean_lead_min"], s["lead_lo"], s["lead_hi"])],
        "False alarms / hour [95%]": [fmt_ci(a, b, c) for a, b, c in
                                      zip(s["fa_per_hour"], s["fa_lo"], s["fa_hi"])],
        "PR-AUC (window)": s["pr_auc_mean"].round(3), "Positive rate": s["positive_rate"].round(3),
    })
    return out


# ---------------------------------------------------------------- pages
def page_overview():
    st.title("Early Forecasting of Network Attack Onset")
    st.caption("CIC-IDS2017 flow telemetry · forecast whether an attack starts in the next H minutes, "
               "under a false-alarm budget")
    info = results("run_info.json")
    s = results("summary_primary.csv")
    if s is None or info is None:
        st.error("No results found. Run scripts/01–05 first (see README).")
        return
    c = st.columns(4)
    meta, X, onsets = dataset()
    c[0].metric("Recorded minutes", f"{len(meta):,}")
    c[1].metric("Attack onsets (all days)", len(onsets))
    c[2].metric("Primary test onsets (Thu+Fri)", int(s["n_onsets"].iloc[0]))
    c[3].metric("False-alarm budget", f"{info['fa_target_per_hour']:g} / hour")
    st.subheader("Headline results (primary folds F1+F2, horizon H = %d min)" % info["H"])
    al = st.radio("Alerter", list(ALERTER_NAMES), format_func=ALERTER_NAMES.get, horizontal=True, index=1)
    st.dataframe(summary_table(s[s["alerter"] == al]), hide_index=True, use_container_width=True)
    st.markdown(
        "**How to read this.** *Detection* = share of test onsets with at least one alarm in the H minutes "
        "before the onset. *Lead* = minutes between first warning and onset. A model is only useful if it "
        "beats the **random-alarm baseline at the same false-alarm rate**. With 10 test onsets, intervals "
        "are wide; overlapping intervals mean the data cannot distinguish the methods.")
    gate = results("power_gate.json")
    if gate:
        st.subheader("Power gate (how much data each fold has)")
        rows = [{"Fold": f, "Role": r["role"], "Test day": r["test_day"],
                 "Train positives": r["train"]["positives"], "Train onsets": r["train"]["onsets_with_positive_minutes"],
                 "Calib onsets": r["calib"]["onsets_with_positive_minutes"],
                 "Test onsets": r["test"]["onsets_with_positive_minutes"],
                 "Test families": ", ".join(r["test"]["families"])} for f, r in gate.items()]
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)


def page_dataset():
    st.title("Dataset facts (measured)")
    rep = processed_json("prepare_report.json")
    meta, X, onsets = dataset()
    c = st.columns(4)
    c[0].metric("Real flow rows", f"{rep['total_real_rows']:,}")
    c[1].metric("Rows dropped (negative duration)", rep["cleaning"]["dropped_negative_or_missing_duration"])
    c[2].metric("Minutes with attack flows", rep["minutes_with_attack_flows"])
    c[3].metric("Model features", X.shape[1])
    day = st.selectbox("Day", sorted(meta["day"].unique()))
    d = meta[meta["day"] == day]
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                        subplot_titles=("All flows becoming available per minute",
                                        "Attack flows starting per minute (log scale)"))
    fig.add_trace(go.Scatter(x=d["minute"], y=d["n_flows"], mode="lines", name="All flows",
                             line=dict(color="#2a78d6", width=2)), 1, 1)
    fig.add_trace(go.Bar(x=d["minute"], y=d["n_attack_flows"].replace(0, np.nan), name="Attack flows",
                         marker_color="#eb6834"), 2, 1)
    fig.update_yaxes(type="log", row=2, col=1)
    for _, o in onsets[onsets["day"] == day].iterrows():
        for r in (1, 2):
            fig.add_vline(x=o["onset"], line=dict(color="#0b0b0b", width=1, dash="dash"), row=r, col=1)
        fig.add_annotation(x=o["onset"], y=1.0, yref="y domain", text=o["family"], showarrow=False,
                           xanchor="left", font=dict(size=10), row=1, col=1)
    fig.update_layout(height=520, hovermode="x unified", showlegend=True, margin=dict(t=40))
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Dashed lines = labelled attack onsets (first minute of a campaign).")
    st.subheader("Onsets")
    st.dataframe(onsets, hide_index=True, use_container_width=True)
    tb = repo_path(cfg(), "processed_dir") / "tiny_bursts.csv"
    st.subheader("Tiny bursts (masked: neither onset nor negative)")
    st.dataframe(pd.read_csv(tb), hide_index=True, use_container_width=True)
    st.subheader("Minute status counts")
    st.dataframe(meta["status"].value_counts().rename_axis("status").reset_index(), hide_index=True)


def replay_figure(mins: pd.DataFrame, onsets_day: pd.DataFrame, upto: int | None, title: str):
    m = mins if upto is None else mins.iloc[:upto + 1]
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.07, row_heights=[0.65, 0.35],
                        subplot_titles=("Model risk score and alert threshold", "Attack flows starting per minute (ground truth, log)"))
    fig.add_trace(go.Scatter(x=m["minute"], y=m["score"], mode="lines", name="Risk score",
                             line=dict(color="#2a78d6", width=2)), 1, 1)
    fig.add_trace(go.Scatter(x=m["minute"], y=m["tau"], mode="lines", name="Threshold τ",
                             line=dict(color="#898781", width=2, dash="dash")), 1, 1)
    for cls, (col, sym, label) in STATUS.items():
        a = m[m["alarm_class"] == cls]
        if len(a):
            fig.add_trace(go.Scatter(x=a["minute"], y=a["score"], mode="markers", name=label,
                                     marker=dict(color=col, symbol=sym, size=11,
                                                 line=dict(color="#fcfcfb", width=2))), 1, 1)
    fig.add_trace(go.Bar(x=m["minute"], y=m["n_attack_flows"].replace(0, np.nan), name="Attack flows",
                         marker_color="#eb6834"), 2, 1)
    fig.update_yaxes(type="log", row=2, col=1)
    fig.update_xaxes(range=[mins["minute"].min(), mins["minute"].max()])
    for _, o in onsets_day.iterrows():
        if upto is None or o["onset"] <= m["minute"].max():
            for r in (1, 2):
                fig.add_vline(x=o["onset"], line=dict(color="#0b0b0b", width=1, dash="dash"), row=r, col=1)
            fig.add_annotation(x=o["onset"], y=1.0, yref="y domain", text=f"onset: {o['family']}",
                               showarrow=False, xanchor="left", font=dict(size=10), row=1, col=1)
    fig.update_layout(height=560, hovermode="x unified", title=title, margin=dict(t=70))
    return fig


def page_replay():
    st.title("Replay of a recorded attack day")
    st.caption("The saved model scores the recorded minutes one by one; the alerter decides causally. "
               "This is the same code path as the evaluation script.")
    c = cfg()
    cols = st.columns(4)
    fold = cols[0].selectbox("Test day", list(c["folds"]["test_days"]),
                             format_func=lambda f: f"{f} · {c['folds']['test_days'][f]}"
                             + (" (exploratory)" if f in c["folds"]["exploratory"] else ""), index=2)
    model = cols[1].selectbox("Model", list(NAMES), format_func=NAMES.get, index=2)
    seeds = c["models"]["seeds"] if model in ("rf", "xgb", "baseline_random") else [0]
    seed = cols[2].selectbox("Seed", seeds)
    alerter = cols[3].selectbox("Alerter", list(ALERTER_NAMES), format_func=ALERTER_NAMES.get, index=2)
    mins, events, metrics = run_replay(fold, model, seed, alerter)
    mins = mins.reset_index(drop=True)
    _, _, onsets = dataset()
    on_day = onsets[onsets["day"] == c["folds"]["test_days"][fold]]

    stored = results("minutes.parquet")
    if stored is not None:
        s = stored[(stored.variant == "forward") & (stored.fold == fold) & (stored.model == model) &
                   (stored.seed == seed) & (stored.alerter == alerter)].sort_values("minute")
        same = len(s) == len(mins) and np.allclose(s["score"], mins["score"]) and \
            np.array_equal(s["alarm"].to_numpy(), mins["alarm"].to_numpy())
        (st.success if same else st.error)(
            "Replay output is identical to the stored evaluation results." if same
            else "Replay differs from stored evaluation — re-run scripts/04_evaluate.py.")

    k = st.columns(4)
    k[0].metric("Onsets warned", f"{metrics['n_warned']} / {metrics['n_onsets']}")
    k[1].metric("Mean lead (min)", "—" if np.isnan(metrics["mean_lead_min"]) else f"{metrics['mean_lead_min']:.1f}")
    k[2].metric("False alarms / hour", f"{metrics['fa_per_hour']:.2f}")
    k[3].metric("Alarms during attacks / near bursts", f"{metrics['n_ongoing_alarms']} / {metrics['n_ambiguous_alarms']}")

    holder = st.empty()
    b = st.columns([1, 1, 4])
    speed = b[2].slider("Playback delay (s per minute)", 0.0, 0.3, 0.03, 0.01)
    if b[0].button("▶ Play minute-by-minute"):
        for t in range(0, len(mins), 2):
            holder.plotly_chart(replay_figure(mins, on_day, t, f"Live replay · {mins['minute'].iloc[t]:%H:%M}"),
                                use_container_width=True, key=f"live{t}")
            time.sleep(speed)
    else:
        holder.plotly_chart(replay_figure(mins, on_day, None, "Full day"), use_container_width=True)
    st.subheader("Per-onset outcome")
    st.dataframe(events, hide_index=True, use_container_width=True)
    with st.expander("Alarm log"):
        st.dataframe(mins.loc[mins["alarm"], ["minute", "score", "tau", "alarm_class", "status"]],
                     hide_index=True, use_container_width=True)


def ci_dot_chart(s: pd.DataFrame, value, lo, hi, title, xlab, target=None, pct=False):
    fig = go.Figure()
    for m in [x for x in NAMES if x in s["model"].unique()]:
        d = s[s["model"] == m]
        y = d["alerter"].map(ALERTER_NAMES)
        fig.add_trace(go.Scatter(
            x=d[value], y=y, mode="markers", name=NAMES[m],
            marker=dict(color=COLOR[m], size=11, symbol="circle" if m in ("logreg", "rf", "xgb") else "diamond",
                        line=dict(color="#fcfcfb", width=2)),
            error_x=dict(type="data", symmetric=False, array=d[hi] - d[value], arrayminus=d[value] - d[lo],
                         color=COLOR[m], thickness=2, width=4),
            hovertemplate="%{y}<br>" + NAMES[m] + ": %{x:.2f}<extra></extra>"))
    if target is not None:
        fig.add_vline(x=target, line=dict(color="#898781", dash="dash"), annotation_text="target")
    fig.update_layout(title=title, xaxis_title=xlab, height=380, scattermode="group",
                      legend=dict(orientation="h", y=-0.25), margin=dict(t=50))
    if pct:
        fig.update_xaxes(tickformat=".0%", range=[0, 1])
    return fig


def page_models():
    st.title("Model comparison (primary folds, 5 seeds for RF/XGBoost)")
    s = results("summary_primary.csv")
    if s is None:
        st.error("Run scripts/04_evaluate.py first.")
        return
    c = st.columns(2)
    c[0].plotly_chart(ci_dot_chart(s, "detection", "det_boot_lo", "det_boot_hi",
                                   "Detection rate (95% event bootstrap)", "share of 10 test onsets warned", pct=True),
                      use_container_width=True)
    c[1].plotly_chart(ci_dot_chart(s, "fa_per_hour", "fa_lo", "fa_hi", "False alarms per quiet hour (95% block bootstrap)",
                                   "false alarms / hour", target=cfg()["alerting"]["fa_target_per_hour"]),
                      use_container_width=True)
    st.subheader("All primary results")
    st.dataframe(summary_table(s), hide_index=True, use_container_width=True)
    st.subheader("Per fold")
    pf = results("summary_per_fold.csv")
    st.dataframe(pd.concat([pf[["folds"]].rename(columns={"folds": "Fold"}), summary_table(pf)], axis=1),
                 hide_index=True, use_container_width=True)
    st.subheader("Per onset (warned share over seeds)")
    fe = results("family_events.csv")
    st.dataframe(fe, hide_index=True, use_container_width=True)
    st.caption("Seed CIs (t-interval over 5 seeds) are in results/summary_primary.csv (det_seed_lo/hi); "
               "they capture model randomness only, not data scarcity.")


def page_drift():
    st.title("Temporal drift and unseen attack families")
    st.markdown("**Forward folds** test on a later day than training: families on Thursday/Friday never "
                "appear in training, so drift and family novelty are mixed together.")
    d = results("drift.csv")
    if d is not None:
        g = d.groupby(["fold", "model", "alerter"], as_index=False)["test_quiet_exceed_rate"].mean()
        g["realised / target"] = g["test_quiet_exceed_rate"] / d["target_exceed_rate"].iloc[0]
        fig = go.Figure()
        for m in [x for x in NAMES if x in g["model"].unique()]:
            x = g[(g["model"] == m) & (g["alerter"] == "conformal")]
            fig.add_trace(go.Bar(x=x["fold"], y=x["realised / target"], name=NAMES[m], marker_color=COLOR[m]))
        fig.add_hline(y=1.0, line=dict(color="#898781", dash="dash"), annotation_text="on target")
        fig.update_layout(barmode="group", bargap=0.3, bargroupgap=0.08, height=360,
                          title="Static conformal: realised quiet-minute exceedance ÷ target, per test day",
                          yaxis_title="ratio")
        st.plotly_chart(fig, use_container_width=True)
        st.caption("> 1 means the threshold calibrated on earlier days fires more often on the test day (drift).")
    sh = results("score_shift.csv")
    if sh is not None:
        st.subheader("Score shift on benign minutes (seed 0)")
        st.dataframe(sh.pivot_table(index=["fold", "model"], columns="block", values="score_median").reset_index(),
                     hide_index=True, use_container_width=True)
    st.subheader("Leave-one-day-out (NON-chronological diagnostic)")
    st.caption("Each attack day is tested with all other attack days in training and attack-free Monday as "
               "calibration. Uses future days for training, so it is not a deployment estimate.")
    ls = results("lodo_summary.csv")
    if ls is not None:
        st.dataframe(summary_table(ls), hide_index=True, use_container_width=True)
        lf = results("lodo_family_events.csv")
        al = st.radio("Alerter", list(ALERTER_NAMES), format_func=ALERTER_NAMES.get, horizontal=True, key="lodo_al")
        p = lf[lf["alerter"] == al].pivot_table(index=["fold", "onset_id", "family"], columns="model",
                                                 values="warned_frac").reset_index()
        st.dataframe(p, hide_index=True, use_container_width=True)
    ab = results("ablation_no_monday.csv")
    if ab is not None:
        st.subheader("Day-identity control: training without attack-free Monday")
        st.dataframe(summary_table(ab), hide_index=True, use_container_width=True)
    dc = results("detection_check_runs.csv")
    if dc is not None:
        st.subheader("Diagnostic: can the same features DETECT running attacks?")
        st.caption("Separate detection models (attack flow in the previous 3 minutes vs quiet), same folds and "
                   "features. Good detection + poor forecasting ⇒ the features see attacks, but there is little "
                   "signal *before* onset.")
        g = dc.groupby(["fold", "model"], as_index=False)["roc_auc"].mean()
        fig = go.Figure()
        for m in ["logreg", "rf", "xgb"]:
            x = g[g["model"] == m]
            fig.add_trace(go.Bar(x=x["fold"], y=x["roc_auc"], name=NAMES[m], marker_color=COLOR[m],
                                 hovertemplate=NAMES[m] + " %{x}: ROC-AUC %{y:.3f}<extra></extra>"))
        fig.add_hline(y=0.5, line=dict(color="#898781", dash="dash"), annotation_text="chance")
        fig.update_layout(barmode="group", bargap=0.3, bargroupgap=0.08, height=340, yaxis_range=[0, 1],
                          yaxis_title="detection ROC-AUC (test day)")
        st.plotly_chart(fig, use_container_width=True)
        de = results("detection_check_events.csv")
        st.dataframe(de.groupby(["fold", "onset_id", "family", "model"]).agg(
            detected_share=("detected", "mean"), mean_delay_min=("delay_min", "mean")).round(2)
            .unstack("model").reset_index(), hide_index=True, use_container_width=True)
    se = results("sensitivity.csv")
    if se is not None:
        st.subheader("Sensitivity to onset gap G and horizon H (primary folds)")
        al = st.radio("Alerter ", ["conformal", "aci"], format_func=ALERTER_NAMES.get, horizontal=True)
        x = se[se["alerter"] == al]
        fig = make_subplots(rows=1, cols=3, shared_yaxes=True, subplot_titles=[f"G = {g} min" for g in sorted(x["G"].unique())])
        for i, G in enumerate(sorted(x["G"].unique()), start=1):
            for m in [k for k in NAMES if k in x["model"].unique()]:
                d = x[(x["G"] == G) & (x["model"] == m)].sort_values("H")
                fig.add_trace(go.Scatter(x=d["H"], y=d["detection"], mode="lines+markers", name=NAMES[m],
                                         line=dict(color=COLOR[m], width=2, dash=DASH.get(m, "solid")),
                                         marker=dict(size=8), showlegend=(i == 1)), 1, i)
        fig.update_yaxes(tickformat=".0%", title_text="detection", row=1, col=1)
        fig.update_xaxes(title_text="horizon H (min)")
        fig.update_layout(height=380, legend=dict(orientation="h", y=-0.3))
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(summary_table(x).assign(G=x["G"].values, H=x["H"].values), hide_index=True,
                     use_container_width=True)


@st.cache_data
def tradeoff(fold, model, seed, budgets):
    c = cfg()
    meta, X, onsets = dataset()
    block = pipeline.fold_block(meta, fold, c)
    rows = (block == "calib") | (block == "test")
    from naf.config import config_hash
    from naf.registry import bundle_dir, load_bundle
    m = None
    if model in pipeline.models.MODEL_NAMES:
        m, _ = load_bundle(bundle_dir(repo_path(c, "models_dir"), fold, model, seed), list(X.columns), config_hash(c))
    scores = pipeline.score_rows(model, seed, m, meta, X, rows, c)
    out = []
    for b in budgets:
        cb = copy.deepcopy(c)
        cb["alerting"]["fa_target_per_hour"] = b
        _, ev, met = pipeline.evaluate_scores(scores, meta, block, onsets, cb, "conformal")
        out.append({"budget": b, "detection": met["n_warned"] / max(met["n_onsets"], 1),
                    "fa_per_hour": met["fa_per_hour"], "mean_lead_min": met["mean_lead_min"]})
    return pd.DataFrame(out)


def page_policy():
    st.title("Alert-policy explorer")
    st.caption("Change the false-alarm budget; the static-conformal threshold is recalibrated on the "
               "calibration block and the saved model is re-evaluated live on the test day.")
    c = cfg()
    cols = st.columns(3)
    fold = cols[0].selectbox("Test day", c["folds"]["primary"], format_func=lambda f: f"{f} · {c['folds']['test_days'][f]}")
    seed = cols[2].selectbox("Seed (RF / XGBoost / random)", c["models"]["seeds"])
    budgets = (0.25, 0.5, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0)
    fig = go.Figure()
    for m in NAMES:
        t = tradeoff(fold, m, seed if m in ("rf", "xgb", "baseline_random") else 0, budgets)
        fig.add_trace(go.Scatter(x=t["fa_per_hour"], y=t["detection"], mode="lines+markers", name=NAMES[m],
                                 line=dict(color=COLOR[m], width=2, dash=DASH.get(m, "solid")), marker=dict(size=8),
                                 customdata=t[["budget", "mean_lead_min"]],
                                 hovertemplate=NAMES[m] + "<br>budget %{customdata[0]}/h<br>realised FA %{x:.2f}/h"
                                 "<br>detection %{y:.0%}<br>mean lead %{customdata[1]:.1f} min<extra></extra>"))
    fig.update_layout(title="Detection vs realised false alarms per hour (test day)", height=460,
                      xaxis_title="realised false alarms per quiet hour", yaxis_title="share of onsets warned",
                      yaxis_tickformat=".0%")
    st.plotly_chart(fig, use_container_width=True)
    st.caption("A useful model sits above the random-alarm baseline at the same false-alarm rate.")


def page_about():
    st.title("About, method and limitations")
    st.markdown(f"""
**Task.** At each minute t, predict whether an attack *onset* occurs in (t, t+H] using only flows that had
**finished** before t (availability = ceil(start + duration) + 1 minute). H = {cfg()['labels']['H']} min.

**Onset.** Attack-active minutes form episodes (gap ≤ 2 min). Episodes with < 3 active minutes are *tiny bursts*
(masked). Valid episodes closer than G = {cfg()['labels']['G']} min form one *campaign*; its first minute is the onset.

**Features.** Whole-network per-minute aggregates of flow statistics, TCP-flag shares, protocol shares and their
previous-10-minute history. **No IP addresses, ports, Flow IDs or clock/calendar features.**

**Evaluation.** Forward-chaining by day (train on earlier days, calibrate alarms on the last 20 % of them, test on the
next day once). Primary test days: Thursday and Friday (10 onsets). Wednesday is exploratory.

**Limitations.** Only 14 onsets in the whole dataset and 3–5 in each primary training block; most test families are
unseen during training; one lab network with a single attacker host; 1-minute timestamps; CIC-IDS2017 has known
labelling/feature-extraction issues. Results must be read as an exploratory event-level study.
""")


PAGES = {"Overview": page_overview, "Dataset": page_dataset, "Replay": page_replay,
         "Model comparison": page_models, "Drift & unseen families": page_drift,
         "Alert-policy explorer": page_policy, "About & limitations": page_about}
choice = st.sidebar.radio("Page", list(PAGES))
st.sidebar.caption("All numbers come from scripts in this repository; nothing is hard-coded.")
PAGES[choice]()
