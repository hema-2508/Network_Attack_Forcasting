"""Alert rules (PROJECT_SPECIFICATION.md §8). All rules are causal.

alpha = fa_target_per_hour / 60 is the allowed per-minute exceedance rate on quiet minutes.
fixed      : tau = empirical (1-alpha) quantile of calibration-negative scores.
conformal  : tau = finite-sample split-conformal quantile of calibration-negative scores
             (the ceil((n+1)(1-alpha))-th smallest), no feedback needed.
aci        : adaptive conformal inference. Threshold from a rolling window of past scores
             whose 'quiet' status is already known; alpha_t updated with
             alpha_{t+1} = alpha_t + gamma * (alpha* - err).
             Feedback for minute s is used only at t >= s + H + 1, and s counts as quiet only if
             no attack-active minute occurred in [s - G, s + H] (all observed by then).
Refractory : an alarm event is raised at an exceedance only if no alarm event happened in the
             previous R-1 minutes.
"""
from __future__ import annotations

import math
from collections import deque

import numpy as np

ALERTERS = ["fixed", "conformal", "aci"]


def alpha_from_cfg(cfg: dict) -> float:
    return cfg["alerting"]["fa_target_per_hour"] / 60.0


def threshold_fixed(cal_neg: np.ndarray, alpha: float) -> float:
    return float(np.quantile(cal_neg, 1 - alpha, method="higher"))


def threshold_conformal(cal_neg: np.ndarray, alpha: float) -> float:
    s = np.sort(np.asarray(cal_neg))
    k = math.ceil((len(s) + 1) * (1 - alpha))
    return float("inf") if k > len(s) else float(s[k - 1])


def known_quiet(active: np.ndarray, H: int, G: int) -> np.ndarray:
    """quiet[s] = no attack-active minute in [s-G, s+H] (within one day's array)."""
    n = len(active)
    c = np.concatenate([[0], np.cumsum(active.astype(int))])
    lo = np.clip(np.arange(n) - G, 0, n)
    hi = np.clip(np.arange(n) + H + 1, 0, n)
    return (c[hi] - c[lo]) == 0


def run_aci(scores: np.ndarray, active: np.ndarray, cal_neg: np.ndarray, alpha: float,
            H: int, G: int, gamma: float, window: int, a_min: float, a_max: float):
    """Online loop over one test day. Returns exceed, tau_trace, alpha_trace."""
    n = len(scores)
    quiet = known_quiet(active, H, G)
    buf = deque(np.asarray(cal_neg)[-window:], maxlen=window)
    a_t = alpha
    exceed = np.zeros(n, bool)
    taus, alphas = np.zeros(n), np.zeros(n)
    for t in range(n):
        s = t - H - 1                      # newest minute whose feedback is available at t
        if s >= 0 and quiet[s]:
            err = float(exceed[s])
            a_t = float(np.clip(a_t + gamma * (alpha - err), a_min, a_max))
            buf.append(scores[s])
        tau = float(np.quantile(np.asarray(buf), 1 - a_t, method="higher"))
        taus[t], alphas[t] = tau, a_t
        exceed[t] = scores[t] > tau
    return exceed, taus, alphas


def refractory(exceed: np.ndarray, R: int) -> np.ndarray:
    alarm = np.zeros(len(exceed), bool)
    last = -10**9
    for t in np.flatnonzero(exceed):
        if t - last >= R:
            alarm[t] = True
            last = t
    return alarm


def apply_alerter(name: str, scores: np.ndarray, active: np.ndarray, cal_neg: np.ndarray, cfg: dict) -> dict:
    """scores/active: one test day in time order. Returns exceed, alarm, tau arrays."""
    alpha = alpha_from_cfg(cfg)
    H, G = cfg["labels"]["H"], cfg["labels"]["G"]
    a = cfg["alerting"]["aci"]
    if name == "fixed":
        tau = np.full(len(scores), threshold_fixed(cal_neg, alpha))
        exceed, alphas = scores > tau, np.full(len(scores), alpha)
    elif name == "conformal":
        tau = np.full(len(scores), threshold_conformal(cal_neg, alpha))
        exceed, alphas = scores > tau, np.full(len(scores), alpha)
    elif name == "aci":
        exceed, tau, alphas = run_aci(scores, active, cal_neg, alpha, H, G, a["gamma"], a["window"],
                                      a["alpha_min"], a["alpha_max"])
    else:
        raise ValueError(name)
    return {"exceed": exceed, "alarm": refractory(exceed, cfg["alerting"]["refractory_minutes"]),
            "tau": tau, "alpha_t": alphas}
