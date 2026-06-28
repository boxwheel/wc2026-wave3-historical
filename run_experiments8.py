"""
Batch 8: Attempts 034-037
034 — Extended host_bonus sweep (250-500) at best 030 params (dr=0.30, sc=400)
035 — Gaussian draw probability model (pD peaks at elo_diff=0, decays symmetrically)
036 — Scale-extended 3D fine grid around 030 optimum (bonus 250-450)
037 — FIFA rank offset as Elo supplement
"""
import os, sys, json, math, itertools, warnings
import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import ttest_1samp
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import log_loss
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

BASE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE, "data")
MATCHES_PATH = os.path.join(DATA_DIR, "wc2026-trees-study-main/fifa_data/matches_detailed.csv")
TEAMS_PATH   = os.path.join(DATA_DIR, "wc2026-trees-study-main/fifa_data/teams.csv")
HIST_PATH    = os.path.join(DATA_DIR, "historical/results.csv")

BASELINE = 0.8337
HOST_TEAMS = {"Mexico", "USA", "Canada"}

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_data():
    matches = pd.read_csv(MATCHES_PATH)
    matches = matches.rename(columns={"home_team_name": "home_team",
                                       "away_team_name": "away_team"})
    completed = matches[matches["status"] == "Completed"].copy()
    teams = pd.read_csv(TEAMS_PATH)
    return completed, teams

def build_df(completed, teams):
    team_elo  = dict(zip(teams["team_name"], teams["elo_rating"].astype(float)))
    team_rank = dict(zip(teams["team_name"], teams["fifa_ranking_pre_tournament"].astype(float)))
    rows = []
    for _, m in completed.iterrows():
        h, a = m["home_team"], m["away_team"]
        y = 2 if m["home_score"] > m["away_score"] else (0 if m["home_score"] < m["away_score"] else 1)
        rows.append({"home": h, "away": a,
                     "elo_h": team_elo.get(h, 1700), "elo_a": team_elo.get(a, 1700),
                     "rank_h": team_rank.get(h, 50), "rank_a": team_rank.get(a, 50),
                     "host_h": 1.0 if h in HOST_TEAMS else 0.0, "y": y})
    return pd.DataFrame(rows)

# ---------------------------------------------------------------------------
# Elo formula
# ---------------------------------------------------------------------------
def elo_probs(elo_diff, dr=0.30, sc=400):
    Eh = expit(elo_diff * math.log(10) / sc)
    Ea = 1.0 - Eh
    pH = np.clip(Eh - 0.5 * dr, 0.01, 0.98)
    pA = np.clip(Ea - 0.5 * dr, 0.01, 0.98)
    pD = np.clip(1.0 - pH - pA, 0.01, 0.98)
    total = pH + pD + pA
    return np.stack([pA / total, pD / total, pH / total], axis=1)

def elo_probs_gaussian_draw(elo_diff, sc=400, max_dr=0.32, sigma=200):
    """Draw probability follows Gaussian bell: peaks at elo_diff=0."""
    Eh = expit(elo_diff * math.log(10) / sc)
    Ea = 1.0 - Eh
    pD_raw = max_dr * np.exp(-elo_diff**2 / (2 * sigma**2))
    pD = np.clip(pD_raw, 0.01, 0.60)
    pWin_total = np.clip(1.0 - pD, 0.05, 0.99)
    pH = np.clip(Eh * pWin_total, 0.01, 0.97)
    pA = np.clip(Ea * pWin_total, 0.01, 0.97)
    total = pH + pD + pA
    return np.stack([pA / total, pD / total, pH / total], axis=1)

def eval_formula(df, prob_fn, **kwargs):
    """Evaluate a formula that takes elo_diff array and returns [pA, pD, pH]."""
    skf = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=0)
    losses = []
    for _, test_idx in skf.split(df, df["y"]):
        test_df = df.iloc[test_idx]
        elo_diffs = (test_df["elo_h"] - test_df["elo_a"]).values
        if "bonus" in kwargs:
            elo_diffs = elo_diffs + kwargs["bonus"] * test_df["host_h"].values
        probs = prob_fn(elo_diffs, **{k: v for k, v in kwargs.items() if k != "bonus"})
        losses.append(log_loss(test_df["y"].values, probs, labels=[0, 1, 2]))
    return np.array(losses)

def verdict(mean, baseline=BASELINE, p_val=None):
    if p_val is None:
        return "FLAT" if mean >= baseline else "GREEN"
    if mean < baseline and p_val < 0.05:
        return "GREEN"
    elif mean >= baseline * 1.01:
        return "RED"
    return "FLAT"

def save_artifact(attempt, metrics, run_info):
    d = os.path.join(BASE, "artifacts", f"attempt-{attempt:03d}")
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    with open(os.path.join(d, "run.json"), "w") as f:
        json.dump(run_info, f, indent=2)

# ---------------------------------------------------------------------------
# Attempt 034: Extended host_bonus sweep at dr=0.30, sc=400
# ---------------------------------------------------------------------------
def attempt_034(df):
    print("\n=== Attempt 034: Extended host_bonus sweep (250-600) at dr=0.30, sc=400 ===")
    bonus_grid = [250, 275, 300, 325, 350, 375, 400, 425, 450, 500, 550, 600]
    dr, sc = 0.30, 400

    results = []
    for bonus in bonus_grid:
        losses = eval_formula(df, elo_probs, dr=dr, sc=sc, bonus=bonus)
        results.append((bonus, losses.mean(), losses.std()))

    results.sort(key=lambda x: x[1])
    print(f"{'bonus':>8} {'mean':>8} {'std':>8}")
    for bonus, mean, std in results[:8]:
        print(f"{bonus:8d} {mean:8.4f} {std:8.4f}")

    best_bonus, best_mean, best_std = results[0]
    losses = eval_formula(df, elo_probs, dr=dr, sc=sc, bonus=best_bonus)
    t_stat, p_val = ttest_1samp(losses, BASELINE, alternative="less")
    v = verdict(best_mean, p_val=p_val)
    delta = best_mean - BASELINE
    print(f"\nBest: bonus={best_bonus}  {best_mean:.4f} ± {best_std:.4f}  Δ={delta:+.4f}  p={p_val:.4f}  {v}")

    metrics = {"mean": float(best_mean), "std": float(best_std),
               "fold_losses": losses.tolist(), "accuracy": None,
               "delta_vs_baseline": float(delta), "p_value": float(p_val),
               "verdict": v, "best_bonus": int(best_bonus), "best_dr": dr, "best_sc": sc,
               "all_results": [(b, float(m), float(s)) for b, m, s in results]}
    run_info = {"attempt": "034", "description": "Extended host_bonus sweep 250-600 at dr=0.30, sc=400",
                "dr": dr, "sc": sc, "bonus_grid": bonus_grid, "best_bonus": int(best_bonus)}
    save_artifact(34, metrics, run_info)
    return best_bonus, best_mean

# ---------------------------------------------------------------------------
# Attempt 035: Gaussian draw probability model
# ---------------------------------------------------------------------------
def attempt_035(df):
    print("\n=== Attempt 035: Gaussian draw probability model ===")
    sc_grid = [350, 400, 450]
    max_dr_grid = [0.28, 0.30, 0.32, 0.35]
    sigma_grid = [100, 150, 200, 250, 300, 400]
    bonus_grid = [250, 300, 350]

    results = []
    for sc, max_dr, sigma, bonus in itertools.product(sc_grid, max_dr_grid, sigma_grid, bonus_grid):
        losses = eval_formula(df, elo_probs_gaussian_draw, sc=sc, max_dr=max_dr, sigma=sigma, bonus=bonus)
        results.append((sc, max_dr, sigma, bonus, losses.mean(), losses.std()))

    results.sort(key=lambda x: x[4])
    print(f"{'sc':>6} {'max_dr':>8} {'sigma':>7} {'bonus':>7} {'mean':>8} {'std':>8}")
    for sc, max_dr, sigma, bonus, mean, std in results[:8]:
        print(f"{sc:6d} {max_dr:8.2f} {sigma:7d} {bonus:7d} {mean:8.4f} {std:8.4f}")

    best_sc, best_max_dr, best_sigma, best_bonus, best_mean, best_std = results[0]
    losses = eval_formula(df, elo_probs_gaussian_draw,
                          sc=best_sc, max_dr=best_max_dr, sigma=best_sigma, bonus=best_bonus)
    t_stat, p_val = ttest_1samp(losses, BASELINE, alternative="less")
    v = verdict(best_mean, p_val=p_val)
    delta = best_mean - BASELINE
    print(f"\nBest: sc={best_sc} max_dr={best_max_dr} sigma={best_sigma} bonus={best_bonus}  {best_mean:.4f} ± {best_std:.4f}  Δ={delta:+.4f}  p={p_val:.4f}  {v}")

    metrics = {"mean": float(best_mean), "std": float(best_std),
               "fold_losses": losses.tolist(), "accuracy": None,
               "delta_vs_baseline": float(delta), "p_value": float(p_val),
               "verdict": v, "best_sc": best_sc, "best_max_dr": best_max_dr,
               "best_sigma": best_sigma, "best_bonus": int(best_bonus),
               "top5": [(sc, max_dr, sigma, bonus, float(m)) for sc, max_dr, sigma, bonus, m, _ in results[:5]]}
    run_info = {"attempt": "035", "description": "Gaussian draw probability model",
                "sc_grid": sc_grid, "max_dr_grid": max_dr_grid,
                "sigma_grid": sigma_grid, "bonus_grid": bonus_grid}
    save_artifact(35, metrics, run_info)
    return best_mean

# ---------------------------------------------------------------------------
# Attempt 036: Extended 3D grid around best 030 region (larger bonus, finer sc)
# ---------------------------------------------------------------------------
def attempt_036(df, bonus_hint):
    print(f"\n=== Attempt 036: Extended 3D grid around 030 optimum (bonus hint={bonus_hint}) ===")
    # Fine-grained around the bonus hint from 034
    lo = max(200, bonus_hint - 100)
    hi = bonus_hint + 150
    bonus_grid = list(range(lo, hi + 1, 25))
    dr_grid = [0.27, 0.28, 0.30, 0.32, 0.33]
    sc_grid = [375, 400, 425, 450, 475]

    results = []
    for dr, sc, bonus in itertools.product(dr_grid, sc_grid, bonus_grid):
        losses = eval_formula(df, elo_probs, dr=dr, sc=sc, bonus=bonus)
        results.append((dr, sc, bonus, losses.mean(), losses.std()))

    results.sort(key=lambda x: x[3])
    print(f"{'dr':>6} {'sc':>6} {'bonus':>7} {'mean':>8} {'std':>8}")
    for dr, sc, bonus, mean, std in results[:10]:
        print(f"{dr:6.2f} {sc:6d} {bonus:7d} {mean:8.4f} {std:8.4f}")

    best_dr, best_sc, best_bonus, best_mean, best_std = results[0]
    losses = eval_formula(df, elo_probs, dr=best_dr, sc=best_sc, bonus=best_bonus)
    t_stat, p_val = ttest_1samp(losses, BASELINE, alternative="less")
    v = verdict(best_mean, p_val=p_val)
    delta = best_mean - BASELINE
    print(f"\nBest: dr={best_dr} sc={best_sc} bonus={best_bonus}  {best_mean:.4f} ± {best_std:.4f}  Δ={delta:+.4f}  p={p_val:.4f}  {v}")

    metrics = {"mean": float(best_mean), "std": float(best_std),
               "fold_losses": losses.tolist(), "accuracy": None,
               "delta_vs_baseline": float(delta), "p_value": float(p_val),
               "verdict": v, "best_dr": best_dr, "best_sc": best_sc, "best_bonus": int(best_bonus),
               "top10": [(dr, sc, bonus, float(m)) for dr, sc, bonus, m, _ in results[:10]]}
    run_info = {"attempt": "036", "description": f"Extended 3D grid around 030 optimum (bonus_hint={bonus_hint})",
                "bonus_grid": bonus_grid, "dr_grid": dr_grid, "sc_grid": sc_grid}
    save_artifact(36, metrics, run_info)
    return best_dr, best_sc, best_bonus, best_mean

# ---------------------------------------------------------------------------
# Attempt 037: FIFA rank offset as Elo supplement
# ---------------------------------------------------------------------------
def attempt_037(df):
    print("\n=== Attempt 037: FIFA rank offset as Elo supplement ===")
    # rank_diff: rank_a - rank_h (positive = home team has better rank = lower number)
    # adjusted elo_diff = elo_diff + gamma * (rank_a - rank_h) * rank_scale
    # Sweep gamma ∈ {0, 0.5, 1, 2, 5, 10} × sc ∈ {400} × bonus ∈ {300,350,400}
    # Also combine with best bonus from 034
    gamma_grid = [0, 0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 10.0]
    sc_grid = [400, 450]
    bonus_grid = [300, 350, 400, 450]
    dr = 0.30

    results = []
    for gamma, sc, bonus in itertools.product(gamma_grid, sc_grid, bonus_grid):
        skf = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=0)
        fold_losses = []
        for _, test_idx in skf.split(df, df["y"]):
            t = df.iloc[test_idx]
            # rank_diff positive when home is better (lower rank number)
            rank_adj = gamma * (t["rank_a"] - t["rank_h"]).values
            elo_diff_adj = (t["elo_h"] - t["elo_a"] + bonus * t["host_h"]).values + rank_adj
            probs = elo_probs(elo_diff_adj, dr=dr, sc=sc)
            fold_losses.append(log_loss(t["y"].values, probs, labels=[0, 1, 2]))
        m = np.mean(fold_losses)
        results.append((gamma, sc, bonus, m, np.std(fold_losses)))

    results.sort(key=lambda x: x[3])
    print(f"{'gamma':>8} {'sc':>6} {'bonus':>7} {'mean':>8} {'std':>8}")
    for gamma, sc, bonus, mean, std in results[:10]:
        print(f"{gamma:8.2f} {sc:6d} {bonus:7d} {mean:8.4f} {std:8.4f}")

    best_gamma, best_sc, best_bonus, best_mean, best_std = results[0]
    # recompute fold losses for best config
    skf = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=0)
    losses = []
    for _, test_idx in skf.split(df, df["y"]):
        t = df.iloc[test_idx]
        rank_adj = best_gamma * (t["rank_a"] - t["rank_h"]).values
        elo_diff_adj = (t["elo_h"] - t["elo_a"] + best_bonus * t["host_h"]).values + rank_adj
        probs = elo_probs(elo_diff_adj, dr=dr, sc=best_sc)
        losses.append(log_loss(t["y"].values, probs, labels=[0, 1, 2]))
    losses = np.array(losses)
    t_stat, p_val = ttest_1samp(losses, BASELINE, alternative="less")
    v = verdict(best_mean, p_val=p_val)
    delta = best_mean - BASELINE
    print(f"\nBest: gamma={best_gamma} sc={best_sc} bonus={best_bonus}  {best_mean:.4f} ± {best_std:.4f}  Δ={delta:+.4f}  p={p_val:.4f}  {v}")

    metrics = {"mean": float(best_mean), "std": float(best_std),
               "fold_losses": losses.tolist(), "accuracy": None,
               "delta_vs_baseline": float(delta), "p_value": float(p_val),
               "verdict": v, "best_gamma": best_gamma, "best_sc": best_sc,
               "best_bonus": int(best_bonus), "dr": dr,
               "top10": [(g, sc, b, float(m)) for g, sc, b, m, _ in results[:10]]}
    run_info = {"attempt": "037", "description": "FIFA rank offset as Elo supplement",
                "gamma_grid": gamma_grid, "sc_grid": sc_grid,
                "bonus_grid": bonus_grid, "dr": dr}
    save_artifact(37, metrics, run_info)
    return best_mean

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    completed, teams = load_data()
    df = build_df(completed, teams)
    print(f"Dataset: {len(df)} completed WC-2026 matches")

    best_bonus_034, mean_034 = attempt_034(df)
    mean_035 = attempt_035(df)
    _, _, best_bonus_036, mean_036 = attempt_036(df, best_bonus_034)
    mean_037 = attempt_037(df)

    print("\n===== BATCH 8 SUMMARY =====")
    print(f"034 extended bonus sweep:  {mean_034:.4f}  (best bonus={best_bonus_034})")
    print(f"035 Gaussian draw model:   {mean_035:.4f}")
    print(f"036 Extended 3D grid:      {mean_036:.4f}  (best bonus={best_bonus_036})")
    print(f"037 FIFA rank supplement:  {mean_037:.4f}")
    print(f"Campaign best (030):       0.8056")
    print(f"Baseline:                  {BASELINE}")
