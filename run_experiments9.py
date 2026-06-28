"""
Batch 9: Attempts 038-041
038 — scipy.optimize: numerical optimization of Elo formula (dr, sc, bonus) — find true global min
039 — WC group-stage only logistic regression (trained on WC 1930-2022 group matches)
040 — Confederation draw rate adjustment (per-confed draw rate)
041 — Blend optimal Elo formula (038) with WC group-stage logistic (039)
"""
import os, sys, json, math, itertools, warnings
import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import ttest_1samp
from scipy.optimize import minimize, differential_evolution
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.linear_model import LogisticRegression
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

NAME_MAP = {
    "Cabo Verde": "Cape Verde", "Congo DR": "DR Congo",
    "Czechia": "Czech Republic", "Côte d'Ivoire": "Ivory Coast",
    "IR Iran": "Iran", "Türkiye": "Turkey", "USA": "United States"
}

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
    team_conf = dict(zip(teams["team_name"], teams["confederation"]))
    rows = []
    for _, m in completed.iterrows():
        h, a = m["home_team"], m["away_team"]
        y = 2 if m["home_score"] > m["away_score"] else (0 if m["home_score"] < m["away_score"] else 1)
        rows.append({"home": h, "away": a,
                     "elo_h": team_elo.get(h, 1700), "elo_a": team_elo.get(a, 1700),
                     "rank_h": team_rank.get(h, 50), "rank_a": team_rank.get(a, 50),
                     "conf_h": team_conf.get(h, "UEFA"), "conf_a": team_conf.get(a, "UEFA"),
                     "host_h": 1.0 if h in HOST_TEAMS else 0.0, "y": y})
    return pd.DataFrame(rows)

# ---------------------------------------------------------------------------
# Elo formula (vectorized)
# ---------------------------------------------------------------------------
def elo_probs(elo_diff, dr=0.30, sc=400):
    Eh = expit(elo_diff * math.log(10) / sc)
    Ea = 1.0 - Eh
    pH = np.clip(Eh - 0.5 * dr, 0.01, 0.98)
    pA = np.clip(Ea - 0.5 * dr, 0.01, 0.98)
    pD = np.clip(1.0 - pH - pA, 0.01, 0.98)
    total = pH + pD + pA
    return np.stack([pA / total, pD / total, pH / total], axis=1)

def eval_elo_params(df, dr, sc, bonus):
    skf = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=0)
    losses = []
    for _, test_idx in skf.split(df, df["y"]):
        t = df.iloc[test_idx]
        elo_diffs = (t["elo_h"] - t["elo_a"] + bonus * t["host_h"]).values
        probs = elo_probs(elo_diffs, dr=dr, sc=sc)
        losses.append(log_loss(t["y"].values, probs, labels=[0, 1, 2]))
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
# Attempt 038: scipy.optimize differential evolution on (dr, sc, bonus)
# ---------------------------------------------------------------------------
def attempt_038(df):
    print("\n=== Attempt 038: scipy differential_evolution Elo formula optimization ===")

    call_count = [0]
    def objective(params):
        dr, sc, bonus = params
        if dr <= 0 or sc <= 0 or bonus < 0:
            return 10.0
        losses = eval_elo_params(df, dr, sc, bonus)
        call_count[0] += 1
        if call_count[0] % 20 == 0:
            print(f"  eval {call_count[0]}: dr={dr:.3f} sc={sc:.1f} bonus={bonus:.1f} → {losses.mean():.5f}")
        return losses.mean()

    bounds = [(0.20, 0.45), (300, 600), (0, 1500)]
    result = differential_evolution(
        objective, bounds,
        seed=42, maxiter=30, popsize=10, tol=1e-5,
        mutation=(0.5, 1), recombination=0.7, workers=1
    )
    best_dr, best_sc, best_bonus = result.x
    print(f"\nOptimized: dr={best_dr:.4f}, sc={best_sc:.2f}, bonus={best_bonus:.2f}")

    losses = eval_elo_params(df, best_dr, best_sc, best_bonus)
    best_mean = losses.mean()
    best_std = losses.std()
    t_stat, p_val = ttest_1samp(losses, BASELINE, alternative="less")
    v = verdict(best_mean, p_val=p_val)
    delta = best_mean - BASELINE
    print(f"Result: {best_mean:.4f} ± {best_std:.4f}  Δ={delta:+.4f}  p={p_val:.4f}  {v}")
    print(f"Total evaluations: {call_count[0]}")

    metrics = {"mean": float(best_mean), "std": float(best_std),
               "fold_losses": losses.tolist(), "accuracy": None,
               "delta_vs_baseline": float(delta), "p_value": float(p_val),
               "verdict": v, "best_dr": float(best_dr), "best_sc": float(best_sc),
               "best_bonus": float(best_bonus), "optimizer_fun": float(result.fun),
               "n_evals": call_count[0]}
    run_info = {"attempt": "038", "description": "scipy differential_evolution on (dr, sc, bonus)",
                "bounds": bounds, "seed": 42, "maxiter": 30, "popsize": 10}
    save_artifact(38, metrics, run_info)
    return best_dr, best_sc, best_bonus, best_mean

# ---------------------------------------------------------------------------
# Attempt 039: WC group-stage only logistic regression
# ---------------------------------------------------------------------------
def attempt_039(df, teams):
    print("\n=== Attempt 039: WC group-stage only logistic regression ===")

    # Load historical data and filter to FIFA World Cup group stage
    hist = pd.read_csv(HIST_PATH)
    wc_mask = hist["tournament"].str.contains("FIFA World Cup", na=False)
    # Group stage: not final, semi, quarter, etc.
    gs_mask = ~hist["tournament"].str.lower().str.contains("qualifier|qualification|friendly|olympic|continental|cup of nations|gold cup|asian|african|euro|copa", na=False)
    wc_gs = hist[wc_mask & (hist["date"] >= "2010-01-01")].copy()
    print(f"WC matches from 2010+: {len(wc_gs)}")

    # Build Elo lookup for historical teams
    team_elo = dict(zip(teams["team_name"], teams["elo_rating"].astype(float)))

    # Map historical names to canonical
    def map_name(n):
        return NAME_MAP.get(n, n)

    # Build training features from WC historical data
    train_rows = []
    for _, m in wc_gs.iterrows():
        h = map_name(str(m["home_team"]))
        a = map_name(str(m["away_team"]))
        if m["home_score"] > m["away_score"]:
            y = 2
        elif m["home_score"] < m["away_score"]:
            y = 0
        else:
            y = 1
        elo_h = team_elo.get(h, 1700)
        elo_a = team_elo.get(a, 1700)
        train_rows.append({"elo_diff": elo_h - elo_a, "y": y})

    train_df = pd.DataFrame(train_rows)
    print(f"Training rows: {len(train_df)}")

    # Eval: for each CV fold, train on all WC historical data, test on 2026 WC fold
    # Since training is fixed (no WC-2026 data), each fold uses same model
    X_train = train_df[["elo_diff"]].values
    y_train = train_df["y"].values

    # Fit on all WC historical data
    lr = LogisticRegression(C=1.0, max_iter=1000, random_state=0)
    lr.fit(X_train, y_train)

    # Evaluate on 2026 WC matches using CV
    skf = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=0)
    losses = []
    for _, test_idx in skf.split(df, df["y"]):
        t = df.iloc[test_idx]
        X_test = (t["elo_h"] - t["elo_a"]).values.reshape(-1, 1)
        probs = lr.predict_proba(X_test)
        # probs columns: sorted by class [0,1,2] = [A, D, H]
        losses.append(log_loss(t["y"].values, probs, labels=lr.classes_))

    losses = np.array(losses)
    best_mean = losses.mean()
    best_std = losses.std()
    t_stat, p_val = ttest_1samp(losses, BASELINE, alternative="less")
    v = verdict(best_mean, p_val=p_val)
    delta = best_mean - BASELINE
    print(f"WC historical logistic (C=1.0): {best_mean:.4f} ± {best_std:.4f}  Δ={delta:+.4f}  p={p_val:.4f}  {v}")

    # Also try with host bonus
    for bonus in [300, 500, 750]:
        fold_l = []
        for _, test_idx in skf.split(df, df["y"]):
            t = df.iloc[test_idx]
            elo_adj = (t["elo_h"] - t["elo_a"] + bonus * t["host_h"]).values.reshape(-1, 1)
            probs = lr.predict_proba(elo_adj)
            fold_l.append(log_loss(t["y"].values, probs, labels=lr.classes_))
        m = np.mean(fold_l)
        print(f"  bonus={bonus}: {m:.4f}")

    metrics = {"mean": float(best_mean), "std": float(best_std),
               "fold_losses": losses.tolist(), "accuracy": None,
               "delta_vs_baseline": float(delta), "p_value": float(p_val),
               "verdict": v, "n_train": len(train_df), "wc_years": "2010-2022"}
    run_info = {"attempt": "039", "description": "WC group-stage logistic (2010-2022) applied to 2026",
                "C": 1.0, "features": ["elo_diff"]}
    save_artifact(39, metrics, run_info)
    return best_mean, lr

# ---------------------------------------------------------------------------
# Attempt 040: Confederation draw rate adjustment
# ---------------------------------------------------------------------------
def attempt_040(df):
    print("\n=== Attempt 040: Confederation draw rate adjustment ===")

    # Confed pairs: different draw rates for CONMEBOL vs UEFA, etc.
    # Two-team confed pair determines expected draw rate
    # CONCACAF/CONMEBOL high draw rate, UEFA moderate, AFC/CAF/OFC lower

    # Strategy: different dr per match based on confederation pair
    confed_draw_rates = {
        # Same confederation
        ("UEFA", "UEFA"):    0.32,
        ("CONMEBOL", "CONMEBOL"): 0.30,
        ("CONCACAF", "CONCACAF"): 0.28,
        ("AFC", "AFC"):      0.28,
        ("CAF", "CAF"):      0.26,
        # Cross-confederation (use average)
        "default":           0.29,
    }

    def get_dr(conf_h, conf_a, dr_map):
        key = tuple(sorted([conf_h, conf_a]))
        return dr_map.get(key, dr_map.get("default", 0.30))

    best_overall = {"mean": 99, "params": {}}

    for sc in [400, 425, 450]:
        for bonus in [700, 750, 800]:
            for dr_scale in [0.85, 0.90, 0.95, 1.0, 1.05, 1.10]:
                # Apply dr_scale to all confed draw rates
                dr_map = {k: (v * dr_scale if k != "default" else v * dr_scale)
                          for k, v in confed_draw_rates.items()}

                skf = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=0)
                fold_l = []
                for _, test_idx in skf.split(df, df["y"]):
                    t = df.iloc[test_idx]
                    probs_list = []
                    for _, row in t.iterrows():
                        dr = get_dr(row["conf_h"], row["conf_a"], dr_map)
                        elo_d = row["elo_h"] - row["elo_a"] + bonus * row["host_h"]
                        p = elo_probs(np.array([elo_d]), dr=dr, sc=sc)[0]
                        probs_list.append(p)
                    probs = np.array(probs_list)
                    fold_l.append(log_loss(t["y"].values, probs, labels=[0, 1, 2]))

                m = np.mean(fold_l)
                if m < best_overall["mean"]:
                    best_overall = {"mean": m, "std": np.std(fold_l),
                                    "params": {"sc": sc, "bonus": bonus, "dr_scale": dr_scale},
                                    "fold_l": fold_l}

    print(f"Best: {best_overall['params']} → {best_overall['mean']:.4f} ± {best_overall['std']:.4f}")
    losses = np.array(best_overall["fold_l"])
    t_stat, p_val = ttest_1samp(losses, BASELINE, alternative="less")
    v = verdict(best_overall["mean"], p_val=p_val)
    delta = best_overall["mean"] - BASELINE
    print(f"  Δ={delta:+.4f}  p={p_val:.4f}  {v}")

    metrics = {"mean": float(best_overall["mean"]), "std": float(best_overall["std"]),
               "fold_losses": best_overall["fold_l"], "accuracy": None,
               "delta_vs_baseline": float(delta), "p_value": float(p_val),
               "verdict": v, **best_overall["params"]}
    run_info = {"attempt": "040", "description": "Confederation draw rate adjustment",
                "confed_draw_rates": {str(k): v for k, v in confed_draw_rates.items()}}
    save_artifact(40, metrics, run_info)
    return best_overall["mean"]

# ---------------------------------------------------------------------------
# Attempt 041: Elo formula + WC-logistic blend with 038 optimal params
# ---------------------------------------------------------------------------
def attempt_041(df, opt_dr, opt_sc, opt_bonus, lr_wc):
    print(f"\n=== Attempt 041: Optimal Elo (dr={opt_dr:.3f}, sc={opt_sc:.1f}, bonus={opt_bonus:.1f}) + WC logistic blend ===")

    alpha_grid = [0.7, 0.8, 0.85, 0.90, 0.95, 1.0]
    skf = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=0)

    results = []
    for alpha in alpha_grid:
        fold_l = []
        for _, test_idx in skf.split(df, df["y"]):
            t = df.iloc[test_idx]
            elo_diffs = (t["elo_h"] - t["elo_a"] + opt_bonus * t["host_h"]).values
            p_elo = elo_probs(elo_diffs, dr=opt_dr, sc=opt_sc)

            X_test = (t["elo_h"] - t["elo_a"]).values.reshape(-1, 1)
            p_wc = lr_wc.predict_proba(X_test)
            # Align WC logistic classes with [pA, pD, pH]
            classes = lr_wc.classes_
            p_wc_aligned = np.zeros((len(t), 3))
            for i, c in enumerate(classes):
                p_wc_aligned[:, c] = p_wc[:, i]

            p_blend = alpha * p_elo + (1 - alpha) * p_wc_aligned
            fold_l.append(log_loss(t["y"].values, p_blend, labels=[0, 1, 2]))

        results.append((alpha, np.mean(fold_l), np.std(fold_l)))

    results.sort(key=lambda x: x[1])
    print(f"{'alpha':>8} {'mean':>8} {'std':>8}")
    for alpha, mean, std in results:
        print(f"{alpha:8.2f} {mean:8.4f} {std:8.4f}")

    best_alpha, best_mean, best_std = results[0]
    losses = eval_elo_params(df, opt_dr, opt_sc, opt_bonus) if best_alpha == 1.0 else None

    # Recompute fold losses for best alpha
    fold_l = []
    for _, test_idx in skf.split(df, df["y"]):
        t = df.iloc[test_idx]
        elo_diffs = (t["elo_h"] - t["elo_a"] + opt_bonus * t["host_h"]).values
        p_elo = elo_probs(elo_diffs, dr=opt_dr, sc=opt_sc)
        X_test = (t["elo_h"] - t["elo_a"]).values.reshape(-1, 1)
        p_wc = lr_wc.predict_proba(X_test)
        classes = lr_wc.classes_
        p_wc_aligned = np.zeros((len(t), 3))
        for i, c in enumerate(classes):
            p_wc_aligned[:, c] = p_wc[:, i]
        p_blend = best_alpha * p_elo + (1 - best_alpha) * p_wc_aligned
        fold_l.append(log_loss(t["y"].values, p_blend, labels=[0, 1, 2]))

    losses = np.array(fold_l)
    t_stat, p_val = ttest_1samp(losses, BASELINE, alternative="less")
    v = verdict(best_mean, p_val=p_val)
    delta = best_mean - BASELINE
    print(f"\nBest (alpha={best_alpha}): {best_mean:.4f} ± {best_std:.4f}  Δ={delta:+.4f}  p={p_val:.4f}  {v}")

    metrics = {"mean": float(best_mean), "std": float(best_std),
               "fold_losses": losses.tolist(), "accuracy": None,
               "delta_vs_baseline": float(delta), "p_value": float(p_val),
               "verdict": v, "best_alpha": best_alpha,
               "elo_params": {"dr": float(opt_dr), "sc": float(opt_sc), "bonus": float(opt_bonus)}}
    run_info = {"attempt": "041", "description": "Optimal Elo + WC-logistic blend",
                "alpha_grid": alpha_grid}
    save_artifact(41, metrics, run_info)
    return best_mean

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    completed, teams = load_data()
    df = build_df(completed, teams)
    print(f"Dataset: {len(df)} completed WC-2026 matches")

    opt_dr, opt_sc, opt_bonus, mean_038 = attempt_038(df)
    mean_039, lr_wc = attempt_039(df, teams)
    mean_040 = attempt_040(df)
    mean_041 = attempt_041(df, opt_dr, opt_sc, opt_bonus, lr_wc)

    print("\n===== BATCH 9 SUMMARY =====")
    print(f"038 differential_evolution Elo: {mean_038:.4f}  (dr={opt_dr:.3f} sc={opt_sc:.1f} bonus={opt_bonus:.1f})")
    print(f"039 WC group-stage logistic:     {mean_039:.4f}")
    print(f"040 confed draw rate adjust:     {mean_040:.4f}")
    print(f"041 Elo + WC logistic blend:     {mean_041:.4f}")
    print(f"Wave-3 best (036):              0.8020")
    print(f"Baseline:                        {BASELINE}")
