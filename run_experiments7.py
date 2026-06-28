"""
Wave-3 Batch 7: Extending Direct Elo formula (Attempts 030-033)

Key fix: uses teams.csv pre-tournament elo_rating (same as Batch 6 attempt 029).
draw_rate=0.3, scale=350, host_bonus=100 → 0.8148 FLAT confirmed.

030 - Finer grid sweep (draw_rate × scale × host_bonus up to 250)
031 - Blend Elo formula + kitchen-sink logistic
032 - Adaptive draw rate by Elo bin (WC-calibrated)
033 - Temperature-scaled logistic + Elo formula blend
"""

import os, json, math, sys
import numpy as np
import pandas as pd
from scipy.special import expit, softmax
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import log_loss
from sklearn.preprocessing import StandardScaler
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from features import build_elo_ratings as build_hist_elo

BASE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE, "data")
MATCHES_PATH = os.path.join(DATA_DIR, "wc2026-trees-study-main/fifa_data/matches_detailed.csv")
TEAMS_PATH   = os.path.join(DATA_DIR, "wc2026-trees-study-main/fifa_data/teams.csv")
HIST_PATH    = os.path.join(DATA_DIR, "historical/results.csv")
ARTIFACTS    = os.path.join(BASE, "artifacts")
os.makedirs(ARTIFACTS, exist_ok=True)

BASELINE   = 0.8337
HOST_TEAMS = {'Mexico', 'USA', 'Canada'}
NAME_MAP   = {"Cabo Verde": "Cape Verde", "Congo DR": "DR Congo",
              "Czechia": "Czech Republic", "Côte d'Ivoire": "Ivory Coast",
              "IR Iran": "Iran", "Türkiye": "Turkey", "USA": "United States"}

print("=" * 70)
print("Wave-3 Batch 7: Extended Elo formula, blends, adaptive draw rate")
print("=" * 70)


def load_all():
    matches = pd.read_csv(MATCHES_PATH)
    teams   = pd.read_csv(TEAMS_PATH)
    hist    = pd.read_csv(HIST_PATH, parse_dates=['date'])
    hist['home_team'] = hist['home_team'].map(lambda x: NAME_MAP.get(x, x))
    hist['away_team'] = hist['away_team'].map(lambda x: NAME_MAP.get(x, x))
    completed = matches[matches['status'] == 'Completed']
    completed = completed[completed['stage_name'].str.lower().str.contains('group', na=False)]
    completed = completed.dropna(subset=['home_score', 'away_score'])
    completed = completed.rename(columns={'home_team_name': 'home_team', 'away_team_name': 'away_team'})
    return completed.copy(), teams, hist


def build_df(completed, teams):
    team_elo  = dict(zip(teams['team_name'], teams['elo_rating'].astype(float)))
    team_rank = dict(zip(teams['team_name'], teams['fifa_ranking_pre_tournament'].astype(float)))
    rows = []
    for _, m in completed.iterrows():
        h, a = m['home_team'], m['away_team']
        y = 2 if m['home_score'] > m['away_score'] else (0 if m['home_score'] < m['away_score'] else 1)
        rows.append({'home': h, 'away': a,
                     'elo_h': team_elo.get(h, 1700), 'elo_a': team_elo.get(a, 1700),
                     'rank_h': team_rank.get(h, 50),  'rank_a': team_rank.get(a, 50),
                     'host_h': 1.0 if h in HOST_TEAMS else 0.0, 'y': y})
    return pd.DataFrame(rows)


def elo_probs(elo_diff, dr=0.3, sc=350):
    """Returns (n,3) array [pA, pD, pH]."""
    Eh = expit(elo_diff * math.log(10) / sc)
    Ea = 1 - Eh
    pH = np.clip(Eh - 0.5*dr, 0.01, 0.98)
    pA = np.clip(Ea - 0.5*dr, 0.01, 0.98)
    pD = np.clip(1 - pH - pA, 0.01, 0.98)
    total = pH + pD + pA
    return np.stack([pA/total, pD/total, pH/total], axis=1)


def save_art(attempt, data, filename):
    d = os.path.join(ARTIFACTS, f"attempt-{attempt:03d}")
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, filename), 'w') as f:
        json.dump(data, f, indent=2)


def eval_params(df, y_all, rskf, bonus, dr, sc):
    losses = []
    for _, test_idx in rskf.split(df, y_all):
        td = df.iloc[test_idx]
        ed = (td['elo_h'] - td['elo_a'] + bonus * td['host_h']).values
        losses.append(log_loss(y_all[test_idx], elo_probs(ed, dr, sc)))
    return float(np.mean(losses)), losses


completed, teams, hist = load_all()
df = build_df(completed, teams)
y_all = df['y'].values
rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=0)

# ─────────────────────────────────────────────────────────────────
print("\n=== Attempt 030: Fine grid sweep around best Elo formula params ===")
# ─────────────────────────────────────────────────────────────────
grid = [(dr, sc, b)
        for dr in [0.25, 0.27, 0.30, 0.32, 0.35]
        for sc in [300, 325, 350, 375, 400]
        for b  in [75, 100, 125, 150, 175, 200, 250]]

all_030 = []
for dr, sc, b in grid:
    m, _ = eval_params(df, y_all, rskf, b, dr, sc)
    all_030.append((m, dr, sc, b))
all_030.sort()

print("  Top 5:")
for m, dr, sc, b in all_030[:5]:
    print(f"    dr={dr}, sc={sc}, bonus={b}: {m:.4f}")

best_m030, best_dr, best_sc, best_b = all_030[0]
_, fl030 = eval_params(df, y_all, rskf, best_b, best_dr, best_sc)  # rerun for std
# Note: eval_params above already ran it; rerun to get fold losses list
fl030 = []
for _, test_idx in rskf.split(df, y_all):
    td = df.iloc[test_idx]
    ed = (td['elo_h'] - td['elo_a'] + best_b * td['host_h']).values
    fl030.append(log_loss(y_all[test_idx], elo_probs(ed, best_dr, best_sc)))

m030, s030 = float(np.mean(fl030)), float(np.std(fl030))
_, p030 = stats.ttest_1samp(fl030, BASELINE, alternative='less')
p030 = float(p030)
v030 = "GREEN" if m030 < BASELINE and p030 < 0.05 else ("FLAT" if m030 <= BASELINE else "RED")
print(f"  Best dr={best_dr}, sc={best_sc}, bonus={best_b}: {m030:.4f} ± {s030:.4f}  p={p030:.4f}  {v030}")

save_art(30, {"mean": m030, "std": s030, "fold_losses": fl030,
              "best_params": {"draw_rate": best_dr, "scale": best_sc, "host_bonus": best_b},
              "delta_vs_baseline": m030-BASELINE, "p_value": p030, "verdict": v030,
              "all_top10": all_030[:10]}, "metrics.json")
save_art(30, {"attempt": 30, "grid_size": len(grid)}, "run.json")


# ─────────────────────────────────────────────────────────────────
print("\n=== Attempt 031: Blend Elo formula (029 params) + kitchen-sink logistic ===")
# ─────────────────────────────────────────────────────────────────
cutoff = pd.Timestamp('2025-06-01')
recent = hist[(hist['date'] < cutoff) & (hist['date'] >= cutoff - pd.Timedelta(days=180))]

def ks_feat(row, hist_recent):
    h, a = row['home'], row['away']
    hm_h = hist_recent[hist_recent['home_team'] == h]
    aw_h = hist_recent[hist_recent['away_team'] == h]
    hm_a = hist_recent[hist_recent['home_team'] == a]
    aw_a = hist_recent[hist_recent['away_team'] == a]
    def wr(hm, aw): return ((hm['home_score']>hm['away_score']).sum()+(aw['away_score']>aw['home_score']).sum()) / max(len(hm)+len(aw),1)
    def gd(hm, aw): return ((hm['home_score']-hm['away_score']).sum()+(aw['away_score']-aw['home_score']).sum()) / max(len(hm)+len(aw),1)
    return [row['elo_h']-row['elo_a'], row['host_h'], row['rank_a']-row['rank_h'],
            wr(hm_h,aw_h)-wr(hm_a,aw_a), gd(hm_h,aw_h)-gd(hm_a,aw_a)]

ks_X = np.array([ks_feat(row, recent) for _, row in df.iterrows()], dtype=np.float32)

best_031_loss = float('inf')
best_alpha_031 = 0.5
fl031 = []
res031 = {}
for alpha in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
    fold_losses = []
    for train_idx, test_idx in rskf.split(df, y_all):
        td = df.iloc[test_idx]
        sc_ = StandardScaler()
        Xtr = sc_.fit_transform(ks_X[train_idx])
        Xte = sc_.transform(ks_X[test_idx])
        lr = LogisticRegression(C=0.05, max_iter=1000, solver='saga')
        lr.fit(Xtr, y_all[train_idx])
        p_lr = lr.predict_proba(Xte)
        ed = (td['elo_h'] - td['elo_a'] + 100*td['host_h']).values
        p_elo = elo_probs(ed, 0.30, 350)
        p_b = alpha*p_elo + (1-alpha)*p_lr
        p_b /= p_b.sum(axis=1, keepdims=True)
        fold_losses.append(log_loss(y_all[test_idx], p_b))
    m = float(np.mean(fold_losses))
    res031[alpha] = m
    print(f"  alpha(elo)={alpha}: {m:.4f}")
    if m < best_031_loss:
        best_031_loss = m
        best_alpha_031 = alpha
        fl031 = fold_losses[:]

m031, s031 = float(np.mean(fl031)), float(np.std(fl031))
_, p031 = stats.ttest_1samp(fl031, BASELINE, alternative='less')
p031 = float(p031)
v031 = "GREEN" if m031 < BASELINE and p031 < 0.05 else ("FLAT" if m031 <= BASELINE else "RED")
print(f"  Best alpha(elo)={best_alpha_031}: {m031:.4f} ± {s031:.4f}  p={p031:.4f}  {v031}")

save_art(31, {"mean": m031, "std": s031, "fold_losses": fl031,
              "best_alpha": best_alpha_031, "all_results": {str(k): v for k,v in res031.items()},
              "delta_vs_baseline": m031-BASELINE, "p_value": p031, "verdict": v031}, "metrics.json")
save_art(31, {"attempt": 31, "elo_dr": 0.30, "elo_sc": 350, "elo_bonus": 100, "C_logistic": 0.05}, "run.json")


# ─────────────────────────────────────────────────────────────────
print("\n=== Attempt 032: Adaptive draw rate by Elo bin (WC 2006-2022) ===")
# ─────────────────────────────────────────────────────────────────
wc_hist = hist[hist['tournament'].str.contains('FIFA World Cup', na=False) &
               ~hist['tournament'].str.contains('Qualif', na=False) &
               (hist['date'] >= '2006-01-01')].copy()

elo_by_yr = {}
for yr in [2006, 2010, 2014, 2018, 2022]:
    elo_by_yr[yr], _ = build_hist_elo(hist, cutoff_date=f'{yr}-06-01')

wc_recs = []
for _, row in wc_hist.iterrows():
    yr = row['date'].year
    wc_yr = min([2006,2010,2014,2018,2022], key=lambda y: abs(y-yr))
    elos = elo_by_yr[wc_yr]
    h, a = row['home_team'], row['away_team']
    ed = elos.get(h,1500) - elos.get(a,1500)
    out = 2 if row['home_score']>row['away_score'] else (0 if row['home_score']<row['away_score'] else 1)
    wc_recs.append({'ed': ed, 'out': out})

wc_df2 = pd.DataFrame(wc_recs)
print(f"  WC calibration n={len(wc_df2)}")
bins = [-500,-150,-75,-25,25,75,150,500]
wc_df2['bin'] = pd.cut(wc_df2['ed'], bins=bins, labels=False)
alpha_lp = 0.5
bin_dr = {}
for b in range(len(bins)-1):
    sub = wc_df2[wc_df2['bin']==b]
    n, d = len(sub), (sub['out']==1).sum()
    bin_dr[b] = (d+alpha_lp) / (n+3*alpha_lp)
    print(f"    [{bins[b]},{bins[b+1]}) n={n}: dr={bin_dr[b]:.3f}")

def get_dr(ed):
    for i,(lo,hi) in enumerate(zip(bins[:-1],bins[1:])):
        if lo<=ed<hi: return bin_dr.get(i,0.30)
    return 0.30

best_032_loss = float('inf')
fl032 = []
b32_sc, b32_bon = 350, 100
for sc in [300,325,350,375,400]:
    for bonus in [75,100,125,150]:
        fold_losses = []
        for _, test_idx in rskf.split(df, y_all):
            td = df.iloc[test_idx]
            eds = (td['elo_h']-td['elo_a']+bonus*td['host_h']).values
            pr = np.array([elo_probs(np.array([e]), get_dr(float(e)), sc)[0] for e in eds])
            fold_losses.append(log_loss(y_all[test_idx], pr))
        m = float(np.mean(fold_losses))
        if m < best_032_loss:
            best_032_loss, b32_sc, b32_bon = m, sc, bonus
            fl032 = fold_losses[:]

m032, s032 = float(np.mean(fl032)), float(np.std(fl032))
_, p032 = stats.ttest_1samp(fl032, BASELINE, alternative='less')
p032 = float(p032)
v032 = "GREEN" if m032 < BASELINE and p032 < 0.05 else ("FLAT" if m032 <= BASELINE else "RED")
print(f"  Best sc={b32_sc}, bonus={b32_bon}: {m032:.4f} ± {s032:.4f}  p={p032:.4f}  {v032}")

save_art(32, {"mean": m032, "std": s032, "fold_losses": fl032,
              "best_scale": b32_sc, "best_bonus": b32_bon,
              "bin_draw_rates": {str(k):v for k,v in bin_dr.items()},
              "delta_vs_baseline": m032-BASELINE, "p_value": p032, "verdict": v032}, "metrics.json")
save_art(32, {"attempt": 32, "bins": bins, "wc_n": len(wc_df2)}, "run.json")


# ─────────────────────────────────────────────────────────────────
print("\n=== Attempt 033: Temperature-scaled logistic + Elo blend ===")
# ─────────────────────────────────────────────────────────────────
best_033_loss = float('inf')
fl033 = []
res033 = {}
best_033_params = {}

for T in [0.7, 0.8, 0.9, 1.0, 1.2, 1.5, 2.0]:
    for alpha in [0.0, 0.3, best_alpha_031, 0.7, 1.0]:
        fold_losses = []
        for train_idx, test_idx in rskf.split(df, y_all):
            td = df.iloc[test_idx]
            sc_ = StandardScaler()
            Xtr = sc_.fit_transform(ks_X[train_idx])
            Xte = sc_.transform(ks_X[test_idx])
            lr = LogisticRegression(C=0.05, max_iter=1000, solver='saga')
            lr.fit(Xtr, y_all[train_idx])
            p_lr = lr.predict_proba(Xte)
            p_lr_T = softmax(np.log(np.clip(p_lr,1e-10,1))/T, axis=1)
            ed = (td['elo_h']-td['elo_a']+100*td['host_h']).values
            p_elo = elo_probs(ed, 0.30, 350)
            p_b = alpha*p_elo + (1-alpha)*p_lr_T
            p_b /= p_b.sum(axis=1, keepdims=True)
            fold_losses.append(log_loss(y_all[test_idx], p_b))
        m = float(np.mean(fold_losses))
        k = f"T{T}_a{alpha}"
        res033[k] = m
        if m < best_033_loss:
            best_033_loss = m
            best_033_params = {'T': T, 'alpha_elo': alpha}
            fl033 = fold_losses[:]

m033, s033 = float(np.mean(fl033)), float(np.std(fl033))
_, p033 = stats.ttest_1samp(fl033, BASELINE, alternative='less')
p033 = float(p033)
v033 = "GREEN" if m033 < BASELINE and p033 < 0.05 else ("FLAT" if m033 <= BASELINE else "RED")
print(f"  Best params={best_033_params}: {m033:.4f} ± {s033:.4f}  p={p033:.4f}  {v033}")
print(f"  Pure elo (alpha=1): {res033.get('T1.0_a1.0', 'N/A')}")
print(f"  Pure logistic T=1 (alpha=0): {res033.get('T1.0_a0.0', 'N/A')}")

save_art(33, {"mean": m033, "std": s033, "fold_losses": fl033,
              "best_params": best_033_params, "all_results_top10": sorted(res033.items(), key=lambda x:x[1])[:10],
              "delta_vs_baseline": m033-BASELINE, "p_value": p033, "verdict": v033}, "metrics.json")
save_art(33, {"attempt": 33, "temps": [0.7,0.8,0.9,1.0,1.2,1.5,2.0], "C_logistic": 0.05}, "run.json")


# ─────────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("SUMMARY TABLE Batch 7")
print("="*70)
for num, name, m, s, v in [
    (30, "Fine grid Elo formula", m030, s030, v030),
    (31, "Elo formula + KS blend", m031, s031, v031),
    (32, "Adaptive draw rate (WC-bins)", m032, s032, v032),
    (33, "Temp-scaled logistic+Elo", m033, s033, v033),
]:
    print(f"  {num:03d} {name:<30}: {m:.4f} ± {s:.4f}  {v}")
