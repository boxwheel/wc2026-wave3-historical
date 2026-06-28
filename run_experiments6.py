"""
Wave-3 Batch 6: Model family diversity + empirical calibration.

Approaches:
  026 — Proportional Odds Model (ordinal logistic, manual scipy optimization)
  027 — Empirical WC bin calibration from historical WC 2014-2022
  028 — Diverse 5-model ensemble (different feature sets)
  029 — Direct Elo formula + fixed draw-rate sweep (no ML)
"""
import sys, os, json, time
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import log_loss, accuracy_score
from scipy.stats import ttest_1samp
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(__file__))
from features import NAME_MAP, build_elo_ratings

DATA_DIR = '/home/user/research/wave3-historical/data'
MATCHES_PATH = f'{DATA_DIR}/wc2026-trees-study-main/fifa_data/matches_detailed.csv'
TEAMS_PATH   = f'{DATA_DIR}/wc2026-trees-study-main/fifa_data/teams.csv'
SQUADS_PATH  = f'{DATA_DIR}/wc2026-trees-study-main/fifa_data/squads_and_players.csv'
HIST_PATH    = f'{DATA_DIR}/historical/results.csv'
ARTIFACTS_DIR = '/home/user/research/wave3-historical/artifacts'
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

BASELINE_LOGLOSS = 0.8337
FRONTIER_LOGLOSS = 0.7608
SEED = 0
HOST_TEAMS = {'Mexico', 'USA', 'Canada'}


def sig_test(fold_losses):
    arr = np.array(fold_losses)
    t, p = ttest_1samp(arr, BASELINE_LOGLOSS)
    mean = arr.mean()
    verdict = ('GREEN' if (mean < BASELINE_LOGLOSS and p < 0.05)
               else ('RED' if mean > BASELINE_LOGLOSS else 'FLAT'))
    return {'delta_vs_baseline': float(mean - BASELINE_LOGLOSS),
            'p_value': float(p), 'verdict': verdict}


def save_artifact(name, metrics, run_info):
    path = os.path.join(ARTIFACTS_DIR, name)
    os.makedirs(path, exist_ok=True)
    with open(f'{path}/metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    with open(f'{path}/run.json', 'w') as f:
        json.dump(run_info, f, indent=2)


def load_base_data():
    matches = pd.read_csv(MATCHES_PATH)
    teams   = pd.read_csv(TEAMS_PATH)
    hist    = pd.read_csv(HIST_PATH, parse_dates=['date'])
    hist['home_team'] = hist['home_team'].map(lambda x: NAME_MAP.get(x, x))
    hist['away_team'] = hist['away_team'].map(lambda x: NAME_MAP.get(x, x))
    completed = matches[matches['status'] == 'Completed'].copy()
    completed = completed[completed['stage_name'].str.lower().str.contains('group', na=False)].copy()
    completed = completed.dropna(subset=['home_score', 'away_score']).copy()
    completed = completed.rename(columns={'home_team_name': 'home_team', 'away_team_name': 'away_team'})
    return completed, teams, hist


def cv_eval_probs(probs, y_str, le=None):
    """Evaluate pre-computed probs (n_samples × 3) on canonical folds."""
    if le is None:
        le = LabelEncoder()
        le.fit(y_str)
    y = le.transform(y_str)
    rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=SEED)
    fold_losses = []
    for _, test_idx in rskf.split(np.zeros(len(y)), y):
        fold_losses.append(log_loss(y[test_idx], probs[test_idx],
                                    labels=list(range(3))))
    return {
        'mean': float(np.mean(fold_losses)),
        'std': float(np.std(fold_losses)),
        'fold_losses': [float(x) for x in fold_losses],
        'accuracy': float(accuracy_score(y, np.argmax(probs, axis=1))),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Attempt 026: Proportional Odds Model (manual L-BFGS-B)
# ──────────────────────────────────────────────────────────────────────────────
def attempt_026_proportional_odds():
    print("\n=== Attempt 026: Proportional Odds Model (ordinal logistic) ===")
    t0 = time.time()
    completed, teams, hist = load_base_data()
    team_map = {r['team_name']: r for _, r in teams.iterrows()}

    rows, y_str = [], []
    for _, m in completed.iterrows():
        h, a = m['home_team'], m['away_team']
        if h not in team_map or a not in team_map:
            continue
        ht, at = team_map[h], team_map[a]
        elo_diff   = float(ht['elo_rating']) - float(at['elo_rating'])
        host_adv   = 1.0 if h in HOST_TEAMS else 0.0
        rank_diff  = float(at['fifa_ranking_pre_tournament']) - float(ht['fifa_ranking_pre_tournament'])
        rows.append([elo_diff, host_adv, rank_diff])
        hs, as_ = int(m['home_score']), int(m['away_score'])
        y_str.append('H' if hs > as_ else 'D' if hs == as_ else 'A')

    le = LabelEncoder()
    le.fit(['A', 'D', 'H'])  # 0=A, 1=D, 2=H (ordered: A < D < H)
    y = le.transform(y_str)
    X_raw = np.array(rows, dtype=float)

    def pom_probs(params, X):
        """Return (n,3) matrix [P(A), P(D), P(H)] for proportional odds model."""
        a0, a1 = params[0], params[1]
        beta = params[2:]
        xb = X @ beta
        p_le0 = expit(a0 - xb)  # P(Y <= 0 = A)
        p_le1 = expit(a1 - xb)  # P(Y <= 1 = D)
        pA = np.clip(p_le0, 1e-9, 1-1e-9)
        pD = np.clip(p_le1 - p_le0, 1e-9, 1-1e-9)
        pH = np.clip(1 - p_le1, 1e-9, 1-1e-9)
        # Re-normalize (numerical safety)
        total = pA + pD + pH
        return np.stack([pA/total, pD/total, pH/total], axis=1)

    def nll(params, X, y_enc, reg=0.01):
        a0, a1 = params[0], params[1]
        if a1 <= a0:  # enforce ordering
            return 1e10
        p = pom_probs(params, X)
        ll = np.mean(np.log(p[np.arange(len(y_enc)), y_enc]))
        penalty = reg * np.sum(params[2:]**2)
        return -ll + penalty

    rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=SEED)
    fold_losses = []
    oof_probs = np.zeros((len(y), 3))

    best_reg, best_loss = 0.01, np.inf
    for reg_sweep in [True, False]:
        if reg_sweep:
            regs = [0.005, 0.01, 0.05, 0.1]
        else:
            break

        for reg in regs:
            fl_tmp, oof_tmp = [], np.zeros((len(y), 3))
            for train_idx, test_idx in rskf.split(X_raw, y):
                sc = StandardScaler()
                X_tr = sc.fit_transform(X_raw[train_idx])
                X_te = sc.transform(X_raw[test_idx])
                y_tr = y[train_idx]
                # Initial params: [a0, a1, beta_1, beta_2, beta_3]
                p0 = np.array([-0.3, 0.3, 0.5, 0.5, 0.0])
                res = minimize(nll, p0, args=(X_tr, y_tr, reg),
                               method='L-BFGS-B',
                               options={'maxiter': 500, 'ftol': 1e-9})
                probs_te = pom_probs(res.x, X_te)
                fl_tmp.append(log_loss(y[test_idx], probs_te, labels=[0,1,2]))
                oof_tmp[test_idx] = probs_te

            mean_loss = np.mean(fl_tmp)
            print(f"  reg={reg}: {mean_loss:.4f}")
            if mean_loss < best_loss:
                best_loss, best_reg = mean_loss, reg
                fold_losses = fl_tmp
                oof_probs = oof_tmp

    sig = sig_test(fold_losses)
    acc = float(accuracy_score(y, np.argmax(oof_probs, axis=1)))
    print(f"  Best reg={best_reg}: {best_loss:.4f} ± {np.std(fold_losses):.4f}  acc={acc:.3f}  {sig['verdict']}")

    metrics = {'mean': best_loss, 'std': float(np.std(fold_losses)),
               'fold_losses': [float(x) for x in fold_losses],
               'accuracy': acc, 'best_reg': best_reg, **sig}
    run_info = {'attempt': 26, 'description': 'Proportional Odds Model (ordinal logistic)',
                'features': ['elo_diff', 'host_advantage', 'rank_diff'],
                'best_reg': best_reg, 'elapsed_s': time.time() - t0}
    save_artifact('attempt-026', metrics, run_info)
    return metrics


# ──────────────────────────────────────────────────────────────────────────────
# Attempt 027: Empirical WC bin calibration (historical WC 2014-2022)
# ──────────────────────────────────────────────────────────────────────────────
def attempt_027_wc_bin_calibration():
    print("\n=== Attempt 027: Empirical WC bin calibration (WC 2014-2022) ===")
    t0 = time.time()
    completed, teams, hist = load_base_data()
    team_map = {r['team_name']: r for _, r in teams.iterrows()}

    # Build Elo for each historical WC separately
    wc_windows = [
        ('2014', '2014-06-12', '2014-07-14'),
        ('2018', '2018-06-14', '2018-07-15'),
        ('2022', '2022-11-20', '2022-12-18'),
    ]

    cal_rows = []  # (elo_diff, outcome_int)
    for wc_year, start_str, end_str in wc_windows:
        elo_at_wc, _ = build_elo_ratings(hist, cutoff_date=start_str)
        wc_matches = hist[
            (hist['tournament'] == 'FIFA World Cup') &
            (hist['date'] >= start_str) &
            (hist['date'] <= end_str)
        ].dropna(subset=['home_score', 'away_score'])
        for _, m in wc_matches.iterrows():
            h, a = m['home_team'], m['away_team']
            eh = elo_at_wc.get(h, 1500)
            ea = elo_at_wc.get(a, 1500)
            diff = eh - ea
            hs, as_ = int(m['home_score']), int(m['away_score'])
            outcome = 2 if hs > as_ else 1 if hs == as_ else 0
            cal_rows.append((diff, outcome))

    cal_df = pd.DataFrame(cal_rows, columns=['elo_diff', 'outcome'])
    print(f"  Historical WC calibration matches: {len(cal_df)}")
    print(f"  Elo diff range: {cal_df['elo_diff'].min():.0f} – {cal_df['elo_diff'].max():.0f}")

    # Build empirical calibration: bin by Elo diff, compute H/D/A rates
    # Use quantile-based bins (7 bins) to ensure roughly equal counts per bin
    n_bins = 7
    cal_df['bin'] = pd.qcut(cal_df['elo_diff'], q=n_bins, duplicates='drop')
    bin_stats = {}
    for name, group in cal_df.groupby('bin', observed=True):
        total = len(group)
        pA = (group['outcome'] == 0).sum() / total
        pD = (group['outcome'] == 1).sum() / total
        pH = (group['outcome'] == 2).sum() / total
        # Laplace smoothing (alpha=0.5)
        pA = (pA * total + 0.5) / (total + 1.5)
        pD = (pD * total + 0.5) / (total + 1.5)
        pH = (pH * total + 0.5) / (total + 1.5)
        mid = name.mid
        bin_stats[mid] = np.array([pA, pD, pH])
        print(f"    bin mid={mid:.0f} (n={total}): H={pH:.3f} D={pD:.3f} A={pA:.3f}")

    bin_mids = sorted(bin_stats.keys())

    def lookup_bin(elo_diff):
        """Find nearest bin midpoint."""
        closest = min(bin_mids, key=lambda m: abs(m - elo_diff))
        return bin_stats[closest]

    # Apply to WC-2026 matches
    le = LabelEncoder()
    le.fit(['A', 'D', 'H'])
    probs = np.zeros((64, 3))
    y_str = []
    idx = 0
    for _, m in completed.iterrows():
        h, a = m['home_team'], m['away_team']
        if h not in team_map or a not in team_map:
            continue
        ht, at = team_map[h], team_map[a]
        elo_diff = float(ht['elo_rating']) - float(at['elo_rating'])
        probs[idx] = lookup_bin(elo_diff)
        hs, as_ = int(m['home_score']), int(m['away_score'])
        y_str.append('H' if hs > as_ else 'D' if hs == as_ else 'A')
        idx += 1
    probs = probs[:idx]

    res = cv_eval_probs(probs, y_str, le)
    sig = sig_test(res['fold_losses'])
    print(f"  Result: {res['mean']:.4f} ± {res['std']:.4f}  acc={res['accuracy']:.3f}  {sig['verdict']}")

    metrics = {**res, **sig, 'n_cal_matches': len(cal_df), 'n_bins': n_bins}
    run_info = {'attempt': 27, 'description': 'Empirical WC bin calibration (WC 2014-2022)',
                'wc_years': ['2014', '2018', '2022'], 'n_bins': n_bins,
                'n_cal_matches': len(cal_df), 'elapsed_s': time.time() - t0}
    save_artifact('attempt-027', metrics, run_info)
    return metrics


# ──────────────────────────────────────────────────────────────────────────────
# Attempt 028: Diverse 5-model ensemble
# ──────────────────────────────────────────────────────────────────────────────
def attempt_028_diverse_ensemble():
    print("\n=== Attempt 028: Diverse 5-model ensemble ===")
    t0 = time.time()
    completed, teams, hist = load_base_data()
    squads = pd.read_csv(SQUADS_PATH)
    team_map = {r['team_name']: r for _, r in teams.iterrows()}
    WC_DATE = pd.Timestamp('2026-06-11')

    def squad_stats(team_id):
        tq = squads[squads['team_id'] == team_id]
        total_mv = tq['market_value_eur'].sum() if 'market_value_eur' in tq.columns else 1.0
        top11_mv = tq.nlargest(11, 'market_value_eur')['market_value_eur'].sum() if 'market_value_eur' in tq.columns else 1.0
        caps = tq['caps'].mean() if 'caps' in tq.columns else 50.0
        goals = tq['goals'].sum() if 'goals' in tq.columns else 0.0
        dob = pd.to_datetime(tq['date_of_birth'], errors='coerce').dropna()
        ages = [(WC_DATE - d).days / 365.25 for d in dob]
        avg_age = float(np.mean(ages)) if ages else 27.0
        return total_mv, top11_mv, caps, goals, avg_age

    squad_cache = {r['team_name']: squad_stats(r['team_id']) for _, r in teams.iterrows()}

    def count_recent_form(team):
        recent = hist[(hist['date'] >= '2024-11-01') & (hist['date'] <= '2026-05-31')]
        hm = recent[recent['home_team'] == team]
        am = recent[recent['away_team'] == team]
        scores, weights = [], []
        max_d = recent['date'].max()
        for _, r in hm.iterrows():
            d = (max_d - r['date']).days
            scores.append(1.0 if r['home_score'] > r['away_score'] else 0.5 if r['home_score'] == r['away_score'] else 0.0)
            weights.append(np.exp(-d / 180))
        for _, r in am.iterrows():
            d = (max_d - r['date']).days
            scores.append(1.0 if r['away_score'] > r['home_score'] else 0.5 if r['away_score'] == r['home_score'] else 0.0)
            weights.append(np.exp(-d / 180))
        return float(np.average(scores, weights=weights)) if weights else 0.5

    form_cache = {r['team_name']: count_recent_form(r['team_name']) for _, r in teams.iterrows()}

    wc_hist = hist[hist['tournament'] == 'FIFA World Cup']
    def wc_apps(team):
        years = pd.to_datetime(wc_hist[(wc_hist['home_team'] == team) | (wc_hist['away_team'] == team)]['date']).dt.year.unique()
        return len([y for y in years if y in {1930,1934,1938,1950,1954,1958,1962,1966,1970,1974,1978,1982,1986,1990,1994,1998,2002,2006,2010,2014,2018,2022}])

    apps_cache = {r['team_name']: wc_apps(r['team_name']) for _, r in teams.iterrows()}

    rows_A, rows_B, rows_C, rows_D, rows_E = [], [], [], [], []
    y_str = []
    for _, m in completed.iterrows():
        h, a = m['home_team'], m['away_team']
        if h not in team_map or a not in team_map:
            continue
        ht, at = team_map[h], team_map[a]
        elo_diff = float(ht['elo_rating']) - float(at['elo_rating'])
        host_adv = 1.0 if h in HOST_TEAMS else 0.0
        rank_diff = float(at['fifa_ranking_pre_tournament']) - float(ht['fifa_ranking_pre_tournament'])

        # Model A: Elo-only
        rows_A.append([elo_diff, host_adv])

        # Model B: Kitchen-sink (Elo + rank + squad values + caps + goals + age)
        hm_v, ht11, hcaps, hgoals, hage = squad_cache.get(h, (1e6, 1e6, 50, 0, 27))
        am_v, at11, acaps, agoals, aage = squad_cache.get(a, (1e6, 1e6, 50, 0, 27))
        rows_B.append([elo_diff, host_adv, rank_diff,
                       np.log1p(hm_v) - np.log1p(am_v),
                       np.log1p(ht11) - np.log1p(at11),
                       hcaps - acaps, hgoals - agoals, hage - aage])

        # Model C: Recent form
        rows_C.append([elo_diff, host_adv, form_cache.get(h, 0.5) - form_cache.get(a, 0.5)])

        # Model D: WC history
        rows_D.append([elo_diff, host_adv, apps_cache.get(h, 0) - apps_cache.get(a, 0)])

        # Model E: Elo + rank (2-feature)
        rows_E.append([elo_diff, host_adv, rank_diff])

        hs, as_ = int(m['home_score']), int(m['away_score'])
        y_str.append('H' if hs > as_ else 'D' if hs == as_ else 'A')

    le = LabelEncoder(); le.fit(y_str)
    y_enc = le.transform(y_str)
    Xs = [pd.DataFrame(r) for r in [rows_A, rows_B, rows_C, rows_D, rows_E]]
    configs = [(1.0, 'A-elo'), (0.05, 'B-ks'), (0.5, 'C-form'), (0.3, 'D-wch'), (0.5, 'E-rank')]
    rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=SEED)

    oofs = [np.zeros((len(y_enc), 3)) for _ in range(5)]
    for train_idx, test_idx in rskf.split(Xs[0], y_enc):
        for k, (X, (C, _)) in enumerate(zip(Xs, configs)):
            sc = StandardScaler()
            X_tr = sc.fit_transform(X.iloc[train_idx])
            X_te = sc.transform(X.iloc[test_idx])
            clf = LogisticRegression(C=C, max_iter=500, solver='saga')
            clf.fit(X_tr, y_enc[train_idx])
            oofs[k][test_idx] = clf.predict_proba(X_te)

    best_loss, best_combo, best_res = np.inf, None, None
    for blend_name, w_vec in [
        ('equal-5', [0.2]*5),
        ('elo-heavy', [0.5, 0.2, 0.1, 0.1, 0.1]),
        ('elo-ks', [0.5, 0.3, 0.1, 0.05, 0.05]),
        ('elo-ks-form', [0.4, 0.3, 0.2, 0.05, 0.05]),
        ('no-C', [0.35, 0.3, 0.0, 0.1, 0.25]),
        ('no-D', [0.35, 0.3, 0.2, 0.0, 0.15]),
    ]:
        w_vec = np.array(w_vec)
        blended = sum(w * oof for w, oof in zip(w_vec, oofs))
        blended /= blended.sum(axis=1, keepdims=True)
        res = cv_eval_probs(blended, y_str, le)
        sig = sig_test(res['fold_losses'])
        print(f"  {blend_name}: {res['mean']:.4f} ± {res['std']:.4f}  {sig['verdict']}")
        if res['mean'] < best_loss:
            best_loss, best_combo, best_res = res['mean'], (blend_name, w_vec.tolist()), (res, sig)

    res, sig = best_res
    print(f"  Best '{best_combo[0]}': {res['mean']:.4f} ± {res['std']:.4f}  acc={res['accuracy']:.3f}  {sig['verdict']}")
    metrics = {**res, **sig, 'best_combo': best_combo[0]}
    run_info = {'attempt': 28, 'description': 'Diverse 5-model ensemble',
                'models': ['A:elo-only', 'B:kitchen-sink', 'C:recent-form', 'D:wc-history', 'E:elo+rank'],
                'best_blend': best_combo, 'elapsed_s': time.time() - t0}
    save_artifact('attempt-028', metrics, run_info)
    return metrics


# ──────────────────────────────────────────────────────────────────────────────
# Attempt 029: Direct Elo formula with fixed draw-rate sweep (no ML)
# ──────────────────────────────────────────────────────────────────────────────
def attempt_029_direct_elo_formula():
    """
    Use the pure Elo formula to compute P(H win) and P(A win),
    then allocate P(D) as a fixed fraction.
    No training — pure model-based probabilities.
    """
    print("\n=== Attempt 029: Direct Elo formula + draw-rate sweep ===")
    t0 = time.time()
    completed, teams, hist = load_base_data()
    team_map = {r['team_name']: r for _, r in teams.iterrows()}

    le = LabelEncoder(); le.fit(['A', 'D', 'H'])
    y_str = []
    elo_diffs = []
    host_advs = []
    for _, m in completed.iterrows():
        h, a = m['home_team'], m['away_team']
        if h not in team_map or a not in team_map:
            continue
        ht, at = team_map[h], team_map[a]
        elo_diff = float(ht['elo_rating']) - float(at['elo_rating'])
        host_bonus = 50.0 if h in HOST_TEAMS else 0.0  # Elo home bonus
        elo_diffs.append(elo_diff + host_bonus)
        host_advs.append(1.0 if h in HOST_TEAMS else 0.0)
        hs, as_ = int(m['home_score']), int(m['away_score'])
        y_str.append('H' if hs > as_ else 'D' if hs == as_ else 'A')

    elo_diffs = np.array(elo_diffs)

    # Elo expected score for home team: P(H_win) + 0.5*P(draw)
    def elo_to_probs(elo_diff, draw_rate=0.25, scale=400):
        E_h = expit(elo_diff * np.log(10) / scale)
        E_a = 1 - E_h
        # Allocate: P(D) = draw_rate, proportionally reduce P(H) and P(A)
        # E_h = P(H) + 0.5*draw_rate → P(H) = E_h - 0.5*draw_rate
        pH = np.clip(E_h - 0.5 * draw_rate, 0.01, 0.98)
        pA = np.clip(E_a - 0.5 * draw_rate, 0.01, 0.98)
        pD = np.clip(1 - pH - pA, 0.01, 0.98)
        # Re-normalize
        total = pH + pD + pA
        return np.stack([pA/total, pD/total, pH/total], axis=1)

    best_loss, best_d, best_scale, best_res = np.inf, None, None, None
    for draw_rate in [0.15, 0.20, 0.22, 0.25, 0.28, 0.30, 0.35]:
        for scale in [350, 400, 450, 500]:
            probs = elo_to_probs(elo_diffs, draw_rate=draw_rate, scale=scale)
            res = cv_eval_probs(probs, y_str, le)
            sig = sig_test(res['fold_losses'])
            if res['mean'] < best_loss:
                best_loss, best_d, best_scale, best_res = res['mean'], draw_rate, scale, (res, sig)

    res, sig = best_res
    print(f"  Best draw_rate={best_d}, scale={best_scale}: {res['mean']:.4f} ± {res['std']:.4f}  acc={res['accuracy']:.3f}  {sig['verdict']}")

    # Also try with host bonus sweep
    print("  Host bonus sweep:")
    for bonus in [0, 25, 50, 75, 100]:
        bonused = elo_diffs.copy()
        for i, h_adv in enumerate(host_advs):
            if h_adv:
                bonused[i] += (bonus - 50)  # adjust from base +50
        probs = elo_to_probs(bonused, draw_rate=best_d, scale=best_scale)
        res2 = cv_eval_probs(probs, y_str, le)
        sig2 = sig_test(res2['fold_losses'])
        print(f"    bonus={bonus}: {res2['mean']:.4f} ± {res2['std']:.4f}  {sig2['verdict']}")
        if res2['mean'] < best_loss:
            best_loss, best_res = res2['mean'], (res2, sig2)

    res, sig = best_res
    print(f"  Final best: {res['mean']:.4f} ± {res['std']:.4f}  acc={res['accuracy']:.3f}  {sig['verdict']}")
    metrics = {**res, **sig, 'best_draw_rate': best_d, 'best_scale': best_scale}
    run_info = {'attempt': 29, 'description': 'Direct Elo formula + draw-rate sweep',
                'approach': 'no ML, pure Elo formula', 'best_draw_rate': best_d,
                'best_scale': best_scale, 'elapsed_s': time.time() - t0}
    save_artifact('attempt-029', metrics, run_info)
    return metrics


if __name__ == '__main__':
    print("=" * 70)
    print("Wave-3 Batch 6: Proportional Odds, WC Calibration, Diverse Ensemble, Elo Formula")
    print("=" * 70)

    results = {}
    results[26] = attempt_026_proportional_odds()
    results[27] = attempt_027_wc_bin_calibration()
    results[28] = attempt_028_diverse_ensemble()
    results[29] = attempt_029_direct_elo_formula()

    print("\n" + "=" * 70)
    print("SUMMARY TABLE Batch 6")
    print("=" * 70)
    labels = {
        26: "Proportional Odds Model    ",
        27: "WC bin calibration (hist)  ",
        28: "Diverse 5-model ensemble   ",
        29: "Direct Elo formula         ",
    }
    for k, r in results.items():
        v = r.get('verdict', '?')
        print(f"  {k:03d} {labels[k]}: {r['mean']:.4f} ± {r['std']:.4f}  acc={r['accuracy']:.3f}  {v}")
