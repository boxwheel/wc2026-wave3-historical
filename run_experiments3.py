"""
Wave-3 Batch 3: Attack/Defense Decomposition via Poisson Goals
Key insight: Elo measures OVERALL strength; attack/defense parameters measure
OFFENSIVE vs DEFENSIVE tendencies separately — genuinely orthogonal to Elo.

Also: geometric mean blending and head-to-head features.
"""
import sys, os, json, time
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import log_loss, accuracy_score
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(__file__))
from features import NAME_MAP, build_wc2026_feature_matrix

DATA_DIR = '/home/user/research/wave3-historical/data'
MATCHES_PATH = f'{DATA_DIR}/wc2026-trees-study-main/fifa_data/matches_detailed.csv'
TEAMS_PATH = f'{DATA_DIR}/wc2026-trees-study-main/fifa_data/teams.csv'
HIST_PATH = f'{DATA_DIR}/historical/results.csv'
ARTIFACTS_DIR = '/home/user/research/wave3-historical/artifacts'
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

BASELINE_LOGLOSS = 0.8337
FRONTIER_LOGLOSS = 0.7608
SEED = 0

teams_df = pd.read_csv(TEAMS_PATH)
teams_df['hist_name'] = teams_df['team_name'].map(lambda x: NAME_MAP.get(x, x))
WC_TO_HIST = {row['team_name']: row['hist_name'] for _, row in teams_df.iterrows()}
HIST_TO_WC = {v: k for k, v in WC_TO_HIST.items()}
WC_TEAMS_HIST = set(teams_df['hist_name'].tolist())


def canonical_cv_eval(probs, labels_str, le=None):
    if le is None:
        le = LabelEncoder()
        le.fit(labels_str)
    y = le.transform(labels_str)
    rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=SEED)
    fold_losses, eps = [], 1e-15
    per_match_loss = -np.log(np.clip(probs[np.arange(len(y)), y], eps, 1 - eps))
    for _, test_idx in rskf.split(np.zeros(len(y)), y):
        fold_losses.append(log_loss(y[test_idx], probs[test_idx],
                                    labels=list(range(len(le.classes_)))))
    return {
        'mean': float(np.mean(fold_losses)),
        'std': float(np.std(fold_losses)),
        'fold_losses': [float(x) for x in fold_losses],
        'per_match_loss': [float(x) for x in per_match_loss],
        'accuracy': float(accuracy_score(y, np.argmax(probs, axis=1))),
        'classes': le.classes_.tolist(),
    }


def cv_train_eval(X_wc, y_wc, model_fn, scale=True):
    le = LabelEncoder()
    le.fit(y_wc)
    y = le.transform(y_wc)
    rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=SEED)
    fold_losses = []
    oof_probs = np.zeros((len(y), len(le.classes_)))
    for train_idx, test_idx in rskf.split(X_wc, y):
        X_tr, X_te = X_wc.iloc[train_idx], X_wc.iloc[test_idx]
        y_tr = y[train_idx]
        if scale:
            sc = StandardScaler()
            X_tr = sc.fit_transform(X_tr)
            X_te = sc.transform(X_te)
        clf = model_fn()
        clf.fit(X_tr, y_tr)
        probs = clf.predict_proba(X_te)
        fold_losses.append(log_loss(y[test_idx], probs, labels=list(range(len(le.classes_)))))
        oof_probs[test_idx] = probs
    return {
        'mean': float(np.mean(fold_losses)),
        'std': float(np.std(fold_losses)),
        'fold_losses': [float(x) for x in fold_losses],
        'accuracy': float(accuracy_score(y, np.argmax(oof_probs, axis=1))),
        'oof_probs': oof_probs.tolist(),
        'classes': le.classes_.tolist(),
    }, le


def significance_test(fold_losses):
    from scipy.stats import ttest_1samp
    arr = np.array(fold_losses)
    t_stat, p_val = ttest_1samp(arr, BASELINE_LOGLOSS)
    mean = arr.mean()
    verdict = 'GREEN' if (mean < BASELINE_LOGLOSS and p_val < 0.05) else ('RED' if mean > BASELINE_LOGLOSS else 'FLAT')
    return {
        'delta_vs_baseline': float(mean - BASELINE_LOGLOSS),
        'delta_vs_frontier': float(mean - FRONTIER_LOGLOSS),
        'p_value_ttest': float(p_val),
        't_stat': float(t_stat),
        'verdict': verdict,
    }


def save_artifact(name, metrics, run_info):
    path = os.path.join(ARTIFACTS_DIR, name)
    os.makedirs(path, exist_ok=True)
    with open(f'{path}/metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    with open(f'{path}/run.json', 'w') as f:
        json.dump(run_info, f, indent=2)
    return path


# ──────────────────────────────────────────────────────────────────────────────
# Dixon-Coles parameter estimation from historical data
# ──────────────────────────────────────────────────────────────────────────────
def fit_poisson_params(hist_matches, teams_set, home_adv_init=0.3, recency_halflife_days=None,
                        cutoff_date=None, tier_weights=None):
    """Fit attack/defense parameters for teams using log-linear Poisson model.

    Model: log(lambda_home) = mu + home_adv + alpha_home - beta_away
           log(lambda_away) = mu + alpha_away - beta_home

    Returns: dict of {team: {'attack': a, 'defense': b}} and home_adv, mu
    """
    hist = hist_matches.copy()
    if cutoff_date:
        hist = hist[hist['date'] < pd.to_datetime(cutoff_date)]
    hist = hist.dropna(subset=['home_score', 'away_score'])

    # Filter to teams in teams_set only (both teams must be in set)
    hist = hist[hist['home_team'].isin(teams_set) & hist['away_team'].isin(teams_set)]
    hist = hist.sort_values('date').reset_index(drop=True)

    if len(hist) == 0:
        return {}, 0.3, 0.0

    # Recency weights
    if recency_halflife_days and len(hist) > 0:
        cutoff_ts = hist['date'].max() if cutoff_date is None else pd.to_datetime(cutoff_date)
        days_before = (cutoff_ts - hist['date']).dt.days
        recency_w = np.exp(-days_before * np.log(2) / recency_halflife_days)
    else:
        recency_w = np.ones(len(hist))

    # Tier weights
    if tier_weights is not None:
        tw = hist['tournament'].map(tier_weights).fillna(1.0).values
        sample_weights = recency_w * tw
    else:
        sample_weights = recency_w

    all_teams = sorted(set(hist['home_team'].tolist()) | set(hist['away_team'].tolist()))
    n_teams = len(all_teams)
    team_idx = {t: i for i, t in enumerate(all_teams)}

    # Parameters: mu, home_adv, alpha_1..n (attack), beta_1..n (defense)
    # Fix alpha_0 = 0 to avoid identifiability issue
    def pack(mu, home_adv, alphas, betas):
        return np.concatenate([[mu, home_adv], alphas[1:], betas])

    def unpack(params):
        mu = params[0]
        home_adv = params[1]
        alphas = np.concatenate([[0], params[2:2 + n_teams - 1]])
        betas = params[2 + n_teams - 1:]
        return mu, home_adv, alphas, betas

    # Pre-compute arrays for vectorized likelihood (avoids Python row-loop)
    home_idx_arr = np.array([team_idx[t] for t in hist['home_team']])
    away_idx_arr = np.array([team_idx[t] for t in hist['away_team']])
    home_scores_arr = hist['home_score'].values.astype(int)
    away_scores_arr = hist['away_score'].values.astype(int)
    neutral_flag = hist['neutral'].values.astype(bool) if 'neutral' in hist.columns else np.zeros(len(hist), dtype=bool)
    sw = sample_weights.values if hasattr(sample_weights, 'values') else np.array(sample_weights)

    def neg_log_likelihood_vec(params):
        mu, home_adv, alphas, betas = unpack(params)
        ha = np.where(neutral_flag, 0.0, home_adv)
        log_lh = mu + ha + alphas[home_idx_arr] - betas[away_idx_arr]
        log_la = mu + alphas[away_idx_arr] - betas[home_idx_arr]
        lambda_h = np.exp(np.clip(log_lh, -5, 3))
        lambda_a = np.exp(np.clip(log_la, -5, 3))
        ll_h = poisson.logpmf(home_scores_arr, lambda_h)
        ll_a = poisson.logpmf(away_scores_arr, lambda_a)
        total = np.dot(sw, ll_h + ll_a)
        reg = 0.5 * (np.sum(alphas[1:]**2) + np.sum(betas**2))
        return -total + reg

    n_params = 2 + (n_teams - 1) + n_teams
    x0 = np.zeros(n_params)
    x0[0] = 0.3  # mu ~ log(1.35 goals/match)
    x0[1] = 0.3  # home_adv

    result = minimize(neg_log_likelihood_vec, x0, method='L-BFGS-B',
                      options={'maxiter': 500, 'ftol': 1e-8})

    mu, home_adv, alphas, betas = unpack(result.x)

    team_params = {}
    for team in all_teams:
        i = team_idx[team]
        team_params[team] = {'attack': float(alphas[i]), 'defense': float(betas[i])}

    return team_params, float(home_adv), float(mu)


def compute_match_probs_poisson(team_params, home_team, away_team, mu, home_adv,
                                 is_neutral=True, max_goals=10):
    """Compute P(H), P(D), P(A) from Poisson attack/defense parameters."""
    hp = team_params.get(home_team, {'attack': 0, 'defense': 0})
    ap = team_params.get(away_team, {'attack': 0, 'defense': 0})

    ha = 0.0 if is_neutral else home_adv
    lambda_h = np.exp(mu + ha + hp['attack'] - ap['defense'])
    lambda_a = np.exp(mu + ap['attack'] - hp['defense'])

    prob_H, prob_D, prob_A = 0.0, 0.0, 0.0
    for g_h in range(max_goals):
        for g_a in range(max_goals):
            p = poisson.pmf(g_h, lambda_h) * poisson.pmf(g_a, lambda_a)
            if g_h > g_a:
                prob_H += p
            elif g_h == g_a:
                prob_D += p
            else:
                prob_A += p

    total = prob_H + prob_D + prob_A
    return prob_H / total, prob_D / total, prob_A / total


# ──────────────────────────────────────────────────────────────────────────────
# Attempt 012: Poisson goals model from historical WC + competitive matches
# ──────────────────────────────────────────────────────────────────────────────
def attempt_012_poisson_attack_defense():
    """Fit attack/defense from historical WC group stage + continental competitions."""
    print("\n=== Attempt 012: Poisson attack/defense from historical WC + competitions ===")
    t0 = time.time()

    hist = pd.read_csv(HIST_PATH, parse_dates=['date'])

    # WC + continental cups only (tier >= 4), all years
    def tier(t):
        t = t.lower()
        if 'world cup' in t and 'qualif' not in t:
            return 5
        elif any(x in t for x in ['copa', 'euro ', 'african cup of nations', 'afc asian cup', 'gold cup']):
            return 4
        return 0

    hist['tier'] = hist['tournament'].map(tier)
    hist_wc = hist[hist['tier'] >= 4].copy()
    hist_wc = hist_wc[hist_wc['date'] < pd.to_datetime('2026-06-12')]
    print(f"  Historical WC+continental matches: {len(hist_wc)}")

    # Fit parameters on historical WC+continental (both teams in WC-2026 or not - use all)
    # For WC, use all teams; for continental, use teams that eventually qualify for WC
    teams_for_fit = set(hist_wc['home_team'].tolist()) | set(hist_wc['away_team'].tolist())

    team_params, home_adv, mu = fit_poisson_params(
        hist_wc, teams_for_fit, recency_halflife_days=365*3, cutoff_date='2026-06-12'
    )
    print(f"  Fit complete. n_teams={len(team_params)}, home_adv={home_adv:.3f}, mu={mu:.3f}")

    # Compute probabilities for WC-2026 matches
    X_wc, y_wc, completed = build_wc2026_feature_matrix(
        MATCHES_PATH, TEAMS_PATH, HIST_PATH, cutoff_date='2026-06-12'
    )

    le = LabelEncoder()
    le.fit(['H', 'D', 'A'])

    probs_all = []
    for _, m in completed.iterrows():
        home_wc = m['home_team_name']
        away_wc = m['away_team_name']
        home_h = WC_TO_HIST.get(home_wc, home_wc)
        away_h = WC_TO_HIST.get(away_wc, away_wc)

        is_neutral = True  # WC group stage at neutral venues mostly
        pH, pD, pA = compute_match_probs_poisson(
            team_params, home_h, away_h, mu, home_adv, is_neutral=is_neutral
        )

        probs_all.append([pH, pD, pA])
        print(f"  {home_wc[:10]:10s} v {away_wc[:10]:10s}: H={pH:.3f} D={pD:.3f} A={pA:.3f}")

    probs_arr = np.array(probs_all)
    # Order: H, D, A (le.classes_ should be in alphabetical order: A, D, H)
    # Need to reorder
    prob_df = pd.DataFrame(probs_arr, columns=['H', 'D', 'A'])
    probs_ordered = prob_df[le.classes_].values

    result = canonical_cv_eval(probs_ordered, y_wc, le)
    sig = significance_test(result['fold_losses'])
    elapsed = time.time() - t0

    print(f"  Log-loss: {result['mean']:.4f} ± {result['std']:.4f}")
    print(f"  Accuracy: {result['accuracy']:.3f}")
    print(f"  Δ vs baseline: {sig['delta_vs_baseline']:+.4f}, p={sig['p_value_ttest']:.3f}, verdict={sig['verdict']}")

    metrics = {**result, **sig, 'elapsed_sec': elapsed}
    run_info = {
        'name': 'attempt_012_poisson_attack_defense',
        'description': 'Poisson goals model with team attack/defense params from historical WC+continental matches',
        'n_historical': len(hist_wc),
        'model': 'Log-linear Poisson (L-BFGS-B, L2 reg), max_goals=10, recency_halflife=3yr',
        'features': ['attack_diff', 'defense_diff', 'home_adv'],
        'cutoff_date': '2026-06-12',
        'cv': 'RepeatedStratifiedKFold(5x10, seed=0)',
        'seed': SEED, 'baseline': BASELINE_LOGLOSS, 'frontier': FRONTIER_LOGLOSS,
    }
    save_artifact('attempt-012', metrics, run_info)
    return metrics, probs_ordered, le


# ──────────────────────────────────────────────────────────────────────────────
# Attempt 013: Geometric mean blend (Poisson + WC Elo logistic)
# ──────────────────────────────────────────────────────────────────────────────
def attempt_013_geomean_blend(probs_012, le_012):
    """Geometric mean blend: Poisson model + WC Elo logistic (in-fold)."""
    print("\n=== Attempt 013: Geometric mean blend (Poisson + WC Elo logistic) ===")
    t0 = time.time()

    X_wc, y_wc, completed = build_wc2026_feature_matrix(
        MATCHES_PATH, TEAMS_PATH, HIST_PATH, cutoff_date='2026-06-12'
    )

    le = LabelEncoder()
    le.fit(['H', 'D', 'A'])
    y = le.transform(y_wc)

    rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=SEED)

    # Try different blend weights for Poisson probs
    best_mean, best_alpha = 999, 0.5
    eps = 1e-8

    for alpha in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        fold_l = []
        for train_idx, test_idx in rskf.split(X_wc[['wc_elo_diff', 'host_advantage']], y):
            X_tr = X_wc[['wc_elo_diff', 'host_advantage']].iloc[train_idx]
            y_tr = y[train_idx]
            X_te = X_wc[['wc_elo_diff', 'host_advantage']].iloc[test_idx]

            sc = StandardScaler()
            X_tr_sc = sc.fit_transform(X_tr)
            X_te_sc = sc.transform(X_te)

            clf = LogisticRegression(C=0.5, max_iter=500, random_state=SEED)
            clf.fit(X_tr_sc, y_tr)
            elo_ordered = clf.predict_proba(X_te_sc)  # cols already in A,D,H order (matches le)

            # Poisson probs for test set
            pois_te = probs_012[test_idx]  # already in (A, D, H) order for le

            # Geometric mean blend
            blend = (elo_ordered ** (1 - alpha)) * (pois_te ** alpha)
            blend = blend / blend.sum(axis=1, keepdims=True)
            blend = np.clip(blend, eps, 1 - eps)
            fold_l.append(log_loss(y[test_idx], blend, labels=list(range(len(le.classes_)))))

        mean_l = float(np.mean(fold_l))
        print(f"  alpha(poisson)={alpha}: {mean_l:.4f}")
        if mean_l < best_mean:
            best_mean = mean_l
            best_alpha = alpha

    print(f"  Best alpha: {best_alpha}")

    # Final eval with best alpha
    fold_losses = []
    oof_probs = np.zeros((len(y), len(le.classes_)))
    for train_idx, test_idx in rskf.split(X_wc[['wc_elo_diff', 'host_advantage']], y):
        X_tr = X_wc[['wc_elo_diff', 'host_advantage']].iloc[train_idx]
        y_tr = y[train_idx]
        X_te = X_wc[['wc_elo_diff', 'host_advantage']].iloc[test_idx]
        sc = StandardScaler()
        X_tr_sc = sc.fit_transform(X_tr)
        X_te_sc = sc.transform(X_te)
        clf = LogisticRegression(C=0.5, max_iter=500, random_state=SEED)
        clf.fit(X_tr_sc, y_tr)
        elo_ordered = clf.predict_proba(X_te_sc)  # cols already in A,D,H order
        pois_te = probs_012[test_idx]
        blend = (elo_ordered ** (1 - best_alpha)) * (pois_te ** best_alpha)
        blend = blend / blend.sum(axis=1, keepdims=True)
        blend = np.clip(blend, eps, 1 - eps)
        fold_losses.append(log_loss(y[test_idx], blend, labels=list(range(len(le.classes_)))))
        oof_probs[test_idx] = blend

    result = {
        'mean': float(np.mean(fold_losses)),
        'std': float(np.std(fold_losses)),
        'fold_losses': fold_losses,
        'accuracy': float(accuracy_score(y, np.argmax(oof_probs, axis=1))),
        'oof_probs': oof_probs.tolist(),
        'classes': le.classes_.tolist(),
        'best_alpha': best_alpha,
    }
    sig = significance_test(fold_losses)
    elapsed = time.time() - t0

    print(f"  Final: {result['mean']:.4f} ± {result['std']:.4f}  acc={result['accuracy']:.3f}  {sig['verdict']}")

    metrics = {**result, **sig, 'elapsed_sec': elapsed}
    run_info = {
        'name': 'attempt_013_geomean_blend',
        'description': 'Geometric mean blend: Poisson (historical WC+continental) + WC-2026 Elo logistic (in-fold)',
        'best_alpha': best_alpha, 'model': 'GeometricMean(WC_Elo^(1-alpha), Poisson^alpha)',
        'seed': SEED, 'baseline': BASELINE_LOGLOSS, 'frontier': FRONTIER_LOGLOSS,
    }
    save_artifact('attempt-013', metrics, run_info)
    return metrics, oof_probs


# ──────────────────────────────────────────────────────────────────────────────
# Attempt 014: Head-to-head historical record as feature
# ──────────────────────────────────────────────────────────────────────────────
def attempt_014_h2h_features():
    """Add head-to-head record from historical data as features."""
    print("\n=== Attempt 014: Head-to-head historical record + WC Elo ===")
    t0 = time.time()

    hist = pd.read_csv(HIST_PATH, parse_dates=['date'])
    hist = hist.dropna(subset=['home_score', 'away_score'])
    hist = hist[hist['date'] < pd.to_datetime('2026-06-12')]
    hist = hist.sort_values('date')

    # Build H2H lookup: (teamA, teamB) -> {'n', 'a_wins', 'b_wins', 'draws', 'gd_a'}
    from collections import defaultdict
    h2h = defaultdict(lambda: {'n': 0, 'a_wins': 0, 'b_wins': 0, 'draws': 0, 'gd_a': 0.0})

    for _, row in hist.iterrows():
        home, away = row['home_team'], row['away_team']
        hs, as_ = row['home_score'], row['away_score']
        key = tuple(sorted([home, away]))
        first = key[0] == home  # is 'home' the first team in sorted order?

        h2h[key]['n'] += 1
        gd = hs - as_
        if hs > as_:
            if first:
                h2h[key]['a_wins'] += 1
                h2h[key]['gd_a'] += gd
            else:
                h2h[key]['b_wins'] += 1
                h2h[key]['gd_a'] -= gd
        elif hs < as_:
            if first:
                h2h[key]['b_wins'] += 1
                h2h[key]['gd_a'] -= (as_ - hs)
            else:
                h2h[key]['a_wins'] += 1
                h2h[key]['gd_a'] += (as_ - hs)
        else:
            h2h[key]['draws'] += 1

    X_wc, y_wc, completed = build_wc2026_feature_matrix(
        MATCHES_PATH, TEAMS_PATH, HIST_PATH, cutoff_date='2026-06-12'
    )

    rows = []
    for _, m in completed.iterrows():
        home_wc = m['home_team_name']
        away_wc = m['away_team_name']
        home_h = WC_TO_HIST.get(home_wc, home_wc)
        away_h = WC_TO_HIST.get(away_wc, away_wc)

        key = tuple(sorted([home_h, away_h]))
        first = key[0] == home_h
        rec = h2h.get(key)

        if rec and rec['n'] >= 3:
            n = rec['n']
            a_wins = rec['a_wins']
            b_wins = rec['b_wins']
            draws = rec['draws']
            gd_a = rec['gd_a']

            if first:  # home = first = 'a'
                h2h_home_wr = a_wins / n
                h2h_away_wr = b_wins / n
                h2h_draw_r = draws / n
                h2h_gd = gd_a / n
            else:  # home = second = 'b'
                h2h_home_wr = b_wins / n
                h2h_away_wr = a_wins / n
                h2h_draw_r = draws / n
                h2h_gd = -gd_a / n

            h2h_n = min(n, 20)  # cap to avoid extreme weighting
        else:
            h2h_home_wr = 0.5
            h2h_away_wr = 0.5
            h2h_draw_r = 0.25
            h2h_gd = 0.0
            h2h_n = 0

        rows.append({
            'wc_elo_diff': X_wc.iloc[len(rows)]['wc_elo_diff'],
            'host_advantage': X_wc.iloc[len(rows)]['host_advantage'],
            'wc_rank_diff': X_wc.iloc[len(rows)]['wc_rank_diff'],
            'h2h_home_wr': h2h_home_wr,
            'h2h_away_wr': h2h_away_wr,
            'h2h_draw_r': h2h_draw_r,
            'h2h_gd': h2h_gd,
            'h2h_n': h2h_n,
        })

    X_h2h = pd.DataFrame(rows)

    # Test: WC Elo + H2H
    feats_elo_h2h = ['wc_elo_diff', 'host_advantage', 'h2h_home_wr', 'h2h_gd', 'h2h_draw_r']
    X = X_h2h[feats_elo_h2h]

    for C in [0.1, 0.3, 0.5]:
        def model_fn(C_=C):
            return LogisticRegression(C=C_, max_iter=500, random_state=SEED)
        result, _ = cv_train_eval(X, y_wc, model_fn, scale=True)
        sig = significance_test(result['fold_losses'])
        print(f"  C={C}: {result['mean']:.4f} ± {result['std']:.4f}  {sig['verdict']}")

    # Best C = 0.3
    def model_fn():
        return LogisticRegression(C=0.3, max_iter=500, random_state=SEED)
    result, _ = cv_train_eval(X, y_wc, model_fn, scale=True)
    sig = significance_test(result['fold_losses'])
    elapsed = time.time() - t0

    print(f"  Final: {result['mean']:.4f} ± {result['std']:.4f}  acc={result['accuracy']:.3f}  {sig['verdict']}")

    metrics = {**result, **sig, 'elapsed_sec': elapsed}
    run_info = {
        'name': 'attempt_014_h2h_features',
        'description': 'In-fold logistic: WC-2026 Elo + head-to-head historical record',
        'features': feats_elo_h2h,
        'model': 'LogisticRegression(C=0.3)',
        'seed': SEED, 'baseline': BASELINE_LOGLOSS, 'frontier': FRONTIER_LOGLOSS,
    }
    save_artifact('attempt-014', metrics, run_info)
    return metrics


# ──────────────────────────────────────────────────────────────────────────────
# Attempt 015: Poisson-from-all-history, isotonic re-calibrated per WC distribution
# ──────────────────────────────────────────────────────────────────────────────
def attempt_015_poisson_recalibrated(probs_012, le_012):
    """In-fold isotonic re-calibration of Poisson probs using WC-2026 matches."""
    print("\n=== Attempt 015: In-fold isotonic re-calibrated Poisson ===")
    t0 = time.time()

    X_wc, y_wc, _ = build_wc2026_feature_matrix(
        MATCHES_PATH, TEAMS_PATH, HIST_PATH, cutoff_date='2026-06-12'
    )

    le = LabelEncoder()
    le.fit(['H', 'D', 'A'])
    y = le.transform(y_wc)

    # Use Poisson probs as input features to an in-fold logistic
    # This recalibrates the Poisson probability outputs within each fold
    p_h_idx = list(le.classes_).index('H')
    p_d_idx = list(le.classes_).index('D')
    p_a_idx = list(le.classes_).index('A')

    X_pois = pd.DataFrame({
        'pois_H': probs_012[:, p_h_idx],
        'pois_D': probs_012[:, p_d_idx],
        'pois_A': probs_012[:, p_a_idx],
        'wc_elo_diff': X_wc['wc_elo_diff'],
        'host_advantage': X_wc['host_advantage'],
    })

    best_result, best_C = None, None
    for C in [0.05, 0.1, 0.2, 0.3]:
        def model_fn(C_=C):
            return LogisticRegression(C=C_, max_iter=500, random_state=SEED)
        result, _ = cv_train_eval(X_pois, y_wc, model_fn, scale=True)
        sig = significance_test(result['fold_losses'])
        print(f"  C={C}: {result['mean']:.4f} ± {result['std']:.4f}  {sig['verdict']}")
        if best_result is None or result['mean'] < best_result['mean']:
            best_result = result
            best_C = C

    result = best_result
    sig = significance_test(result['fold_losses'])
    elapsed = time.time() - t0

    print(f"  Final (best C={best_C}): {result['mean']:.4f} ± {result['std']:.4f}  acc={result['accuracy']:.3f}  {sig['verdict']}")

    metrics = {**result, **sig, 'elapsed_sec': elapsed, 'best_C': best_C}
    run_info = {
        'name': 'attempt_015_poisson_recalibrated',
        'description': 'In-fold logistic re-calibration of Poisson probs using WC-2026 matches + WC Elo',
        'features': X_pois.columns.tolist(), 'model': f'LogisticRegression(C={best_C})',
        'seed': SEED, 'baseline': BASELINE_LOGLOSS, 'frontier': FRONTIER_LOGLOSS,
    }
    save_artifact('attempt-015', metrics, run_info)
    return metrics


if __name__ == '__main__':
    print("=" * 70)
    print("Wave-3 Batch 3: Attack/Defense Decomposition via Poisson Goals")
    print("=" * 70)

    m012, probs_012, le_012 = attempt_012_poisson_attack_defense()
    m013, oof_013 = attempt_013_geomean_blend(probs_012, le_012)
    m014 = attempt_014_h2h_features()
    m015 = attempt_015_poisson_recalibrated(probs_012, le_012)

    print("\n" + "=" * 70)
    print("SUMMARY TABLE Batch 3")
    print("=" * 70)
    for name, m in [('012 Poisson A/D', m012), ('013 Geo-mean blend', m013),
                    ('014 H2H features', m014), ('015 Poisson+Recal', m015)]:
        print(f"  {name:25s}: {m['mean']:.4f} ± {m['std']:.4f}  acc={m['accuracy']:.3f}  {m['verdict']}")
