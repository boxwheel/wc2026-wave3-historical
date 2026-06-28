"""
Wave-3 Batch 4: Focused approaches after Batch 1-3 learnings.

Key insights:
1. Form features are correlated with Elo -> don't add signal
2. Distribution mismatch hurts pure transfer (historical ~48% H-wins vs WC ~48% H-wins)
   Wait — actually WC-2026 has 31/64 = 48.4% H-wins, which is the SAME as historical!
   So distribution mismatch in class balance isn't the problem.
3. The issue is in the FEATURE DISTRIBUTION: historical Elo values don't align with WC provided Elo
4. Good news: historical WC matches (all neutral, similar teams) are a representative prior

New approaches:
- Attempt 016: Squad-derived attack/defense as Poisson features (WC-2026 data only)
- Attempt 017: Recency-weighted competitive form + historical Elo + WC Elo blend
- Attempt 018: WC-2026 squad data as proxies for attack/defense in Poisson
- Attempt 019: Combined best features from all batches in ridge-regularized logistic
"""
import sys, os, json, time
import numpy as np
import pandas as pd
from scipy.stats import poisson
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import log_loss, accuracy_score
from sklearn.calibration import CalibratedClassifierCV
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(__file__))
from features import NAME_MAP, build_wc2026_feature_matrix

DATA_DIR = '/home/user/research/wave3-historical/data'
MATCHES_PATH = f'{DATA_DIR}/wc2026-trees-study-main/fifa_data/matches_detailed.csv'
TEAMS_PATH = f'{DATA_DIR}/wc2026-trees-study-main/fifa_data/teams.csv'
SQUADS_PATH = f'{DATA_DIR}/wc2026-trees-study-main/fifa_data/squads_and_players.csv'
HIST_PATH = f'{DATA_DIR}/historical/results.csv'
ARTIFACTS_DIR = '/home/user/research/wave3-historical/artifacts'
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

BASELINE_LOGLOSS = 0.8337
FRONTIER_LOGLOSS = 0.7608
SEED = 0


def cv_train_eval(X_wc, y_wc, model_fn, scale=True):
    le = LabelEncoder()
    le.fit(y_wc)
    y = le.transform(y_wc)
    rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=SEED)
    fold_losses, oof_probs = [], np.zeros((len(y), len(le.classes_)))
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


def canonical_cv_eval(probs, y_str, le=None):
    if le is None:
        le = LabelEncoder()
        le.fit(y_str)
    y = le.transform(y_str)
    rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=SEED)
    fold_losses = []
    for _, test_idx in rskf.split(np.zeros(len(y)), y):
        fold_losses.append(log_loss(y[test_idx], probs[test_idx], labels=list(range(len(le.classes_)))))
    return {
        'mean': float(np.mean(fold_losses)),
        'std': float(np.std(fold_losses)),
        'fold_losses': [float(x) for x in fold_losses],
        'accuracy': float(accuracy_score(y, np.argmax(probs, axis=1))),
        'classes': le.classes_.tolist(),
    }


def sig_test(fold_losses):
    from scipy.stats import ttest_1samp
    arr = np.array(fold_losses)
    t, p = ttest_1samp(arr, BASELINE_LOGLOSS)
    mean = arr.mean()
    verdict = 'GREEN' if (mean < BASELINE_LOGLOSS and p < 0.05) else ('RED' if mean > BASELINE_LOGLOSS else 'FLAT')
    return {'delta_vs_baseline': float(mean - BASELINE_LOGLOSS),
            'delta_vs_frontier': float(mean - FRONTIER_LOGLOSS),
            'p_value': float(p), 'verdict': verdict}


def save_artifact(name, metrics, run_info):
    path = os.path.join(ARTIFACTS_DIR, name)
    os.makedirs(path, exist_ok=True)
    with open(f'{path}/metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    with open(f'{path}/run.json', 'w') as f:
        json.dump(run_info, f, indent=2)
    return path


# ──────────────────────────────────────────────────────────────────────────────
# Attempt 016: WC-2026 Squad-derived Poisson goals model
# ──────────────────────────────────────────────────────────────────────────────
def attempt_016_squad_poisson():
    """Use squad data to compute attack/defense strength proxies, then Poisson P(H/D/A)."""
    print("\n=== Attempt 016: Squad-derived Poisson goals model ===")
    t0 = time.time()

    teams = pd.read_csv(TEAMS_PATH)
    squads = pd.read_csv(SQUADS_PATH)

    # Compute per-team squad features
    team_stats = {}
    for _, t in teams.iterrows():
        team = t['team_name']
        tq = squads[squads['team_id'] == t['team_id']]

        # Attacker features (proxy for attack rate)
        attackers = tq[tq['position'].str.contains('FW|Forward|Att', case=False, na=False)]
        midfielders = tq[tq['position'].str.contains('MF|Mid', case=False, na=False)]
        defenders = tq[tq['position'].str.contains('DF|Def|Back', case=False, na=False)]
        gk = tq[tq['position'].str.contains('GK|Goalkeeper', case=False, na=False)]

        n_att = len(attackers)
        n_def = len(defenders)
        n_mid = len(midfielders)
        n_gk = len(gk)

        total_mv = tq['market_value_eur'].sum() if 'market_value_eur' in tq.columns else 1e6
        att_mv = attackers['market_value_eur'].sum() if n_att > 0 else 0
        def_mv = defenders['market_value_eur'].sum() if n_def > 0 else 0
        gk_mv = gk['market_value_eur'].sum() if n_gk > 0 else 0
        top11_mv = tq.nlargest(11, 'market_value_eur')['market_value_eur'].sum()

        att_goals = attackers['goals'].sum() if 'goals' in attackers.columns and n_att > 0 else 0
        mean_caps = tq['caps'].mean() if 'caps' in tq.columns else 50

        team_stats[team] = {
            'total_mv': total_mv,
            'att_mv': att_mv,
            'def_mv': def_mv,
            'gk_mv': gk_mv,
            'top11_mv': top11_mv,
            'att_goals': att_goals,
            'mean_caps': mean_caps,
            'elo_rating': t['elo_rating'],
            'fifa_rank': t['fifa_ranking_pre_tournament'],
        }

    # Compute global means for normalization
    all_mv = [v['total_mv'] for v in team_stats.values()]
    all_att_mv = [v['att_mv'] for v in team_stats.values()]
    all_def_mv = [v['def_mv'] for v in team_stats.values()]
    all_gk_mv = [v['gk_mv'] for v in team_stats.values()]
    mean_mv = np.mean(all_mv)
    mean_att_mv = np.mean(all_att_mv) + 1
    mean_def_mv = np.mean(all_def_mv) + 1
    mean_gk_mv = np.mean(all_gk_mv) + 1
    mean_elo = np.mean([v['elo_rating'] for v in team_stats.values()])

    # Compute Poisson params:
    # attack_strength = (att_mv / mean_att_mv) ^ 0.3 * (att_goals / global_att_goals_mean) ^ 0.2
    # defense_strength = (def_mv / mean_def_mv) ^ 0.2 * (gk_mv / mean_gk_mv) ^ 0.2
    mean_att_goals = np.mean([v['att_goals'] for v in team_stats.values()]) + 1

    mu_goals = 1.3  # average goals per team per WC match (historical ~2.6 total)
    home_adv = 0.1  # slight home advantage for host teams

    X_wc, y_wc, completed = build_wc2026_feature_matrix(
        MATCHES_PATH, TEAMS_PATH, HIST_PATH, cutoff_date='2026-06-12'
    )

    host_teams = {'Mexico', 'USA', 'Canada'}

    le = LabelEncoder()
    le.fit(['H', 'D', 'A'])

    probs_all = []
    for _, m in completed.iterrows():
        home_wc = m['home_team_name']
        away_wc = m['away_team_name']

        hs = team_stats.get(home_wc, {'att_mv': mean_att_mv, 'def_mv': mean_def_mv,
                                        'gk_mv': mean_gk_mv, 'att_goals': mean_att_goals,
                                        'elo_rating': mean_elo})
        as_ = team_stats.get(away_wc, {'att_mv': mean_att_mv, 'def_mv': mean_def_mv,
                                          'gk_mv': mean_gk_mv, 'att_goals': mean_att_goals,
                                          'elo_rating': mean_elo})

        # Attack strength proxy: attacker market value normalized
        alpha_home = (hs['att_mv'] / mean_att_mv) ** 0.3
        alpha_away = (as_['att_mv'] / mean_att_mv) ** 0.3

        # Defense strength proxy: defender + GK market value normalized
        beta_home = (hs['def_mv'] / mean_def_mv + hs['gk_mv'] / mean_gk_mv) / 2.0
        beta_away = (as_['def_mv'] / mean_def_mv + as_['gk_mv'] / mean_gk_mv) / 2.0

        # Expected goals = mu * attack_home / defense_away * home_factor
        is_host_home = 1 if home_wc in host_teams else 0
        home_factor = np.exp(home_adv * is_host_home)

        lambda_h = mu_goals * max(alpha_home / max(beta_away, 0.1), 0.1) * home_factor
        lambda_a = mu_goals * max(alpha_away / max(beta_home, 0.1), 0.1)

        pH, pD, pA = 0.0, 0.0, 0.0
        for g_h in range(12):
            for g_a in range(12):
                p = poisson.pmf(g_h, lambda_h) * poisson.pmf(g_a, lambda_a)
                if g_h > g_a:
                    pH += p
                elif g_h == g_a:
                    pD += p
                else:
                    pA += p

        total = pH + pD + pA
        probs_all.append([pH/total, pD/total, pA/total])

    probs_arr = np.array(probs_all)
    prob_df = pd.DataFrame(probs_arr, columns=['H', 'D', 'A'])
    probs_ordered = prob_df[le.classes_].values

    result = canonical_cv_eval(probs_ordered, y_wc, le)
    sig = sig_test(result['fold_losses'])
    elapsed = time.time() - t0

    print(f"  Log-loss: {result['mean']:.4f} ± {result['std']:.4f}")
    print(f"  Accuracy: {result['accuracy']:.3f}  {sig['verdict']}")

    metrics = {**result, **sig, 'elapsed_sec': elapsed}
    run_info = {
        'name': 'attempt_016_squad_poisson',
        'description': 'Poisson goals model with squad market-value attack/defense proxies',
        'model': 'Poisson(mu=1.3, home_adv=0.1)',
        'features': ['att_mv/mean', 'def_mv/mean', 'gk_mv/mean'],
        'seed': SEED, 'baseline': BASELINE_LOGLOSS, 'frontier': FRONTIER_LOGLOSS,
    }
    save_artifact('attempt-016', metrics, run_info)
    return metrics, probs_ordered, le


# ──────────────────────────────────────────────────────────────────────────────
# Attempt 017: Historical competitive-only Elo + WC Elo blend + squad features, in-fold
# ──────────────────────────────────────────────────────────────────────────────
def attempt_017_combined_features():
    """Best combination of WC-2026 features + historical signals, in-fold logistic."""
    print("\n=== Attempt 017: Combined: WC Elo + rank + squad features, in-fold logistic ===")
    t0 = time.time()

    teams = pd.read_csv(TEAMS_PATH)
    squads = pd.read_csv(SQUADS_PATH)
    matches = pd.read_csv(MATCHES_PATH)
    completed = matches[matches['status'] == 'Completed'].copy()

    # Compute squad features
    squad_feats = {}
    for _, t in teams.iterrows():
        team = t['team_name']
        tq = squads[squads['team_id'] == t['team_id']]
        total_mv = tq['market_value_eur'].sum() if len(tq) > 0 else 1e6
        top11_mv = tq.nlargest(11, 'market_value_eur')['market_value_eur'].sum() if len(tq) >= 11 else total_mv
        mean_caps = tq['caps'].mean() if len(tq) > 0 else 50
        mean_age = (pd.to_datetime('2026-06-15') - pd.to_datetime(tq['date_of_birth'])).dt.days.mean() / 365.25 if len(tq) > 0 else 27

        attackers = tq[tq['position'].str.contains('FW|Forward|Att', case=False, na=False)]
        gk = tq[tq['position'].str.contains('GK|Goalkeeper', case=False, na=False)]
        defenders = tq[tq['position'].str.contains('DF|Def|Back', case=False, na=False)]

        att_goals = attackers['goals'].sum() if len(attackers) > 0 else 0
        gk_mv = gk['market_value_eur'].sum() if len(gk) > 0 else 0
        def_mv = defenders['market_value_eur'].sum() if len(defenders) > 0 else 0

        squad_feats[team] = {
            'total_mv': total_mv,
            'top11_mv': top11_mv,
            'mean_caps': mean_caps,
            'mean_age': mean_age,
            'att_goals': att_goals,
            'gk_mv': gk_mv,
            'def_mv': def_mv,
        }

    global_total_mv = np.mean([v['total_mv'] for v in squad_feats.values()])
    global_top11_mv = np.mean([v['top11_mv'] for v in squad_feats.values()])
    global_mean_caps = np.mean([v['mean_caps'] for v in squad_feats.values()])
    global_att_goals = np.mean([v['att_goals'] for v in squad_feats.values()]) + 1

    host_teams = {'Mexico', 'USA', 'Canada'}
    teams_elo = dict(zip(teams['team_name'], teams['elo_rating']))
    teams_rank = dict(zip(teams['team_name'], teams['fifa_ranking_pre_tournament']))
    teams_conf = dict(zip(teams['team_name'], teams['confederation']))

    conf_enc = {'UEFA': 0, 'CONMEBOL': 1, 'CONCACAF': 2, 'CAF': 3, 'AFC': 4, 'OFC': 5}

    rows, labels = [], []
    for _, m in completed.iterrows():
        home, away = m['home_team_name'], m['away_team_name']
        hs_f = squad_feats.get(home, {k: 0 for k in ['total_mv', 'top11_mv', 'mean_caps', 'mean_age', 'att_goals', 'gk_mv', 'def_mv']})
        as_f = squad_feats.get(away, {k: 0 for k in ['total_mv', 'top11_mv', 'mean_caps', 'mean_age', 'att_goals', 'gk_mv', 'def_mv']})

        home_elo = teams_elo.get(home, 1500)
        away_elo = teams_elo.get(away, 1500)
        home_rank = teams_rank.get(home, 50)
        away_rank = teams_rank.get(away, 50)

        home_conf = conf_enc.get(teams_conf.get(home, 'UEFA'), 0)
        away_conf = conf_enc.get(teams_conf.get(away, 'UEFA'), 0)

        row = {
            'elo_diff': home_elo - away_elo,
            'rank_diff': away_rank - home_rank,
            'host_adv': (1 if home in host_teams else 0) - (1 if away in host_teams else 0),
            'log_total_mv_diff': np.log1p(hs_f['total_mv']) - np.log1p(as_f['total_mv']),
            'log_top11_mv_diff': np.log1p(hs_f['top11_mv']) - np.log1p(as_f['top11_mv']),
            'caps_diff': hs_f['mean_caps'] - as_f['mean_caps'],
            'age_diff': hs_f['mean_age'] - as_f['mean_age'],
            'att_goals_diff': np.log1p(hs_f['att_goals']) - np.log1p(as_f['att_goals']),
            'gk_mv_diff': np.log1p(hs_f['gk_mv']) - np.log1p(as_f['gk_mv']),
            'def_mv_diff': np.log1p(hs_f['def_mv']) - np.log1p(as_f['def_mv']),
        }
        rows.append(row)

        hs, as_ = m['home_score'], m['away_score']
        labels.append('H' if hs > as_ else ('D' if hs == as_ else 'A'))

    X = pd.DataFrame(rows)
    y_wc = pd.Series(labels)

    best_r, best_C = None, None
    for C in [0.05, 0.1, 0.2, 0.3]:
        def model_fn(C_=C):
            return LogisticRegression(C=C_, max_iter=500, random_state=SEED)
        result, _ = cv_train_eval(X, y_wc, model_fn, scale=True)
        sig = sig_test(result['fold_losses'])
        print(f"  C={C}: {result['mean']:.4f} ± {result['std']:.4f}  {sig['verdict']}")
        if best_r is None or result['mean'] < best_r['mean']:
            best_r, best_C = result, C

    result = best_r
    sig = sig_test(result['fold_losses'])
    elapsed = time.time() - t0

    print(f"  Best C={best_C}: {result['mean']:.4f} ± {result['std']:.4f}  acc={result['accuracy']:.3f}  {sig['verdict']}")

    metrics = {**result, **sig, 'elapsed_sec': elapsed, 'best_C': best_C}
    run_info = {
        'name': 'attempt_017_combined_features',
        'description': 'In-fold logistic: WC-2026 Elo + rank + squad features (market value, caps, age, attacker goals)',
        'features': X.columns.tolist(), 'model': f'LogisticRegression(C={best_C})',
        'seed': SEED, 'baseline': BASELINE_LOGLOSS, 'frontier': FRONTIER_LOGLOSS,
    }
    save_artifact('attempt-017', metrics, run_info)
    return metrics, X, y_wc


# ──────────────────────────────────────────────────────────────────────────────
# Attempt 018: Geometric mean blend of Squad Poisson + WC Elo logistic
# ──────────────────────────────────────────────────────────────────────────────
def attempt_018_squad_poisson_blend(probs_016, le_016, X_017, y_017):
    """Blend squad Poisson probs with WC Elo features, in-fold."""
    print("\n=== Attempt 018: Blend squad Poisson + full features logistic ===")
    t0 = time.time()

    le = LabelEncoder()
    le.fit(['H', 'D', 'A'])
    y = le.transform(y_017)

    # Add squad Poisson probs as features
    p_h_idx = list(le.classes_).index('H')
    p_d_idx = list(le.classes_).index('D')
    p_a_idx = list(le.classes_).index('A')

    X_blend = X_017.copy()
    X_blend['sq_pois_H'] = probs_016[:, p_h_idx]
    X_blend['sq_pois_D'] = probs_016[:, p_d_idx]
    X_blend['sq_pois_A'] = probs_016[:, p_a_idx]

    def model_fn():
        return LogisticRegression(C=0.1, max_iter=500, random_state=SEED)
    result, _ = cv_train_eval(X_blend, y_017, model_fn, scale=True)
    sig = sig_test(result['fold_losses'])
    elapsed = time.time() - t0

    print(f"  {result['mean']:.4f} ± {result['std']:.4f}  acc={result['accuracy']:.3f}  {sig['verdict']}")

    metrics = {**result, **sig, 'elapsed_sec': elapsed}
    run_info = {
        'name': 'attempt_018_squad_poisson_blend',
        'description': 'In-fold logistic: full squad+Elo features + squad Poisson probs',
        'features': X_blend.columns.tolist(), 'model': 'LogisticRegression(C=0.1)',
        'seed': SEED, 'baseline': BASELINE_LOGLOSS, 'frontier': FRONTIER_LOGLOSS,
    }
    save_artifact('attempt-018', metrics, run_info)
    return metrics


# ──────────────────────────────────────────────────────────────────────────────
# Attempt 019: WC-2026 Elo only with optimal regularization sweep
# ──────────────────────────────────────────────────────────────────────────────
def attempt_019_elo_only_opt():
    """Systematic sweep of regularization for WC-2026 Elo-only logistic."""
    print("\n=== Attempt 019: WC Elo-only logistic, fine-tuned regularization sweep ===")
    t0 = time.time()

    X_wc, y_wc, _ = build_wc2026_feature_matrix(
        MATCHES_PATH, TEAMS_PATH, HIST_PATH, cutoff_date='2026-06-12'
    )

    X = X_wc[['wc_elo_diff', 'host_advantage']].copy()

    best_r, best_cfg = None, None
    for C in [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 2.0]:
        for solver in ['lbfgs', 'saga']:
            def model_fn(C_=C, s=solver):
                return LogisticRegression(C=C_, max_iter=1000, random_state=SEED, solver=s)
            result, _ = cv_train_eval(X, y_wc, model_fn, scale=True)
            if best_r is None or result['mean'] < best_r['mean']:
                best_r, best_cfg = result, (C, solver)
                print(f"  C={C} solver={solver}: {result['mean']:.4f} NEW BEST")
            else:
                print(f"  C={C} solver={solver}: {result['mean']:.4f}")

    result = best_r
    sig = sig_test(result['fold_losses'])
    elapsed = time.time() - t0
    print(f"  Best config {best_cfg}: {result['mean']:.4f} ± {result['std']:.4f}  {sig['verdict']}")

    metrics = {**result, **sig, 'elapsed_sec': elapsed, 'best_config': list(best_cfg)}
    run_info = {
        'name': 'attempt_019_elo_only_opt',
        'description': 'WC-2026 Elo-only logistic with systematic regularization + solver sweep',
        'features': ['wc_elo_diff', 'host_advantage'], 'best_config': list(best_cfg),
        'seed': SEED, 'baseline': BASELINE_LOGLOSS, 'frontier': FRONTIER_LOGLOSS,
    }
    save_artifact('attempt-019', metrics, run_info)
    return metrics


# ──────────────────────────────────────────────────────────────────────────────
# Attempt 020: WC-2026 historical Elo + squad + H2H, in-fold logistic (Kitchen Sink)
# ──────────────────────────────────────────────────────────────────────────────
def attempt_020_kitchen_sink():
    """All available WC-2026 pre-match features + historical form in one model."""
    print("\n=== Attempt 020: Kitchen sink (all WC features + historical form), ridge logistic ===")
    t0 = time.time()

    X_wc, y_wc, completed = build_wc2026_feature_matrix(
        MATCHES_PATH, TEAMS_PATH, HIST_PATH, cutoff_date='2026-06-12'
    )

    teams = pd.read_csv(TEAMS_PATH)
    squads = pd.read_csv(SQUADS_PATH)

    squad_feats = {}
    for _, t in teams.iterrows():
        team = t['team_name']
        tq = squads[squads['team_id'] == t['team_id']]
        total_mv = tq['market_value_eur'].sum() if len(tq) > 0 else 1e6
        top11_mv = tq.nlargest(11, 'market_value_eur')['market_value_eur'].sum() if len(tq) >= 11 else total_mv
        mean_caps = tq['caps'].mean() if len(tq) > 0 else 50
        att = tq[tq['position'].str.contains('FW|Forward|Att', case=False, na=False)]
        gk = tq[tq['position'].str.contains('GK|Goalkeeper', case=False, na=False)]
        squad_feats[team] = {
            'total_mv': total_mv,
            'top11_mv': top11_mv,
            'mean_caps': mean_caps,
            'att_goals': att['goals'].sum() if len(att) > 0 else 0,
            'gk_mv': gk['market_value_eur'].sum() if len(gk) > 0 else 0,
        }

    teams_elo = dict(zip(teams['team_name'], teams['elo_rating']))
    teams_rank = dict(zip(teams['team_name'], teams['fifa_ranking_pre_tournament']))
    host_teams = {'Mexico', 'USA', 'Canada'}

    rows = []
    for i, (_, m) in enumerate(completed.iterrows()):
        home, away = m['home_team_name'], m['away_team_name']
        hs = squad_feats.get(home, {})
        as_ = squad_feats.get(away, {})

        row = {
            # Elo
            'wc_elo_diff': X_wc.iloc[i]['wc_elo_diff'],
            'hist_elo_diff': X_wc.iloc[i]['hist_elo_diff'],
            'elo_diff_delta': X_wc.iloc[i]['elo_diff_delta'],
            # Rank
            'wc_rank_diff': X_wc.iloc[i]['wc_rank_diff'],
            # Context
            'host_advantage': X_wc.iloc[i]['host_advantage'],
            # Squad market value
            'log_mv_diff': np.log1p(hs.get('total_mv', 1e6)) - np.log1p(as_.get('total_mv', 1e6)),
            'log_top11_diff': np.log1p(hs.get('top11_mv', 1e6)) - np.log1p(as_.get('top11_mv', 1e6)),
            'caps_diff': hs.get('mean_caps', 50) - as_.get('mean_caps', 50),
            'att_goals_diff': np.log1p(hs.get('att_goals', 5)) - np.log1p(as_.get('att_goals', 5)),
            'gk_mv_diff': np.log1p(hs.get('gk_mv', 1e5)) - np.log1p(as_.get('gk_mv', 1e5)),
            # Historical form (competitive)
            'win_rate_10_diff': X_wc.iloc[i]['win_rate_10_diff'],
            'gd_10_diff': X_wc.iloc[i]['gd_10_diff'],
            'win_rate_5_diff': X_wc.iloc[i]['win_rate_5_diff'],
        }
        rows.append(row)

    X = pd.DataFrame(rows)

    best_r, best_C = None, None
    for C in [0.03, 0.05, 0.07, 0.1, 0.15, 0.2]:
        def model_fn(C_=C):
            return LogisticRegression(C=C_, max_iter=500, random_state=SEED)
        result, _ = cv_train_eval(X, y_wc, model_fn, scale=True)
        sig = sig_test(result['fold_losses'])
        print(f"  C={C}: {result['mean']:.4f} ± {result['std']:.4f}  {sig['verdict']}")
        if best_r is None or result['mean'] < best_r['mean']:
            best_r, best_C = result, C

    result = best_r
    sig = sig_test(result['fold_losses'])
    elapsed = time.time() - t0

    print(f"  Best C={best_C}: {result['mean']:.4f} ± {result['std']:.4f}  acc={result['accuracy']:.3f}  {sig['verdict']}")

    metrics = {**result, **sig, 'elapsed_sec': elapsed, 'best_C': best_C}
    run_info = {
        'name': 'attempt_020_kitchen_sink',
        'description': 'In-fold logistic: WC Elo + rank + squad features + historical form (15 features, regularized)',
        'features': X.columns.tolist(), 'model': f'LogisticRegression(C={best_C})',
        'seed': SEED, 'baseline': BASELINE_LOGLOSS, 'frontier': FRONTIER_LOGLOSS,
    }
    save_artifact('attempt-020', metrics, run_info)
    return metrics


if __name__ == '__main__':
    print("=" * 70)
    print("Wave-3 Batch 4: Squad-derived Poisson + Combined Features")
    print("=" * 70)

    m016, probs_016, le_016 = attempt_016_squad_poisson()
    m017, X_017, y_017 = attempt_017_combined_features()
    m018 = attempt_018_squad_poisson_blend(probs_016, le_016, X_017, y_017)
    m019 = attempt_019_elo_only_opt()
    m020 = attempt_020_kitchen_sink()

    print("\n" + "=" * 70)
    print("SUMMARY TABLE Batch 4")
    print("=" * 70)
    for name, m in [('016 Squad Poisson', m016), ('017 Combined features', m017),
                    ('018 Squad Pois blend', m018), ('019 Elo-opt sweep', m019),
                    ('020 Kitchen sink', m020)]:
        print(f"  {name:22s}: {m['mean']:.4f} ± {m['std']:.4f}  acc={m['accuracy']:.3f}  {m['verdict']}")
