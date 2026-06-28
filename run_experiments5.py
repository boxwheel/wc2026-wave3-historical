"""
Wave-3 Batch 5: New angles after all Batch 1-4 features proved RED/FLAT.

Key insight from analysis: beating 0.8337 requires genuinely orthogonal pre-computed
signals (not in-fold estimates, which have too little data). We try:
  - Recent form: weighted win rate from last 12 months of historical matches
  - Confederation strength bias: different confederations systematically differ at same Elo
  - WC tournament history: institutional experience
  - 3-feature clean logistic: Elo + FIFA rank + recent form (no kitchen-sink noise)
  - Ensemble: average OOF probs from our two best FLAT models (019 + 020)
"""
import sys, os, json, time
import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import log_loss, accuracy_score
from scipy.stats import ttest_1samp
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


def canonical_cv_eval_probs(probs, y_str, le=None):
    """Evaluate pre-computed probability matrix over canonical folds."""
    if le is None:
        le = LabelEncoder()
        le.fit(y_str)
    y = le.transform(y_str)
    rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=SEED)
    fold_losses = []
    for _, test_idx in rskf.split(np.zeros(len(y)), y):
        fold_losses.append(log_loss(y[test_idx], probs[test_idx],
                                    labels=list(range(len(le.classes_)))))
    return {
        'mean': float(np.mean(fold_losses)),
        'std': float(np.std(fold_losses)),
        'fold_losses': [float(x) for x in fold_losses],
        'accuracy': float(accuracy_score(y, np.argmax(probs, axis=1))),
    }


def sig_test(fold_losses):
    arr = np.array(fold_losses)
    t, p = ttest_1samp(arr, BASELINE_LOGLOSS)
    mean = arr.mean()
    verdict = ('GREEN' if (mean < BASELINE_LOGLOSS and p < 0.05)
               else ('RED' if mean > BASELINE_LOGLOSS else 'FLAT'))
    return {
        'delta_vs_baseline': float(mean - BASELINE_LOGLOSS),
        'p_value': float(p),
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


HOST_TEAMS = {'Mexico', 'USA', 'Canada'}

def load_base_data():
    matches = pd.read_csv(MATCHES_PATH)
    teams = pd.read_csv(TEAMS_PATH)
    hist = pd.read_csv(HIST_PATH)
    hist['date'] = pd.to_datetime(hist['date'])
    # matches_detailed.csv uses home_team_name / away_team_name
    # historical results.csv uses home_team / away_team
    hist['home_team'] = hist['home_team'].map(lambda x: NAME_MAP.get(x, x))
    hist['away_team'] = hist['away_team'].map(lambda x: NAME_MAP.get(x, x))
    # Completed group-stage matches only
    completed = matches[matches['status'] == 'Completed'].copy()
    completed = completed[completed['stage_name'].str.lower().str.contains('group', na=False)].copy()
    completed = completed.dropna(subset=['home_score', 'away_score']).copy()
    # Standardize column names for convenience
    completed = completed.rename(columns={'home_team_name': 'home_team', 'away_team_name': 'away_team'})
    return completed, teams, hist


def compute_recent_form(hist, team_names, start_date='2025-01-01', end_date='2026-05-31'):
    """
    Pre-compute a weighted win rate for each team from recent international matches.
    Weighting: exponential recency decay with 180-day halflife.
    Returns dict: team_name -> form_score (0-1 scale, 0.5 = average).
    """
    recent = hist[(hist['date'] >= start_date) & (hist['date'] <= end_date)].copy()
    if len(recent) == 0:
        return {t: 0.5 for t in team_names}
    max_date = recent['date'].max()
    form = {}
    for team in team_names:
        h_rows = recent[recent['home_team'] == team]
        a_rows = recent[recent['away_team'] == team]
        scores = []
        weights = []
        for _, row in h_rows.iterrows():
            days = (max_date - row['date']).days
            w = np.exp(-days / 180.0)
            s = 1.0 if row['home_score'] > row['away_score'] else \
                0.5 if row['home_score'] == row['away_score'] else 0.0
            scores.append(s); weights.append(w)
        for _, row in a_rows.iterrows():
            days = (max_date - row['date']).days
            w = np.exp(-days / 180.0)
            s = 1.0 if row['away_score'] > row['home_score'] else \
                0.5 if row['away_score'] == row['home_score'] else 0.0
            scores.append(s); weights.append(w)
        if weights:
            form[team] = float(np.average(scores, weights=weights))
        else:
            form[team] = 0.5
    return form


def count_wc_appearances(hist, team_names):
    """Count distinct FIFA World Cups each team appeared in from historical data."""
    wc = hist[hist['tournament'] == 'FIFA World Cup'].copy()
    counts = {}
    for team in team_names:
        team_wc = wc[(wc['home_team'] == team) | (wc['away_team'] == team)]
        # Group by year to count distinct WC tournaments
        years = pd.to_datetime(team_wc['date']).dt.year.unique()
        # WC years: 1930,1934,...,2022 — each WC spans 1 calendar year
        wc_years = [y for y in years if y in {1930,1934,1938,1950,1954,1958,1962,1966,
                                                1970,1974,1978,1982,1986,1990,1994,1998,
                                                2002,2006,2010,2014,2018,2022}]
        counts[team] = len(wc_years)
    return counts


# ──────────────────────────────────────────────────────────────────────────────
# Attempt 021: Pre-computed recent form (last 18 months) + WC Elo
# ──────────────────────────────────────────────────────────────────────────────
def attempt_021_recent_form():
    print("\n=== Attempt 021: Recent form (last 18 months) + WC Elo logistic ===")
    t0 = time.time()
    completed, teams, hist = load_base_data()

    team_names = list(teams['team_name'])
    form = compute_recent_form(hist, team_names, start_date='2024-11-01', end_date='2026-05-31')
    print(f"  Teams with form data: {sum(1 for v in form.values() if v != 0.5)} / {len(team_names)}")
    print(f"  Form range: {min(form.values()):.3f} – {max(form.values()):.3f}")

    team_map = {r['team_name']: r for _, r in teams.iterrows()}

    rows = []
    y = []
    for _, m in completed.iterrows():
        h, a = m['home_team'], m['away_team']
        if h not in team_map or a not in team_map:
            continue
        ht, at = team_map[h], team_map[a]
        elo_diff = float(ht['elo_rating']) - float(at['elo_rating'])
        host_adv = 1.0 if h in HOST_TEAMS else 0.0
        form_diff = form.get(h, 0.5) - form.get(a, 0.5)
        rows.append({'elo_diff': elo_diff, 'host_advantage': host_adv, 'form_diff': form_diff})
        hs, as_ = int(m['home_score']), int(m['away_score'])
        y.append('H' if hs > as_ else 'D' if hs == as_ else 'A')

    X = pd.DataFrame(rows)
    print(f"  Samples: {len(X)}, form_diff mean={X['form_diff'].mean():.4f} std={X['form_diff'].std():.4f}")

    best_loss, best_C, best_res = np.inf, None, None
    for C in [0.05, 0.1, 0.3, 0.5, 1.0]:
        res, le = cv_train_eval(X, y, lambda c=C: LogisticRegression(C=c, max_iter=500, solver='saga'))
        sig = sig_test(res['fold_losses'])
        print(f"  C={C}: {res['mean']:.4f} ± {res['std']:.4f}  {sig['verdict']}")
        if res['mean'] < best_loss:
            best_loss, best_C, best_res = res['mean'], C, (res, le, sig)

    res, le, sig = best_res
    print(f"  Best C={best_C}: {res['mean']:.4f} ± {res['std']:.4f}  acc={res['accuracy']:.3f}  {sig['verdict']}")
    metrics = {**res, **sig, 'best_C': best_C}
    run_info = {'attempt': 21, 'description': 'Recent form (18m) + WC Elo logistic',
                'features': ['elo_diff', 'host_advantage', 'form_diff'],
                'form_window': '2024-11-01 to 2026-05-31',
                'best_C': best_C, 'elapsed_s': time.time() - t0}
    save_artifact('attempt-021', metrics, run_info)
    return metrics


# ──────────────────────────────────────────────────────────────────────────────
# Attempt 022: Confederation strength bias + WC Elo
# ──────────────────────────────────────────────────────────────────────────────
def attempt_022_confederation():
    print("\n=== Attempt 022: Confederation strength bias + WC Elo ===")
    t0 = time.time()
    completed, teams, hist = load_base_data()

    # Confederation mean Elo (proxy for tier)
    conf_elo = teams.groupby('confederation')['elo_rating'].mean().to_dict()
    conf_rank = teams.groupby('confederation')['fifa_ranking_pre_tournament'].mean().to_dict()

    # Ordinal confederation encoding: rank confederations by mean Elo
    sorted_confs = sorted(conf_elo.items(), key=lambda x: x[1])
    conf_tier = {c: i for i, (c, _) in enumerate(sorted_confs)}
    print(f"  Confederation tiers: { {c: f'{v:.0f}' for c,v in conf_elo.items()} }")

    team_map = {r['team_name']: r for _, r in teams.iterrows()}

    rows = []
    y = []
    for _, m in completed.iterrows():
        h, a = m['home_team'], m['away_team']
        if h not in team_map or a not in team_map:
            continue
        ht, at = team_map[h], team_map[a]
        elo_diff = float(ht['elo_rating']) - float(at['elo_rating'])
        host_adv = 1.0 if h in HOST_TEAMS else 0.0
        h_conf = str(ht.get('confederation', ''))
        a_conf = str(at.get('confederation', ''))
        conf_tier_diff = conf_tier.get(h_conf, 3) - conf_tier.get(a_conf, 3)
        conf_elo_diff = conf_elo.get(h_conf, 1500) - conf_elo.get(a_conf, 1500)
        rows.append({
            'elo_diff': elo_diff,
            'host_advantage': host_adv,
            'conf_tier_diff': conf_tier_diff,
            'conf_elo_diff': conf_elo_diff,
        })
        hs, as_ = int(m['home_score']), int(m['away_score'])
        y.append('H' if hs > as_ else 'D' if hs == as_ else 'A')

    X = pd.DataFrame(rows)
    print(f"  Samples: {len(X)}")

    best_loss, best_C, best_res = np.inf, None, None
    for C in [0.05, 0.1, 0.3, 0.5, 1.0]:
        res, le = cv_train_eval(X, y, lambda c=C: LogisticRegression(C=c, max_iter=500, solver='saga'))
        sig = sig_test(res['fold_losses'])
        print(f"  C={C}: {res['mean']:.4f} ± {res['std']:.4f}  {sig['verdict']}")
        if res['mean'] < best_loss:
            best_loss, best_C, best_res = res['mean'], C, (res, le, sig)

    res, le, sig = best_res
    print(f"  Best C={best_C}: {res['mean']:.4f} ± {res['std']:.4f}  acc={res['accuracy']:.3f}  {sig['verdict']}")
    metrics = {**res, **sig, 'best_C': best_C}
    run_info = {'attempt': 22, 'description': 'Confederation strength bias + WC Elo',
                'features': ['elo_diff', 'host_advantage', 'conf_tier_diff', 'conf_elo_diff'],
                'best_C': best_C, 'elapsed_s': time.time() - t0}
    save_artifact('attempt-022', metrics, run_info)
    return metrics


# ──────────────────────────────────────────────────────────────────────────────
# Attempt 023: WC tournament appearance count + WC Elo
# ──────────────────────────────────────────────────────────────────────────────
def attempt_023_wc_history():
    print("\n=== Attempt 023: WC tournament history count + WC Elo ===")
    t0 = time.time()
    completed, teams, hist = load_base_data()

    team_names = list(teams['team_name'])
    wc_apps = count_wc_appearances(hist, team_names)
    print(f"  WC apps range: {min(wc_apps.values())} – {max(wc_apps.values())}")
    print(f"  Most experienced: {max(wc_apps, key=wc_apps.get)} ({max(wc_apps.values())})")

    team_map = {r['team_name']: r for _, r in teams.iterrows()}

    rows = []
    y = []
    for _, m in completed.iterrows():
        h, a = m['home_team'], m['away_team']
        if h not in team_map or a not in team_map:
            continue
        ht, at = team_map[h], team_map[a]
        elo_diff = float(ht['elo_rating']) - float(at['elo_rating'])
        host_adv = 1.0 if h in HOST_TEAMS else 0.0
        apps_diff = wc_apps.get(h, 0) - wc_apps.get(a, 0)
        rows.append({'elo_diff': elo_diff, 'host_advantage': host_adv, 'wc_apps_diff': apps_diff})
        hs, as_ = int(m['home_score']), int(m['away_score'])
        y.append('H' if hs > as_ else 'D' if hs == as_ else 'A')

    X = pd.DataFrame(rows)
    print(f"  Samples: {len(X)}, wc_apps_diff range: {X['wc_apps_diff'].min()} – {X['wc_apps_diff'].max()}")

    best_loss, best_C, best_res = np.inf, None, None
    for C in [0.05, 0.1, 0.3, 0.5, 1.0]:
        res, le = cv_train_eval(X, y, lambda c=C: LogisticRegression(C=c, max_iter=500, solver='saga'))
        sig = sig_test(res['fold_losses'])
        print(f"  C={C}: {res['mean']:.4f} ± {res['std']:.4f}  {sig['verdict']}")
        if res['mean'] < best_loss:
            best_loss, best_C, best_res = res['mean'], C, (res, le, sig)

    res, le, sig = best_res
    print(f"  Best C={best_C}: {res['mean']:.4f} ± {res['std']:.4f}  acc={res['accuracy']:.3f}  {sig['verdict']}")
    metrics = {**res, **sig, 'best_C': best_C}
    run_info = {'attempt': 23, 'description': 'WC history count + WC Elo',
                'features': ['elo_diff', 'host_advantage', 'wc_apps_diff'],
                'best_C': best_C, 'elapsed_s': time.time() - t0}
    save_artifact('attempt-023', metrics, run_info)
    return metrics


# ──────────────────────────────────────────────────────────────────────────────
# Attempt 024: Squad age features + WC Elo
# ──────────────────────────────────────────────────────────────────────────────
def attempt_024_squad_age():
    print("\n=== Attempt 024: Squad age profile + WC Elo ===")
    t0 = time.time()
    completed, teams, hist = load_base_data()
    squads = pd.read_csv(SQUADS_PATH)

    WC_DATE = pd.Timestamp('2026-06-11')  # approximate WC-2026 start

    def squad_age_features(team_id):
        tq = squads[squads['team_id'] == team_id].copy()
        if len(tq) == 0:
            return {'avg_age': 27.0, 'peak_ratio': 0.33, 'age_std': 3.0}
        dob = pd.to_datetime(tq['date_of_birth'], errors='coerce').dropna()
        ages = [(WC_DATE - d).days / 365.25 for d in dob]
        if not ages:
            return {'avg_age': 27.0, 'peak_ratio': 0.33, 'age_std': 3.0}
        avg_age = float(np.mean(ages))
        peak_ratio = float(np.mean([(24 <= a <= 29) for a in ages]))  # prime years
        age_std = float(np.std(ages))
        return {'avg_age': avg_age, 'peak_ratio': peak_ratio, 'age_std': age_std}

    team_map = {r['team_name']: r for _, r in teams.iterrows()}
    age_feats = {r['team_name']: squad_age_features(r['team_id']) for _, r in teams.iterrows()}

    rows = []
    y = []
    for _, m in completed.iterrows():
        h, a = m['home_team'], m['away_team']
        if h not in team_map or a not in team_map:
            continue
        ht, at = team_map[h], team_map[a]
        elo_diff = float(ht['elo_rating']) - float(at['elo_rating'])
        host_adv = 1.0 if h in HOST_TEAMS else 0.0
        hf, af = age_feats.get(h, {}), age_feats.get(a, {})
        age_diff = hf.get('avg_age', 27) - af.get('avg_age', 27)
        peak_diff = hf.get('peak_ratio', 0.33) - af.get('peak_ratio', 0.33)
        rows.append({
            'elo_diff': elo_diff,
            'host_advantage': host_adv,
            'age_diff': age_diff,
            'peak_ratio_diff': peak_diff,
        })
        hs, as_ = int(m['home_score']), int(m['away_score'])
        y.append('H' if hs > as_ else 'D' if hs == as_ else 'A')

    X = pd.DataFrame(rows)
    print(f"  Samples: {len(X)}, avg_age examples: {X['age_diff'].describe().to_dict()}")

    best_loss, best_C, best_res = np.inf, None, None
    for C in [0.05, 0.1, 0.3, 0.5, 1.0]:
        res, le = cv_train_eval(X, y, lambda c=C: LogisticRegression(C=c, max_iter=500, solver='saga'))
        sig = sig_test(res['fold_losses'])
        print(f"  C={C}: {res['mean']:.4f} ± {res['std']:.4f}  {sig['verdict']}")
        if res['mean'] < best_loss:
            best_loss, best_C, best_res = res['mean'], C, (res, le, sig)

    res, le, sig = best_res
    print(f"  Best C={best_C}: {res['mean']:.4f} ± {res['std']:.4f}  acc={res['accuracy']:.3f}  {sig['verdict']}")
    metrics = {**res, **sig, 'best_C': best_C}
    run_info = {'attempt': 24, 'description': 'Squad age profile + WC Elo',
                'features': ['elo_diff', 'host_advantage', 'age_diff', 'peak_ratio_diff'],
                'best_C': best_C, 'elapsed_s': time.time() - t0}
    save_artifact('attempt-024', metrics, run_info)
    return metrics


# ──────────────────────────────────────────────────────────────────────────────
# Attempt 025: Ensemble arithmetic mean of best FLAT models (019 + 020)
# ──────────────────────────────────────────────────────────────────────────────
def attempt_025_ensemble():
    """
    Re-run 019 (Elo only C=1.0) and 020 (kitchen sink C=0.05) to get OOF probs,
    then blend with weight w: w*019 + (1-w)*020.
    """
    print("\n=== Attempt 025: Ensemble of 019 + 020 OOF probs ===")
    t0 = time.time()
    completed, teams, hist = load_base_data()
    squads = pd.read_csv(SQUADS_PATH)

    # Build 019 features (Elo only)
    WC_DATE = pd.Timestamp('2026-06-11')
    team_map = {r['team_name']: r for _, r in teams.iterrows()}

    def squad_stats(team_id):
        tq = squads[squads['team_id'] == team_id]
        total_mv = tq['market_value_eur'].sum() if 'market_value_eur' in tq.columns else 1.0
        mean_caps = tq['caps'].mean() if 'caps' in tq.columns else 50.0
        top11_mv = tq.nlargest(11, 'market_value_eur')['market_value_eur'].sum() if 'market_value_eur' in tq.columns else 1.0
        goals = tq['goals'].sum() if 'goals' in tq.columns else 0.0
        dob = pd.to_datetime(tq['date_of_birth'], errors='coerce').dropna()
        ages = [(WC_DATE - d).days / 365.25 for d in dob]
        avg_age = float(np.mean(ages)) if ages else 27.0
        peak_ratio = float(np.mean([(24 <= a <= 29) for a in ages])) if ages else 0.33
        return total_mv, mean_caps, top11_mv, goals, avg_age, peak_ratio

    squad_cache = {r['team_name']: squad_stats(r['team_id']) for _, r in teams.iterrows()}

    rows_019, rows_020, y = [], [], []
    for _, m in completed.iterrows():
        h, a = m['home_team'], m['away_team']
        if h not in team_map or a not in team_map:
            continue
        ht, at = team_map[h], team_map[a]
        elo_diff = float(ht['elo_rating']) - float(at['elo_rating'])
        host_adv = 1.0 if h in HOST_TEAMS else 0.0
        rank_diff = float(at['fifa_ranking_pre_tournament']) - float(ht['fifa_ranking_pre_tournament'])

        # 019 features
        rows_019.append({'elo_diff': elo_diff, 'host_advantage': host_adv})

        # 020 kitchen sink features
        h_mv, h_caps, h_top11, h_goals, h_age, h_peak = squad_cache.get(h, (1e6, 50, 1e6, 0, 27, 0.33))
        a_mv, a_caps, a_top11, a_goals, a_age, a_peak = squad_cache.get(a, (1e6, 50, 1e6, 0, 27, 0.33))
        mv_diff = np.log1p(h_mv) - np.log1p(a_mv)
        mv_ratio = np.log1p(h_mv) / np.log1p(a_mv + 1)
        top11_diff = np.log1p(h_top11) - np.log1p(a_top11)
        caps_diff = h_caps - a_caps
        goals_diff = h_goals - a_goals
        age_diff = h_age - a_age
        peak_diff = h_peak - a_peak
        rows_020.append({
            'elo_diff': elo_diff, 'host_advantage': host_adv, 'rank_diff': rank_diff,
            'mv_diff': mv_diff, 'mv_ratio': mv_ratio, 'top11_diff': top11_diff,
            'caps_diff': caps_diff, 'goals_diff': goals_diff,
            'age_diff': age_diff, 'peak_diff': peak_diff,
        })
        hs, as_ = int(m['home_score']), int(m['away_score'])
        y.append('H' if hs > as_ else 'D' if hs == as_ else 'A')

    X019 = pd.DataFrame(rows_019)
    X020 = pd.DataFrame(rows_020)

    le = LabelEncoder()
    le.fit(y)
    y_enc = le.transform(y)
    rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=SEED)

    # Collect OOF probs for both models
    oof_019 = np.zeros((len(y), 3))
    oof_020 = np.zeros((len(y), 3))

    for train_idx, test_idx in rskf.split(X019, y_enc):
        # 019
        sc019 = StandardScaler()
        X_tr019 = sc019.fit_transform(X019.iloc[train_idx])
        X_te019 = sc019.transform(X019.iloc[test_idx])
        clf019 = LogisticRegression(C=1.0, max_iter=500, solver='saga')
        clf019.fit(X_tr019, y_enc[train_idx])
        oof_019[test_idx] = clf019.predict_proba(X_te019)

        # 020
        sc020 = StandardScaler()
        X_tr020 = sc020.fit_transform(X020.iloc[train_idx])
        X_te020 = sc020.transform(X020.iloc[test_idx])
        clf020 = LogisticRegression(C=0.05, max_iter=500, solver='saga')
        clf020.fit(X_tr020, y_enc[train_idx])
        oof_020[test_idx] = clf020.predict_proba(X_te020)

    best_loss, best_w, best_res = np.inf, None, None
    for w019 in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        blended = w019 * oof_019 + (1 - w019) * oof_020
        blended = blended / blended.sum(axis=1, keepdims=True)
        res = canonical_cv_eval_probs(blended, y, le)
        sig = sig_test(res['fold_losses'])
        print(f"  w(019)={w019:.1f}: {res['mean']:.4f} ± {res['std']:.4f}  {sig['verdict']}")
        if res['mean'] < best_loss:
            best_loss, best_w, best_res = res['mean'], w019, (res, sig)

    res, sig = best_res
    print(f"  Best w(019)={best_w}: {res['mean']:.4f} ± {res['std']:.4f}  acc={res['accuracy']:.3f}  {sig['verdict']}")
    metrics = {**res, **sig, 'best_w019': best_w}
    run_info = {'attempt': 25, 'description': 'Ensemble OOF probs: 019 (Elo C=1.0) + 020 (kitchen sink C=0.05)',
                'weight_019': best_w, 'weight_020': 1 - best_w, 'elapsed_s': time.time() - t0}
    save_artifact('attempt-025', metrics, run_info)
    return metrics


if __name__ == '__main__':
    print("=" * 70)
    print("Wave-3 Batch 5: Recent Form, Confederation, WC History, Age, Ensemble")
    print("=" * 70)

    results = {}
    results[21] = attempt_021_recent_form()
    results[22] = attempt_022_confederation()
    results[23] = attempt_023_wc_history()
    results[24] = attempt_024_squad_age()
    results[25] = attempt_025_ensemble()

    print("\n" + "=" * 70)
    print("SUMMARY TABLE Batch 5")
    print("=" * 70)
    labels = {
        21: "Recent form (18m) + Elo",
        22: "Confederation bias + Elo",
        23: "WC history count + Elo",
        24: "Squad age profile + Elo",
        25: "Ensemble 019+020",
    }
    for k, r in results.items():
        v = r.get('verdict', '?')
        print(f"  {k:03d} {labels[k]:30s}: {r['mean']:.4f} ± {r['std']:.4f}  acc={r['accuracy']:.3f}  {v}")
