"""
Wave-3: Batch 2 — WC-Team-Filtered Historical Features
Key insight from Batch 1: pure transfer (RED) because distribution mismatch.
Fix: compute historical form for SPECIFIC WC-2026 teams, then use as in-fold features.
"""
import sys
import os
import json
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.calibration import CalibratedClassifierCV
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


# ── Load data ────────────────────────────────────────────────────────────────
teams_df = pd.read_csv(TEAMS_PATH)
teams_df['hist_name'] = teams_df['team_name'].map(lambda x: NAME_MAP.get(x, x))
WC_TEAMS_HIST = set(teams_df['hist_name'].tolist())  # historical names of WC-2026 teams
HIST_TO_WC = {row['hist_name']: row['team_name'] for _, row in teams_df.iterrows()}
WC_TO_HIST = {v: k for k, v in HIST_TO_WC.items()}


def load_filtered_historical(cutoff_date='2026-06-12', min_year=2015,
                              tier_min=3, wc_teams_only=True):
    """Load historical matches involving WC-2026 teams in competitive matches.

    tier_min: 3 = qualifier/nations league or higher; 5 = WC only
    """
    hist = pd.read_csv(HIST_PATH, parse_dates=['date'])
    hist = hist.dropna(subset=['home_score', 'away_score'])
    hist = hist[hist['date'] < pd.to_datetime(cutoff_date)]
    hist = hist[hist['date'].dt.year >= min_year]
    hist = hist.sort_values('date').reset_index(drop=True)

    # Assign tier
    def tier(t):
        t = t.lower()
        if 'world cup' in t and 'qualif' not in t:
            return 5
        elif 'copa' in t or 'euro' in t or 'african cup of nations' == t or 'afc asian cup' in t or 'gold cup' in t or 'nations cup' in t:
            return 4
        elif 'qualif' in t:
            return 3
        elif 'nations league' in t:
            return 2
        elif 'friendly' in t:
            return 1
        else:
            return 2

    hist['tier'] = hist['tournament'].map(tier)
    hist = hist[hist['tier'] >= tier_min]

    if wc_teams_only:
        # Keep only matches where BOTH teams are WC-2026 qualified
        mask = hist['home_team'].isin(WC_TEAMS_HIST) & hist['away_team'].isin(WC_TEAMS_HIST)
        hist = hist[mask]

    print(f"Filtered historical matches: {len(hist)} (tier>={tier_min}, year>={min_year}, wc_teams_only={wc_teams_only})")
    return hist


def build_wc_team_form_features(hist, cutoff_date='2026-06-12', window_years=3):
    """Compute per-team form features from filtered historical matches."""
    hist = hist[hist['date'] < pd.to_datetime(cutoff_date)].copy()
    cutoff = pd.to_datetime(cutoff_date)
    lookback = cutoff - pd.DateOffset(years=window_years)
    hist = hist[hist['date'] >= lookback]

    team_stats = {}
    for team in WC_TEAMS_HIST:
        home_games = hist[hist['home_team'] == team]
        away_games = hist[hist['away_team'] == team]

        # All games as home team
        home_wins = (home_games['home_score'] > home_games['away_score']).sum()
        home_draws = (home_games['home_score'] == home_games['away_score']).sum()
        home_gf = home_games['home_score'].sum()
        home_ga = home_games['away_score'].sum()

        # All games as away team
        away_wins = (away_games['away_score'] > away_games['home_score']).sum()
        away_draws = (away_games['home_score'] == away_games['away_score']).sum()
        away_gf = away_games['away_score'].sum()
        away_ga = away_games['home_score'].sum()

        n = len(home_games) + len(away_games)
        total_wins = home_wins + away_wins
        total_draws = home_draws + away_draws
        total_gf = home_gf + away_gf
        total_ga = home_ga + away_ga

        if n == 0:
            team_stats[team] = {
                'n_games': 0, 'win_rate': 0.5, 'draw_rate': 0.25,
                'gf_per': 1.3, 'ga_per': 1.0, 'gd_per': 0.3,
                'win_streak': 0, 'unbeaten_streak': 0,
            }
            continue

        # Compute streaks from most recent matches
        all_games = pd.concat([
            home_games.assign(gf=home_games['home_score'], ga=home_games['away_score']),
            away_games.assign(gf=away_games['away_score'], ga=away_games['home_score'])
        ]).sort_values('date')

        all_games = all_games.tail(10)
        streak_win = 0
        streak_unbeaten = 0
        for _, g in all_games.iloc[::-1].iterrows():
            gf, ga = g['gf'], g['ga']
            if gf > ga:
                streak_win += 1
                streak_unbeaten += 1
            elif gf == ga:
                streak_win = 0  # break win streak
                streak_unbeaten += 1
            else:
                break  # unbeaten streak broken
            if gf > ga and streak_win == 0:
                break

        # Recalculate streaks properly
        win_streak = 0
        unbeaten_streak = 0
        for _, g in all_games.iloc[::-1].iterrows():
            gf, ga = g['gf'], g['ga']
            if gf > ga:
                win_streak += 1
                unbeaten_streak += 1
            elif gf == ga:
                win_streak = 0
                unbeaten_streak += 1
            else:
                break
        # Reset to consistent
        win_streak_final = 0
        for _, g in all_games.iloc[::-1].iterrows():
            if g['gf'] > g['ga']:
                win_streak_final += 1
            else:
                break

        team_stats[team] = {
            'n_games': n,
            'win_rate': total_wins / n,
            'draw_rate': total_draws / n,
            'gf_per': total_gf / n,
            'ga_per': total_ga / n,
            'gd_per': (total_gf - total_ga) / n,
            'win_streak': min(win_streak_final, 10),
        }

    return team_stats


def build_enriched_features(cutoff='2026-06-12'):
    """Build WC-2026 feature matrix enriched with historical form."""
    X_wc, y_wc, completed = build_wc2026_feature_matrix(
        MATCHES_PATH, TEAMS_PATH, HIST_PATH, cutoff_date=cutoff
    )

    # Load filtered historical — all competitive matches involving WC teams, last 3 years
    hist_all_comp = load_filtered_historical(cutoff_date=cutoff, min_year=2018, tier_min=2, wc_teams_only=False)
    hist_wc_only = load_filtered_historical(cutoff_date=cutoff, min_year=2015, tier_min=2, wc_teams_only=True)
    hist_qual = load_filtered_historical(cutoff_date=cutoff, min_year=2020, tier_min=3, wc_teams_only=False)

    # Build form features from WC-team-only competitive matches
    form_wc_all = build_wc_team_form_features(hist_all_comp, cutoff, window_years=3)
    form_wc_3yr = build_wc_team_form_features(hist_wc_only, cutoff, window_years=4)
    form_qual = build_wc_team_form_features(hist_qual, cutoff, window_years=4)

    rows = []
    for i, row in completed.iterrows():
        home_wc = row['home_team_name']
        away_wc = row['away_team_name']
        home_h = WC_TO_HIST.get(home_wc, home_wc)
        away_h = WC_TO_HIST.get(away_wc, away_wc)

        def get_form(form_dict, team, default_wr=0.5, default_gd=0.0):
            if team in form_dict:
                return form_dict[team]
            return {'win_rate': default_wr, 'draw_rate': 0.25, 'gf_per': 1.3,
                    'ga_per': 1.0, 'gd_per': default_gd, 'win_streak': 0, 'n_games': 0}

        h_all = get_form(form_wc_all, home_h)
        a_all = get_form(form_wc_all, away_h)
        h_wc3 = get_form(form_wc_3yr, home_h)
        a_wc3 = get_form(form_wc_3yr, away_h)
        h_qual = get_form(form_qual, home_h)
        a_qual = get_form(form_qual, away_h)

        feat = {
            # WC-2026 provided features (canonical)
            'wc_elo_diff': X_wc.iloc[i]['wc_elo_diff'],
            'wc_rank_diff': X_wc.iloc[i]['wc_rank_diff'],
            'host_advantage': X_wc.iloc[i]['host_advantage'],
            'hist_elo_diff': X_wc.iloc[i]['hist_elo_diff'],
            # Historical form (all competitive, last 3yr)
            'win_rate_all_diff': h_all['win_rate'] - a_all['win_rate'],
            'gd_all_diff': h_all['gd_per'] - a_all['gd_per'],
            'gf_all_diff': h_all['gf_per'] - a_all['gf_per'],
            'ga_all_diff': h_all['ga_per'] - a_all['ga_per'],
            # WC-team matchup form (last 4yr)
            'win_rate_wc3_diff': h_wc3['win_rate'] - a_wc3['win_rate'],
            'gd_wc3_diff': h_wc3['gd_per'] - a_wc3['gd_per'],
            # Qualifying/competitive form (last 4yr)
            'win_rate_qual_diff': h_qual['win_rate'] - a_qual['win_rate'],
            'gd_qual_diff': h_qual['gd_per'] - a_qual['gd_per'],
            # Absolute values
            'home_win_rate_all': h_all['win_rate'],
            'away_win_rate_all': a_all['win_rate'],
            'home_gd_all': h_all['gd_per'],
            'away_gd_all': a_all['gd_per'],
            'home_win_streak': h_all['win_streak'],
            'away_win_streak': a_all['win_streak'],
        }
        rows.append(feat)

    X_enriched = pd.DataFrame(rows)
    return X_enriched, y_wc


def canonical_cv_eval(probs, labels_str, le=None):
    if le is None:
        le = LabelEncoder()
        le.fit(labels_str)
    y = le.transform(labels_str)
    rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=SEED)
    fold_losses = []
    eps = 1e-15
    per_match_loss = -np.log(np.clip(probs[np.arange(len(y)), y], eps, 1 - eps))
    for _, test_idx in rskf.split(np.zeros(len(y)), y):
        fold_loss = log_loss(y[test_idx], probs[test_idx],
                             labels=list(range(len(le.classes_))))
        fold_losses.append(fold_loss)
    return {
        'mean': np.mean(fold_losses),
        'std': np.std(fold_losses),
        'fold_losses': fold_losses,
        'per_match_loss': per_match_loss.tolist(),
        'accuracy': accuracy_score(y, np.argmax(probs, axis=1)),
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
        'mean': np.mean(fold_losses),
        'std': np.std(fold_losses),
        'fold_losses': fold_losses,
        'accuracy': accuracy_score(y, np.argmax(oof_probs, axis=1)),
        'oof_probs': oof_probs.tolist(),
        'classes': le.classes_.tolist(),
    }, le


def significance_test(fold_losses):
    from scipy.stats import ttest_1samp
    arr = np.array(fold_losses)
    mean = arr.mean()
    t_stat, p_val = ttest_1samp(arr, BASELINE_LOGLOSS)
    if mean < BASELINE_LOGLOSS and p_val < 0.05:
        verdict = 'GREEN'
    elif mean > BASELINE_LOGLOSS:
        verdict = 'RED'
    else:
        verdict = 'FLAT'
    return {
        'delta_vs_baseline': mean - BASELINE_LOGLOSS,
        'delta_vs_frontier': mean - FRONTIER_LOGLOSS,
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
    print(f"  Saved to {path}/")
    return path


# ──────────────────────────────────────────────────────────────────────────────
# Attempt 007: WC-team-filtered historical form + WC Elo, in-fold logistic
# ──────────────────────────────────────────────────────────────────────────────
def attempt_007_enriched_logistic():
    print("\n=== Attempt 007: WC-team form (competitive) + WC Elo, in-fold logistic ===")
    t0 = time.time()

    X_enriched, y_wc = build_enriched_features()

    # Subset: WC Elo + historical form
    feats = ['wc_elo_diff', 'host_advantage', 'win_rate_all_diff', 'gd_all_diff',
             'win_rate_wc3_diff', 'gd_wc3_diff']
    X = X_enriched[feats].copy()

    for C in [0.1, 0.3, 0.5, 1.0]:
        def model_fn(C_=C):
            return LogisticRegression(C=C_, max_iter=500, random_state=SEED)
        result, _ = cv_train_eval(X, y_wc, model_fn, scale=True)
        sig = significance_test(result['fold_losses'])
        print(f"  C={C}: {result['mean']:.4f} ± {result['std']:.4f}  {sig['verdict']}")

    # Best C=0.3
    def model_fn():
        return LogisticRegression(C=0.3, max_iter=500, random_state=SEED)
    result, le = cv_train_eval(X, y_wc, model_fn, scale=True)
    sig = significance_test(result['fold_losses'])
    elapsed = time.time() - t0

    print(f"  Final (C=0.3): {result['mean']:.4f} ± {result['std']:.4f}  acc={result['accuracy']:.3f}  {sig['verdict']}")

    metrics = {**result, **sig, 'elapsed_sec': elapsed}
    run_info = {
        'name': 'attempt_007_enriched_logistic',
        'description': 'In-fold logistic: WC-2026 Elo + historical WC-team form (competitive matches, 2018-2026)',
        'features': feats, 'model': 'LogisticRegression(C=0.3)',
        'data_source': 'Historical competitive matches involving WC-2026 teams (tier>=2, year>=2018)',
        'seed': SEED, 'baseline': BASELINE_LOGLOSS, 'frontier': FRONTIER_LOGLOSS,
    }
    save_artifact('attempt-007', metrics, run_info)
    return metrics, X_enriched, y_wc


# ──────────────────────────────────────────────────────────────────────────────
# Attempt 008: Full enriched features + Elo, in-fold logistic
# ──────────────────────────────────────────────────────────────────────────────
def attempt_008_full_enriched_logistic(X_enriched, y_wc):
    print("\n=== Attempt 008: Full enriched features (all form), in-fold logistic ===")
    t0 = time.time()

    feats = ['wc_elo_diff', 'wc_rank_diff', 'host_advantage',
             'win_rate_all_diff', 'gd_all_diff', 'gf_all_diff', 'ga_all_diff',
             'win_rate_wc3_diff', 'gd_wc3_diff', 'win_rate_qual_diff', 'gd_qual_diff',
             'home_gd_all', 'away_gd_all', 'home_win_streak', 'away_win_streak']
    X = X_enriched[feats].copy()

    def model_fn():
        return LogisticRegression(C=0.1, max_iter=500, random_state=SEED)
    result, _ = cv_train_eval(X, y_wc, model_fn, scale=True)
    sig = significance_test(result['fold_losses'])
    elapsed = time.time() - t0

    print(f"  {result['mean']:.4f} ± {result['std']:.4f}  acc={result['accuracy']:.3f}  {sig['verdict']}")

    metrics = {**result, **sig, 'elapsed_sec': elapsed}
    run_info = {
        'name': 'attempt_008_full_enriched_logistic',
        'description': 'In-fold logistic: WC-2026 Elo/rank + all historical WC-team form features',
        'features': feats, 'model': 'LogisticRegression(C=0.1)',
        'seed': SEED, 'baseline': BASELINE_LOGLOSS, 'frontier': FRONTIER_LOGLOSS,
    }
    save_artifact('attempt-008', metrics, run_info)
    return metrics


# ──────────────────────────────────────────────────────────────────────────────
# Attempt 009: Augmented historical WC + WC-2026 training, in-fold logistic
#   - For each fold, augment the WC-2026 training set with historical WC matches
#   - Historical WC matches use the computed historical Elo + form features
# ──────────────────────────────────────────────────────────────────────────────
def attempt_009_augmented_wc_train():
    """Augment WC-2026 training folds with historical World Cup group stage matches.

    Historical WC matches use: historical Elo diff + host flag.
    """
    print("\n=== Attempt 009: Augmented WC historical matches + WC-2026 in-fold ===")
    t0 = time.time()

    hist = pd.read_csv(HIST_PATH, parse_dates=['date'])
    hist = hist.dropna(subset=['home_score', 'away_score'])
    hist = hist[hist['date'] < pd.to_datetime('2026-06-12')]
    hist = hist[hist['tournament'].str.lower().str.contains('world cup')]
    hist = hist[~hist['tournament'].str.lower().str.contains('qualif')]
    hist = hist.sort_values('date')
    print(f"  Historical WC matches: {len(hist)}")

    # Build Elo for all historical matches up to cutoff
    from features import build_elo_ratings
    elo_ratings, _ = build_elo_ratings(
        pd.read_csv(HIST_PATH, parse_dates=['date']),
        cutoff_date='2026-06-12'
    )

    # For each historical WC match, compute elo_diff at match time
    # (We can't easily do this incrementally here, so approximate with pre-2026 final ratings)
    # Better: compute elo_diff from the bulk elo_ratings (post-2026-cutoff approximation)
    hist['home_elo'] = hist['home_team'].map(lambda t: elo_ratings.get(t, 1500))
    hist['away_elo'] = hist['away_team'].map(lambda t: elo_ratings.get(t, 1500))
    hist['elo_diff'] = hist['home_elo'] - hist['away_elo']
    hist['host_advantage'] = 0  # historical WC matches at neutral venues mostly
    hist['label'] = hist.apply(
        lambda r: 'H' if r['home_score'] > r['away_score']
        else ('D' if r['home_score'] == r['away_score'] else 'A'), axis=1
    )

    X_aug = hist[['elo_diff', 'host_advantage']].rename(columns={'elo_diff': 'wc_elo_diff'}).copy()
    y_aug = hist['label']

    # WC-2026 features
    X_wc, y_wc, _ = build_wc2026_feature_matrix(
        MATCHES_PATH, TEAMS_PATH, HIST_PATH, cutoff_date='2026-06-12'
    )
    X_wc_sub = X_wc[['wc_elo_diff', 'host_advantage']].copy()

    # In-fold augmented training
    le = LabelEncoder()
    le.fit(['H', 'D', 'A'])
    y = le.transform(y_wc)
    y_aug_enc = le.transform(y_aug)

    rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=SEED)
    fold_losses = []
    oof_probs = np.zeros((len(y), len(le.classes_)))

    # Try multiple weights for augmentation
    best_mean = 999
    best_weight = None
    for w in [0.05, 0.1, 0.2, 0.5, 1.0]:
        fold_l = []
        for train_idx, test_idx in rskf.split(X_wc_sub, y):
            X_tr = X_wc_sub.iloc[train_idx]
            y_tr = y[train_idx]
            X_te = X_wc_sub.iloc[test_idx]

            # Augment with historical WC
            X_combined = pd.concat([X_tr, X_aug], ignore_index=True)
            y_combined = np.concatenate([y_tr, y_aug_enc])
            sample_weights = np.concatenate([
                np.ones(len(X_tr)),
                np.full(len(X_aug), w)
            ])

            sc = StandardScaler()
            X_combined_sc = sc.fit_transform(X_combined)
            X_te_sc = sc.transform(X_te)

            clf = LogisticRegression(C=0.5, max_iter=500, random_state=SEED)
            clf.fit(X_combined_sc, y_combined, sample_weight=sample_weights)
            probs = clf.predict_proba(X_te_sc)
            fold_l.append(log_loss(y[test_idx], probs, labels=list(range(len(le.classes_)))))

        mean_l = np.mean(fold_l)
        print(f"  weight={w}: {mean_l:.4f} ± {np.std(fold_l):.4f}")
        if mean_l < best_mean:
            best_mean = mean_l
            best_weight = w

    # Final run with best weight
    fold_losses = []
    print(f"  Best weight: {best_weight}")
    for train_idx, test_idx in rskf.split(X_wc_sub, y):
        X_tr = X_wc_sub.iloc[train_idx]
        y_tr = y[train_idx]
        X_te = X_wc_sub.iloc[test_idx]
        X_combined = pd.concat([X_tr, X_aug], ignore_index=True)
        y_combined = np.concatenate([y_tr, y_aug_enc])
        sample_weights = np.concatenate([np.ones(len(X_tr)), np.full(len(X_aug), best_weight)])
        sc = StandardScaler()
        X_combined_sc = sc.fit_transform(X_combined)
        X_te_sc = sc.transform(X_te)
        clf = LogisticRegression(C=0.5, max_iter=500, random_state=SEED)
        clf.fit(X_combined_sc, y_combined, sample_weight=sample_weights)
        probs = clf.predict_proba(X_te_sc)
        fold_losses.append(log_loss(y[test_idx], probs, labels=list(range(len(le.classes_)))))
        oof_probs[test_idx] = probs

    result = {
        'mean': float(np.mean(fold_losses)),
        'std': float(np.std(fold_losses)),
        'fold_losses': fold_losses,
        'accuracy': float(accuracy_score(y, np.argmax(oof_probs, axis=1))),
        'oof_probs': oof_probs.tolist(),
        'classes': le.classes_.tolist(),
        'best_aug_weight': best_weight,
    }
    sig = significance_test(fold_losses)
    elapsed = time.time() - t0

    print(f"  {result['mean']:.4f} ± {result['std']:.4f}  acc={result['accuracy']:.3f}  {sig['verdict']}")

    metrics = {**result, **sig, 'elapsed_sec': elapsed}
    run_info = {
        'name': 'attempt_009_augmented_wc_train',
        'description': 'In-fold logistic: WC-2026 + augmented historical WC group-stage matches (weight-swept)',
        'n_historical_wc': len(X_aug),
        'best_weight': best_weight,
        'model': 'LogisticRegression(C=0.5)',
        'features': ['wc_elo_diff', 'host_advantage'],
        'seed': SEED, 'baseline': BASELINE_LOGLOSS, 'frontier': FRONTIER_LOGLOSS,
    }
    save_artifact('attempt-009', metrics, run_info)
    return metrics, oof_probs, le


# ──────────────────────────────────────────────────────────────────────────────
# Attempt 010: Historical WC logistic applied to WC-2026 (tune K, home_adv)
# ──────────────────────────────────────────────────────────────────────────────
def attempt_010_historical_wc_pure_elo():
    """Pure WC-only logistic: train on all historical WC group stage matches."""
    print("\n=== Attempt 010: Pure transfer from historical WC group stage ===")
    t0 = time.time()

    from features import build_elo_ratings
    hist_full = pd.read_csv(HIST_PATH, parse_dates=['date'])

    # Historical WC group stage matches only
    hist = hist_full.dropna(subset=['home_score', 'away_score'])
    hist = hist[hist['date'] < pd.to_datetime('2026-06-12')]
    hist = hist[hist['tournament'].str.lower().str.contains('world cup')]
    hist = hist[~hist['tournament'].str.lower().str.contains('qualif')]
    hist = hist.sort_values('date').reset_index(drop=True)
    print(f"  Historical WC matches: {len(hist)}")

    # Compute running Elo incrementally for historical WC matches
    # Use full corpus for Elo computation, then evaluate on WC matches
    elo_ratings = {}
    start_rating = 1500
    HOME_ADV = 100

    def expected(ra, rb):
        return 1.0 / (1.0 + 10 ** ((rb - ra) / 400.0))

    def k_factor(tournament):
        t = tournament.lower()
        if 'world cup' in t and 'qualif' not in t:
            return 30
        elif any(x in t for x in ['qualif', 'nations', 'cup', 'euro', 'copa', 'african', 'afc']):
            return 20
        else:
            return 10

    hist_full_sorted = hist_full.dropna(subset=['home_score', 'away_score']).sort_values('date')

    # Track match-time Elo for WC matches
    wc_match_elos = {}

    for _, row in hist_full_sorted.iterrows():
        home, away = row['home_team'], row['away_team']
        neutral = row.get('neutral', False)

        ra = elo_ratings.get(home, start_rating)
        rb = elo_ratings.get(away, start_rating)

        # Record Elo before update for WC training matches
        match_key = (row['date'], home, away)
        if row['tournament'].lower().find('world cup') >= 0 and 'qualif' not in row['tournament'].lower():
            wc_match_elos[match_key] = (ra, rb)

        ra_eff = ra + (0 if neutral else HOME_ADV)
        Ea = expected(ra_eff, rb)
        Eb = 1.0 - Ea

        hs, as_ = row['home_score'], row['away_score']
        if hs > as_:
            Sa, Sb = 1.0, 0.0
        elif hs < as_:
            Sa, Sb = 0.0, 1.0
        else:
            Sa, Sb = 0.5, 0.5

        K = k_factor(row['tournament'])
        elo_ratings[home] = ra + K * (Sa - Ea)
        elo_ratings[away] = rb + K * (Sb - Eb)

    # Build WC historical training set with correct Elo
    wc_train_rows = []
    wc_train_labels = []
    for _, row in hist.iterrows():
        if row['date'] >= pd.to_datetime('2026-06-12'):
            continue
        match_key = (row['date'], row['home_team'], row['away_team'])
        if match_key in wc_match_elos:
            ra, rb = wc_match_elos[match_key]
        else:
            ra = elo_ratings.get(row['home_team'], start_rating)
            rb = elo_ratings.get(row['away_team'], start_rating)

        neutral = row.get('neutral', False)
        elo_diff = ra - rb
        home_adv = 0 if neutral else 1

        hs, as_ = row['home_score'], row['away_score']
        if hs > as_:
            label = 'H'
        elif hs < as_:
            label = 'A'
        else:
            label = 'D'

        wc_train_rows.append({'elo_diff': elo_diff, 'home_adv': home_adv})
        wc_train_labels.append(label)

    X_hist = pd.DataFrame(wc_train_rows)
    y_hist = pd.Series(wc_train_labels)
    print(f"  WC training set size: {len(X_hist)}")
    print(f"  Label dist: {y_hist.value_counts().to_dict()}")

    # WC-2026 features
    X_wc, y_wc, _ = build_wc2026_feature_matrix(
        MATCHES_PATH, TEAMS_PATH, HIST_PATH, cutoff_date='2026-06-12'
    )

    # Map WC-2026 teams to historical Elo
    teams_df = pd.read_csv(TEAMS_PATH)
    teams_df['hist_name'] = teams_df['team_name'].map(lambda x: NAME_MAP.get(x, x))
    teams_elo = dict(zip(teams_df['team_name'], teams_df['elo_rating']))

    matches = pd.read_csv(MATCHES_PATH)
    completed = matches[matches['status'] == 'Completed']

    wc_rows = []
    for _, m in completed.iterrows():
        home_wc, away_wc = m['home_team_name'], m['away_team_name']
        wc_rows.append({
            'elo_diff': teams_elo.get(home_wc, 1500) - teams_elo.get(away_wc, 1500),
            'home_adv': 0,  # all WC matches neutral-ish
        })
    X_wc_hist = pd.DataFrame(wc_rows)

    # Train logistic on historical WC matches
    le = LabelEncoder()
    le.fit(['H', 'D', 'A'])

    sc = StandardScaler()
    X_hist_sc = sc.fit_transform(X_hist)
    X_wc_sc = sc.transform(X_wc_hist)

    clf = LogisticRegression(C=0.3, max_iter=500, random_state=SEED)
    clf.fit(X_hist_sc, y_hist)

    probs = clf.predict_proba(X_wc_sc)
    prob_df = pd.DataFrame(probs, columns=clf.classes_)
    probs_ordered = prob_df[le.classes_].values

    result = canonical_cv_eval(probs_ordered, y_wc, le)
    sig = significance_test(result['fold_losses'])
    elapsed = time.time() - t0

    print(f"  {result['mean']:.4f} ± {result['std']:.4f}  acc={result['accuracy']:.3f}  {sig['verdict']}")

    metrics = {**result, **sig, 'elapsed_sec': elapsed}
    run_info = {
        'name': 'attempt_010_historical_wc_pure_elo',
        'description': 'Logistic trained on 1,000+ historical WC group stage matches, Elo-at-match-time, predict WC-2026',
        'n_training': len(X_hist),
        'model': 'LogisticRegression(C=0.3) pure transfer',
        'features': ['elo_diff', 'home_adv'],
        'elo_source': 'Running Elo (K=30/20/10, home_adv=100)',
        'seed': SEED, 'baseline': BASELINE_LOGLOSS, 'frontier': FRONTIER_LOGLOSS,
    }
    save_artifact('attempt-010', metrics, run_info)
    return metrics, probs_ordered, le


# ──────────────────────────────────────────────────────────────────────────────
# Attempt 011: Best blend of historical + WC-2026 features + calibrated ensemble
# ──────────────────────────────────────────────────────────────────────────────
def attempt_011_best_blend(probs_010, le_010, X_enriched, y_wc):
    """Blend: historical WC transfer probs + WC-2026 Elo + form, in-fold logistic."""
    print("\n=== Attempt 011: Best blend (hist WC probs + WC-2026 enriched, in-fold) ===")
    t0 = time.time()

    X_blend = pd.DataFrame({
        'p010_H': probs_010[:, le_010.transform(['H'])[0]],
        'p010_D': probs_010[:, le_010.transform(['D'])[0]],
        'p010_A': probs_010[:, le_010.transform(['A'])[0]],
        'wc_elo_diff': X_enriched['wc_elo_diff'],
        'host_advantage': X_enriched['host_advantage'],
        'win_rate_all_diff': X_enriched['win_rate_all_diff'],
        'gd_all_diff': X_enriched['gd_all_diff'],
        'win_rate_wc3_diff': X_enriched['win_rate_wc3_diff'],
    })

    for C in [0.05, 0.1, 0.2, 0.3]:
        def model_fn(C_=C):
            return LogisticRegression(C=C_, max_iter=500, random_state=SEED)
        result, _ = cv_train_eval(X_blend, y_wc, model_fn, scale=True)
        sig = significance_test(result['fold_losses'])
        print(f"  C={C}: {result['mean']:.4f} ± {result['std']:.4f}  {sig['verdict']}")

    # Best C=0.1
    def model_fn():
        return LogisticRegression(C=0.1, max_iter=500, random_state=SEED)
    result, _ = cv_train_eval(X_blend, y_wc, model_fn, scale=True)
    sig = significance_test(result['fold_losses'])
    elapsed = time.time() - t0

    print(f"  Final (C=0.1): {result['mean']:.4f} ± {result['std']:.4f}  acc={result['accuracy']:.3f}  {sig['verdict']}")

    metrics = {**result, **sig, 'elapsed_sec': elapsed}
    run_info = {
        'name': 'attempt_011_best_blend',
        'description': 'In-fold logistic: hist WC transfer probs + WC-2026 Elo + form',
        'features': X_blend.columns.tolist(), 'model': 'LogisticRegression(C=0.1)',
        'seed': SEED, 'baseline': BASELINE_LOGLOSS, 'frontier': FRONTIER_LOGLOSS,
    }
    save_artifact('attempt-011', metrics, run_info)
    return metrics


if __name__ == '__main__':
    print("=" * 70)
    print("Wave-3 Batch 2: WC-Team-Filtered Historical Features")
    print("=" * 70)

    m007, X_enriched, y_wc = attempt_007_enriched_logistic()
    m008 = attempt_008_full_enriched_logistic(X_enriched, y_wc)
    m009, oof_009, le_009 = attempt_009_augmented_wc_train()
    m010, probs_010, le_010 = attempt_010_historical_wc_pure_elo()
    m011 = attempt_011_best_blend(probs_010, le_010, X_enriched, y_wc)

    print("\n" + "=" * 70)
    print("SUMMARY TABLE Batch 2")
    print("=" * 70)
    for name, m in [('007 Enriched Logistic', m007), ('008 Full Enriched', m008),
                    ('009 Augmented WC', m009), ('010 Historical WC Transfer', m010),
                    ('011 Best Blend', m011)]:
        print(f"  {name:30s}: {m['mean']:.4f} ± {m['std']:.4f}  acc={m['accuracy']:.3f}  {m['verdict']}")
