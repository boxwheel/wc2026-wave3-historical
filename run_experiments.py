"""
Wave-3: Full-Scale Historical Training — Experiment Runner
All experiments follow canonical eval: RepeatedStratifiedKFold(5x10, seed=0)
"""
import sys
import os
import json
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import log_loss, accuracy_score
from scipy.stats import wilcoxon, ttest_rel
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(__file__))
from features import (
    build_wc2026_feature_matrix,
    build_historical_training_set,
)

DATA_DIR = '/home/user/research/wave3-historical/data'
MATCHES_PATH = f'{DATA_DIR}/wc2026-trees-study-main/fifa_data/matches_detailed.csv'
TEAMS_PATH = f'{DATA_DIR}/wc2026-trees-study-main/fifa_data/teams.csv'
HIST_PATH = f'{DATA_DIR}/historical/results.csv'
ARTIFACTS_DIR = '/home/user/research/wave3-historical/artifacts'
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

BASELINE_LOGLOSS = 0.8337
FRONTIER_LOGLOSS = 0.7608
SEED = 0


def canonical_cv_eval(probs, labels_str, le=None):
    """Run RepeatedStratifiedKFold(5x10) eval on fixed probability predictions.

    For pure-transfer models (fixed predictions), just compute fold-level statistics.
    Returns: mean, std, per_fold_losses, per_match_losses
    """
    if le is None:
        le = LabelEncoder()
        le.fit(labels_str)

    y = le.transform(labels_str)

    rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=SEED)

    fold_losses = []
    # Compute per-match losses manually
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


def cv_train_eval(X_wc, y_wc, model_fn, scale=True, meta_features=None):
    """CV evaluation for models that TRAIN inside each fold on WC data.

    meta_features: optional additional fixed features to append to X_wc in each fold.
    Returns: mean, std, fold_losses, oof_probs
    """
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

        # Only first 50 folds contribute to OOF (10 repeats × 5 folds — last assignment wins)
        oof_probs[test_idx] = probs

    return {
        'mean': np.mean(fold_losses),
        'std': np.std(fold_losses),
        'fold_losses': fold_losses,
        'accuracy': accuracy_score(y, np.argmax(oof_probs, axis=1)),
        'oof_probs': oof_probs.tolist(),
        'classes': le.classes_.tolist(),
    }, le


def significance_test(fold_losses, baseline_name='Elo-logistic (0.8337)'):
    """Paired t-test and Wilcoxon against known baseline fold losses.
    Since baseline fold losses aren't available, report vs scalar."""
    arr = np.array(fold_losses)
    mean = arr.mean()
    delta = mean - BASELINE_LOGLOSS
    vs_frontier = mean - FRONTIER_LOGLOSS

    # Can't do paired test without baseline fold losses, so report t-test on fold means
    from scipy.stats import ttest_1samp
    t_stat, p_val = ttest_1samp(arr, BASELINE_LOGLOSS)

    if mean < BASELINE_LOGLOSS and p_val < 0.05:
        verdict = 'GREEN'
    elif mean > BASELINE_LOGLOSS:
        verdict = 'RED'
    else:
        verdict = 'FLAT'

    return {
        'delta_vs_baseline': delta,
        'delta_vs_frontier': vs_frontier,
        'p_value_ttest': p_val,
        't_stat': t_stat,
        'verdict': verdict,
    }


def save_artifact(name, metrics, run_info):
    path = os.path.join(ARTIFACTS_DIR, name)
    os.makedirs(path, exist_ok=True)
    with open(f'{path}/metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    with open(f'{path}/run.json', 'w') as f:
        json.dump(run_info, f, indent=2)
    print(f"  Saved artifacts to {path}/")
    return path


# ─────────────────────────────────────────────────────────────────────────────
# ATTEMPT 001: Pure Transfer — Logistic fit on full historical corpus
# ─────────────────────────────────────────────────────────────────────────────
def attempt_001_pure_transfer_logistic():
    """Train logistic on 49K historical matches, predict 64 WC-2026 matches."""
    print("\n=== Attempt 001: Pure Transfer Logistic (full historical corpus) ===")
    t0 = time.time()

    # Build historical training set (features computed incrementally)
    X_hist, y_hist, w_hist, final_elos = build_historical_training_set(
        HIST_PATH, cutoff_date='2026-06-12', min_year=1990, tier_weight=False
    )

    # Build WC-2026 feature matrix with same features
    X_wc, y_wc, completed = build_wc2026_feature_matrix(
        MATCHES_PATH, TEAMS_PATH, HIST_PATH, cutoff_date='2026-06-12'
    )

    # Align features — use intersection
    hist_features = ['elo_diff', 'home_adv', 'win_rate_10_diff', 'gd_10_diff',
                     'win_rate_5_diff', 'gd_5_diff']
    wc_features = ['hist_elo_diff', 'home_is_host', 'win_rate_10_diff', 'gd_10_diff',
                   'win_rate_5_diff', 'gd_5_diff']

    X_hist_sub = X_hist[hist_features].copy()
    X_wc_sub = X_wc[wc_features].copy()
    X_wc_sub.columns = hist_features  # rename to match

    # Train on historical corpus
    le = LabelEncoder()
    le.fit(['H', 'D', 'A'])

    sc = StandardScaler()
    X_hist_scaled = sc.fit_transform(X_hist_sub)
    X_wc_scaled = sc.transform(X_wc_sub)

    clf = LogisticRegression(C=0.5, max_iter=1000, random_state=SEED)
    clf.fit(X_hist_scaled, y_hist)

    # Fixed predictions on WC-2026
    probs = clf.predict_proba(X_wc_scaled)
    # Align with le order (H, D, A) — sklearn may reorder
    prob_df = pd.DataFrame(probs, columns=clf.classes_)
    probs_ordered = prob_df[le.classes_].values

    # Canonical CV eval (fixed predictions, fold-level stats)
    result = canonical_cv_eval(probs_ordered, y_wc, le)
    sig = significance_test(result['fold_losses'])

    elapsed = time.time() - t0
    print(f"  Log-loss: {result['mean']:.4f} ± {result['std']:.4f}")
    print(f"  Accuracy: {result['accuracy']:.3f}")
    print(f"  Δ vs baseline: {sig['delta_vs_baseline']:+.4f}, p={sig['p_value_ttest']:.3f}, verdict={sig['verdict']}")

    metrics = {**result, **sig, 'elapsed_sec': elapsed}
    run_info = {
        'name': 'attempt_001_pure_transfer_logistic',
        'description': 'Logistic regression trained on full historical corpus (post-1990, ~24K matches), predict WC-2026',
        'n_historical': len(X_hist),
        'n_wc': len(X_wc),
        'features': hist_features,
        'model': 'LogisticRegression(C=0.5, multinomial)',
        'scaler': 'StandardScaler',
        'cutoff_date': '2026-06-12',
        'cv': 'RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=0)',
        'seed': SEED,
        'baseline': BASELINE_LOGLOSS,
        'frontier': FRONTIER_LOGLOSS,
    }
    save_artifact('attempt-001', metrics, run_info)
    return metrics, probs_ordered, le


# ─────────────────────────────────────────────────────────────────────────────
# ATTEMPT 002: Full GBM on historical corpus, predict WC-2026
# ─────────────────────────────────────────────────────────────────────────────
def attempt_002_gbm_transfer():
    """HistGBM trained on recent historical corpus, predict WC-2026 matches."""
    print("\n=== Attempt 002: GBM Transfer (post-2000, rich features) ===")
    t0 = time.time()

    X_hist, y_hist, w_hist, final_elos = build_historical_training_set(
        HIST_PATH, cutoff_date='2026-06-12', min_year=2000, tier_weight=True
    )

    X_wc, y_wc, completed = build_wc2026_feature_matrix(
        MATCHES_PATH, TEAMS_PATH, HIST_PATH, cutoff_date='2026-06-12'
    )

    # Use all common features
    hist_feats = ['elo_diff', 'home_adv', 'tier', 'win_rate_10_diff', 'gf_10_diff',
                  'ga_10_diff', 'gd_10_diff', 'draw_rate_10_diff', 'win_rate_5_diff',
                  'gf_5_diff', 'ga_5_diff', 'gd_5_diff',
                  'home_win_rate_10', 'away_win_rate_10', 'home_gd_10', 'away_gd_10']
    wc_feats = ['hist_elo_diff', 'home_is_host', 'win_rate_10_diff', 'win_rate_10_diff',
                'gf_10_diff', 'ga_10_diff', 'gd_10_diff', 'draw_rate_10_diff',
                'win_rate_5_diff', 'gf_5_diff', 'ga_5_diff', 'gd_5_diff',
                'home_win_rate_10', 'away_win_rate_10', 'home_gd_10', 'away_gd_10']

    # For WC features, create tier column = 5 (World Cup)
    X_wc_sub = X_wc.copy()
    X_wc_sub['tier'] = 5

    hist_sub = X_hist[hist_feats].copy()
    wc_sub = X_wc_sub[['hist_elo_diff', 'home_is_host', 'win_rate_10_diff', 'gf_10_diff',
                        'ga_10_diff', 'gd_10_diff', 'draw_rate_10_diff', 'win_rate_5_diff',
                        'gf_5_diff', 'ga_5_diff', 'gd_5_diff',
                        'home_win_rate_10', 'away_win_rate_10', 'home_gd_10', 'away_gd_10']].copy()
    wc_sub.insert(0, 'elo_diff', X_wc_sub['hist_elo_diff'])
    wc_sub.insert(1, 'home_adv', X_wc_sub['home_is_host'])
    wc_sub.insert(2, 'tier', 5)
    wc_sub = wc_sub[hist_feats]

    le = LabelEncoder()
    le.fit(['H', 'D', 'A'])

    clf = HistGradientBoostingClassifier(
        max_iter=300, max_depth=4, min_samples_leaf=20,
        l2_regularization=2.0, learning_rate=0.05,
        random_state=SEED, class_weight='balanced'
    )
    clf.fit(hist_sub, y_hist, sample_weight=w_hist)

    probs = clf.predict_proba(wc_sub)
    prob_df = pd.DataFrame(probs, columns=clf.classes_)
    probs_ordered = prob_df[le.classes_].values

    result = canonical_cv_eval(probs_ordered, y_wc, le)
    sig = significance_test(result['fold_losses'])

    elapsed = time.time() - t0
    print(f"  Log-loss: {result['mean']:.4f} ± {result['std']:.4f}")
    print(f"  Accuracy: {result['accuracy']:.3f}")
    print(f"  Δ vs baseline: {sig['delta_vs_baseline']:+.4f}, p={sig['p_value_ttest']:.3f}, verdict={sig['verdict']}")

    metrics = {**result, **sig, 'elapsed_sec': elapsed}
    run_info = {
        'name': 'attempt_002_gbm_transfer',
        'description': 'HistGBM trained on historical corpus (post-2000, tier-weighted), predict WC-2026',
        'n_historical': len(X_hist),
        'n_wc': len(X_wc),
        'features': hist_feats,
        'model': 'HistGradientBoostingClassifier(max_depth=4, min_samples_leaf=20, l2=2.0)',
        'tier_weight': True,
        'cutoff_date': '2026-06-12',
        'cv': 'RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=0)',
        'seed': SEED,
        'baseline': BASELINE_LOGLOSS,
        'frontier': FRONTIER_LOGLOSS,
    }
    save_artifact('attempt-002', metrics, run_info)
    return metrics, probs_ordered, le


# ─────────────────────────────────────────────────────────────────────────────
# ATTEMPT 003: Historical transfer probs as features in WC-2026 in-fold model
# ─────────────────────────────────────────────────────────────────────────────
def attempt_003_stacked_blend(probs_001, probs_002, le):
    """Blend historical model predictions with WC-2026 Elo features via in-fold logistic."""
    print("\n=== Attempt 003: Stacked Blend (historical probs + WC Elo, in-fold logistic) ===")
    t0 = time.time()

    X_wc, y_wc, _ = build_wc2026_feature_matrix(
        MATCHES_PATH, TEAMS_PATH, HIST_PATH, cutoff_date='2026-06-12'
    )

    # Stack historical probs with WC-2026 Elo features
    X_blend = pd.DataFrame({
        'p001_H': probs_001[:, le.transform(['H'])[0]],
        'p001_D': probs_001[:, le.transform(['D'])[0]],
        'p001_A': probs_001[:, le.transform(['A'])[0]],
        'p002_H': probs_002[:, le.transform(['H'])[0]],
        'p002_D': probs_002[:, le.transform(['D'])[0]],
        'p002_A': probs_002[:, le.transform(['A'])[0]],
        'wc_elo_diff': X_wc['wc_elo_diff'],
        'host_advantage': X_wc['host_advantage'],
        'wc_rank_diff': X_wc['wc_rank_diff'],
    })

    def model_fn():
        return LogisticRegression(C=0.1, max_iter=500, random_state=SEED)

    result, le2 = cv_train_eval(X_blend, y_wc, model_fn, scale=True)
    sig = significance_test(result['fold_losses'])

    elapsed = time.time() - t0
    print(f"  Log-loss: {result['mean']:.4f} ± {result['std']:.4f}")
    print(f"  Accuracy: {result['accuracy']:.3f}")
    print(f"  Δ vs baseline: {sig['delta_vs_baseline']:+.4f}, p={sig['p_value_ttest']:.3f}, verdict={sig['verdict']}")

    metrics = {**result, **sig, 'elapsed_sec': elapsed}
    run_info = {
        'name': 'attempt_003_stacked_blend',
        'description': 'In-fold logistic meta-learner blending Attempt-001 + 002 historical probs with WC-2026 Elo/rank features',
        'n_wc': len(X_wc),
        'features': X_blend.columns.tolist(),
        'model': 'LogisticRegression(C=0.1, multinomial) in-fold',
        'cv': 'RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=0)',
        'seed': SEED,
        'baseline': BASELINE_LOGLOSS,
        'frontier': FRONTIER_LOGLOSS,
    }
    save_artifact('attempt-003', metrics, run_info)
    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# ATTEMPT 004: WC-2026 In-fold GBM with ALL features (WC Elo + historical form)
# ─────────────────────────────────────────────────────────────────────────────
def attempt_004_infold_gbm(probs_001, probs_002, le):
    """In-fold GBM on WC-2026 features + historical model probs."""
    print("\n=== Attempt 004: In-fold GBM (all WC features + historical probs) ===")
    t0 = time.time()

    X_wc, y_wc, _ = build_wc2026_feature_matrix(
        MATCHES_PATH, TEAMS_PATH, HIST_PATH, cutoff_date='2026-06-12'
    )

    # Full feature set: WC-2026 features + historical model probs as features
    X_full = X_wc.copy()
    X_full['p001_H'] = probs_001[:, le.transform(['H'])[0]]
    X_full['p001_D'] = probs_001[:, le.transform(['D'])[0]]
    X_full['p001_A'] = probs_001[:, le.transform(['A'])[0]]
    X_full['p002_H'] = probs_002[:, le.transform(['H'])[0]]
    X_full['p002_D'] = probs_002[:, le.transform(['D'])[0]]
    X_full['p002_A'] = probs_002[:, le.transform(['A'])[0]]

    def model_fn():
        return HistGradientBoostingClassifier(
            max_iter=100, max_depth=2, min_samples_leaf=15,
            l2_regularization=5.0, learning_rate=0.1, random_state=SEED
        )

    result, _ = cv_train_eval(X_full, y_wc, model_fn, scale=False)
    sig = significance_test(result['fold_losses'])

    elapsed = time.time() - t0
    print(f"  Log-loss: {result['mean']:.4f} ± {result['std']:.4f}")
    print(f"  Accuracy: {result['accuracy']:.3f}")
    print(f"  Δ vs baseline: {sig['delta_vs_baseline']:+.4f}, p={sig['p_value_ttest']:.3f}, verdict={sig['verdict']}")

    metrics = {**result, **sig, 'elapsed_sec': elapsed}
    run_info = {
        'name': 'attempt_004_infold_gbm',
        'description': 'In-fold HistGBM on all WC-2026 features + historical transfer probs (001+002) as inputs',
        'n_wc': len(X_wc),
        'features': X_full.columns.tolist(),
        'model': 'HistGradientBoostingClassifier(depth=2, min_leaf=15, l2=5.0) in-fold',
        'cv': 'RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=0)',
        'seed': SEED,
        'baseline': BASELINE_LOGLOSS,
        'frontier': FRONTIER_LOGLOSS,
    }
    save_artifact('attempt-004', metrics, run_info)
    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# ATTEMPT 005: Historical corpus GBM with Platt calibration (pure transfer)
# ─────────────────────────────────────────────────────────────────────────────
def attempt_005_gbm_calibrated():
    """GBM trained on historical corpus, Platt-calibrated on full historical data."""
    print("\n=== Attempt 005: Historical GBM + Isotonic/Platt calibration ===")
    t0 = time.time()

    # Use post-2005 data for tighter distribution
    X_hist, y_hist, w_hist, _ = build_historical_training_set(
        HIST_PATH, cutoff_date='2026-06-12', min_year=2005, tier_weight=True
    )

    X_wc, y_wc, _ = build_wc2026_feature_matrix(
        MATCHES_PATH, TEAMS_PATH, HIST_PATH, cutoff_date='2026-06-12'
    )

    hist_feats = ['elo_diff', 'home_adv', 'tier', 'win_rate_10_diff', 'gf_10_diff',
                  'ga_10_diff', 'gd_10_diff', 'win_rate_5_diff', 'gd_5_diff',
                  'home_win_rate_10', 'away_win_rate_10', 'home_gd_10', 'away_gd_10']

    X_wc_sub = X_wc.copy()
    X_wc_sub['tier'] = 5
    X_wc_sub['elo_diff'] = X_wc_sub['hist_elo_diff']
    X_wc_sub['home_adv'] = X_wc_sub['home_is_host']
    wc_sub = X_wc_sub[hist_feats]
    hist_sub = X_hist[hist_feats]

    le = LabelEncoder()
    le.fit(['H', 'D', 'A'])

    # Base GBM
    base = HistGradientBoostingClassifier(
        max_iter=500, max_depth=4, min_samples_leaf=15,
        l2_regularization=1.0, learning_rate=0.03,
        random_state=SEED
    )

    # Calibrate using CalibratedClassifierCV with isotonic on training data
    # Note: This uses cross-validation internally for calibration
    from sklearn.calibration import CalibratedClassifierCV
    clf = CalibratedClassifierCV(base, method='isotonic', cv=5)
    clf.fit(hist_sub, y_hist)

    probs = clf.predict_proba(wc_sub)
    # CalibratedClassifierCV returns probabilities in order of classes_ attribute
    prob_df = pd.DataFrame(probs, columns=clf.classes_)
    probs_ordered = prob_df[le.classes_].values

    result = canonical_cv_eval(probs_ordered, y_wc, le)
    sig = significance_test(result['fold_losses'])

    elapsed = time.time() - t0
    print(f"  Log-loss: {result['mean']:.4f} ± {result['std']:.4f}")
    print(f"  Accuracy: {result['accuracy']:.3f}")
    print(f"  Δ vs baseline: {sig['delta_vs_baseline']:+.4f}, p={sig['p_value_ttest']:.3f}, verdict={sig['verdict']}")

    metrics = {**result, **sig, 'elapsed_sec': elapsed}
    run_info = {
        'name': 'attempt_005_gbm_calibrated',
        'description': 'HistGBM+Isotonic calibration trained on historical corpus (post-2005), predict WC-2026',
        'n_historical': len(X_hist),
        'n_wc': len(X_wc),
        'features': hist_feats,
        'model': 'CalibratedClassifierCV(HistGBM, isotonic, cv=5)',
        'cutoff_date': '2026-06-12',
        'cv': 'RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=0)',
        'seed': SEED,
        'baseline': BASELINE_LOGLOSS,
        'frontier': FRONTIER_LOGLOSS,
    }
    save_artifact('attempt-005', metrics, run_info)
    return metrics, probs_ordered, le


# ─────────────────────────────────────────────────────────────────────────────
# ATTEMPT 006: WC-only Elo-logistic (campaign canonical baseline, for verification)
# ─────────────────────────────────────────────────────────────────────────────
def attempt_006_wc_elo_baseline():
    """Canonical Elo-logistic baseline on WC-2026 data (Elo diff + host, in-fold fit)."""
    print("\n=== Attempt 006: WC-2026 Elo-logistic (campaign baseline verification) ===")
    t0 = time.time()

    X_wc, y_wc, _ = build_wc2026_feature_matrix(
        MATCHES_PATH, TEAMS_PATH, HIST_PATH, cutoff_date='2026-06-12'
    )

    X_elo = X_wc[['wc_elo_diff', 'host_advantage']].copy()

    def model_fn():
        return LogisticRegression(C=0.5, max_iter=500, random_state=SEED)

    result, _ = cv_train_eval(X_elo, y_wc, model_fn, scale=True)
    sig = significance_test(result['fold_losses'])

    elapsed = time.time() - t0
    print(f"  Log-loss: {result['mean']:.4f} ± {result['std']:.4f}")
    print(f"  Accuracy: {result['accuracy']:.3f}")
    print(f"  Δ vs baseline: {sig['delta_vs_baseline']:+.4f}, p={sig['p_value_ttest']:.3f}, verdict={sig['verdict']}")

    metrics = {**result, **sig, 'elapsed_sec': elapsed}
    run_info = {
        'name': 'attempt_006_wc_elo_baseline',
        'description': 'Campaign baseline verification: in-fold logistic on WC-2026 Elo diff + host advantage',
        'features': ['wc_elo_diff', 'host_advantage'],
        'model': 'LogisticRegression(C=0.5, multinomial) in-fold',
        'cv': 'RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=0)',
        'seed': SEED,
    }
    save_artifact('attempt-006', metrics, run_info)
    return metrics


if __name__ == '__main__':
    print("=" * 70)
    print("Wave-3: Full-Scale Historical Training Experiments")
    print("=" * 70)
    print(f"Baseline: {BASELINE_LOGLOSS} | Frontier: {FRONTIER_LOGLOSS}")

    # Run baseline verification first
    m006 = attempt_006_wc_elo_baseline()

    # Run core experiments
    m001, probs_001, le_001 = attempt_001_pure_transfer_logistic()
    m002, probs_002, le_002 = attempt_002_gbm_transfer()
    m003 = attempt_003_stacked_blend(probs_001, probs_002, le_001)
    m004 = attempt_004_infold_gbm(probs_001, probs_002, le_001)
    m005, probs_005, le_005 = attempt_005_gbm_calibrated()

    print("\n" + "=" * 70)
    print("SUMMARY TABLE")
    print("=" * 70)
    results = [
        ('006 WC-Elo baseline', m006),
        ('001 Transfer Logistic', m001),
        ('002 GBM Transfer', m002),
        ('003 Stacked Blend', m003),
        ('004 In-fold GBM', m004),
        ('005 GBM Calibrated', m005),
    ]
    for name, m in results:
        verdict = m.get('verdict', '?')
        print(f"  {name:30s}: {m['mean']:.4f} ± {m['std']:.4f}  acc={m['accuracy']:.3f}  {verdict}")

    # Save combined summary
    summary = {
        exp: {k: v for k, v in m.items()
              if k not in ('fold_losses', 'per_match_loss', 'oof_probs')}
        for exp, m in results
    }
    with open(f'{ARTIFACTS_DIR}/summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to {ARTIFACTS_DIR}/summary.json")
