# WC2026 Trees Study — Cluster: Trees Workstream

FIFA World Cup 2026 match-outcome prediction using tree ensemble models.
Part of the Flywheel campaign: **black-butterfly-5248**.

## Summary

Across 8 experiments (RandomForest, ExtraTrees, HistGradientBoosting, calibration ablations),
**ExtraTrees with top-6 features + sigmoid calibration** achieves the best result:

- **ET-top6-sigmoid**: log-loss 0.8558 ± 0.085, accuracy 65.3%
- **Elo-logistic baseline**: log-loss 0.8393 ± 0.275, accuracy 63.7%
- Verdict: **RED** — no tree beats Elo-logistic by mean log-loss on n=64

Trees are 3× more consistent (lower std), but 0.016 log-loss above the Elo baseline.

## Key findings

1. Feature reduction (34→6) cuts ET log-loss by 0.019 — more effective than tuning regularisation
2. Top-6 features: `elo_diff`, `mv_diff`, `mv_top11_diff`, `rank_diff`, `home_elo_rating`, `home_is_host`
3. ExtraTrees > RandomForest at n=64 (random splits = better implicit regularisation)
4. HGB fails at n=64 even with heavy regularisation
5. Isotonic calibration collapses (~17 samples/fold for 3-class); sigmoid is safe

## Reproducing

```bash
cd ~/research/code
pip install scikit-learn numpy pandas scipy

# Baseline
python3 run_experiment.py elo-baseline elo_logistic '{"C": 1.0}' '["elo_diff", "home_is_host"]'

# RF-v1 (34 features)
python3 run_experiment.py rf-v1 rf '{"n_estimators": 100, "max_depth": 3, "min_samples_leaf": 8, "max_features": 0.5}'

# ET-v1 (34 features)
python3 run_experiment.py et-v1 et '{"n_estimators": 200, "max_depth": 3, "min_samples_leaf": 8, "max_features": 0.5}'

# HGB-v1 (34 features)
python3 run_experiment.py hgb-v1 hgb '{"max_depth": 3, "min_samples_leaf": 20, "l2_regularization": 5.0, "max_iter": 100}'

# RF-elo-only (8 Elo/rank features)
python3 run_experiment.py rf-elo-only rf '{"n_estimators": 100, "max_depth": 3, "min_samples_leaf": 8}' \
  '["elo_diff", "rank_diff", "home_elo_rating", "away_elo_rating", "home_fifa_ranking_pre_tournament", "away_fifa_ranking_pre_tournament", "home_is_host", "away_is_host"]'

# ET-top6-nocal (top-6 features, no calibration)
python3 run_experiment.py et-top6-nocal et '{"n_estimators": 300, "max_depth": 3, "min_samples_leaf": 8, "max_features": 0.8}' \
  '["elo_diff", "mv_diff", "mv_top11_diff", "rank_diff", "home_elo_rating", "home_is_host"]'

# Calibrated runs
python3 run_calibrated.py
```

## Files

- `code/features.py` — feature engineering (all pre-match features, no leakage)
- `code/run_experiment.py` — CV experiment runner (5×10 RepeatedStratifiedKFold)
- `code/run_calibrated.py` — calibrated ET/HGB variants (sigmoid and isotonic)
- `artifacts/<exp_id>/metrics.json` — CV results per experiment
- `artifacts/<exp_id>/run.json` — reproducibility metadata

## CV protocol

- `RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=0)` — 50 folds
- All encoders/scalers fit on training split only (no fold leakage)
- Seed 0 everywhere; CPU-only (no GPU used)
- Data: 64 completed WC2026 matches (`status==Completed`)
