# WC-2026 Wave-3: Full-Scale Historical Training

Wave-3 workstream for the [FIFA World Cup 2026 Match-Outcome Prediction](https://github.com/boxwheel/wc2026-trees-study) campaign.

## Approach
Train full-scale supervised models on ~49K historical international matches with rich pre-match features, then apply to predict WC-2026 group-stage matches.

**Distinct from Wave-2 Transfer:** Wave-2 only low-weight augmented WC-only folds. Wave-3 trains at full scale on the entire historical corpus.

## Data
- WC-2026 canonical data: `boxwheel/wc2026-trees-study`
- Historical: `martj42/international-football-results-from-1872-to-2017` (~49K matches, 1872-2026)

## Experiments

| Attempt | Model | Log-loss | Verdict |
|---------|-------|----------|---------|
| 006 | WC-2026 Elo-logistic (baseline verify) | 0.8374 ± 0.114 | FLAT |
| 001 | Transfer Logistic (full historical) | 0.8959 ± 0.102 | RED |
| 002 | GBM Transfer (tier-weighted) | 0.9265 ± 0.104 | RED |
| 003 | Stacked Blend | 0.8406 ± 0.101 | RED |
| 004 | In-fold GBM | 1.0028 ± 0.241 | RED |
| 005 | GBM Calibrated | 0.8870 ± 0.090 | RED |
| 007 | WC-team form + WC Elo, logistic | 0.9170 ± 0.123 | RED |
| 008 | Full enriched features | 0.9937 ± 0.124 | RED |
| 009 | Augmented WC historical (weight=0.05) | 0.8573 ± 0.117 | RED |
| 010 | Historical WC pure transfer | 0.8914 ± 0.132 | RED |
| 011 | Best blend | 0.8719 ± 0.108 | RED |

**Campaign baseline**: 0.8337 | **Wave-2 ensemble frontier**: 0.7608

## Key Finding
Historical form features (win rate, goal difference) are largely correlated with Elo ratings (Elo IS a running form average). Adding them doesn't provide independent signal. Genuinely orthogonal signals require different information (e.g., attack/defense decomposition, head-to-head records).

## Eval Protocol
RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=0) on 64 completed WC-2026 group-stage matches.

## Reproduce
```bash
pip install pandas numpy scikit-learn scipy
python code/run_experiments.py   # Batch 1: attempts 001-006
python code/run_experiments2.py  # Batch 2: attempts 007-011
```
