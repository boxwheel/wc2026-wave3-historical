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
| 012 | Dixon-Coles Poisson (historical WC+continental) | 0.9954 ± 0.100 | RED |
| 013 | Geo-mean blend (Poisson + WC Elo, α=0.1) | 0.8403 ± 0.108 | RED |
| 014 | H2H features + WC Elo logistic | 0.8574 ± 0.112 | RED |
| 015 | Isotonic-recalibrated Poisson blend | 0.8716 ± 0.125 | RED |
| 016 | Squad-derived Poisson (market value params) | 1.5627 ± 0.534 | RED |
| 017 | Combined WC Elo + squad features logistic | 0.8451 ± 0.104 | RED |
| 018 | Squad Poisson + logistic blend | 0.8647 ± 0.160 | RED |
| 019 | WC Elo-only, C+solver sweep (best C=1.0) | 0.8337 ± 0.134 | FLAT |
| 020 | Kitchen sink (15 features, ridge, C=0.05) | 0.8295 ± 0.117 | FLAT |

**Campaign baseline**: 0.8337 | **Wave-2 ensemble frontier**: 0.7608

## Key Findings
1. **Historical form ≡ Elo**: Win rate, GD, and form features are proxies for the same signal Elo already encodes. Adding them doesn't beat the baseline.
2. **Dixon-Coles Poisson is stale**: Historical attack/defense params from 1960-2026 don't reflect current team quality; recency weighting helps but can't fix stale base rates.
3. **Squad market values need MLE calibration**: Raw market value as a Poisson rate parameter is wildly uncalibrated (loss=1.56).
4. **Kitchen sink (020) is numerically best (0.8295)** but not statistically significant (p=0.80). The best FLAT result suggests extremely tight prior on ridge.
5. **Wave-3 gap from frontier**: Best Wave-3 is 0.8295 vs Wave-2 frontier 0.7608 — a 0.069 gap still to close.

## Eval Protocol
RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=0) on 64 completed WC-2026 group-stage matches.

## Reproduce
```bash
pip install pandas numpy scikit-learn scipy
python code/run_experiments.py    # Batch 1: attempts 001-006
python code/run_experiments2.py   # Batch 2: attempts 007-011
python run_experiments3.py        # Batch 3: attempts 012-015 (Dixon-Coles, blend, H2H, calibration)
python run_experiments4.py        # Batch 4: attempts 016-020 (squad Poisson, combined, sweep)
```
