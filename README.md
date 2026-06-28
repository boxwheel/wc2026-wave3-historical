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
| 021 | Recent form (90-day, C=0.5) | 0.8628 ± 0.107 | RED |
| 022 | Confederation strength logistic | 0.8726 ± 0.090 | RED |
| 023 | WC experience + Elo | 0.8438 ± 0.119 | RED |
| 024 | Squad age features | 0.8567 ± 0.112 | RED |
| 025 | Blend 70% Elo + 30% kitchen-sink | 0.8254 ± 0.110 | FLAT |
| 026 | Proportional Odds Model (ordinal logistic) | 0.8961 ± 0.115 | RED |
| 027 | Empirical WC bin calibration (2014-2022) | 0.9705 ± 0.151 | RED |
| 028 | Diverse 5-model ensemble (elo-ks blend) | 0.8240 ± 0.121 | FLAT |
| 029 | Direct Elo formula (draw_rate=0.3, scale=350, host_bonus=100) | **0.8148 ± 0.126** | FLAT |

**Campaign baseline**: 0.8337 | **Wave-3 best**: 0.8148 | **Wave-2 ensemble frontier**: 0.7608

## Key Findings
1. **Historical form ≡ Elo**: Win rate, GD, and form features are proxies for the same signal Elo already encodes. Adding them doesn't beat the baseline.
2. **Dixon-Coles Poisson is stale**: Historical attack/defense params from 1960-2026 don't reflect current team quality; recency weighting helps but can't fix stale base rates.
3. **Squad market values need MLE calibration**: Raw market value as a Poisson rate parameter is wildly uncalibrated (loss=1.56).
4. **Direct Elo formula beats all ML (029, 0.8148)**: Skipping logistic regression and using the Elo sigmoid directly with draw_rate=0.30, scale=350, and host_bonus=100 is the campaign best. Host teams (Mexico/USA/Canada) get a substantial +100 Elo bonus.
5. **Ordinal logistic adds no value**: Treating H/D/A as ordinal (026) hurts calibration vs multinomial logistic.
6. **WC bin calibration needs large samples**: 192 WC matches is insufficient for 7-bin empirical calibration (027, loss=0.9705).
7. **Ensemble ceiling ≈ 0.824**: The 5-model ensemble (028) and 2-model blend (025) converge around 0.824; diminishing returns on more models.
8. **Wave-3 gap from frontier**: Best Wave-3 is 0.8148 vs Wave-2 frontier 0.7608 — a 0.054 gap still to close.

## Eval Protocol
RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=0) on 64 completed WC-2026 group-stage matches.

## Reproduce
```bash
pip install pandas numpy scikit-learn scipy
python code/run_experiments.py    # Batch 1: attempts 001-006
python code/run_experiments2.py   # Batch 2: attempts 007-011
python run_experiments3.py        # Batch 3: attempts 012-015 (Dixon-Coles, blend, H2H, calibration)
python run_experiments4.py        # Batch 4: attempts 016-020 (squad Poisson, combined, sweep)
python run_experiments5.py        # Batch 5: attempts 021-025 (form, confederation, WC exp, age, ensemble)
python run_experiments6.py        # Batch 6: attempts 026-029 (ordinal, WC bins, 5-model ensemble, direct Elo)
```
