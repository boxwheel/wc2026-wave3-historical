"""
Wave-3: Full-Scale Historical Training — Feature Engineering Pipeline
Builds pre-match features from ~49K historical international results.
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# ── Team name mapping: WC-2026 → historical results ─────────────────────────
NAME_MAP = {
    "Cabo Verde": "Cape Verde",
    "Congo DR": "DR Congo",
    "Czechia": "Czech Republic",
    "Côte d'Ivoire": "Ivory Coast",
    "IR Iran": "Iran",
    "Türkiye": "Turkey",
    "USA": "United States",
}

# Competition tier weights for training sample weights
TIER = {
    "FIFA World Cup": 5,
    "Copa América": 4,
    "UEFA Euro": 4,
    "African Cup of Nations": 4,
    "AFC Asian Cup": 4,
    "CONCACAF Gold Cup": 4,
    "FIFA World Cup qualification": 3,
    "UEFA Euro qualification": 3,
    "African Cup of Nations qualification": 3,
    "AFC Asian Cup qualification": 3,
    "CONCACAF Nations League": 2,
    "UEFA Nations League": 2,
    "Friendly": 1,
}

def _tier(t):
    for k, v in TIER.items():
        if k.lower() in t.lower():
            return v
    return 2  # default for other competitive matches


def build_elo_ratings(hist_df, K_wc=30, K_qual=20, K_friendly=10, home_adv=100,
                      start_rating=1500, cutoff_date=None):
    """Build running Elo ratings from historical match results.

    Returns dict: team_name -> final Elo rating.
    Also returns per-match elo_before_home, elo_before_away columns added to hist_df.
    """
    if cutoff_date is not None:
        df = hist_df[hist_df['date'] < cutoff_date].copy()
    else:
        df = hist_df.copy()

    df = df.sort_values('date').reset_index(drop=True)
    df = df.dropna(subset=['home_score', 'away_score'])

    ratings = {}
    match_elos = []

    def get_rating(team):
        return ratings.get(team, start_rating)

    def expected(ra, rb):
        return 1.0 / (1.0 + 10 ** ((rb - ra) / 400.0))

    def k_factor(tournament):
        t = tournament.lower()
        if 'world cup' in t and 'qualif' not in t:
            return K_wc
        elif 'qualif' in t or 'nations' in t or 'cup' in t or 'euro' in t or 'copa' in t or 'african' in t or 'afc' in t:
            return K_qual
        else:
            return K_friendly

    for _, row in df.iterrows():
        home = row['home_team']
        away = row['away_team']
        neutral = row.get('neutral', False)

        ra = get_rating(home)
        rb = get_rating(away)

        # Apply home advantage for non-neutral matches
        ra_eff = ra + (0 if neutral else home_adv)

        Ea = expected(ra_eff, rb)
        Eb = 1.0 - Ea

        hs = row['home_score']
        as_ = row['away_score']

        if hs > as_:
            Sa, Sb = 1.0, 0.0
        elif hs < as_:
            Sa, Sb = 0.0, 1.0
        else:
            Sa, Sb = 0.5, 0.5

        K = k_factor(row['tournament'])

        match_elos.append({'home_elo_before': ra, 'away_elo_before': rb})

        ratings[home] = ra + K * (Sa - Ea)
        ratings[away] = rb + K * (Sb - Eb)

    return ratings, pd.DataFrame(match_elos, index=df.index)


def build_rolling_form(hist_df, cutoff_date, window=10):
    """Compute rolling form for each team as of cutoff_date.

    Returns dict: team_name -> {win_rate_N, goals_for_N, goals_against_N, gd_N}
    """
    if cutoff_date is not None:
        df = hist_df[hist_df['date'] < cutoff_date].copy()
    else:
        df = hist_df.copy()

    df = df.sort_values('date').reset_index(drop=True)
    df = df.dropna(subset=['home_score', 'away_score'])

    from collections import defaultdict, deque

    team_history = defaultdict(list)

    for _, row in df.iterrows():
        home = row['home_team']
        away = row['away_team']
        hs = row['home_score']
        as_ = row['away_score']

        team_history[home].append({
            'gf': hs, 'ga': as_, 'gd': hs - as_,
            'win': 1 if hs > as_ else 0,
            'draw': 1 if hs == as_ else 0,
        })
        team_history[away].append({
            'gf': as_, 'ga': hs, 'gd': as_ - hs,
            'win': 1 if as_ > hs else 0,
            'draw': 1 if hs == as_ else 0,
        })

    form = {}
    for team, history in team_history.items():
        last_n = history[-window:]
        if len(last_n) == 0:
            form[team] = {
                f'win_rate_{window}': 0.5,
                f'gf_{window}': 1.0,
                f'ga_{window}': 1.0,
                f'gd_{window}': 0.0,
                f'draw_rate_{window}': 0.25,
            }
        else:
            form[team] = {
                f'win_rate_{window}': np.mean([m['win'] for m in last_n]),
                f'gf_{window}': np.mean([m['gf'] for m in last_n]),
                f'ga_{window}': np.mean([m['ga'] for m in last_n]),
                f'gd_{window}': np.mean([m['gd'] for m in last_n]),
                f'draw_rate_{window}': np.mean([m['draw'] for m in last_n]),
            }

    return form


def build_wc2026_feature_matrix(
    matches_detailed_path,
    teams_path,
    hist_results_path,
    cutoff_date='2026-06-11',
    elo_K_wc=30,
    elo_K_qual=20,
    elo_K_friendly=10,
    elo_home_adv=100,
    form_window=10,
):
    """Build the full WC-2026 feature matrix for the 64 completed matches."""
    matches = pd.read_csv(matches_detailed_path)
    teams = pd.read_csv(teams_path)
    hist = pd.read_csv(hist_results_path, parse_dates=['date'])

    completed = matches[matches['status'] == 'Completed'].copy()

    # Map WC names to historical names
    teams['hist_name'] = teams['team_name'].map(lambda x: NAME_MAP.get(x, x))
    name_to_hist = dict(zip(teams['team_name'], teams['hist_name']))

    # Build Elo ratings
    elo_ratings, _ = build_elo_ratings(
        hist, K_wc=elo_K_wc, K_qual=elo_K_qual, K_friendly=elo_K_friendly,
        home_adv=elo_home_adv, cutoff_date=cutoff_date
    )

    # Build rolling form (windows 5 and 10)
    form_10 = build_rolling_form(hist, cutoff_date=cutoff_date, window=10)
    form_5 = build_rolling_form(hist, cutoff_date=cutoff_date, window=5)

    # Host teams (home advantage is playing at own country)
    host_teams = {'Mexico', 'USA', 'Canada'}

    rows = []
    labels = []

    for _, m in completed.iterrows():
        home_wc = m['home_team_name']
        away_wc = m['away_team_name']
        home_hist = name_to_hist.get(home_wc, home_wc)
        away_hist = name_to_hist.get(away_wc, away_wc)

        # WC-2026 provided Elo and rank
        home_row = teams[teams['team_name'] == home_wc]
        away_row = teams[teams['team_name'] == away_wc]
        wc_home_elo = home_row['elo_rating'].values[0] if len(home_row) else 1500
        wc_away_elo = away_row['elo_rating'].values[0] if len(away_row) else 1500
        wc_home_rank = home_row['fifa_ranking_pre_tournament'].values[0] if len(home_row) else 100
        wc_away_rank = away_row['fifa_ranking_pre_tournament'].values[0] if len(away_row) else 100

        # Historical Elo
        hist_home_elo = elo_ratings.get(home_hist, 1500)
        hist_away_elo = elo_ratings.get(away_hist, 1500)

        # Form features
        h10 = form_10.get(home_hist, {f'win_rate_10': 0.5, 'gf_10': 1.0, 'ga_10': 1.0, 'gd_10': 0.0, 'draw_rate_10': 0.25})
        a10 = form_10.get(away_hist, {f'win_rate_10': 0.5, 'gf_10': 1.0, 'ga_10': 1.0, 'gd_10': 0.0, 'draw_rate_10': 0.25})
        h5 = form_5.get(home_hist, {f'win_rate_5': 0.5, 'gf_5': 1.0, 'ga_5': 1.0, 'gd_5': 0.0, 'draw_rate_5': 0.25})
        a5 = form_5.get(away_hist, {f'win_rate_5': 0.5, 'gf_5': 1.0, 'ga_5': 1.0, 'gd_5': 0.0, 'draw_rate_5': 0.25})

        home_is_host = 1 if home_wc in host_teams else 0
        away_is_host = 1 if away_wc in host_teams else 0

        row = {
            # WC-2026 provided features
            'wc_elo_diff': wc_home_elo - wc_away_elo,
            'wc_rank_diff': wc_away_rank - wc_home_rank,  # positive = home team better
            'home_is_host': home_is_host,
            'away_is_host': away_is_host,
            'host_advantage': home_is_host - away_is_host,
            # Historical Elo
            'hist_elo_diff': hist_home_elo - hist_away_elo,
            'hist_home_elo': hist_home_elo,
            'hist_away_elo': hist_away_elo,
            # Elo disagreement (historical vs WC provided)
            'elo_diff_delta': (wc_home_elo - wc_away_elo) - (hist_home_elo - hist_away_elo),
            # Form features (home - away diffs)
            'win_rate_10_diff': h10.get('win_rate_10', 0.5) - a10.get('win_rate_10', 0.5),
            'gf_10_diff': h10.get('gf_10', 1.0) - a10.get('gf_10', 1.0),
            'ga_10_diff': h10.get('ga_10', 1.0) - a10.get('ga_10', 1.0),
            'gd_10_diff': h10.get('gd_10', 0.0) - a10.get('gd_10', 0.0),
            'draw_rate_10_diff': h10.get('draw_rate_10', 0.25) - a10.get('draw_rate_10', 0.25),
            'win_rate_5_diff': h5.get('win_rate_5', 0.5) - a5.get('win_rate_5', 0.5),
            'gf_5_diff': h5.get('gf_5', 1.0) - a5.get('gf_5', 1.0),
            'ga_5_diff': h5.get('ga_5', 1.0) - a5.get('ga_5', 1.0),
            'gd_5_diff': h5.get('gd_5', 0.0) - a5.get('gd_5', 0.0),
            # Absolute values
            'home_win_rate_10': h10.get('win_rate_10', 0.5),
            'away_win_rate_10': a10.get('win_rate_10', 0.5),
            'home_gd_10': h10.get('gd_10', 0.0),
            'away_gd_10': a10.get('gd_10', 0.0),
        }
        rows.append(row)

        hs, as_ = m['home_score'], m['away_score']
        label = 'H' if hs > as_ else ('D' if hs == as_ else 'A')
        labels.append(label)

    X = pd.DataFrame(rows)
    y = pd.Series(labels, name='outcome')

    return X, y, completed.reset_index(drop=True)


def build_historical_training_set(hist_results_path, cutoff_date='2026-06-11',
                                   min_year=2000, tier_weight=True):
    """Build training features for the historical corpus (for full supervised training).

    Each row is a match. Features are computed from history-so-far at match time.
    This is expensive for the full 49K corpus, so we subsample to post-min_year.
    """
    hist = pd.read_csv(hist_results_path, parse_dates=['date'])
    hist = hist.dropna(subset=['home_score', 'away_score'])
    hist = hist[hist['date'] < pd.to_datetime(cutoff_date)]
    hist = hist.sort_values('date').reset_index(drop=True)

    print(f"Total historical matches up to {cutoff_date}: {len(hist)}")

    # For efficiency, compute Elo incrementally and collect features
    ratings = {}
    start_rating = 1500

    def get_rating(team):
        return ratings.get(team, start_rating)

    def expected(ra, rb):
        return 1.0 / (1.0 + 10 ** ((rb - ra) / 400.0))

    def k_factor(tournament):
        t = tournament.lower()
        if 'world cup' in t and 'qualif' not in t:
            return 30
        elif 'qualif' in t or 'nations' in t or 'cup' in t or 'euro' in t or 'copa' in t or 'african' in t or 'afc' in t:
            return 20
        else:
            return 10

    # Also track rolling form incrementally
    from collections import defaultdict
    team_last_10 = defaultdict(list)
    team_last_5 = defaultdict(list)

    HOME_ADV = 100
    rows = []
    labels = []
    weights = []

    for _, row in hist.iterrows():
        home = row['home_team']
        away = row['away_team']
        neutral = row.get('neutral', False)
        tournament = row['tournament']

        ra = get_rating(home)
        rb = get_rating(away)
        ra_eff = ra + (0 if neutral else HOME_ADV)

        Ea = expected(ra_eff, rb)
        Eb = 1.0 - Ea

        hs = row['home_score']
        as_ = row['away_score']

        if hs > as_:
            Sa, Sb = 1.0, 0.0
            label = 'H'
        elif hs < as_:
            Sa, Sb = 0.0, 1.0
            label = 'A'
        else:
            Sa, Sb = 0.5, 0.5
            label = 'D'

        K = k_factor(tournament)
        tier = _tier(tournament)

        # Only include post-min_year matches as training rows
        if row['date'].year >= min_year:
            h_last10 = team_last_10[home][-10:]
            a_last10 = team_last_10[away][-10:]
            h_last5 = team_last_5[home][-5:]
            a_last5 = team_last_5[away][-5:]

            def form_stats(last_n, suffix):
                if not last_n:
                    return {f'win_rate_{suffix}': 0.5, f'gf_{suffix}': 1.0,
                            f'ga_{suffix}': 1.0, f'gd_{suffix}': 0.0, f'draw_rate_{suffix}': 0.25}
                return {
                    f'win_rate_{suffix}': np.mean([m['win'] for m in last_n]),
                    f'gf_{suffix}': np.mean([m['gf'] for m in last_n]),
                    f'ga_{suffix}': np.mean([m['ga'] for m in last_n]),
                    f'gd_{suffix}': np.mean([m['gd'] for m in last_n]),
                    f'draw_rate_{suffix}': np.mean([m['draw'] for m in last_n]),
                }

            h10 = form_stats(h_last10, '10')
            a10 = form_stats(a_last10, '10')
            h5 = form_stats(h_last5, '5')
            a5 = form_stats(a_last5, '5')

            feat = {
                'elo_diff': ra - rb,
                'home_elo': ra,
                'away_elo': rb,
                'home_adv': 0 if neutral else 1,
                'tier': tier,
                'win_rate_10_diff': h10['win_rate_10'] - a10['win_rate_10'],
                'gf_10_diff': h10['gf_10'] - a10['gf_10'],
                'ga_10_diff': h10['ga_10'] - a10['ga_10'],
                'gd_10_diff': h10['gd_10'] - a10['gd_10'],
                'draw_rate_10_diff': h10['draw_rate_10'] - a10['draw_rate_10'],
                'win_rate_5_diff': h5['win_rate_5'] - a5['win_rate_5'],
                'gf_5_diff': h5['gf_5'] - a5['gf_5'],
                'ga_5_diff': h5['ga_5'] - a5['ga_5'],
                'gd_5_diff': h5['gd_5'] - a5['gd_5'],
                'home_win_rate_10': h10['win_rate_10'],
                'away_win_rate_10': a10['win_rate_10'],
                'home_gd_10': h10['gd_10'],
                'away_gd_10': a10['gd_10'],
            }

            rows.append(feat)
            labels.append(label)
            weights.append(float(tier) if tier_weight else 1.0)

        # Update Elo
        ratings[home] = ra + K * (Sa - Ea)
        ratings[away] = rb + K * (Sb - Eb)

        # Update rolling form
        for team, gf, ga in [(home, hs, as_), (away, as_, hs)]:
            win = 1 if gf > ga else 0
            draw = 1 if gf == ga else 0
            team_last_10[team].append({'gf': gf, 'ga': ga, 'gd': gf - ga, 'win': win, 'draw': draw})
            team_last_5[team].append({'gf': gf, 'ga': ga, 'gd': gf - ga, 'win': win, 'draw': draw})
            if len(team_last_10[team]) > 10:
                team_last_10[team] = team_last_10[team][-10:]
            if len(team_last_5[team]) > 5:
                team_last_5[team] = team_last_5[team][-5:]

    X_hist = pd.DataFrame(rows)
    y_hist = pd.Series(labels, name='outcome')
    w_hist = pd.Series(weights, name='weight')

    print(f"Training rows (post-{min_year}): {len(X_hist)}")
    print(f"Label distribution: {y_hist.value_counts().to_dict()}")

    return X_hist, y_hist, w_hist, ratings


if __name__ == '__main__':
    DATA_DIR = '/home/user/research/wave3-historical/data'
    X, y, _ = build_wc2026_feature_matrix(
        f'{DATA_DIR}/wc2026-trees-study-main/fifa_data/matches_detailed.csv',
        f'{DATA_DIR}/wc2026-trees-study-main/fifa_data/teams.csv',
        f'{DATA_DIR}/historical/results.csv',
    )
    print("WC-2026 feature matrix shape:", X.shape)
    print(X.describe())
    print("Label counts:", y.value_counts().to_dict())
