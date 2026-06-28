"""Feature engineering for WC2026 prediction - pre-match features ONLY."""
import pandas as pd
import numpy as np
from datetime import datetime

HOSTS = {"USA", "MEX", "CAN"}

def load_data(data_dir="~/research/fifa_data"):
    import os
    data_dir = os.path.expanduser(data_dir)
    matches = pd.read_csv(f"{data_dir}/matches_detailed.csv")
    teams = pd.read_csv(f"{data_dir}/teams.csv")
    squads = pd.read_csv(f"{data_dir}/squads_and_players.csv")
    venues = pd.read_csv(f"{data_dir}/venues.csv")
    return matches, teams, squads, venues

def compute_squad_features(squads, teams):
    """Aggregate per-team squad features (all pre-match)."""
    # merge team_id -> fifa_code
    squad = squads.merge(teams[["team_id","fifa_code"]], on="team_id", how="left")
    # compute age from date_of_birth (reference tournament date 2026-06-11)
    ref = datetime(2026, 6, 11)
    squad["dob"] = pd.to_datetime(squad["date_of_birth"], errors="coerce")
    squad["age"] = (ref - squad["dob"]).dt.days / 365.25
    squad["mv"] = pd.to_numeric(squad["market_value_eur"], errors="coerce").fillna(0)
    squad["goals_val"] = pd.to_numeric(squad["goals"], errors="coerce").fillna(0)
    squad["caps_val"] = pd.to_numeric(squad["caps"], errors="coerce").fillna(0)
    squad["height_val"] = pd.to_numeric(squad["height_cm"], errors="coerce")
    
    feats = squad.groupby("fifa_code").agg(
        squad_total_mv=("mv","sum"),
        squad_mean_mv=("mv","mean"),
        squad_top11_mv=("mv", lambda x: x.nlargest(11).sum()),
        squad_mean_age=("age","mean"),
        squad_mean_caps=("caps_val","mean"),
        squad_total_goals=("goals_val","sum"),
        squad_mean_height=("height_val","mean"),
    ).reset_index()
    
    # GK value
    gk = squad[squad["position"]=="GK"].groupby("fifa_code")["mv"].sum().rename("gk_mv")
    feats = feats.merge(gk, on="fifa_code", how="left")
    feats["gk_mv"] = feats["gk_mv"].fillna(0)
    
    # attacker goals
    att = squad[squad["position"].isin(["FW","MF"])].groupby("fifa_code")["goals_val"].sum().rename("att_goals")
    feats = feats.merge(att, on="fifa_code", how="left")
    feats["att_goals"] = feats["att_goals"].fillna(0)
    
    return feats

def build_match_features(data_dir="~/research/fifa_data"):
    matches, teams, squads, venues = load_data(data_dir)
    
    # Only completed matches
    df = matches[matches["status"]=="Completed"].copy()
    assert len(df) == 64, f"Expected 64 completed, got {len(df)}"
    
    # Target
    df["label"] = np.where(df["home_score"] > df["away_score"], "H",
                  np.where(df["home_score"] == df["away_score"], "D", "A"))
    
    # Squad features
    sq_feats = compute_squad_features(squads, teams)
    
    # Team info: merge elo, ranking, confederation
    team_info = teams[["fifa_code","elo_rating","fifa_ranking_pre_tournament","confederation"]].copy()
    team_info["elo_rating"] = pd.to_numeric(team_info["elo_rating"], errors="coerce")
    team_info["fifa_ranking_pre_tournament"] = pd.to_numeric(team_info["fifa_ranking_pre_tournament"], errors="coerce")
    team_info = team_info.merge(sq_feats, on="fifa_code", how="left")
    
    # Venue features
    ven = venues[["stadium_name","capacity","elevation_meters"]].copy()
    ven["capacity"] = pd.to_numeric(ven["capacity"], errors="coerce")
    ven["elevation_meters"] = pd.to_numeric(ven["elevation_meters"], errors="coerce").fillna(0)
    df = df.merge(ven, on="stadium_name", how="left")
    
    # Merge home team
    home_cols = {c: f"home_{c}" for c in team_info.columns if c != "fifa_code"}
    home_cols["confederation"] = "home_confederation"
    df = df.merge(team_info.rename(columns={c: f"home_{c}" if c!="fifa_code" else c
                                             for c in team_info.columns}),
                  left_on="home_fifa_code", right_on="fifa_code", how="left").drop(columns=["fifa_code"])
    
    # Merge away team
    df = df.merge(team_info.rename(columns={c: f"away_{c}" if c!="fifa_code" else c
                                             for c in team_info.columns}),
                  left_on="away_fifa_code", right_on="fifa_code", how="left").drop(columns=["fifa_code"])
    
    # Difference features
    df["elo_diff"] = df["home_elo_rating"] - df["away_elo_rating"]
    df["rank_diff"] = df["home_fifa_ranking_pre_tournament"] - df["away_fifa_ranking_pre_tournament"]
    df["mv_diff"] = df["home_squad_total_mv"] - df["away_squad_total_mv"]
    df["mv_top11_diff"] = df["home_squad_top11_mv"] - df["away_squad_top11_mv"]
    df["caps_diff"] = df["home_squad_mean_caps"] - df["away_squad_mean_caps"]
    df["goals_diff"] = df["home_squad_total_goals"] - df["away_squad_total_goals"]
    df["age_diff"] = df["home_squad_mean_age"] - df["away_squad_mean_age"]
    df["gk_mv_diff"] = df["home_gk_mv"] - df["away_gk_mv"]
    df["att_goals_diff"] = df["home_att_goals"] - df["away_att_goals"]
    
    # Host flag
    df["home_is_host"] = df["home_fifa_code"].isin(HOSTS).astype(int)
    df["away_is_host"] = df["away_fifa_code"].isin(HOSTS).astype(int)
    
    # Stage encoding
    stage_order = {"Group Stage":0, "Round of 16":1, "Quarter-final":2, "Semi-final":3, "Third-place play-off":3, "Final":4}
    df["stage_enc"] = df["stage_name"].map(stage_order).fillna(0)
    
    return df

if __name__ == "__main__":
    df = build_match_features()
    print("Feature matrix shape:", df.shape)
    print("Label distribution:", df["label"].value_counts().to_dict())
    print("Elo diff stats:", df["elo_diff"].describe())
