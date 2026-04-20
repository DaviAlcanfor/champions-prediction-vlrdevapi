import glob
import os
import pandas as pd
from src.config.constants import VCT_2025_PATH, TEAM_NAME_ALIASES, INVALID_TEAMS

WINDOW_DAYS = 60
STAT_COLUMNS = ["match_date", "player_team", "acs", "kast", "adr"]


def _load_raw_player_stats() -> pd.DataFrame:
    """
    Loads the raw data from player_stats dataset
    """

    dfs = [
        pd.read_csv(filepath)
        for filepath in glob.glob(os.path.join(VCT_2025_PATH, "**", "detailed_matches_player_stats.csv"), recursive=True)
    ]
    return pd.concat(dfs, ignore_index=True)


def _clean_player_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filters from type of stats, clean name aliases and patternize them, remove invalid teams,
    clean KAST % format and transforms datetime col
    """

    df = df[df["stat_type"] == "overall"]
    df["player_team"] = df["player_team"].replace(TEAM_NAME_ALIASES)
    df = df[~df["player_team"].isin(INVALID_TEAMS)]
    df["kast"] = df["kast"].str.replace("%", "").astype(float)
    df["match_date"] = pd.to_datetime(df["match_date"])
    return df[STAT_COLUMNS]


def load_player_stats() -> pd.DataFrame:
    """
    Loads and cleans all detailed_matches_player_stats.csv from vct_2025.
    """
    raw = _load_raw_player_stats()
    return _clean_player_stats(raw)


def compute_team_stats(player_df: pd.DataFrame, match_date: pd.Timestamp, team: str) -> dict | None:
    """
    Computes mean ACS, KAST, ADR for a team in the 60 days before a given match date.
    Returns None if no data is available in the window.
    """
    window_start = match_date - pd.Timedelta(days=WINDOW_DAYS)

    window = player_df[
        (player_df["player_team"] == team) &
        (player_df["match_date"] >= window_start) &
        (player_df["match_date"] < match_date)
    ]

    if window.empty:
        return None

    mean_acs = round(window["acs"].mean(), 2)
    mean_kast = round(window["kast"].mean(), 2)
    mean_adr = round(window["adr"].mean(), 2)

    return {
        "acs": mean_acs, 
        "kast": mean_kast, 
        "adr": mean_adr
    }


def build_player_features(elo_df: pd.DataFrame, player_df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds team player stats features to each match in elo_df.
    Uses only data from the 60 days before each match to avoid leakage.
    """
    rows = []

    for _, match in elo_df.iterrows():
        date = match["date"]
        team1 = match["team1"]
        team2 = match["team2"]

        stats1 = compute_team_stats(player_df, date, team1) or {}
        stats2 = compute_team_stats(player_df, date, team2) or {}

        rows.append({
            "date": date,
            "team1": team1,
            "team2": team2,
            "acs_team1": stats1.get("acs"),
            "kast_team1": stats1.get("kast"),
            "adr_team1": stats1.get("adr"),
            "acs_team2": stats2.get("acs"),
            "kast_team2": stats2.get("kast"),
            "adr_team2": stats2.get("adr"),
        })

    return pd.DataFrame(rows)


if __name__ == "__main__":
    elo_df = pd.read_parquet("data/processed/elo.parquet")
    player_df = load_player_stats()

    print(f"Player stats loaded: {player_df.shape}")

    features_df = build_player_features(elo_df, player_df)

    print(features_df.shape)
    print(features_df.head(10))

    features_df.to_parquet("data/features/player_features.parquet", index=False)
    print("Saved to data/features/player_features.parquet")