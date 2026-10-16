import pandas as pd
from src.config.constants import KAGGLE_2026_PLAYER_STATS_PATH, TEAM_NAME_ALIASES


def _load_tournament_totals() -> pd.DataFrame:
    """One row per player per tournament; skips players listed under two teams."""
    raw = pd.read_csv(KAGGLE_2026_PLAYER_STATS_PATH)

    is_tournament_total = raw["Stage"] == "All Stages"
    has_single_team = ~raw["Teams"].str.contains(",")

    return raw[is_tournament_total & has_single_team]


def load_player_stats_2026(event_end_dates: pd.Series) -> pd.DataFrame:
    """
    Loads 2026 player stats from Kaggle in the same schema as load_player_stats().

    Kaggle only has per-tournament aggregates (no match dates), so each row
    is dated at its event's last match, taken from event_end_dates.
    """
    raw = _load_tournament_totals()

    df = pd.DataFrame({
        "match_date": raw["Tournament"].map(event_end_dates),
        "player_team": raw["Teams"].replace(TEAM_NAME_ALIASES),
        "acs": raw["Average Combat Score"],
        "kast": raw["Kill, Assist, Trade, Survive %"].str.rstrip("%").astype(float),
        "adr": raw["Average Damage Per Round"],
    })

    return df.dropna(subset=["match_date"])
