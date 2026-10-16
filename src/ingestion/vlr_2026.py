import pandas as pd
import vlrdevapi as vlr
from src.config.constants import TEAM_NAME_ALIASES, STAGE_ALIASES
from src.processing.placements import classify_event

SEASON = "2026"


def _team_name(team) -> str:
    return TEAM_NAME_ALIASES.get(team.name, team.name)


def _match_to_row(match, event_name: str) -> dict:
    """Converts a vlrdevapi Match into the same schema as data/processed/matches.parquet."""
    team1, team2 = match.teams
    winner = team1 if team1.is_winner else team2

    return {
        "date": pd.Timestamp(match.date),
        "team1": _team_name(team1),
        "team2": _team_name(team2),
        "winner": _team_name(winner),
        "stage": STAGE_ALIASES.get(match.stage, match.stage),
        "event": event_name,
    }


def fetch_2026_matches() -> pd.DataFrame:
    """
    Fetches every completed 2026 official VCT match (regional, Masters, Champions) from vlr.gg.
    Returns a DataFrame sorted by date.
    """
    events = vlr.events.list_events(tier="vct", status="all", limit=60)
    rows = []

    for event in events:
        is_official = classify_event(event.name) is not None
        if not (is_official and SEASON in event.name):
            continue

        for match in vlr.events.matches(event.id):
            if match.status == "completed":
                rows.append(_match_to_row(match, event.name))

        print(f"  fetched: {event.name}")

    df = pd.DataFrame(rows)
    df = df.sort_values("date", kind="stable").reset_index(drop=True)

    return df


if __name__ == "__main__":
    df = fetch_2026_matches()

    print(df.shape)
    print(df.groupby("event").size())
