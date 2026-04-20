import pandas as pd
from src.config.constants import STAGE_K, EVENT_WEIGHTS
from src.config.enums import EventType
from src.processing.placements import classify_event

BASE_ELO = 0
ELO_SCALE = 100


def calculate_initial_elo(placements_df: pd.DataFrame) -> dict[str, float]:
    """Sums weighted_score per team to get initial Elo rating. Scaled by ELO_SCALE."""
    return (
        placements_df
        .groupby("team")["weighted_score"]
        .sum()
        .mul(ELO_SCALE)
        .to_dict()
    )


def expected_score(elo_team: float, elo_opponent: float) -> float:
    """
    Calculates the expected probability of winning based on Elo difference.
    Formula: 1 / (1 + 10 ** ((elo_opponent - elo_team) / 400))
    
    The 400 is a scaling factor — a team with 400 more Elo points
    is expected to win ~90% of the time.
    """
    return 1 / (1 + 10 ** ((elo_opponent - elo_team) / 400))


def get_k(stage: str, event_type: EventType) -> float:
    """
    Returns the K factor for a match — how much Elo can change.
    K = base stage value * event importance weight.
    Higher stage (Playoffs) + more important event (Champions) = bigger Elo swing.
    """
    base_k = STAGE_K.get(stage, 20)  # default 20 if stage unknown
    event_weight = EVENT_WEIGHTS.get(event_type, 0.7)  # default regional'

    return base_k * event_weight


def update_elo(
    elo_team: float,
    elo_opponent: float,
    result: int,
    stage: str,
    event_type: EventType,
) -> float:
    """
    Updates Elo after a match.
    result = 1 if team won, 0 if lost.
    Returns the new Elo value.
    """
    k = get_k(stage, event_type)
    expected = expected_score(elo_team, elo_opponent)
    
    new_elo = elo_team + k * (result - expected)
    return new_elo


def calculate_dynamic_elo(
    matches_df: pd.DataFrame,
    initial_elo: dict[str, float],
) -> pd.DataFrame:
    """
    Iterates all matches chronologically and updates Elo after each one.
    Returns a DataFrame with elo_before and elo_after for each match.
    """
    elo = dict(initial_elo)
    rows = []

    for _, match in matches_df.iterrows():
        team1 = match["team1"]
        team2 = match["team2"]
        winner = match["winner"]
        stage = match["stage"]
        event = match["event"]

        # skip showmatches and unknown stages
        if stage not in STAGE_K:
            continue

        # classify event type
        event_type = classify_event(event) or EventType.REGIONAL

        elo1 = elo.get(team1, BASE_ELO)
        elo2 = elo.get(team2, BASE_ELO)

        result1 = 1 if winner == team1 else 0
        result2 = 1 - result1

        new_elo1 = update_elo(elo1, elo2, result1, stage, event_type)
        new_elo2 = update_elo(elo2, elo1, result2, stage, event_type)
        win_prob = round(expected_score(elo1, elo2), 4)

        rows.append({
            "date": match["date"],
            "event": event,
            "stage": stage,
            "team1": team1,
            "team2": team2,
            "winner": winner,
            "elo1_before": round(elo1, 2),
            "elo2_before": round(elo2, 2),
            "elo1_after": round(new_elo1, 2),
            "elo2_after": round(new_elo2, 2),
            "elo_diff": round(elo1 - elo2, 2),
            "win_prob_team1": win_prob,
            "result": result1,
        })

        elo[team1] = new_elo1
        elo[team2] = new_elo2

    return pd.DataFrame(rows)


if __name__ == "__main__":
    matches_df = pd.read_parquet("data/processed/matches.parquet")
    placements_df = pd.read_parquet("data/processed/placements.parquet")

    initial_elo = calculate_initial_elo(placements_df)
    df_elo = calculate_dynamic_elo(matches_df, initial_elo)

    print(df_elo.shape)
    print(df_elo.head(10))

    df_elo.to_parquet("data/processed/elo.parquet", index=False)
    print("Saved to data/processed/elo.parquet")

    team1_elo = df_elo[["team1", "elo1_after"]].rename(columns={"team1": "team", "elo1_after": "elo"})
    team2_elo = df_elo[["team2", "elo2_after"]].rename(columns={"team2": "team", "elo2_after": "elo"})

    final_elo = (
        pd.concat([team1_elo, team2_elo])
        .groupby("team")["elo"]
        .last()
        .reset_index()
    )

    final_elo.to_parquet("data/processed/elo_final.parquet", index=False)
    print("Saved to data/processed/elo_final.parquet")