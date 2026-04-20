import pandas as pd
from src.features.elo import expected_score


def build_match_features(
    matches_df: pd.DataFrame,
    elo_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Builds a feature table for each match.
    Uses elo_before values to avoid data leakage.
    """
    rows = []

    for _, match in matches_df.iterrows():
        team1 = match["team1"]
        team2 = match["team2"]
        date = match["date"]

        # get elo_before for both teams at this specific match
        match_elo = elo_df[
            (elo_df["team1"] == team1) &
            (elo_df["team2"] == team2) &
            (elo_df["date"] == date)
        ]

        if match_elo.empty:
            continue

        elo1 = match_elo["elo1_before"].values[0]
        elo2 = match_elo["elo2_before"].values[0]

        rows.append({
            "date": date,
            "event": match["event"],
            "stage": match["stage"],
            "team1": team1,
            "team2": team2,
            "elo_team1": elo1,
            "elo_team2": elo2,
            "elo_diff": round(elo1 - elo2, 2),
            "win_prob_team1": round(expected_score(elo1, elo2), 4),
            "result": 1 if match["winner"] == team1 else 0,
        })

    df = pd.DataFrame(rows)
    df = df.sort_values("date").reset_index(drop=True)

    return df


if __name__ == "__main__":
    matches_df = pd.read_parquet("data/processed/matches.parquet")
    elo_df = pd.read_parquet("data/processed/elo.parquet")

    features_df = build_match_features(matches_df, elo_df)

    print(features_df.shape)
    print(features_df.head(10))

    features_df.to_parquet("data/features/match_features.parquet", index=False)
    print("Saved to data/features/match_features.parquet")