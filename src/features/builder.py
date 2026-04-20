import pandas as pd


def build_match_features(elo_df: pd.DataFrame, player_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merges Elo features and player stats into a single feature table ready for modeling.
    Joins on date, team1, team2.
    """
    df = elo_df.merge(
        player_df, 
        on=["date", "team1", "team2"], 
        how="left"
    )
    return df


if __name__ == "__main__":
    elo_df = pd.read_parquet("data/processed/elo.parquet")
    player_df = pd.read_parquet("data/features/player_features.parquet")

    features = build_match_features(elo_df, player_df)

    print(features.shape)
    print(features.columns.tolist())
    print(features.tail(5))

    features.to_parquet("data/features/match_features.parquet", index=False)
    print("Saved to data/features/match_features.parquet")