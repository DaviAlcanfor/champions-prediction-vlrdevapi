import os
from glob import glob
import pandas as pd
from src.config.constants import (
    VCT_2025_PATH,
    RELEVANT_COLUMNS,
    PROCESSED_PATH,
)

def normalize_matches() -> pd.DataFrame:
    all_matches = []   

    for filepath in glob(os.path.join(VCT_2025_PATH, "**", "matches.csv"), recursive=True):
        df = pd.read_csv(filepath)
        df = df[RELEVANT_COLUMNS]

        event_name = os.path.basename(os.path.dirname(filepath))
        df["event"] = event_name

        all_matches.append(df)

    if not all_matches:
        raise FileNotFoundError(f"No matches.csv found in {VCT_2025_PATH}")

    combined = pd.concat(all_matches, ignore_index=True)  # <- faltava isso

    combined["date"] = pd.to_datetime(combined["date"], errors="coerce")
    combined = combined.dropna(subset=["date"])

    # fix types
    combined["match_id"] = combined["match_id"].astype(int)
    combined["score1"] = combined["score1"].astype("int8")
    combined["score2"] = combined["score2"].astype("int8")

    # sort by date
    combined = combined.sort_values("date").reset_index(drop=True)

    return combined


if __name__ == "__main__":
    os.makedirs(PROCESSED_PATH, exist_ok=True)
    
    df = normalize_matches()
    df.to_parquet(os.path.join(PROCESSED_PATH, "matches.parquet"), index=False)

    print(f"Saved {len(df)} matches to data/processed/matches.parquet")
    print(df.dtypes)