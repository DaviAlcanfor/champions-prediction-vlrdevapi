import pandas as pd
from src.processing.placements import placements_before


def test_placements_before_drops_cutoff_year_and_later():
    df = pd.DataFrame({
        "team": ["LOUD", "LOUD", "LOUD"],
        "year": [2023, 2024, 2025],
        "weighted_score": [1.0, 1.0, 1.0],
    })

    assert placements_before(df, 2025)["year"].tolist() == [2023, 2024]
