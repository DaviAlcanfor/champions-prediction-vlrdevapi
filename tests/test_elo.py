import pandas as pd
import pytest
from src.config.constants import SEASON_REGRESSION
from src.features.elo import calculate_dynamic_elo, regress_to_mean


def test_regress_to_mean_pulls_ratings_toward_average():
    assert regress_to_mean({"A": 100.0, "B": 0.0}, 0.25) == {"A": 87.5, "B": 12.5}


def test_dynamic_elo_regresses_between_seasons():
    matches = pd.DataFrame({
        "date": pd.to_datetime(["2025-06-01", "2026-01-10"]),
        "team1": ["A", "A"],
        "team2": ["B", "B"],
        "winner": ["A", "A"],
        "stage": ["Playoffs", "Playoffs"],
        "event": ["VCT 2025: EMEA Stage 1", "VCT 2026: EMEA Kickoff"],
    })

    elo_df = calculate_dynamic_elo(matches, {"A": 100.0, "B": 0.0})
    first, second = elo_df.iloc[0], elo_df.iloc[1]

    mean = (first["elo1_after"] + first["elo2_after"]) / 2
    expected_a = mean + (first["elo1_after"] - mean) * (1 - SEASON_REGRESSION)

    assert second["elo1_before"] == pytest.approx(expected_a, abs=0.01)
