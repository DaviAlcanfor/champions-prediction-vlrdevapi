import pandas as pd
from src.features.player_stats import WINDOW_DAYS, compute_team_stats


def test_compute_team_stats_uses_only_the_window_before_the_match():
    match_date = pd.Timestamp("2026-09-24")
    player_df = pd.DataFrame({
        "match_date": [
            match_date - pd.Timedelta(days=WINDOW_DAYS - 1),  # inside the window
            match_date - pd.Timedelta(days=WINDOW_DAYS + 1),  # too old
            match_date,                                       # the match itself: future
        ],
        "player_team": ["LOUD"] * 3,
        "acs": [200.0, 999.0, 999.0],
        "kast": [70.0, 0.0, 0.0],
        "adr": [130.0, 0.0, 0.0],
    })

    assert compute_team_stats(player_df, match_date, "LOUD") == {"acs": 200.0, "kast": 70.0, "adr": 130.0}


def test_window_reaches_stage_1_before_champions_2026():
    # Stage 1 2026 ended on 2026-05-10 (China); Champions started on 2026-09-24
    assert pd.Timestamp("2026-09-24") - pd.Timedelta(days=WINDOW_DAYS) <= pd.Timestamp("2026-05-10")
