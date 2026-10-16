import pandas as pd
import pytest
from src.modeling.train import elo_baseline_metrics, walk_forward_folds


def test_elo_baseline_metrics_scores_the_elo_favorite():
    df = pd.DataFrame({
        "win_prob_team1": [0.8, 0.3, 0.6, 0.4],
        "result": [1, 0, 0, 1],
    })

    metrics = elo_baseline_metrics(df)

    assert metrics["elo_accuracy"] == 0.5
    assert metrics["elo_log_loss"] == pytest.approx(0.6031, abs=1e-4)


def test_walk_forward_folds_never_train_on_future_events():
    df = pd.DataFrame({
        "date": pd.date_range("2025-01-01", periods=30, freq="D"),
        "event": ["A"] * 12 + ["B"] * 12 + ["C"] * 6,
    })

    folds = list(walk_forward_folds(df))

    assert [event for event, _, _ in folds] == ["B", "C"]
    for _, train, test in folds:
        assert train["date"].max() < test["date"].min()
