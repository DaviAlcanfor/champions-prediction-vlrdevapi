import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import log_loss, accuracy_score
from src.config.constants import EVENT_WEIGHTS, STAGE_K
from src.config.enums import EventType
from src.processing.placements import classify_event


FEATURE_COLS = [
    "elo_diff",
    "win_prob_team1",
    "acs_team1", "kast_team1", "adr_team1",
    "acs_team2", "kast_team2", "adr_team2",
]
TARGET_COL = "result"


def get_tournament_order(df: pd.DataFrame) -> list[str]:
    """Returns events sorted chronologically by their first match date."""
    return (
        df.groupby("event")["date"]
        .min()
        .sort_values()
        .index.tolist()
    )


def fill_missing_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Fills NaN player stats with median of each column."""
    stat_cols = ["acs_team1", "kast_team1", "adr_team1", "acs_team2", "kast_team2", "adr_team2"]
    
    for col in stat_cols:
        df[col] = df[col].fillna(df[col].median())
    
    return df


def walk_forward_folds(df: pd.DataFrame):
    """
    Yields (test_event, train, test): trains on all past tournaments, tests on the next one.
    Skips folds too small to evaluate.
    """
    events = get_tournament_order(df)

    for i in range(1, len(events)):
        train = df[df["event"].isin(events[:i])]
        test = df[df["event"] == events[i]]

        if len(train) < 10 or len(test) < 3:
            continue

        yield events[i], train, test


def elo_baseline_metrics(df: pd.DataFrame) -> dict:
    """
    Accuracy and log loss of the Elo formula alone (win_prob_team1), no model.
    The model is only worth it if it beats these numbers.
    """
    y_true = df[TARGET_COL]
    y_prob = df["win_prob_team1"]

    return {
        "elo_accuracy": round(accuracy_score(y_true, (y_prob >= 0.5).astype(int)), 4),
        "elo_log_loss": round(log_loss(y_true, y_prob, labels=[0, 1]), 4),
    }


def elo_baseline_log_loss(df: pd.DataFrame) -> float:
    """Mean walk-forward log loss of the Elo formula alone, comparable to the tuning objective."""
    return float(np.mean([
        elo_baseline_metrics(test)["elo_log_loss"]
        for _, _, test in walk_forward_folds(df)
    ]))


def compute_sample_weights(
    df: pd.DataFrame,
    alpha: float = 0.5,
    decay_days: int = 180
) -> np.ndarray:
    """
    Computes sample weights based on time decay and event importance.
    weight = (time_weight ** alpha) * (event_weight ** (1 - alpha))
    """

    if df.empty:
        return np.array([])

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    reference_date = df["date"].max()
    days_ago = (reference_date - df["date"]).dt.days

    time_weight = np.exp(-days_ago / decay_days)

    event_series = df["event"].map(classify_event)
    event_series = event_series.fillna(EventType.REGIONAL)

    event_weight = event_series.map(EVENT_WEIGHTS).fillna(0.7)

    weights = (time_weight ** alpha) * (event_weight ** (1 - alpha))
    return weights.to_numpy()


def walk_forward_validation(df: pd.DataFrame) -> list[dict]:
    """
    Validates model using walk-forward by tournament.
    Trains on all past tournaments, tests on the next one.
    Returns list of results per fold.
    """
    results = []

    for test_event, train, test in walk_forward_folds(df):
        X_train = train[FEATURE_COLS]
        y_train = train[TARGET_COL]
        X_test = test[FEATURE_COLS]
        y_test = test[TARGET_COL]

        model = XGBClassifier(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.1,
            eval_metric="logloss",
            verbosity=0,
        )
        weights = compute_sample_weights(train, alpha=0.5)
        model.fit(X_train, y_train, sample_weight=weights)

        y_prob = model.predict_proba(X_test)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)

        results.append({
            "test_event": test_event,
            "train_size": len(train),
            "test_size": len(test),
            "accuracy": round(accuracy_score(y_test, y_pred), 4),
            "log_loss": round(log_loss(y_test, y_prob), 4),
            **elo_baseline_metrics(test),
        })

        print(f"  [{test_event}] acc={results[-1]['accuracy']} logloss={results[-1]['log_loss']}")

    return results




if __name__ == "__main__":
    df = pd.read_parquet("data/features/match_features.parquet")
    df = fill_missing_stats(df)

    print(f"Dataset: {df.shape}")
    print("Running walk-forward validation...\n")

    results = walk_forward_validation(df)

    results_df = pd.DataFrame(results)

    print("\nSummary:")
    print(results_df)
    print(f"\nMean accuracy: {results_df['accuracy'].mean():.4f}")
    print(f"Mean log_loss: {results_df['log_loss'].mean():.4f}")
    print(f"Elo baseline accuracy: {results_df['elo_accuracy'].mean():.4f}")
    print(f"Elo baseline log_loss: {results_df['elo_log_loss'].mean():.4f}")