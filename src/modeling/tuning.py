import optuna
import yaml
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import log_loss

from src.modeling.train import (
    get_tournament_order,
    fill_missing_stats,
    compute_sample_weights,
    FEATURE_COLS,
    TARGET_COL,
)


def objective(trial: optuna.Trial, df: pd.DataFrame) -> float:
    """
    Optuna objective function — returns mean log_loss across walk-forward folds.
    Lower is better.
    """
    alpha = trial.suggest_float("alpha", 0.0, 1.0)
    decay_days = trial.suggest_int("decay_days", 60, 365)

    max_depth = trial.suggest_int("max_depth", 2, 6)
    learning_rate = trial.suggest_float("learning_rate", 0.01, 0.3, log=True)
    n_estimators = trial.suggest_int("n_estimators", 50, 300)
    subsample = trial.suggest_float("subsample", 0.5, 1.0)
    colsample_bytree = trial.suggest_float("colsample_bytree", 0.5, 1.0)

    events = get_tournament_order(df)
    fold_losses = []

    for i in range(1, len(events)):
        train_events = events[:i]
        test_event = events[i]

        train = df[df["event"].isin(train_events)]
        test = df[df["event"] == test_event]

        if len(train) < 10 or len(test) < 3:
            continue

        X_train = train[FEATURE_COLS]
        y_train = train[TARGET_COL]
        X_test = test[FEATURE_COLS]
        y_test = test[TARGET_COL]

        weights = compute_sample_weights(train, alpha=alpha, decay_days=decay_days)

        model = XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            eval_metric="logloss",
            verbosity=0,
        )
        model.fit(X_train, y_train, sample_weight=weights)

        y_prob = model.predict_proba(X_test)[:, 1]
        fold_losses.append(log_loss(y_test, y_prob))

    return np.mean(fold_losses) if fold_losses else float("inf")


if __name__ == "__main__":
    df = pd.read_parquet("data/features/match_features.parquet")
    df = fill_missing_stats(df)

    study = optuna.create_study(direction="minimize")
    study.optimize(lambda trial: objective(trial, df), n_trials=50, show_progress_bar=True)

    print("\nBest params:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")
    print(f"Best log_loss: {study.best_value:.4f}")

    # saving the params
    best_params = study.best_params
    with open("models/best_params.yaml", "w") as f:
        yaml.dump(best_params, f, default_flow_style=False)
    print("Saved to models/best_params.yaml")