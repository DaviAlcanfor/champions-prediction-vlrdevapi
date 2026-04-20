import yaml
import joblib
import pandas as pd
import numpy as np
from xgboost import XGBClassifier

from src.modeling.train import (
    fill_missing_stats,
    compute_sample_weights,
    FEATURE_COLS,
    TARGET_COL,
)


def load_best_params() -> dict:
    """Loads best hyperparameters from Optuna tuning."""
    with open("models/best_params.yaml") as f:
        return yaml.safe_load(f)


def train_final_model(df: pd.DataFrame, params: dict) -> XGBClassifier:
    """Trains final model on all available data using best params."""
    alpha = params.pop("alpha")
    decay_days = params.pop("decay_days")

    X = df[FEATURE_COLS]
    y = df[TARGET_COL]
    
    weights = compute_sample_weights(
        df, 
        alpha=alpha, 
        decay_days=decay_days
    )

    model = XGBClassifier(
        **params, 
        eval_metric="logloss", 
        verbosity=0
    )
    model.fit(X, y, sample_weight=weights)

    return model


def predict_match(model: XGBClassifier, features: dict) -> float:
    """
    Predicts win probability for team1 given match features.
    Returns a float between 0 and 1.
    """
    X = pd.DataFrame([features])[FEATURE_COLS]
    return model.predict_proba(X)[0][1]


if __name__ == "__main__":
    df = pd.read_parquet("data/features/match_features.parquet")
    df = fill_missing_stats(df)

    params = load_best_params()
    model = train_final_model(df, params)

    joblib.dump(model, "models/final_model.pkl")
    print("Model saved to models/final_model.pkl")