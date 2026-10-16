import os
from itertools import combinations

import optuna
import pandas as pd
from xgboost import XGBClassifier

from src.config.constants import (
    CHAMPIONS_2026_EVENT,
    CHAMPIONS_2026_START,
    PROCESSED_PATH,
    RANDOM_SEED,
    TUNING_TRIALS,
)
from src.features.builder import build_match_features
from src.features.elo import BASE_ELO, calculate_dynamic_elo, calculate_initial_elo
from src.features.player_stats import build_player_features, compute_team_stats, load_player_stats
from src.modeling.predict import train_final_model
from src.modeling.train import FEATURE_COLS, elo_baseline_log_loss, fill_missing_stats
from src.modeling.tuning import objective
from src.processing.placements import placements_before
from src.processing.player_stats_2026 import load_player_stats_2026

STAT_NAMES = ["acs", "kast", "adr"]


def split_matches(matches_2026: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Splits 2026 matches into (before Champions, Champions)."""
    is_champions = matches_2026["event"] == CHAMPIONS_2026_EVENT
    is_before = matches_2026["date"] < pd.Timestamp(CHAMPIONS_2026_START)

    return matches_2026[~is_champions & is_before], matches_2026[is_champions]


def _load_training_matches(pre_champions: pd.DataFrame) -> pd.DataFrame:
    """2025 Kaggle matches + 2026 matches before Champions, in date order."""
    matches_2025 = pd.read_parquet(os.path.join(PROCESSED_PATH, "matches.parquet"))
    matches = pd.concat([matches_2025, pre_champions], ignore_index=True)

    return matches.sort_values("date", kind="stable").reset_index(drop=True)


def _initial_elo(matches: pd.DataFrame) -> dict[str, float]:
    """Initial Elo from placements before the first training match (no leakage)."""
    placements = pd.read_parquet(os.path.join(PROCESSED_PATH, "placements.parquet"))
    first_year = matches["date"].min().year

    return calculate_initial_elo(placements_before(placements, first_year))


def _final_elo(elo_df: pd.DataFrame) -> dict[str, float]:
    """Last known Elo of each team."""
    columns = ["date", "team", "elo"]
    team1 = elo_df[["date", "team1", "elo1_after"]].set_axis(columns, axis=1)
    team2 = elo_df[["date", "team2", "elo2_after"]].set_axis(columns, axis=1)

    return (
        pd.concat([team1, team2])
        .sort_values("date", kind="stable")
        .groupby("team")["elo"]
        .last()
        .to_dict()
    )


def build_training_data(pre_champions: pd.DataFrame) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    """
    Runs the feature pipeline over 2025 + 2026 pre-Champions matches.
    Returns (feature table, final Elo per team, player stats).
    """
    matches = _load_training_matches(pre_champions)

    elo_df = calculate_dynamic_elo(matches, _initial_elo(matches))

    event_end_dates = pre_champions.groupby("event")["date"].max()
    player_df = pd.concat(
        [load_player_stats(), load_player_stats_2026(event_end_dates)],
        ignore_index=True,
    )

    features = build_match_features(elo_df, build_player_features(elo_df, player_df))

    return features, _final_elo(elo_df), player_df


def tune_and_train(features: pd.DataFrame) -> tuple[XGBClassifier, float]:
    """Optuna walk-forward tuning, then trains on all data. Returns (model, best log_loss)."""
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED),
    )
    study.optimize(lambda trial: objective(trial, features), n_trials=TUNING_TRIALS)

    model = train_final_model(features, dict(study.best_params))

    return model, study.best_value


def team_stats_at_start(player_df: pd.DataFrame, teams: list[str]) -> dict[str, dict]:
    """Mean ACS/KAST/ADR of each team in the WINDOW_DAYS before Champions ({} if no data)."""
    start = pd.Timestamp(CHAMPIONS_2026_START)

    return {team: compute_team_stats(player_df, start, team) or {} for team in teams}


def _matchup_features(team_a: str, team_b: str, elo: dict, team_stats: dict, medians: pd.Series) -> dict:
    """Feature row for team_a (team1) vs team_b (team2). Missing stats -> training median."""
    elo_a = elo.get(team_a, BASE_ELO)
    elo_b = elo.get(team_b, BASE_ELO)

    row = {"elo_diff": elo_a - elo_b}

    for stat in STAT_NAMES:
        row[f"{stat}_team1"] = team_stats[team_a].get(stat, medians[f"{stat}_team1"])
        row[f"{stat}_team2"] = team_stats[team_b].get(stat, medians[f"{stat}_team2"])

    return row


def build_win_prob_table(
    model: XGBClassifier,
    teams: list[str],
    elo: dict,
    team_stats: dict,
    medians: pd.Series,
) -> dict[tuple[str, str], float]:
    """
    P(team_a beats team_b) for every pair, with features frozen at Champions start.
    Each pair is averaged over both team orders, so P(a, b) == 1 - P(b, a).
    """
    table = {}

    for team_a, team_b in combinations(teams, 2):
        X = pd.DataFrame([
            _matchup_features(team_a, team_b, elo, team_stats, medians),
            _matchup_features(team_b, team_a, elo, team_stats, medians),
        ])[FEATURE_COLS]

        p_a_first, p_b_first = model.predict_proba(X)[:, 1]
        p = (p_a_first + 1 - p_b_first) / 2

        table[(team_a, team_b)] = p
        table[(team_b, team_a)] = 1 - p

    return table


def train_champions_model(pre_champions: pd.DataFrame, teams: list[str]) -> dict:
    """
    Full pipeline: features -> tuning -> final model -> win probability table.
    Returns everything the reports need.
    """
    features, elo, player_df = build_training_data(pre_champions)

    medians = features[FEATURE_COLS].median()
    features = fill_missing_stats(features)

    model, best_log_loss = tune_and_train(features)

    team_stats = team_stats_at_start(player_df, teams)
    win_prob = build_win_prob_table(model, teams, elo, team_stats, medians)

    return {
        "elo": elo,
        "team_stats": team_stats,
        "win_prob": win_prob,
        "best_log_loss": best_log_loss,
        "elo_log_loss": elo_baseline_log_loss(features),
    }
