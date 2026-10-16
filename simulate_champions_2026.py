"""
Champions 2026 validation.

Retrains the model with 2026 data from before Champions, predicts the full bracket
(groups -> grand final) without seeing any Champions result and compares it with reality.

Run from project root after main.py:
    python simulate_champions_2026.py
"""
import pandas as pd

from src.config.constants import CHAMPIONS_2026_GROUPS, MONTE_CARLO_SIMULATIONS
from src.ingestion.vlr_2026 import fetch_2026_matches
from src.modeling.champions_2026 import split_matches, train_champions_model
from src.modeling.evaluation import bracket_comparison, match_predictions, team_ratings, title_odds
from src.modeling.simulation import monte_carlo, predicted_bracket, replay_bracket

TEAMS = [team for group in CHAMPIONS_2026_GROUPS.values() for team in group]


def print_section(title: str, df: pd.DataFrame) -> None:
    print(f"\n{'=' * 20} {title} {'=' * 20}\n")
    print(df.to_string(index=False))


def print_accuracy(label: str, hits: pd.Series) -> None:
    played = hits.dropna().astype(bool)
    print(f"\n{label}: {int(played.sum())}/{len(played)} ({played.mean():.1%})")


if __name__ == "__main__":
    print("Fetching 2026 matches from vlr.gg...")
    pre_champions, champions = split_matches(fetch_2026_matches())

    print(f"\n2026 training matches: {len(pre_champions)} (last: {pre_champions['date'].max().date()})")
    print("Training model (feature pipeline + Optuna)...")

    trained = train_champions_model(pre_champions, TEAMS)
    win_prob = trained["win_prob"]

    print(f"Best walk-forward log_loss: {trained['best_log_loss']:.4f}")

    # team ratings
    ratings = team_ratings(TEAMS, trained["elo"], trained["team_stats"])
    print_section("Ratings at Champions start", ratings)

    # predicted bracket vs real
    results = list(champions[["team1", "team2", "winner"]].itertuples(index=False, name=None))

    bracket = bracket_comparison(
        predicted_bracket(win_prob, CHAMPIONS_2026_GROUPS),
        replay_bracket(results, CHAMPIONS_2026_GROUPS),
    )
    print_section("Predicted bracket vs real", bracket)

    final = bracket.iloc[-1]
    print(f"\nPredicted champion: {final['predicted_winner']}  |  real champion: {final['real_winner']}")
    print(f"Same matchup as real: {bracket['same_match'].sum()}/{len(bracket)}")
    print_accuracy("Right winner per slot", bracket["hit"])

    # model pick on the real matchups
    predictions = match_predictions(champions, win_prob)
    print_section("Model pick on real matchups", predictions)
    print_accuracy("Accuracy", predictions["hit"])

    # monte carlo
    titles, playoffs = monte_carlo(win_prob, CHAMPIONS_2026_GROUPS, MONTE_CARLO_SIMULATIONS)

    odds = title_odds(titles, playoffs, TEAMS, MONTE_CARLO_SIMULATIONS)
    print_section(f"Monte Carlo ({MONTE_CARLO_SIMULATIONS:,} runs)", odds)
