from collections import Counter
import pandas as pd
from src.features.elo import BASE_ELO
from src.modeling.simulation import TBD, WinProbTable


def _to_percent(prob: float) -> float:
    return round(float(prob) * 100, 1)  # model returns float32


def team_ratings(teams: list[str], elo: dict, team_stats: dict) -> pd.DataFrame:
    """Elo and player stats availability of each team at Champions start."""
    df = pd.DataFrame({
        "team": teams,
        "elo": [round(elo.get(team, BASE_ELO), 1) for team in teams],
        "has_player_stats": [bool(team_stats[team]) for team in teams],
    })

    return df.sort_values("elo", ascending=False).reset_index(drop=True)


def bracket_comparison(predicted: list[dict], real: list[dict]) -> pd.DataFrame:
    """Predicted bracket next to the real one, slot by slot. hit is None for matches not played yet."""
    rows = []

    for pred, act in zip(predicted, real):
        is_played = act["winner"] != TBD

        rows.append({
            "slot": str(pred["slot"]),
            "predicted_match": f"{pred['team1']} x {pred['team2']}",
            "predicted_winner": pred["winner"],
            "prob_%": _to_percent(pred["prob"]),
            "real_match": f"{act['team1']} x {act['team2']}",
            "real_winner": act["winner"],
            "same_match": {pred["team1"], pred["team2"]} == {act["team1"], act["team2"]},
            "hit": pred["winner"] == act["winner"] if is_played else None,
        })

    return pd.DataFrame(rows)


def match_predictions(champions: pd.DataFrame, win_prob: WinProbTable) -> pd.DataFrame:
    """Model pick for every real Champions matchup already played."""
    rows = []

    for _, match in champions.iterrows():
        team1, team2 = match["team1"], match["team2"]
        p_team1 = win_prob[(team1, team2)]

        pick, prob = (team1, p_team1) if p_team1 >= 0.5 else (team2, 1 - p_team1)

        rows.append({
            "stage": match["stage"],
            "match": f"{team1} x {team2}",
            "pick": pick,
            "prob_%": _to_percent(prob),
            "winner": match["winner"],
            "hit": pick == match["winner"],
        })

    return pd.DataFrame(rows)


def title_odds(titles: Counter, playoffs: Counter, teams: list[str], n_simulations: int) -> pd.DataFrame:
    """Monte Carlo playoff and title probability of each team."""
    df = pd.DataFrame({
        "team": teams,
        "playoffs_%": [_to_percent(playoffs[team] / n_simulations) for team in teams],
        "title_%": [_to_percent(titles[team] / n_simulations) for team in teams],
    })

    return df.sort_values("title_%", ascending=False).reset_index(drop=True)
