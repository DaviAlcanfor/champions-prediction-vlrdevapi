import random
from collections import Counter
from typing import Callable

from src.config.enums import BracketSlot

TBD = "TBD"

# play(slot, team_a, team_b) -> (winner, loser)
PlayFn = Callable[[str, str, str], tuple[str, str]]

# {(team_a, team_b): probability that team_a wins}
WinProbTable = dict[tuple[str, str], float]


def simulate_group(play: PlayFn, name: str, teams: tuple[str, str, str, str]) -> tuple[str, str]:
    """
    GSL group (double elimination, 4 teams). Opening matches: teams[0] x teams[1], teams[2] x teams[3].
    Returns (1st, 2nd): winners match winner and decider winner.
    """
    def slot(group_slot: BracketSlot) -> str:
        return f"Group {name} - {group_slot}"

    winner1, loser1 = play(slot(BracketSlot.GROUP_OPENING), teams[0], teams[1])
    winner2, loser2 = play(slot(BracketSlot.GROUP_OPENING), teams[2], teams[3])

    first, winners_match_loser = play(slot(BracketSlot.GROUP_WINNERS), winner1, winner2)
    survivor, _ = play(slot(BracketSlot.GROUP_ELIMINATION), loser1, loser2)
    second, _ = play(slot(BracketSlot.GROUP_DECIDER), winners_match_loser, survivor)

    return first, second


def simulate_playoffs(play: PlayFn, opening_matches: list[tuple[str, str]]) -> str:
    """
    8-team double elimination bracket. Returns the champion.
    Lower round 2 crosses sides: upper semi 2 loser meets lower round 1 match 1 winner.
    """
    (upper1, lower1), (upper2, lower2), (upper3, lower3), (upper4, lower4) = (
        play(BracketSlot.UPPER_R1, a, b) for a, b in opening_matches
    )

    # upper bracket
    semi1_winner, semi1_loser = play(BracketSlot.UPPER_R2, upper1, upper2)
    semi2_winner, semi2_loser = play(BracketSlot.UPPER_R2, upper3, upper4)

    # lower bracket
    lower_r1_a, _ = play(BracketSlot.LOWER_R1, lower1, lower2)
    lower_r1_b, _ = play(BracketSlot.LOWER_R1, lower3, lower4)

    lower_r2_a, _ = play(BracketSlot.LOWER_R2, semi2_loser, lower_r1_a)
    lower_r2_b, _ = play(BracketSlot.LOWER_R2, semi1_loser, lower_r1_b)

    lower_semi_winner, _ = play(BracketSlot.LOWER_SEMIFINAL, lower_r2_a, lower_r2_b)

    # finals
    upper_champion, upper_final_loser = play(BracketSlot.UPPER_FINAL, semi1_winner, semi2_winner)
    lower_champion, _ = play(BracketSlot.LOWER_FINAL, upper_final_loser, lower_semi_winner)

    champion, _ = play(BracketSlot.GRAND_FINAL, upper_champion, lower_champion)
    return champion


def simulate_tournament(play: PlayFn, groups: dict[str, tuple]) -> tuple[str, list[str]]:
    """
    4 GSL groups (A-D) into an 8-team bracket.
    Returns (champion, teams that reached playoffs).
    """
    (a1, a2), (b1, b2), (c1, c2), (d1, d2) = (
        simulate_group(play, name, teams) for name, teams in groups.items()
    )

    opening_matches = [(d1, a2), (c1, b2), (b1, d2), (a1, c2)]
    champion = simulate_playoffs(play, opening_matches)

    return champion, [a1, a2, b1, b2, c1, c2, d1, d2]


def monte_carlo(
    win_prob: WinProbTable,
    groups: dict[str, tuple],
    n_simulations: int,
    seed: int = 42,
) -> tuple[Counter, Counter]:
    """Runs the tournament n times, drawing each match. Returns (title counts, playoff counts)."""
    random.seed(seed)

    def random_play(slot, team_a, team_b):
        if random.random() < win_prob[(team_a, team_b)]:
            return team_a, team_b
        return team_b, team_a

    titles = Counter()
    playoffs = Counter()

    for _ in range(n_simulations):
        champion, qualified = simulate_tournament(random_play, groups)

        titles[champion] += 1
        playoffs.update(qualified)

    return titles, playoffs


def predicted_bracket(win_prob: WinProbTable, groups: dict[str, tuple]) -> list[dict]:
    """Most likely bracket: the favorite wins every match. Returns every match in bracket order."""
    matches = []

    def favorite_play(slot, team_a, team_b):
        prob_a = win_prob[(team_a, team_b)]

        if prob_a >= 0.5:
            winner, loser, prob = team_a, team_b, prob_a
        else:
            winner, loser, prob = team_b, team_a, 1 - prob_a

        matches.append({"slot": slot, "team1": team_a, "team2": team_b, "winner": winner, "prob": prob})
        return winner, loser

    simulate_tournament(favorite_play, groups)
    return matches


def replay_bracket(results: list[tuple[str, str, str]], groups: dict[str, tuple]) -> list[dict]:
    """
    Real bracket from played results [(team1, team2, winner), ...] in date order.
    Matches not played yet come out as TBD. Returns every match in bracket order.
    """
    remaining = list(results)
    matches = []

    def real_play(slot, team_a, team_b):
        played = next(
            (r for r in remaining if {r[0], r[1]} == {team_a, team_b}),
            None,
        )

        if played is None:
            winner, loser = TBD, TBD
        else:
            remaining.remove(played)  # same pair can meet twice (groups + playoffs)
            winner = played[2]
            loser = team_b if winner == team_a else team_a

        matches.append({"slot": slot, "team1": team_a, "team2": team_b, "winner": winner})
        return winner, loser

    simulate_tournament(real_play, groups)
    return matches
