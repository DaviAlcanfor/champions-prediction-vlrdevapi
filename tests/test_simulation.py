from src.config.constants import CHAMPIONS_2026_GROUPS
from src.config.enums import BracketSlot
from src.modeling.simulation import TBD, predicted_bracket, replay_bracket, simulate_group, simulate_playoffs

TEAMS = [team for group in CHAMPIONS_2026_GROUPS.values() for team in group]


def test_group_first_is_winners_match_winner_and_second_is_decider_winner():
    # alphabetical order decides every match: A beats everyone, B beats C and D...
    def play(slot, team_a, team_b):
        return min(team_a, team_b), max(team_a, team_b)

    assert simulate_group(play, "X", ("A", "B", "C", "D")) == ("A", "B")


def test_predicted_bracket_crowns_the_team_that_beats_everyone():
    win_prob = {}
    for a in TEAMS:
        for b in TEAMS:
            if a != b:
                win_prob[(a, b)] = 0.9 if a == "LOUD" else 0.1 if b == "LOUD" else 0.5

    bracket = predicted_bracket(win_prob, CHAMPIONS_2026_GROUPS)

    assert len(bracket) == 34  # 4 groups x 5 matches + 14 playoff matches
    assert bracket[-1]["winner"] == "LOUD"


def test_replay_without_results_leaves_every_match_tbd():
    bracket = replay_bracket([], CHAMPIONS_2026_GROUPS)

    assert len(bracket) == 34
    assert all(match["winner"] == TBD for match in bracket)


def test_playoffs_lower_round_2_crosses_sides():
    played = []

    def play(slot, team_a, team_b):
        played.append((slot, team_a, team_b))
        return min(team_a, team_b), max(team_a, team_b)

    simulate_playoffs(play, [("A", "B"), ("C", "D"), ("E", "F"), ("G", "H")])

    # upper semi losers: C (semi 1) and G (semi 2); lower round 1 winners: B (match 1) and F (match 2)
    lower_r2 = [(a, b) for slot, a, b in played if slot == BracketSlot.LOWER_R2]
    assert lower_r2 == [("G", "B"), ("C", "F")]
