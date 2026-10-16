from enum import StrEnum


class EventType(StrEnum):
    CHAMPIONS = "champions"
    MASTERS = "masters"
    REGIONAL = "regional"

class EventKeyword(StrEnum):
    CHAMPIONS = "valorant champions"
    MASTERS = "masters"
    ASCENSION = "ascension"

class BracketSlot(StrEnum):
    GROUP_OPENING = "Opening"
    GROUP_WINNERS = "Winners"
    GROUP_ELIMINATION = "Elimination"
    GROUP_DECIDER = "Decider"
    UPPER_R1 = "Upper R1"
    UPPER_R2 = "Upper R2"
    UPPER_FINAL = "Upper Final"
    LOWER_R1 = "Lower R1"
    LOWER_R2 = "Lower R2"
    LOWER_SEMIFINAL = "Lower Semifinal"
    LOWER_FINAL = "Lower Final"
    GRAND_FINAL = "Grand Final"