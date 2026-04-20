from enum import StrEnum


class EventType(StrEnum):
    CHAMPIONS = "champions"
    MASTERS = "masters"
    REGIONAL = "regional"

class EventKeyword(StrEnum):
    CHAMPIONS = "valorant champions"
    MASTERS = "masters"
    ASCENSION = "ascension"