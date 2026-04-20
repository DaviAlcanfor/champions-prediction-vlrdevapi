import pandas as pd
from src.config.enums import EventType, EventKeyword
from src.config.constants import VCT_KEYWORDS,EVENT_WEIGHTS


def parse_place(place: str) -> int | None:
    """
    Converts place string ('1st', '3rd') to int. 
    Returns None if unparseable.
    """

    try:
        return int(place[:-2])
    except ValueError:
        return None


def classify_event(event_name: str) -> EventType | None:
    """
    Classifies event as champions, masters or regional. 
    Returns None if not a VCT official event.
    """

    name = event_name.lower()

    if not any(k.lower() in name for k in VCT_KEYWORDS):
        return None
    if EventKeyword.ASCENSION in name:
        return None
    if EventKeyword.MASTERS in name:
        return EventType.MASTERS
    if EventKeyword.CHAMPIONS in name:
        return EventType.CHAMPIONS
    return EventType.REGIONAL


def time_decay(
    event_year: int, 
    reference_year: int = 2026, 
    half_life: float = 2.0
) -> float:
    """
    Exponential decay based on years ago. 
    half_life=2.0 means 2 years ago = half weight.
    """

    years_ago = reference_year - event_year
    return 0.5 ** (years_ago / half_life)


def _parse_event_rows(team_name: str, event) -> list[dict]:
    """
    Parses a single EventPlacement into a list of row dicts. 
    Returns empty list if event is irrelevant.
    """

    event_type = classify_event(event.event_name)
    if event_type is None:
        return []

    rows = []
    event_weight = EVENT_WEIGHTS[event_type]
    decay = time_decay(int(event.year))

    for detail in event.placements:
        place = parse_place(detail.place)
        if place is None:
            continue

        weighted_score = round(event_weight * decay / place, 4)

        rows.append({
            "team": team_name,
            "event_name": event.event_name,
            "event_type": event_type,
            "year": int(event.year),
            "place": place,
            "event_weight": event_weight,
            "decay": round(decay, 4),
            "weighted_score": weighted_score,
        })

    return rows


def process_placements(raw: dict) -> pd.DataFrame:
    """
    Processes raw placements dict into a filtered, 
    weighted DataFrame ready for Elo calculation.
    """

    rows = []

    for team_name, placements in raw.items():
        for event in placements:
            rows.extend(_parse_event_rows(team_name, event))

    df = pd.DataFrame(rows)
    df = df.sort_values(["team", "year"])
    df = df.reset_index(drop=True)

    return df


if __name__ == "__main__":
    from src.ingestion.vlr_api import fetch_team_placements

    raw = fetch_team_placements()
    df = process_placements(raw)
    
    print(df.shape)
    print(df.head(10))

    df.to_parquet("data/processed/placements.parquet", index=False)
    print("Saved to data/processed/placements.parquet")