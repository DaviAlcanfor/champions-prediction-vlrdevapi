import time
import vlrdevapi as vlr
from src.config.constants import TEAM_IDS

def fetch_team_placements() -> dict:
    """
    Fetches historical event placements for all mapped VCT teams.
    Returns a dict: {team_name: [EventPlacement, ...]}
    """
    placements = {}

    for team_name, team_id in TEAM_IDS.items():
        try:
            result = vlr.teams.placements(team_id=team_id)
            placements[team_name] = result

            print(f"  fetched: {team_name} ({len(result)} events)")
            
        except Exception as e:
            print(f"  failed: {team_name} — {e}")
            placements[team_name] = []

        time.sleep(0.5)

    return placements


if __name__ == "__main__":
    print("Fetching team placements...")

    data = fetch_team_placements()
    print(f"\nDone. {sum(len(v) for v in data.values())} total placements fetched.")