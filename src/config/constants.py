import os
from src.config.enums import EventType

# Dataset slugs
DATASETS = {
    "vct_2025": "piyush86kumar/valorant-vct-2025-all-events",
    "vct_historic": "ryanluong1/valorant-champion-tour-2021-2023-data"
}

# Raw data paths
VCT_2025_PATH = os.path.join("data", "raw", "vct_2025")
VCT_HISTORIC_PATH = os.path.join("data", "raw", "vct_historic")

# Event paths
MASTERS_BANGKOK_2025_PATH = os.path.join(VCT_2025_PATH, "Valorant Masters Bangkok 2025_csvs")
MASTERS_TORONTO_2025_PATH = os.path.join(VCT_2025_PATH, "Valorant Masters Toronto 2025_csvs")
CHAMPIONS_2025_PATH = os.path.join(VCT_2025_PATH, "Valorant Champions 2025_csvs")

# Folder paths
PROCESSED_PATH = os.path.join("data", "processed")
FEATURES_PATH = os.path.join("data", "features")
MODELS_PATH = "models"

RELEVANT_COLUMNS = ["date", "match_id", "team1", "score1", "team2", "score2", "winner", "stage"]

# Showmatch teams
INVALID_TEAMS = {
    "Team tarik", "Team Toast", "Team World", "Team Thailand",
    "Team International", "Precise Defeat", "Pure Aim", "Glory Once Again",
    "Team Alpha", "Team Omega", "Team France", "Team EMEA"
}

# All valid teams with name, id
TEAM_IDS = {
    "100 Thieves": 19510,
    "2Game Esports": 15072,
    "All Gamers": 1119,
    "Apeks": 11479,
    "BBL Esports": 397,
    "BOOM Esports": 466,
    "Bilibili Gaming": 12010,
    "Cloud9": 188,
    "DRX": 9749,
    "DetonatioN FocusMe": 278,
    "Dragon Ranger Gaming": 11981,
    "EDward Gaming": 1120,
    "Evil Geniuses": 9547,
    "FNATIC": 2593,
    "FURIA": 2406,
    "FUT Esports": 20697,
    "FunPlus Phoenix": 11328,
    "G2 Esports": 257,
    "GIANTX": 15119,
    "Gen.G": 17,
    "Gentle Mates": 21093,
    "Global Esports": 918,
    "JDG Esports": 13576,
    "KOI": 7035,
    "KRÜ Esports": 2355,
    "Karmine Corp": 8877,
    "LEVIATÁN": 2359,
    "LOUD": 6961,
    "MIBR": 16606,
    "NRG": 1034,
    "Natus Vincere": 4915,
    "Nongshim RedForce": 11060,
    "Nova Esports": 12064,
    "Paper Rex": 624,
    "Rex Regum Qeon": 878,
    "Sentinels": 2,
    "T1": 14,
    "TALON": 8304,
    "TYLOO": 731,
    "Team Heretics": 1001,
    "Team Liquid": 7055,
    "Team Secret": 6199,
    "Team Vitality": 2059,
    "Titan Esports Club": 14137,
    "Trace Esports": 12685,
    "Wolves Esports": 13790,
    "Xi Lai Gaming": 13581,
    "ZETA DIVISION": 6997,
}

TEAM_NAME_ALIASES = {
    "Guangzhou Huadu Bilibili Gaming(Bilibili Gaming)": "Bilibili Gaming",
    "VISA KRÜ(KRÜ Esports)": "KRÜ Esports",
    "JD Mall JDG Esports(JDG Esports)": "JDG Esports",
    "Movistar KOI(KOI)": "KOI",
    "JD Gaming": "JDG Esports",
    "KIWOOM DRX": "DRX",
    "Mega Minors": "NRG",  # Kaggle 2026 mislabels NRG
}

# Stages missing from STAGE_K would be skipped by the Elo calculation
STAGE_ALIASES = {
    "Play-Ins": "Group Stage",
}


# Event type weights for Elo initial calculation
EVENT_WEIGHTS = {
    EventType.CHAMPIONS: 1.5,
    EventType.MASTERS: 1.2,
    EventType.REGIONAL: 0.7,
}


# Keywords to identify official VCT events
VCT_KEYWORDS = ["Valorant Champions", "Valorant Masters", "VCT", "Champions Tour"]

# Stage pontuations 
STAGE_K = {
    "Group Stage": 20,
    "Swiss Stage": 20,
    "Main Event": 24,
    "Playoffs": 32,
}

# Share of the distance to the mean Elo every team loses when a new season starts
SEASON_REGRESSION = 0.25


# Champions 2026 validation
KAGGLE_2026_PLAYER_STATS_PATH = os.path.join(VCT_HISTORIC_PATH, "vct_2026", "players_stats", "players_stats.csv")

CHAMPIONS_2026_EVENT = "Valorant Champions 2026"
CHAMPIONS_2026_START = "2026-09-24"

# GSL groups, in opening match order: (0 x 1), (2 x 3)
CHAMPIONS_2026_GROUPS = {
    "A": ("Team Liquid", "Paper Rex", "TYLOO", "G2 Esports"),
    "B": ("Nongshim RedForce", "NRG", "Karmine Corp", "Xi Lai Gaming"),
    "C": ("Global Esports", "Team Vitality", "LOUD", "EDward Gaming"),
    "D": ("100 Thieves", "T1", "JDG Esports", "FUT Esports"),
}

TUNING_TRIALS = 50
MONTE_CARLO_SIMULATIONS = 10_000
RANDOM_SEED = 42