import os
import subprocess
import sys
from src.config.constants import PROCESSED_PATH, FEATURES_PATH, MODELS_PATH

OUTPUT_DIRS = [PROCESSED_PATH, FEATURES_PATH, MODELS_PATH]

PIPELINE = [
    "src.ingestion.kaggle_loader",
    "src.processing.normalizer",
    "src.ingestion.vlr_api",
    "src.processing.placements",
    "src.features.elo",
    "src.features.player_stats",
    "src.features.builder",
    "src.modeling.tuning",
    "src.modeling.predict",
]

if __name__ == "__main__":
    
    for path in OUTPUT_DIRS:
        os.makedirs(path, exist_ok=True)

    for module in PIPELINE:
        result = subprocess.run([sys.executable, "-m", module])

        if result.returncode != 0:
            print(f"Failed at {module}")
            sys.exit(1)