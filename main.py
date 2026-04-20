import subprocess
import sys

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
    
    for module in PIPELINE:
        result = subprocess.run([sys.executable, "-m", module])

        if result.returncode != 0:
            print(f"Failed at {module}")
            sys.exit(1)