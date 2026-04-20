import kagglehub
import shutil
import os
from src.config.constants import DATASETS

RAW_DATA_PATH = os.path.join("data", "raw")

def download_datasets():
    for name, slug in DATASETS.items():
        print(f"Downloading {name}...")
        
        cache_path = kagglehub.dataset_download(slug)
        
        destination = os.path.join(RAW_DATA_PATH, name)
        os.makedirs(destination, exist_ok=True)
        
        for root, dirs, files in os.walk(cache_path):
            csvs = [f for f in files if f.endswith(".csv")]
            if not csvs:
                continue
            
            # get the relative path of the subfolder in relation to the cache
            relative_subfolder = os.path.relpath(root, cache_path)
            subfolder_destination = os.path.join(destination, relative_subfolder)
            os.makedirs(subfolder_destination, exist_ok=True)
            
            for file in csvs:
                origin = os.path.join(root, file)
                shutil.copy2(origin, subfolder_destination)
                print(f"  copied: {relative_subfolder}/{file}")
        
        print(f"  saved at: {destination}\n")

if __name__ == "__main__":
    download_datasets()