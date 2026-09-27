"""
Milestone 1 pipeline runner: download -> clean -> feature engineering.

Run: python main.py
"""

from src.data.download import main as download_main
from src.data.clean import main as clean_main
from src.features.engineer import main as feature_main


def run_pipeline():
    print("=== Step 1: Downloading raw data ===")
    download_main()

    print("\n=== Step 2: Cleaning data ===")
    clean_main()

    print("\n=== Step 3: Engineering features + targets ===")
    feature_main()

    print("\nPipeline complete. Check data/processed/*_features.csv")
    print("Next: open notebooks/01_data_collection.ipynb to sanity-check the output.")


if __name__ == "__main__":
    run_pipeline()
