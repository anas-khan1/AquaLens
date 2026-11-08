"""Download and prepare datasets for Marine Reef Vision.

Checks for Kaggle credentials (environment variables or ~/.kaggle/kaggle.json).
If available, downloads:
    - Fish dataset: crowai/large-scale-fish-dataset
    - Coral dataset: sonainjamil/coral-reef-bleaching-dataset

If credentials are not found, or if --placeholder is specified, it generates
a structured synthetic dataset using OpenCV so the entire pipeline can be
trained, evaluated, and tested immediately.

Usage:
    python data/download_data.py [--placeholder] [--task {fish,coral,all}]
"""

import argparse
import logging
import os
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset import KaggleDownloader, PlaceholderDatasetGenerator
from src.utils import load_config, setup_logging

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download or generate marine & coral datasets."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(PROJECT_ROOT / "config.yaml"),
        help="Path to config.yaml",
    )
    parser.add_argument(
        "--placeholder",
        action="store_true",
        help="Force generation of synthetic placeholder dataset without calling Kaggle API.",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="all",
        choices=["all", "fish", "coral"],
        help="Which dataset to download or generate.",
    )
    parser.add_argument(
        "--num-samples-per-class",
        type=int,
        default=40,
        help="Number of placeholder samples per class if generating synthetic data.",
    )
    return parser.parse_args()


def print_kaggle_instructions() -> None:
    """Print clear, step-by-step instructions for Kaggle setup."""
    instructions = """
================================================================================
                    KAGGLE API CREDENTIALS SETUP GUIDE
================================================================================

To download the real Kaggle datasets automatically:

Option 1: Environment Variables (Recommended for CI / Colab)
------------------------------------------------------------
Set the following environment variables in your terminal or Colab notebook:

    On Linux / macOS / Google Colab:
        export KAGGLE_USERNAME="your_kaggle_username"
        export KAGGLE_KEY="your_kaggle_api_key"

    On Windows PowerShell:
        $env:KAGGLE_USERNAME="your_kaggle_username"
        $env:KAGGLE_KEY="your_kaggle_api_key"

Option 2: Kaggle API Token File (kaggle.json)
---------------------------------------------
1. Log in to your Kaggle account at https://www.kaggle.com
2. Go to 'Account' / 'Settings' -> Scroll to 'API' section
3. Click 'Create New Token'. This downloads `kaggle.json`.
4. Place `kaggle.json` in:
       Windows:  C:\\Users\\<YourUsername>\\.kaggle\\kaggle.json
       Linux/Mac: ~/.kaggle/kaggle.json
5. Ensure permissions on Linux/Mac: chmod 600 ~/.kaggle/kaggle.json

Option 3: Manual Browser Download & Extraction
----------------------------------------------
If you prefer downloading manually:
1. Fish Dataset: https://www.kaggle.com/datasets/crowai/large-scale-fish-dataset
   Extract into: data/fish/
   Folder structure:
       data/fish/
           Gilt-Head Bream/
           Red Sea Bream/
           Sea Bass/
           ...

2. Coral Dataset: https://www.kaggle.com/datasets/sonainjamil/coral-reef-bleaching-dataset
   Extract into: data/coral/
   Folder structure:
       data/coral/
           healthy/
           bleached/

================================================================================
"""
    print(instructions)


def main() -> None:
    setup_logging()
    args = parse_args()
    cfg = load_config(args.config)

    logger.info("Initializing dataset acquisition pipeline...")

    # Check for credentials
    has_creds = KaggleDownloader._has_credentials()

    if args.placeholder or not has_creds:
        if not has_creds and not args.placeholder:
            logger.warning("Kaggle credentials not detected!")
            print_kaggle_instructions()
            logger.info("Proceeding with synthetic placeholder dataset generation so the pipeline remains 100% testable.")
        else:
            logger.info("Generating synthetic placeholder datasets as requested.")

        samples_per_class = args.num_samples_per_class
        generator = PlaceholderDatasetGenerator(cfg, images_per_class=samples_per_class)

        if args.task in ("all", "fish"):
            logger.info("Generating synthetic fish dataset (%d samples/class)...", samples_per_class)
            generator.generate_fish_images()
            logger.info("Successfully generated fish placeholder images.")

        if args.task in ("all", "coral"):
            logger.info("Generating synthetic coral dataset (%d samples/class)...", samples_per_class)
            generator.generate_coral_images()
            logger.info("Successfully generated coral placeholder images.")

        logger.info("Placeholder dataset ready in %s", cfg.paths.data_root)
        return

    # If credentials exist and not forced placeholder, attempt download
    downloader = KaggleDownloader(cfg)
    try:
        if args.task in ("all", "fish"):
            logger.info("Downloading fish dataset from Kaggle (%s)...", cfg.data.fish_dataset_slug)
            downloader.download_fish()
        if args.task in ("all", "coral"):
            logger.info("Downloading coral dataset from Kaggle (%s)...", cfg.data.coral_dataset_slug)
            downloader.download_coral()
        logger.info("Kaggle dataset download complete.")
    except Exception as exc:
        logger.error("Failed to download datasets from Kaggle: %s", exc)
        logger.info("Falling back to synthetic placeholder dataset...")
        generator = PlaceholderDatasetGenerator(cfg, images_per_class=args.num_samples_per_class)
        generator.generate_fish_images()
        generator.generate_coral_images()
        logger.info("Placeholder dataset generated as fallback.")


if __name__ == "__main__":
    main()
