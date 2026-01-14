"""
Dataset Download Script for VSF Research Platform

Usage:
    python scripts/download_datasets.py [--dataset DATASET_NAME]

Available datasets:
    - metr-la: METR-LA traffic speed (207 sensors)
    - pems-bay: PEMS-BAY traffic speed (325 sensors)
    - solar: Solar energy production (137 plants)
    - traffic: California freeway occupancy (862 sensors)
    - all: Download all datasets
"""

import os
import sys
import argparse
import urllib.request
import zipfile
import tarfile
import gzip
import shutil

# Project paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")

# Dataset sources
DATASETS = {
    "solar": {
        "url": "https://github.com/laiguokun/multivariate-time-series-data/raw/master/solar-energy/solar_AL.txt.gz",
        "filename": "solar.txt.gz",
        "output_filename": "solar.txt",
        "description": "Solar energy production from 137 PV plants in Alabama",
        "compressed": True
    },
    "traffic": {
        "url": "https://github.com/laiguokun/multivariate-time-series-data/raw/master/traffic/traffic.txt.gz",
        "filename": "traffic.txt.gz",
        "output_filename": "traffic.txt",
        "description": "California freeway occupancy rates (862 sensors)",
        "compressed": True
    },
    "exchange": {
        "url": "https://github.com/laiguokun/multivariate-time-series-data/raw/master/exchange_rate/exchange_rate.txt.gz",
        "filename": "exchange_rate.txt.gz",
        "output_filename": "exchange_rate.txt",
        "description": "Daily exchange rates for 8 currencies",
        "compressed": True
    },
    "electricity": {
        "url": "https://github.com/laiguokun/multivariate-time-series-data/raw/master/electricity/electricity.txt.gz",
        "filename": "electricity.txt.gz",
        "output_filename": "electricity.txt",
        "description": "Electricity consumption of 321 clients",
        "compressed": True
    },
    "metr-la": {
        "url": "https://drive.google.com/uc?id=10FOTa6HXPqX8Pf5WRoRwcFnW9BrNZEIX",
        "filename": "metr-la.h5",
        "description": "METR-LA traffic speed (207 sensors)",
        "requires_gdown": True
    },
    "pems-bay": {
        "url": "https://drive.google.com/uc?id=1wD-mHlqAb2mtHOe_68fZvDh1LpDegMMq",
        "filename": "pems-bay.h5",
        "description": "PEMS-BAY traffic speed (325 sensors)",
        "requires_gdown": True
    },
    "adj_mx": {
        "url": "https://raw.githubusercontent.com/nnzhan/MTGNN/master/data/sensor_graph/adj_mx.pkl",
        "filename": "adj_mx.pkl",
        "description": "METR-LA adjacency matrix",
        "target_dir": "sensor_graph"
    }
}


def ensure_dirs():
    """Create necessary directories."""
    os.makedirs(RAW_DIR, exist_ok=True)
    os.makedirs(os.path.join(DATA_DIR, "sensor_graph"), exist_ok=True)
    os.makedirs(os.path.join(DATA_DIR, "METR-LA"), exist_ok=True)
    os.makedirs(os.path.join(DATA_DIR, "SOLAR"), exist_ok=True)
    os.makedirs(os.path.join(DATA_DIR, "TRAFFIC"), exist_ok=True)


def download_file(url, filepath, use_gdown=False):
    """Download a file from URL."""
    print(f"Downloading: {os.path.basename(filepath)}")

    if use_gdown:
        try:
            import gdown
            gdown.download(url, filepath, quiet=False)
        except Exception as e:
            print(f"  gdown failed: {e}")
            print(f"  Please download manually from: {url}")
            return False
    else:
        try:
            urllib.request.urlretrieve(url, filepath)
            print(f"  Saved to: {filepath}")
        except Exception as e:
            print(f"  Download failed: {e}")
            return False

    return os.path.exists(filepath)


def download_dataset(name):
    """Download a specific dataset."""
    if name not in DATASETS:
        print(f"Unknown dataset: {name}")
        print(f"Available: {', '.join(DATASETS.keys())}")
        return False

    info = DATASETS[name]
    target_dir = os.path.join(DATA_DIR, info.get("target_dir", "raw"))
    os.makedirs(target_dir, exist_ok=True)

    # Check if final output file already exists
    output_filename = info.get("output_filename", info["filename"])
    final_filepath = os.path.join(target_dir, output_filename)

    if os.path.exists(final_filepath):
        print(f"Already exists: {final_filepath}")
        return True

    filepath = os.path.join(target_dir, info["filename"])

    print(f"\n=== {name.upper()} ===")
    print(f"Description: {info['description']}")

    use_gdown = info.get("requires_gdown", False)
    success = download_file(info["url"], filepath, use_gdown)

    # Decompress if needed
    if success and info.get("compressed", False):
        print(f"  Decompressing...")
        try:
            with gzip.open(filepath, 'rb') as f_in:
                with open(final_filepath, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            os.remove(filepath)  # Remove compressed file
            print(f"  Decompressed to: {final_filepath}")
        except Exception as e:
            print(f"  Decompression failed: {e}")
            return False

    return success


def main():
    parser = argparse.ArgumentParser(description="Download VSF datasets")
    parser.add_argument("--dataset", "-d", default="all",
                       help="Dataset to download (default: all)")
    parser.add_argument("--list", "-l", action="store_true",
                       help="List available datasets")
    args = parser.parse_args()

    if args.list:
        print("\nAvailable datasets:")
        for name, info in DATASETS.items():
            print(f"  {name:12} - {info['description']}")
        return

    ensure_dirs()

    if args.dataset == "all":
        # Download all (except Google Drive ones which often fail)
        success = []
        failed = []

        for name in DATASETS:
            if download_dataset(name):
                success.append(name)
            else:
                failed.append(name)

        print(f"\n=== Summary ===")
        print(f"Success: {', '.join(success) if success else 'None'}")
        print(f"Failed: {', '.join(failed) if failed else 'None'}")

        if failed:
            print("\nFor failed downloads, try manually:")
            for name in failed:
                print(f"  {name}: {DATASETS[name]['url']}")
    else:
        download_dataset(args.dataset)


if __name__ == "__main__":
    main()
