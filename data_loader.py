"""
Data loading module for the analytics assistant.
"""
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, Optional

import pandas as pd


def download_data_from_gdrive(
    url: str = "https://drive.google.com/file/d/1v4aDGZsXmNsFAxQ9D7IoPes-pko7QRzc/view?usp=share_link",
    output_path: str = "data.json"
) -> bool:
    """
    Download data from Google Drive using gdown.
    
    Args:
        url: Google Drive file URL
        output_path: Path to save the downloaded file
        
    Returns:
        True if download successful, False otherwise
    """
    try:
        subprocess.run(
            ["gdown", "--fuzzy", url, "-O", output_path],
            check=True,
            capture_output=True
        )
        print(f"✅ Data downloaded successfully to {output_path}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to download data: {e}")
        return False
    except FileNotFoundError:
        print("❌ gdown not found. Install it with: pip install gdown")
        return False


def load_data(data_path: str = "data.json") -> Optional[pd.DataFrame]:
    """
    Load and preprocess the JSON data into a pandas DataFrame.
    
    Args:
        data_path: Path to the JSON data file
        
    Returns:
        Preprocessed DataFrame or None if loading fails
    """
    path = Path(data_path)
    
    if not path.exists():
        print(f"⚠️ Data file not found at {data_path}")
        print("Attempting to download from Google Drive...")
        
        if not download_data_from_gdrive(output_path=data_path):
            return None
    
    try:
        data = pd.read_json(path)
        data = pd.json_normalize(data['data'])
        data = data.dropna(axis=1, how="all")
        print(f"✅ Loaded {len(data)} records from {data_path}")
        return data
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return None


def load_metadata(metadata_path: str = "metadata.json") -> Optional[Dict]:
    """
    Load metadata from JSON file.
    
    Args:
        metadata_path: Path to the metadata JSON file
        
    Returns:
        Metadata dictionary or None if loading fails
    """
    path = Path(metadata_path)
    
    if not path.exists():
        print(f"❌ Metadata file not found at {metadata_path}")
        return None
    
    try:
        with open(path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        print(f"✅ Loaded metadata with {len(metadata)} fields")
        return metadata
    except Exception as e:
        print(f"❌ Error loading metadata: {e}")
        return None


if __name__ == "__main__":
    # Test data loading
    df = load_data()
    if df is not None:
        print(f"\nDataFrame shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
    
    metadata = load_metadata()
    if metadata is not None:
        print(f"\nMetadata fields: {list(metadata.keys())}")
