"""
Data loading module for the analytics assistant.
"""
import json
import re
import subprocess
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd


# Columns to drop during preprocessing
COLUMNS_TO_DROP = [
    "type", 
    "offer_type", 
    "description_html", 
    "remote_options", 
    "short_description", 
    "stack_description",
    "short_info",
    "company.logotype",
    "company.url",
    "company.post_to_job_aggregators",
    "recruiter.photo",
    "og.title",
    "og.description",
    "og.image_url",
    "og.image_width",
    "og.image_height",
    "og.site_name",
    "relocation_options",
    "english_level.name",
    "english_level.vacancy_description",
    "one_day_offer_content.version",
    "one_day_offer_content.block_one.header",
    "one_day_offer_content.block_one.last_date",
    "one_day_offer_content.block_one.event_dates",
    "one_day_offer_content.block_one.applications_before",
    "one_day_offer_content.block_two.stack",
    "one_day_offer_content.block_two.header",
    "one_day_offer_content.block_two.short_description",
    "one_day_offer_content.advantages.items",
    "one_day_offer_content.advantages.header",
    "one_day_offer_content_v3.date",
    "one_day_offer_content_v3.teams.items",
    "one_day_offer_content_v3.teams.header",
    "one_day_offer_content_v3.format",
    "one_day_offer_content_v3.schedule.items",
    "one_day_offer_content_v3.schedule.header",
    "one_day_offer_content_v3.block_one.header",
    "one_day_offer_content_v3.block_one.short_description",
    "one_day_offer_content_v3.block_two.stack",
    "one_day_offer_content_v3.block_two.header",
    "one_day_offer_content_v3.block_two.short_description",
    "one_day_offer_content_v3.advantages.items",
    "one_day_offer_content_v3.advantages.header",
    "description",
    "offer_description",
    "analytics_id",
    "office_options",
    "url",
    "company.short_description",
]

# Currency exchange rates to RUB
CURRENCY_RATES = {
    '₽': 1,
    '$': 80,
    '€': 94,
}


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


def calculate_salary_rub(row: pd.Series, rates: dict = CURRENCY_RATES) -> float:
    """
    Calculate salary in RUB from various salary fields.
    
    Args:
        row: DataFrame row with salary fields
        rates: Currency exchange rates dictionary
        
    Returns:
        Salary in RUB or NaN
    """
    # 1. Determine salary value
    frm = row.get('salary_display_from')
    to = row.get('salary_display_to')

    if pd.notna(frm) and pd.notna(to):
        salary = (frm + to) / 2
    elif pd.notna(frm):
        salary = frm
    elif pd.notna(to):
        salary = to
    else:
        # Try to parse from salary_description
        text = row.get('salary_description')
        if pd.isna(text):
            return np.nan
        
        match = re.search(r'([\d\s]+)\s*(₽|\$|€)', str(text))
        if not match:
            return np.nan
        
        salary = int(match.group(1).replace(' ', ''))

    # 2. Determine currency
    currency = row.get('salary_currency')

    if pd.isna(currency):
        text = row.get('salary_description')
        if pd.isna(text):
            return np.nan
        
        cur_match = re.search(r'(₽|\$|€)', str(text))
        if not cur_match:
            return np.nan
        
        currency = cur_match.group(1)

    # 3. Convert to RUB
    rate = rates.get(currency)
    if rate is None:
        return np.nan

    return salary * rate


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and preprocess the DataFrame.
    
    Args:
        df: Raw DataFrame
        
    Returns:
        Preprocessed DataFrame
    """
    # Drop columns that exist in the dataframe
    cols_to_drop = [col for col in COLUMNS_TO_DROP if col in df.columns]
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)
        print(f"  ↳ Dropped {len(cols_to_drop)} unnecessary columns")
    
    # Calculate salary in RUB
    salary_cols = ['salary_display_from', 'salary_display_to', 'salary_description', 'salary_currency']
    if all(col in df.columns for col in ['salary_display_from', 'salary_display_to']):
        print("  ↳ Calculating salary in RUB...")
        df['salary'] = df.apply(calculate_salary_rub, axis=1)
        
        # Drop original salary columns
        salary_cols_to_drop = [col for col in salary_cols if col in df.columns]
        df = df.drop(columns=salary_cols_to_drop)
        print(f"  ↳ Created 'salary' column, dropped {len(salary_cols_to_drop)} original salary columns")
    
    return df


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
        print(f"📂 Loading data from {data_path}...")
        data = pd.read_json(path)
        data = pd.json_normalize(data['data'])
        data = data.dropna(axis=1, how="all")
        print(f"  ↳ Loaded {len(data)} records, {len(data.columns)} columns")
        
        # Apply preprocessing
        print("🔧 Preprocessing data...")
        data = preprocess_data(data)
        
        print(f"✅ Final dataset: {len(data)} records, {len(data.columns)} columns")
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
        print(f"\nSalary stats:")
        print(df['salary'].describe())
    
    metadata = load_metadata()
    if metadata is not None:
        print(f"\nMetadata fields: {list(metadata.keys())}")
