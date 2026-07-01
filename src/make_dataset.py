import os
import sys
import pandas as pd
from typing import NoReturn

# Ensure project root is in path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.config_loader import CONFIG
from src.logger_config import logger

def load_and_clean_data(raw_file_path: str, output_path: str) -> None:
    """Loads raw Excel transaction logs, applies data cleaning pipeline, and saves processed CSV.

    Args:
        raw_file_path: Path to the raw Excel dataset.
        output_path: Path to save the cleaned CSV dataset.
    """
    logger.info("Loading Raw Excel Data...")
    if not os.path.exists(raw_file_path):
        logger.error(f"Raw file not found: {raw_file_path}")
        raise FileNotFoundError(f"File not found at {raw_file_path}")
        
    try:
        df = pd.read_excel(raw_file_path, sheet_name='Year 2009-2010')
        logger.info(f"Successfully loaded raw data. Row count: {len(df)}")
        
        logger.info("Starting cleaning pipeline: dropping missing customer IDs...")
        df_clean = df.dropna(subset=['Customer ID']).copy()
        
        logger.info("Filtering out refunds and returns (Quantity/Price <= 0)...")
        df_clean = df_clean[(df_clean['Quantity'] > 0) & (df_clean['Price'] > 0)]
        
        logger.info("Formatting customer IDs and datetimes...")
        df_clean['Customer ID'] = df_clean['Customer ID'].astype(int).astype(str)
        df_clean['InvoiceDate'] = pd.to_datetime(df_clean['InvoiceDate'])
        
        logger.info("Calculating total transactional revenue per row...")
        df_clean['Total Price'] = df_clean['Quantity'] * df_clean['Price']
        
        # Ensure parent directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        logger.info(f"Saving cleaned dataset to {output_path}...")
        df_clean.to_csv(output_path, index=False)
        logger.info("Data loading and cleaning pipeline complete!")
        
    except Exception as e:
        logger.error(f"Data cleaning failed with error: {str(e)}")
        raise e

if __name__ == "__main__":
    RAW_PATH = CONFIG["paths"]["raw_data"]
    PROCESSED_PATH = CONFIG["paths"]["clean_data"]
    
    try:
        load_and_clean_data(RAW_PATH, PROCESSED_PATH)
    except Exception:
        sys.exit(1)