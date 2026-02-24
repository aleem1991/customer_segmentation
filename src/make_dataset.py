import pandas as pd
import os

def load_and_clean_data(raw_file_path, output_path):
    print("Step 1: Loading Data...")
    if not os.path.exists(raw_file_path):
        raise FileNotFoundError(f"File not found at {raw_file_path}")
        
    df = pd.read_excel(raw_file_path, sheet_name='Year 2009-2010')
    
    print("Step 2: Cleaning Data...")
    # Drop missing IDs
    df_clean = df.dropna(subset=['Customer ID']).copy()
    
    # Filter negatives (Returns)
    df_clean = df_clean[(df_clean['Quantity'] > 0) & (df_clean['Price'] > 0)]
    
    # Types
    df_clean['Customer ID'] = df_clean['Customer ID'].astype(int).astype(str)
    df_clean['InvoiceDate'] = pd.to_datetime(df_clean['InvoiceDate'])
    
    # Calculate Total
    df_clean['Total Price'] = df_clean['Quantity'] * df_clean['Price']
    
    print(f"Step 3: Saving Processed Data to {output_path}")
    df_clean.to_csv(output_path, index=False)
    print("Data processing complete!")

if __name__ == "__main__":
    # Define paths relative to this script
    RAW_PATH = os.path.join("data", "raw", "online_retail_II.xlsx")
    PROCESSED_PATH = os.path.join("data", "processed", "online_retail_clean.csv")
    
    # Run
    load_and_clean_data(RAW_PATH, PROCESSED_PATH)