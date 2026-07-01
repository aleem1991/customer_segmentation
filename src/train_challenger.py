import os
import sys
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Ensure project root is in path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.config_loader import CONFIG
from src.logger_config import logger

def train_challenger_rf(data_path: str, model_output_path: str) -> None:
    """Trains a Random Forest classifier as a Challenger model shadow candidate.

    Args:
        data_path: Path to the clean CSV dataset.
        model_output_path: Output path for the serialized RF model.
    """
    logger.info(f"Loading cleaned dataset from {data_path} for Challenger training...")
    try:
        df = pd.read_csv(data_path)
        df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
        
        cutoff_days = CONFIG["parameters"]["cutoff_offset_days"]
        cutoff_date = df['InvoiceDate'].max() - pd.DateOffset(days=cutoff_days)
        
        train_data = df[df['InvoiceDate'] < cutoff_date].copy()
        test_target_data = df[df['InvoiceDate'] >= cutoff_date].copy()
        
        active_customers = [str(int(x)) for x in test_target_data['Customer ID'].dropna().unique()]
        
        # Aggregate features
        features = train_data.groupby('Customer ID').agg({
            'InvoiceDate': lambda x: (cutoff_date - x.max()).days,
            'Invoice': 'nunique',
            'Total Price': 'mean',
            'Quantity': 'mean'
        }).reset_index()
        
        features.rename(columns={
            'InvoiceDate': 'Recency', 
            'Invoice': 'Frequency', 
            'Total Price': 'Monetary',
            'Quantity': 'AvgBucketSize'
        }, inplace=True)
        
        features['Customer ID'] = features['Customer ID'].astype(str)
        
        # Advanced features
        df_sorted = train_data.sort_values(['Customer ID', 'InvoiceDate'])
        invoices = df_sorted.drop_duplicates(subset=['Customer ID', 'Invoice']).copy()
        invoices['PrevInvoiceDate'] = invoices.groupby('Customer ID')['InvoiceDate'].shift(1)
        invoices['DaysBetween'] = (invoices['InvoiceDate'] - invoices['PrevInvoiceDate']).dt.days
        
        avg_days_between = invoices.groupby('Customer ID')['DaysBetween'].mean().reset_index()
        avg_days_between.rename(columns={'DaysBetween': 'AvgDaysBetween'}, inplace=True)
        avg_days_between['Customer ID'] = avg_days_between['Customer ID'].astype(str)
        
        features = pd.merge(features, avg_days_between, on='Customer ID', how='left')
        single_buyer_impute = CONFIG["parameters"]["single_order_imputation_days"]
        features['AvgDaysBetween'] = features['AvgDaysBetween'].fillna(single_buyer_impute)
        
        features['Recency_to_AvgDaysRatio'] = features['Recency'] / (features['AvgDaysBetween'] + 1e-5)
        
        recent_window = CONFIG["parameters"]["recent_purchase_window_days"]
        recent_cutoff = cutoff_date - pd.DateOffset(days=recent_window)
        recent_invoices = train_data[train_data['InvoiceDate'] >= recent_cutoff].groupby('Customer ID')['Invoice'].nunique().reset_index()
        recent_invoices.rename(columns={'Invoice': 'RecentInvoices'}, inplace=True)
        recent_invoices['Customer ID'] = recent_invoices['Customer ID'].astype(str)
        
        features = pd.merge(features, recent_invoices, on='Customer ID', how='left')
        features['RecentInvoices'] = features['RecentInvoices'].fillna(0)
        features['Recent_Orders_Ratio'] = features['RecentInvoices'] / features['Frequency']
        features.drop(columns=['RecentInvoices'], inplace=True)
        
        customer_country = train_data.groupby('Customer ID')['Country'].first().reset_index()
        customer_country['Customer ID'] = customer_country['Customer ID'].astype(str)
        customer_country['Is_UK'] = (customer_country['Country'] == 'United Kingdom').astype(int)
        
        features = pd.merge(features, customer_country[['Customer ID', 'Is_UK']], on='Customer ID', how='left')
        
        features['Is_Churn'] = features['Customer ID'].apply(lambda x: 0 if x in active_customers else 1)
        
        feature_cols = [
            'Recency', 'Frequency', 'Monetary', 'AvgBucketSize', 
            'AvgDaysBetween', 'Recency_to_AvgDaysRatio', 'Recent_Orders_Ratio', 'Is_UK'
        ]
        X = features[feature_cols]
        y = features['Is_Churn']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        logger.info("Initializing RandomForest challenger classifier...")
        rf = RandomForestClassifier(
            n_estimators=100,
            max_depth=6,
            class_weight="balanced",
            random_state=42
        )
        
        logger.info("Fitting Random Forest challenger on training set...")
        rf.fit(X_train, y_train)
        
        logger.info("Evaluating Challenger model on hold-out validation:")
        y_pred = rf.predict(X_test)
        report = classification_report(y_test, y_pred)
        logger.info(f"\n{report}")
        
        os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
        logger.info(f"Saving serialized Random Forest challenger model to {model_output_path}...")
        joblib.dump(rf, model_output_path)
        logger.info("Challenger RF model training complete!")
        
    except Exception as e:
        logger.error(f"Challenger training failed: {str(e)}")
        raise e

if __name__ == "__main__":
    DATA_PATH = CONFIG["paths"]["clean_data"]
    MODEL_OUT = os.path.join(project_root, "models", "churn_rf_model.pkl")
    
    try:
        train_challenger_rf(DATA_PATH, MODEL_OUT)
    except Exception:
        sys.exit(1)
