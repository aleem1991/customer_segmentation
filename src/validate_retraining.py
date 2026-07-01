import os
import sys
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score, classification_report

# Ensure project root is in path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.config_loader import CONFIG
from src.logger_config import logger

def validate_model_performance(data_path: str, model_path: str, recall_threshold: float = 0.80) -> bool:
    """Evaluates the trained model recall performance on holdout test set to gate deployment.

    Args:
        data_path: Path to the clean CSV dataset.
        model_path: Path to the serialized XGBoost model.
        recall_threshold: Minimum acceptable recall score on the churn class.

    Returns:
        bool: True if the model performance meets or exceeds the threshold, False otherwise.
    """
    logger.info("Initializing model performance validation check...")
    
    if not os.path.exists(model_path):
        logger.error(f"Model binary not found at {model_path}. Cannot validate.")
        return False
        
    try:
        df = pd.read_csv(data_path)
        df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
        
        cutoff_days = CONFIG["parameters"]["cutoff_offset_days"]
        cutoff_date = df['InvoiceDate'].max() - pd.DateOffset(days=cutoff_days)
        
        train_data = df[df['InvoiceDate'] < cutoff_date].copy()
        test_target_data = df[df['InvoiceDate'] >= cutoff_date].copy()
        
        active_customers = [str(int(x)) for x in test_target_data['Customer ID'].dropna().unique()]
        
        # Aggregations
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
        
        # Advanced Features
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
        
        # Validation Hold-out Split
        _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Load active model
        model = joblib.load(model_path)
        
        # Predict on holdout
        y_pred = model.predict(X_test)
        
        # Compute Recall for Churn (label=1)
        recall = recall_score(y_test, y_pred)
        
        logger.info(f"Retrained Model validation result: Recall = {recall:.2%}")
        logger.info(f"Target Performance Threshold: Recall >= {recall_threshold:.2%}")
        logger.info(f"\n{classification_report(y_test, y_pred)}")
        
        if recall >= recall_threshold:
            logger.info("Validation PASSED! Model is eligible for release.")
            return True
        else:
            logger.warning("Validation FAILED! Recall performance does not meet threshold.")
            return False
            
    except Exception as e:
        logger.error(f"Error validating model performance: {str(e)}")
        return False

if __name__ == "__main__":
    DATA_PATH = CONFIG["paths"]["clean_data"]
    MODEL_PATH = CONFIG["paths"]["model"]
    
    success = validate_model_performance(DATA_PATH, MODEL_PATH, recall_threshold=0.80)
    if success:
        sys.exit(0)
    else:
        sys.exit(1)
