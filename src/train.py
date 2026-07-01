import os
import sys
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import classification_report

# Ensure project root is in path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.config_loader import CONFIG
from src.logger_config import logger

def train_churn_model(data_path: str, model_output_path: str) -> None:
    """Performs feature engineering on clean customer history, trains an XGBoost model,

    saves the trained model `.pkl` to disk.

    Args:
        data_path: Path to the clean CSV dataset.
        model_output_path: Output path for the serialized XGBoost model.
    """
    logger.info(f"Loading cleaned dataset from {data_path}...")
    try:
        df = pd.read_csv(data_path)
        df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
        
        logger.info("Computing cutoff date for prediction window...")
        cutoff_days = CONFIG["parameters"]["cutoff_offset_days"]
        cutoff_date = df['InvoiceDate'].max() - pd.DateOffset(days=cutoff_days)
        logger.info(f"Reference snapshot cutoff date: {cutoff_date.strftime('%Y-%m-%d')}")
        
        logger.info("Splitting dataset into history vs target prediction window...")
        train_data = df[df['InvoiceDate'] < cutoff_date].copy()
        test_target_data = df[df['InvoiceDate'] >= cutoff_date].copy()
        
        # Identify active customers (members of active list are non-churners)
        active_customers = [str(int(x)) for x in test_target_data['Customer ID'].dropna().unique()]
        
        logger.info("Engineering base RFM features...")
        features = train_data.groupby('Customer ID').agg({
            'InvoiceDate': lambda x: (cutoff_date - x.max()).days, # Recency
            'Invoice': 'nunique',       # Frequency
            'Total Price': 'mean',      # Avg Monetary
            'Quantity': 'mean'          # Avg Basket Size
        }).reset_index()
        
        features.rename(columns={
            'InvoiceDate': 'Recency', 
            'Invoice': 'Frequency', 
            'Total Price': 'Monetary',
            'Quantity': 'AvgBucketSize'
        }, inplace=True)
        
        features['Customer ID'] = features['Customer ID'].astype(str)
        
        logger.info("Engineering advanced MLE features...")
        
        # 1. Calculate Inter-purchase Time (AvgDaysBetween)
        df_sorted = train_data.sort_values(['Customer ID', 'InvoiceDate'])
        invoices = df_sorted.drop_duplicates(subset=['Customer ID', 'Invoice']).copy()
        invoices['PrevInvoiceDate'] = invoices.groupby('Customer ID')['InvoiceDate'].shift(1)
        invoices['DaysBetween'] = (invoices['InvoiceDate'] - invoices['PrevInvoiceDate']).dt.days
        
        avg_days_between = invoices.groupby('Customer ID')['DaysBetween'].mean().reset_index()
        avg_days_between.rename(columns={'DaysBetween': 'AvgDaysBetween'}, inplace=True)
        avg_days_between['Customer ID'] = avg_days_between['Customer ID'].astype(str)
        
        features = pd.merge(features, avg_days_between, on='Customer ID', how='left')
        
        # Impute single purchase buyers with config default
        single_buyer_impute = CONFIG["parameters"]["single_order_imputation_days"]
        features['AvgDaysBetween'] = features['AvgDaysBetween'].fillna(single_buyer_impute)
        
        # 2. Recency to AvgDaysBetween Ratio
        features['Recency_to_AvgDaysRatio'] = features['Recency'] / (features['AvgDaysBetween'] + 1e-5)
        
        # 3. Recent Orders Ratio (last 60 days of training window)
        recent_window = CONFIG["parameters"]["recent_purchase_window_days"]
        recent_cutoff = cutoff_date - pd.DateOffset(days=recent_window)
        recent_invoices = train_data[train_data['InvoiceDate'] >= recent_cutoff].groupby('Customer ID')['Invoice'].nunique().reset_index()
        recent_invoices.rename(columns={'Invoice': 'RecentInvoices'}, inplace=True)
        recent_invoices['Customer ID'] = recent_invoices['Customer ID'].astype(str)
        
        features = pd.merge(features, recent_invoices, on='Customer ID', how='left')
        features['RecentInvoices'] = features['RecentInvoices'].fillna(0)
        features['Recent_Orders_Ratio'] = features['RecentInvoices'] / features['Frequency']
        features.drop(columns=['RecentInvoices'], inplace=True)
        
        # 4. Is UK Customer
        customer_country = train_data.groupby('Customer ID')['Country'].first().reset_index()
        customer_country['Customer ID'] = customer_country['Customer ID'].astype(str)
        customer_country['Is_UK'] = (customer_country['Country'] == 'United Kingdom').astype(int)
        
        features = pd.merge(features, customer_country[['Customer ID', 'Is_UK']], on='Customer ID', how='left')
        
        # Generate target labels
        features['Is_Churn'] = features['Customer ID'].apply(lambda x: 0 if x in active_customers else 1)
        logger.info(f"Target Label Generation complete. Churn class ratio: {features['Is_Churn'].mean():.2%}")
        
        # Split features and labels
        feature_cols = [
            'Recency', 'Frequency', 'Monetary', 'AvgBucketSize', 
            'AvgDaysBetween', 'Recency_to_AvgDaysRatio', 'Recent_Orders_Ratio', 'Is_UK'
        ]
        X = features[feature_cols]
        y = features['Is_Churn']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        logger.info("Initializing XGBoost classifier with config hyperparameters...")
        hyperparams = CONFIG["model_hyperparameters"]
        xgb = XGBClassifier(
            n_estimators=hyperparams["n_estimators"],
            learning_rate=hyperparams["learning_rate"],
            max_depth=hyperparams["max_depth"],
            subsample=hyperparams["subsample"],
            colsample_bytree=hyperparams["colsample_bytree"],
            scale_pos_weight=hyperparams["scale_pos_weight"],
            random_state=hyperparams["random_state"]
        )
        
        logger.info("Fitting model on training set...")
        xgb.fit(X_train, y_train)
        
        logger.info("Evaluating model on validation hold-out set:")
        y_pred = xgb.predict(X_test)
        report = classification_report(y_test, y_pred)
        logger.info(f"\n{report}")
        
        # Ensure parent folder exists
        os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
        
        logger.info(f"Saving serialized model to {model_output_path}...")
        joblib.dump(xgb, model_output_path)
        logger.info("Model training pipeline complete!")
        
    except Exception as e:
        logger.error(f"Model training failed with error: {str(e)}")
        raise e

if __name__ == "__main__":
    DATA_PATH = CONFIG["paths"]["clean_data"]
    MODEL_PATH = CONFIG["paths"]["model"]
    
    try:
        train_churn_model(DATA_PATH, MODEL_PATH)
    except Exception:
        sys.exit(1)