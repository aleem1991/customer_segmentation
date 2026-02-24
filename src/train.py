import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import classification_report

def train_churn_model(data_path, model_output_path):
    print("Step 1: Loading Clean Data...")
    df = pd.read_csv(data_path)
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    
    print("Step 2: Feature Engineering (Snapshotting)...")
    # Define prediction cutoff (last 90 days)
    cutoff_date = df['InvoiceDate'].max() - pd.DateOffset(days=90)
    
    # Split History vs Target
    train_data = df[df['InvoiceDate'] < cutoff_date]
    test_target_data = df[df['InvoiceDate'] >= cutoff_date]
    
    # Identify Active Customers in Target Window
    active_customers = test_target_data['Customer ID'].unique()
    
    # Create RFM Features from History
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
    
    # Label Generation (Churn = 1 if NOT in active list)
    features['Is_Churn'] = features['Customer ID'].apply(lambda x: 0 if x in active_customers else 1)
    
    print(f"Features created. Total Churn Rate: {features['Is_Churn'].mean():.2%}")

    print("Step 3: Training Model...")
    X = features.drop(['Customer ID', 'Is_Churn'], axis=1)
    y = features['Is_Churn']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train XGBoost
    xgb = XGBClassifier(n_estimators=100, scale_pos_weight=1.5, random_state=42)
    xgb.fit(X_train, y_train)
    
    print("Evaluation on Hold-out set:")
    y_pred = xgb.predict(X_test)
    print(classification_report(y_test, y_pred))
    
    print(f"Step 4: Saving model to {model_output_path}")
    joblib.dump(xgb, model_output_path)

if __name__ == "__main__":
    DATA_PATH = "data/processed/online_retail_clean.csv"
    MODEL_PATH = "models/churn_xgb_model.pkl"
    
    train_churn_model(DATA_PATH, MODEL_PATH)