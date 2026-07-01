import os
import sys
import pandas as pd
import numpy as np
import joblib
import xgboost as xgb
from typing import Optional

# Ensure project root is in the path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.config_loader import CONFIG
from src.logger_config import logger
from src.make_dataset import load_and_clean_data

def run_predictions() -> None:
    """Executes customer churn forecasts, merges segments, and exports marketing reports."""
    RAW_PATH = CONFIG["paths"]["raw_data"]
    PROCESSED_PATH = CONFIG["paths"]["clean_data"]
    MODEL_PATH = CONFIG["paths"]["model"]
    SEGMENTS_PATH = CONFIG["paths"]["segments"]
    
    OUTPUT_REPORT_PATH = CONFIG["paths"]["predictions_report"]
    OUTPUT_TARGET_PATH = CONFIG["paths"]["high_value_report"]

    # Step 1: Ensure Clean Data Exists
    if not os.path.exists(PROCESSED_PATH):
        logger.warning(f"Cleaned dataset not found at {PROCESSED_PATH}. Running loader...")
        if not os.path.exists(RAW_PATH):
            logger.error(f"Raw data file not found at {RAW_PATH}. Cannot proceed.")
            raise FileNotFoundError(f"Raw data file not found at {RAW_PATH}")
        load_and_clean_data(RAW_PATH, PROCESSED_PATH)

    logger.info("Loading cleaned dataset...")
    try:
        df = pd.read_csv(PROCESSED_PATH)
        df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

        # Step 2: Feature Engineering (Full History Snapshot)
        logger.info("Engineering customer RFM features...")
        today = df['InvoiceDate'].max()
        logger.info(f"Current snapshot date (Reference 'Today'): {today.strftime('%Y-%m-%d')}")
        
        features = df.groupby('Customer ID').agg({
            'InvoiceDate': lambda x: (today - x.max()).days, # Recency
            'Invoice': 'nunique',       # Frequency
            'Total Price': 'mean',      # Avg Monetary Spend
            'Quantity': 'mean'          # Avg Basket Size
        }).reset_index()
        
        features.rename(columns={
            'InvoiceDate': 'Recency', 
            'Invoice': 'Frequency', 
            'Total Price': 'Monetary',
            'Quantity': 'AvgBucketSize'
        }, inplace=True)
        
        features['Customer ID'] = features['Customer ID'].astype(str)

        # 1. Inter-purchase Time (AvgDaysBetween)
        logger.info("Calculating average days between purchases...")
        df_sorted = df.sort_values(['Customer ID', 'InvoiceDate'])
        invoices = df_sorted.drop_duplicates(subset=['Customer ID', 'Invoice']).copy()
        invoices['PrevInvoiceDate'] = invoices.groupby('Customer ID')['InvoiceDate'].shift(1)
        invoices['DaysBetween'] = (invoices['InvoiceDate'] - invoices['PrevInvoiceDate']).dt.days
        
        avg_days_between = invoices.groupby('Customer ID')['DaysBetween'].mean().reset_index()
        avg_days_between.rename(columns={'DaysBetween': 'AvgDaysBetween'}, inplace=True)
        avg_days_between['Customer ID'] = avg_days_between['Customer ID'].astype(str)
        
        features = pd.merge(features, avg_days_between, on='Customer ID', how='left')
        
        single_buyer_impute = CONFIG["parameters"]["single_order_imputation_days"]
        features['AvgDaysBetween'] = features['AvgDaysBetween'].fillna(single_buyer_impute)
        
        # 2. Recency to AvgDaysBetween Ratio
        features['Recency_to_AvgDaysRatio'] = features['Recency'] / (features['AvgDaysBetween'] + 1e-5)
        
        # 3. Recent Orders Ratio (last 60 days)
        logger.info("Calculating order frequency ratios in recent days...")
        recent_window = CONFIG["parameters"]["recent_purchase_window_days"]
        recent_cutoff = today - pd.DateOffset(days=recent_window)
        recent_invoices = df[df['InvoiceDate'] >= recent_cutoff].groupby('Customer ID')['Invoice'].nunique().reset_index()
        recent_invoices.rename(columns={'Invoice': 'RecentInvoices'}, inplace=True)
        recent_invoices['Customer ID'] = recent_invoices['Customer ID'].astype(str)
        
        features = pd.merge(features, recent_invoices, on='Customer ID', how='left')
        features['RecentInvoices'] = features['RecentInvoices'].fillna(0)
        features['Recent_Orders_Ratio'] = features['RecentInvoices'] / features['Frequency']
        features.drop(columns=['RecentInvoices'], inplace=True)
        
        # 4. Is UK Customer
        customer_country = df.groupby('Customer ID')['Country'].first().reset_index()
        customer_country['Customer ID'] = customer_country['Customer ID'].astype(str)
        customer_country['Is_UK'] = (customer_country['Country'] == 'United Kingdom').astype(int)
        
        features = pd.merge(features, customer_country[['Customer ID', 'Is_UK']], on='Customer ID', how='left')

        # Step 3: Loading Model and Predicting
        logger.info(f"Loading trained XGBoost model from {MODEL_PATH}...")
        if not os.path.exists(MODEL_PATH):
            logger.error("Serialized model file missing.")
            raise FileNotFoundError(f"Model file not found at {MODEL_PATH}. Please train the model first.")
        
        model = joblib.load(MODEL_PATH)
        
        feature_cols = [
            'Recency', 'Frequency', 'Monetary', 'AvgBucketSize', 
            'AvgDaysBetween', 'Recency_to_AvgDaysRatio', 'Recent_Orders_Ratio', 'Is_UK'
        ]
        X = features[feature_cols]
        
        logger.info("Running predictions...")
        features['Churn_Probability'] = model.predict_proba(X)[:, 1]
        
        logger.info("Calculating TreeSHAP contributions...")
        booster = model.get_booster()
        dmat = xgb.DMatrix(X, feature_names=feature_cols)
        contribs = booster.predict(dmat, pred_contribs=True)
        for i, col in enumerate(feature_cols):
            features[f'SHAP_{col}'] = contribs[:, i]
        
        # Define Risk Tiers
        def get_risk_tier(prob: float) -> str:
            if prob >= 0.70:
                return "High Risk"
            elif prob >= 0.30:
                return "Medium Risk"
            else:
                return "Low Risk"
                
        features['Risk_Tier'] = features['Churn_Probability'].apply(get_risk_tier)

        # Step 4: Merging with Customer Segments
        logger.info("Merging predictions with offline segments...")
        if os.path.exists(SEGMENTS_PATH):
            segments_df = pd.read_csv(SEGMENTS_PATH)[['Customer ID', 'Segment']]
            segments_df['Customer ID'] = segments_df['Customer ID'].astype(str)
            
            merged_df = pd.merge(features, segments_df, on='Customer ID', how='left')
            merged_df['Segment'] = merged_df['Segment'].fillna('New / Unclassified')
        else:
            logger.warning(f"Segments database missing at {SEGMENTS_PATH}.")
            merged_df = features.copy()
            merged_df['Segment'] = 'Unclassified'

        # Step 5: Assign Business Recommendations
        logger.info("Generating targeted marketing recommendations...")
        def get_recommendation(row: pd.Series) -> str:
            segment = row['Segment']
            tier = row['Risk_Tier']
            
            if tier == "High Risk":
                if "Champion" in segment:
                    return "At-Risk Champion: High historical spend. Assign a personal account manager for direct outreach. Do not send automated discount spam."
                elif "Loyalist" in segment or "Loyal" in segment:
                    return "At-Risk Loyalist: Dedicated customer showing signs of leaving. Offer a special loyalty reward or high-value incentive."
                elif "Hibernating" in segment or "About to Sleep" in segment:
                    return "Hibernating Win-back: Aggressive discount offer or 'We Miss You' promotion with limited validity to re-engage."
                elif "New" in segment or "Promising" in segment:
                    return "Immediate Activation: One-time buyer showing low activity. Trigger welcome sequence or first-repeat-purchase incentive."
                else:
                    return "Standard Re-engagement: Target with standard product updates and a mild discount."
            elif tier == "Medium Risk":
                if "Champion" in segment or "Loyal" in segment:
                    return "Proactive VIP Retention: High-value showing drop-off signs. Send customized recommendations based on past purchases. Avoid direct discount spam."
                elif "Hibernating" in segment or "About to Sleep" in segment:
                    return "Nurture Campaign: Include in standard promotional newsletters and generic sale announcements."
                else:
                    return "Standard Retention: Monitor activity. Send standard seasonal discount codes."
            else:  # Low Risk
                if "Champion" in segment or "Loyal" in segment:
                    return "Maintain & Protect: Do nothing. Keep regular service quality high. Exclude from aggressive discount lists to preserve margin."
                else:
                    return "Standard Relationship Management: Keep engaged with standard updates."

        merged_df['Actionable_Recommendation'] = merged_df.apply(get_recommendation, axis=1)

        # Step 6: Exporting Reports
        logger.info("Exporting CSV reports...")
        merged_df = merged_df.sort_values(by='Churn_Probability', ascending=False)
        
        os.makedirs(os.path.dirname(OUTPUT_REPORT_PATH), exist_ok=True)
        merged_df.to_csv(OUTPUT_REPORT_PATH, index=False)
        logger.info(f" -> Complete Churn Report saved to: {OUTPUT_REPORT_PATH}")
        
        # Filter and export top 100 high-value high-risk customers
        high_risk_df = merged_df[merged_df['Risk_Tier'] == 'High Risk']
        high_value_at_risk = high_risk_df.sort_values(by='Monetary', ascending=False).head(100)
        
        high_value_at_risk.to_csv(OUTPUT_TARGET_PATH, index=False)
        logger.info(f" -> Top 100 High-Value At-Risk Customers saved to: {OUTPUT_TARGET_PATH}")

        # Summary logging
        logger.info("\n" + "="*50)
        logger.info("                CHURN ANALYSIS SUMMARY")
        logger.info("="*50)
        logger.info(f"Total Customers Analyzed:        {len(merged_df)}")
        
        risk_counts = merged_df['Risk_Tier'].value_counts()
        high_count = risk_counts.get('High Risk', 0)
        med_count = risk_counts.get('Medium Risk', 0)
        low_count = risk_counts.get('Low Risk', 0)
        
        logger.info(f"High Risk (Churn Prob >= 70%):   {high_count} ({high_count/len(merged_df):.1%})")
        logger.info(f"Medium Risk (30% <= Prob < 70%): {med_count} ({med_count/len(merged_df):.1%})")
        logger.info(f"Low Risk (Churn Prob < 30%):    {low_count} ({low_count/len(merged_df):.1%})")
        logger.info("-"*50)
        
        high_risk_revenue = (high_risk_df['Monetary'] * high_risk_df['Frequency']).sum()
        logger.info(f"Total Revenue At Risk (High Risk): ${high_risk_revenue:,.2f}")
        logger.info("="*50 + "\n")

    except Exception as e:
        logger.error(f"Prediction flow failed with error: {str(e)}")
        raise e

if __name__ == "__main__":
    try:
        run_predictions()
    except Exception:
        sys.exit(1)
