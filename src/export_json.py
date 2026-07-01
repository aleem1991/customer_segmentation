import os
import sys
import pandas as pd
import json
from typing import Dict, Any

# Ensure project root is in path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.config_loader import CONFIG
from src.logger_config import logger

def export_data() -> None:
    """Formats churn model prediction outcomes into browser-ready JSON assets for the dashboard."""
    REPORT_PATH = CONFIG["paths"]["predictions_report"]
    
    # Target folders inside React project
    DATA_DIR = os.path.join(project_root, "dashboard", "public", "data")
    os.makedirs(DATA_DIR, exist_ok=True)
    
    SUMMARY_OUT = os.path.join(DATA_DIR, "summary.json")
    CUSTOMERS_OUT = os.path.join(DATA_DIR, "customers.json")
    
    # Fallback to run prediction script if missing
    if not os.path.exists(REPORT_PATH):
        logger.warning(f"Report file not found at {REPORT_PATH}. Running predict.py first...")
        from src.predict import run_predictions
        run_predictions()
        
    logger.info("Reading prediction outputs for JSON export...")
    try:
        df = pd.read_csv(REPORT_PATH)
        
        # Calculate stats
        total_customers = len(df)
        
        risk_counts = df['Risk_Tier'].value_counts()
        high_risk_count = int(risk_counts.get('High Risk', 0))
        med_risk_count = int(risk_counts.get('Medium Risk', 0))
        low_risk_count = int(risk_counts.get('Low Risk', 0))
        
        # Revenue at Risk (High Risk customers)
        high_risk_df = df[df['Risk_Tier'] == 'High Risk']
        high_risk_df_copy = high_risk_df.copy()
        high_risk_df_copy['TotalValue'] = high_risk_df_copy['Monetary'] * high_risk_df_copy['Frequency']
        revenue_at_risk = float(high_risk_df_copy['TotalValue'].sum())
        
        # Calculate segments count
        segment_counts = df['Segment'].value_counts().to_dict()
        
        # Churn risk per segment
        segment_churn: Dict[str, Dict[str, Any]] = {}
        for seg in df['Segment'].unique():
            seg_df = df[df['Segment'] == seg]
            total_seg = len(seg_df)
            if total_seg > 0:
                high_risk_seg = len(seg_df[seg_df['Risk_Tier'] == 'High Risk'])
                segment_churn[seg] = {
                    "total": total_seg,
                    "high_risk": high_risk_seg,
                    "high_risk_pct": float(high_risk_seg / total_seg)
                }

        summary_data = {
            "total_customers": total_customers,
            "revenue_at_risk": round(revenue_at_risk, 2),
            "risk_tiers": {
                "High Risk": high_risk_count,
                "Medium Risk": med_risk_count,
                "Low Risk": low_risk_count
            },
            "segment_distribution": {str(k): int(v) for k, v in segment_counts.items()},
            "segment_churn": segment_churn
        }
        
        logger.info(f"Saving summary statistics to {SUMMARY_OUT}...")
        with open(SUMMARY_OUT, 'w', encoding='utf-8') as f:
            json.dump(summary_data, f, indent=2)
            
        logger.info("Formatting individual customer records for frontend table...")
        customers_list = []
        for _, row in df.iterrows():
            customers_list.append({
                "id": str(row['Customer ID']),
                "recency": int(row['Recency']),
                "frequency": int(row['Frequency']),
                "monetary": round(float(row['Monetary']), 2),
                "basketSize": round(float(row['AvgBucketSize']), 2),
                "churnProb": round(float(row['Churn_Probability']), 4),
                "riskTier": str(row['Risk_Tier']),
                "segment": str(row['Segment']),
                "recommendation": str(row['Actionable_Recommendation']),
                "shapValues": {
                    "Recency": round(float(row['SHAP_Recency']), 4),
                    "Frequency": round(float(row['SHAP_Frequency']), 4),
                    "Monetary": round(float(row['SHAP_Monetary']), 4),
                    "AvgBucketSize": round(float(row['SHAP_AvgBucketSize']), 4),
                    "AvgDaysBetween": round(float(row['SHAP_AvgDaysBetween']), 4),
                    "Recency_to_AvgDaysRatio": round(float(row['SHAP_Recency_to_AvgDaysRatio']), 4),
                    "Recent_Orders_Ratio": round(float(row['SHAP_Recent_Orders_Ratio']), 4),
                    "Is_UK": round(float(row['SHAP_Is_UK']), 4)
                }
            })
            
        logger.info(f"Saving {len(customers_list)} formatted records to {CUSTOMERS_OUT}...")
        with open(CUSTOMERS_OUT, 'w', encoding='utf-8') as f:
            json.dump(customers_list, f) # No indent to minimize payload size
            
        logger.info("JSON Data Export Completed successfully!")
        
    except Exception as e:
        logger.error(f"JSON export failed with error: {str(e)}")
        raise e

if __name__ == "__main__":
    export_data()
