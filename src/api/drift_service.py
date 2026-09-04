import os
import pandas as pd
from scipy.stats import ks_2samp
from typing import Dict, Any
from src.api.config import HISTORY_PATH, CLEANED_DATA_PATH, CONFIG, logger

def run_drift_analysis() -> Dict[str, Any]:
    """Runs a Kolmogorov-Smirnov test to detect data drift between baseline and production data."""
    # Check production history file
    if not os.path.exists(HISTORY_PATH):
        return {
            "drift_detected": False,
            "drift_status": "Insufficient Data",
            "message": "Production inference history file is missing."
        }
        
    try:
        prod_df = pd.read_csv(HISTORY_PATH)
    except Exception as e:
        logger.error(f"Error reading inference history: {str(e)}")
        return {
            "drift_detected": False,
            "drift_status": "Error",
            "message": f"Could not load production logs: {str(e)}"
        }
        
    # We require a minimum of 10 samples to run statistical checks
    min_samples = 10
    prod_size = len(prod_df)
    if prod_size < min_samples:
        return {
            "drift_detected": False,
            "drift_status": "Insufficient Data",
            "message": f"Awaiting production predictions. Need at least {min_samples} requests to run statistical test (current: {prod_size}).",
            "sample_sizes": {
                "baseline": 4312,
                "production": prod_size
            }
        }
        
    # Load baseline dataset
    baseline_path = CLEANED_DATA_PATH
    if not os.path.exists(baseline_path):
        baseline_path = CONFIG["paths"]["clean_data"]
    if not os.path.exists(baseline_path):
        from src.api.config import project_root
        baseline_path = os.path.join(project_root, "data", "processed", "churn_predictions_report.csv")
    if not os.path.exists(baseline_path):
        baseline_path = os.path.join(project_root, "data", "processed", "rfm_segments.csv")
        
    try:
        base_df = pd.read_csv(baseline_path)
    except Exception as e:
        logger.error(f"Error loading baseline clean dataset: {str(e)}")
        return {
            "drift_detected": False,
            "drift_status": "Error",
            "message": f"Could not load baseline training data: {str(e)}"
        }
        
    # Mapping of column names: baseline vs production history
    features_to_test = {
        "Recency": "Recency",
        "Frequency": "Frequency",
        "Monetary": "Monetary",
        "AvgBucketSize": "BasketSize"
    }
    
    drift_details = {}
    drift_detected = False
    
    for base_col, prod_col in features_to_test.items():
        if base_col not in base_df.columns or prod_col not in prod_df.columns:
            logger.warning(f"Feature columns not found: {base_col} in base or {prod_col} in prod.")
            continue
            
        base_arr = base_df[base_col].dropna().values
        prod_arr = prod_df[prod_col].dropna().values
        
        # Run Kolmogorov-Smirnov test (2-sample)
        stat, pval = ks_2samp(base_arr, prod_arr)
        
        # Standard 5% significance level
        has_drifted = pval < 0.05
        if has_drifted:
            drift_detected = True
            
        drift_details[base_col] = {
            "p_value": round(float(pval), 5),
            "drift_status": "Drifted" if has_drifted else "Stable",
            "baseline_mean": round(float(base_arr.mean()), 2),
            "production_mean": round(float(prod_arr.mean()), 2)
        }
        
    status = "Drift Detected" if drift_detected else "Stable"
    message = "Production distribution has shifted statistically from baseline training distributions. Model performance may degrade." if drift_detected else "Incoming request distributions align with baseline training distributions."
    
    return {
        "drift_detected": drift_detected,
        "drift_status": status,
        "message": message,
        "sample_sizes": {
            "baseline": len(base_df),
            "production": prod_size
        },
        "features": drift_details
    }
