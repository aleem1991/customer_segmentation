import os
import joblib
import json
import numpy as np
import xgboost as xgb
import pandas as pd
from typing import Dict, Any, List
from src.api.config import MODEL_PATH, CHALLENGER_PATH, CUSTOMERS_JSON_PATH, CONFIG, logger
from src.api.models import CustomerInput

# Global variables for models and in-memory customer data
champion_model = None
challenger_model = None
memory_customers = {}

def load_models_and_data() -> None:
    """Loads all models and loads customer record assets into memory."""
    global champion_model, challenger_model, memory_customers
    
    # Load Champion model
    logger.info(f"Attempting to load champion model from {MODEL_PATH}...")
    if os.path.exists(MODEL_PATH):
        try:
            champion_model = joblib.load(MODEL_PATH)
            logger.info("Successfully loaded champion XGBoost model.")
        except Exception as e:
            logger.error(f"Error loading champion model: {str(e)}")
    else:
        logger.warning(f"Champion model file missing at {MODEL_PATH}.")
        
    # Load Challenger model
    logger.info(f"Attempting to load challenger model from {CHALLENGER_PATH}...")
    if os.path.exists(CHALLENGER_PATH):
        try:
            challenger_model = joblib.load(CHALLENGER_PATH)
            logger.info("Successfully loaded challenger Random Forest model.")
        except Exception as e:
            logger.error(f"Error loading challenger model: {str(e)}")
    else:
        logger.warning(f"Challenger model file missing at {CHALLENGER_PATH}.")

    # Load customer records into memory for live WebSocket streaming
    if os.path.exists(CUSTOMERS_JSON_PATH):
        try:
            with open(CUSTOMERS_JSON_PATH, "r") as f:
                data = json.load(f)
                for cust in data:
                    memory_customers[str(cust["id"])] = cust
            logger.info(f"Successfully loaded {len(memory_customers)} customer records into memory for WebSocket streaming.")
        except Exception as e:
            logger.error(f"Failed to load customers.json into memory: {str(e)}")
    else:
        logger.warning(f"customers.json missing at {CUSTOMERS_JSON_PATH}. WebSocket streaming fallback mock data will be used.")

def get_realtime_recommendation(recency: int, frequency: int, monetary: float, churn_prob: float) -> str:
    """Generates actionable retention strategy based on client segment classification."""
    is_high_value = (frequency >= 3) or (monetary >= 300.0)
    
    if churn_prob >= 0.70:
        if is_high_value:
            return "At-Risk VIP: High historic value. Route to customer relations manager for direct feedback outreach. Offer priority recovery benefits."
        else:
            return "Hibernating Win-back: Inactive standard customer. Target with automated email re-engagement flow offering aggressive discount vouchers."
    elif churn_prob >= 0.30:
        if is_high_value:
            return "Proactive VIP Retention: High-value showing drop-off signs. Send customized recommendations based on past purchases. Avoid direct discount spam."
        else:
            return "Standard Retention: Nurture with standard newsletter promotions and seasonal discounts."
    else:
        if is_high_value:
            return "Maintain & Upsell: Core loyal customer. Exclude from margin-diluting discount codes. Send early access and premium alerts."
        else:
            return "Nurture Campaign: Keep engaged with standard marketing updates."

def preprocess_features(customer: CustomerInput) -> np.ndarray:
    """Preprocesses input parameters and fills in feature engineered attributes."""
    single_buyer_impute = CONFIG["parameters"]["single_order_imputation_days"]
    default_uk = CONFIG["parameters"]["default_is_uk"]
    
    avg_days = customer.avg_days_between if customer.avg_days_between is not None else (single_buyer_impute if customer.frequency == 1 else 30.0)
    recent_ratio = customer.recent_orders_ratio if customer.recent_orders_ratio is not None else (1.0 if customer.recency <= 60 else 0.0)
    recency_ratio = customer.recency / (avg_days + 1e-5)
    is_uk = customer.is_uk if customer.is_uk is not None else default_uk
    
    return np.array([[
        customer.recency,
        customer.frequency,
        customer.monetary,
        customer.basket_size,
        avg_days,
        recency_ratio,
        recent_ratio,
        is_uk
    ]])

def run_champion_inference(features: np.ndarray) -> float:
    """Computes prediction probability using the Champion XGBoost model."""
    if champion_model is None:
        raise ValueError("Champion model is not loaded.")
    return float(champion_model.predict_proba(features)[:, 1][0])

def run_challenger_inference(features: np.ndarray) -> float:
    """Computes prediction probability using the Challenger Random Forest model."""
    if challenger_model is None:
        return 0.0
    df_features = pd.DataFrame(features, columns=[
        'Recency', 'Frequency', 'Monetary', 'AvgBucketSize', 
        'AvgDaysBetween', 'Recency_to_AvgDaysRatio', 'Recent_Orders_Ratio', 'Is_UK'
    ])
    return float(challenger_model.predict_proba(df_features)[:, 1][0])

def compute_shap_values(features: np.ndarray) -> Dict[str, float]:
    """Calculates TreeSHAP feature attribution scores for explainability plots."""
    if champion_model is None:
        return {}
    booster = champion_model.get_booster()
    dmat = xgb.DMatrix(features, feature_names=[
        'Recency', 'Frequency', 'Monetary', 'AvgBucketSize', 
        'AvgDaysBetween', 'Recency_to_AvgDaysRatio', 'Recent_Orders_Ratio', 'Is_UK'
    ])
    contribs = booster.predict(dmat, pred_contribs=True)[0]
    return {
        "Recency": float(contribs[0]),
        "Frequency": float(contribs[1]),
        "Monetary": float(contribs[2]),
        "AvgBucketSize": float(contribs[3]),
        "AvgDaysBetween": float(contribs[4]),
        "Recency_to_AvgDaysRatio": float(contribs[5]),
        "Recent_Orders_Ratio": float(contribs[6]),
        "Is_UK": float(contribs[7])
    }
