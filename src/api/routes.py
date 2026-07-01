import os
import sqlite3
import numpy as np
from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from typing import Dict, Any

from src.api.config import DB_PATH, HISTORY_PATH, logger
from src.api.models import CustomerInput, BatchInput
from src.api import ml_services
from src.api.database import log_inference, log_shadow_prediction
from src.api.drift_service import run_drift_analysis
from src.api.websocket_manager import ws_manager

router = APIRouter()

@router.get("/health", tags=["System"])
def health_check() -> Dict[str, Any]:
    """Returns API health status and checks if the model is loaded."""
    logger.info("Executing health check request...")
    return {
        "status": "healthy",
        "model_loaded": ml_services.champion_model is not None
    }

@router.post("/predict", tags=["Predictions"])
def predict_churn(customer: CustomerInput) -> Dict[str, Any]:
    """Calculates churn probability and risk tier for a single customer."""
    # Log incoming prediction inputs to CSV
    log_inference(customer.recency, customer.frequency, customer.monetary, customer.basket_size)

    if ml_services.champion_model is None:
        logger.error("Inference requested but model is not loaded.")
        raise HTTPException(
            status_code=503, 
            detail="Machine learning model is not loaded. Please train the model first."
        )
    
    try:
        # Preprocess features
        features = ml_services.preprocess_features(customer)
        
        # Calculate probabilities
        prob = ml_services.run_champion_inference(features)
        challenger_prob = ml_services.run_challenger_inference(features)
        
        # Compute explainability values
        shap_values = ml_services.compute_shap_values(features)
        
        # Log to SQLite DB
        log_shadow_prediction(
            customer.recency, customer.frequency, customer.monetary, customer.basket_size,
            prob, challenger_prob
        )
    except Exception as e:
        logger.error(f"Inference prediction process failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")
        
    if prob >= 0.70:
        tier = "High Risk"
    elif prob >= 0.30:
        tier = "Medium Risk"
    else:
        tier = "Low Risk"
        
    recommendation = ml_services.get_realtime_recommendation(
        customer.recency,
        customer.frequency,
        customer.monetary,
        prob
    )
    
    return {
        "churn_probability": round(prob, 4),
        "risk_tier": tier,
        "recommendation": recommendation,
        "shap_values": shap_values
    }

@router.post("/predict_batch", tags=["Predictions"])
def predict_churn_batch(batch: BatchInput) -> Dict[str, Any]:
    """Calculates churn predictions in batch for a list of customer records."""
    if ml_services.champion_model is None:
        logger.error("Inference batch requested but model is not loaded.")
        raise HTTPException(
            status_code=503, 
            detail="Machine learning model is not loaded. Please train the model first."
        )
        
    if not batch.customers:
        return {"predictions": []}
        
    features_list = []
    for c in batch.customers:
        features = ml_services.preprocess_features(c)
        features_list.append(features[0])
    
    logger.info(f"Running batch inference for {len(batch.customers)} profiles...")
    try:
        probs = ml_services.champion_model.predict_proba(np.array(features_list))[:, 1].tolist()
    except Exception as e:
        logger.error(f"XGBoost batch prediction failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")
        
    results = []
    for customer, prob in zip(batch.customers, probs):
        if prob >= 0.70:
            tier = "High Risk"
        elif prob >= 0.30:
            tier = "Medium Risk"
        else:
            tier = "Low Risk"
            
        recommendation = ml_services.get_realtime_recommendation(
            customer.recency,
            customer.frequency,
            customer.monetary,
            prob
        )
        
        results.append({
            "churn_probability": round(prob, 4),
            "risk_tier": tier,
            "recommendation": recommendation
        })
        
    return {"predictions": results}

@router.get("/monitor", tags=["System"])
def monitor_drift() -> Dict[str, Any]:
    """Runs a Kolmogorov-Smirnov test to detect data drift between baseline and production data."""
    return run_drift_analysis()

@router.get("/shadow_stats", tags=["System"])
def get_shadow_stats() -> Dict[str, Any]:
    """Retrieves side-by-side performance metrics for Champion vs Challenger models in shadow deployment."""
    if not os.path.exists(DB_PATH):
        return {
            "total_predictions": 0,
            "champion_mean": 0.0,
            "challenger_mean": 0.0,
            "mean_absolute_deviation": 0.0,
            "agreement_rate": 1.0,
            "recent_logs": []
        }
        
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # Load all shadow logs
        cursor.execute("SELECT * FROM shadow_predictions ORDER BY id DESC")
        rows = cursor.fetchall()
        conn.close()
    except Exception as e:
        logger.error(f"Error reading shadow predictions SQLite table: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Database query failed: {str(e)}")
        
    total_preds = len(rows)
    if total_preds == 0:
        return {
            "total_predictions": 0,
            "champion_mean": 0.0,
            "challenger_mean": 0.0,
            "mean_absolute_deviation": 0.0,
            "agreement_rate": 1.0,
            "recent_logs": []
        }
        
    champion_probs = [r["champion_prob"] for r in rows]
    challenger_probs = [r["challenger_prob"] for r in rows]
    
    champion_mean = sum(champion_probs) / total_preds
    challenger_mean = sum(challenger_probs) / total_preds
    
    # Calculate Mean Absolute Deviation (MAD)
    mad = sum(abs(champ - chall) for champ, chall in zip(champion_probs, challenger_probs)) / total_preds
    
    # Calculate decision agreement (agreement on binary threshold 0.50 risk split)
    agreements = 0
    for champ, chall in zip(champion_probs, challenger_probs):
        champ_class = 1 if champ >= 0.5 else 0
        chall_class = 1 if chall >= 0.5 else 0
        if champ_class == chall_class:
            agreements += 1
            
    agreement_rate = agreements / total_preds
    
    # Extract last 5 logs for front-end rendering
    recent_logs = []
    for r in rows[:5]:
        recent_logs.append({
            "id": r["id"],
            "timestamp": r["timestamp"],
            "recency": r["recency"],
            "frequency": r["frequency"],
            "monetary": r["monetary"],
            "basket_size": r["basket_size"],
            "champion_prob": round(float(r["champion_prob"]), 4),
            "challenger_prob": round(float(r["challenger_prob"]), 4)
        })
        
    return {
        "total_predictions": total_preds,
        "champion_mean": round(champion_mean, 4),
        "challenger_mean": round(challenger_mean, 4),
        "mean_absolute_deviation": round(mad, 4),
        "agreement_rate": round(agreement_rate, 4),
        "recent_logs": recent_logs
    }

@router.websocket("/ws/transactions")
async def websocket_transactions(websocket: WebSocket):
    """WebSocket connection that generates a live transaction stream and pushes real-time updates."""
    await ws_manager.connect(websocket)
    try:
        await ws_manager.stream_live_transactions(websocket)
    except WebSocketDisconnect:
        ws_manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket execution error: {str(e)}")
        ws_manager.disconnect(websocket)
