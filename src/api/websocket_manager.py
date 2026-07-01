import asyncio
import random
import numpy as np
import pandas as pd
from fastapi import WebSocket, WebSocketDisconnect
from src.api.config import logger
from src.api.ml_services import memory_customers, champion_model, challenger_model
from src.api.database import log_inference, log_shadow_prediction

class WebSocketManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"WebSocket client connected. Total connections: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
            logger.info(f"WebSocket client disconnected. Total connections: {len(self.active_connections)}")

    async def stream_live_transactions(self, websocket: WebSocket):
        """Generates mock transaction stream, executes predictions, and pushes live JSON socket updates."""
        try:
            while True:
                # Select random customer
                if not memory_customers:
                    cust_id = "19999"
                    cust = {
                        "id": cust_id,
                        "recency": 45,
                        "frequency": 3,
                        "monetary": 120.0,
                        "basketSize": 4.5,
                        "isUk": 1,
                        "avgDaysBetween": 30.0
                    }
                else:
                    cust_id = random.choice(list(memory_customers.keys()))
                    cust = memory_customers[cust_id]

                # Simulate transaction values
                invoice_value = round(random.uniform(15.0, 250.0), 2)
                quantity = random.randint(1, 10)
                
                old_freq = cust.get("frequency", 3)
                old_mon = cust.get("monetary", 100.0)
                old_basket = cust.get("basketSize", cust.get("basket_size", 4.0))
                is_uk = cust.get("isUk", cust.get("is_uk", 1))
                avg_days = cust.get("avgDaysBetween", cust.get("avg_days_between", 30.0))
                
                new_freq = old_freq + 1
                new_mon = (old_mon * old_freq + invoice_value) / new_freq
                new_basket = (old_basket * old_freq + quantity) / new_freq
                
                # Update memory cache
                cust["recency"] = 0
                cust["frequency"] = new_freq
                cust["monetary"] = new_mon
                cust["basketSize"] = new_basket
                memory_customers[cust_id] = cust

                # Features ordering: Recency, Frequency, Monetary, AvgBucketSize, AvgDaysBetween, Recency_to_AvgDaysRatio, Recent_Orders_Ratio, Is_UK
                features = np.array([[
                    0.0,
                    new_freq,
                    new_mon,
                    new_basket,
                    avg_days,
                    0.0,
                    1.0,
                    is_uk
                ]])
                
                champion_prob = 0.15
                challenger_prob = 0.20
                
                # Model predicts
                if champion_model is not None:
                    try:
                        champion_prob = float(champion_model.predict_proba(features)[:, 1][0])
                    except Exception as ex:
                        logger.error(f"XGBoost WS prediction failed: {str(ex)}")
                
                if challenger_model is not None:
                    try:
                        df_features = pd.DataFrame(features, columns=[
                            'Recency', 'Frequency', 'Monetary', 'AvgBucketSize', 
                            'AvgDaysBetween', 'Recency_to_AvgDaysRatio', 'Recent_Orders_Ratio', 'Is_UK'
                        ])
                        challenger_prob = float(challenger_model.predict_proba(df_features)[:, 1][0])
                    except Exception as ex:
                        logger.error(f"RF WS prediction failed: {str(ex)}")

                if champion_prob >= 0.70:
                    new_risk_tier = "High Risk"
                elif champion_prob >= 0.30:
                    new_risk_tier = "Medium Risk"
                else:
                    new_risk_tier = "Low Risk"

                # Log predictions
                log_inference(0.0, new_freq, new_mon, new_basket)
                log_shadow_prediction(0.0, new_freq, new_mon, new_basket, champion_prob, challenger_prob)

                payload = {
                    "id": cust_id,
                    "type": "TRANSACTION",
                    "invoiceValue": invoice_value,
                    "quantity": quantity,
                    "newMetrics": {
                        "recency": 0,
                        "frequency": int(new_freq),
                        "monetary": round(float(new_mon), 2),
                        "basketSize": round(float(new_basket), 1),
                        "churnProb": round(float(champion_prob), 4),
                        "riskTier": new_risk_tier
                    }
                }
                
                await websocket.send_json(payload)
                await asyncio.sleep(random.uniform(3.0, 6.0))
                
        except WebSocketDisconnect:
            pass
        except Exception as e:
            logger.error(f"Error streaming live transactions: {str(e)}")

ws_manager = WebSocketManager()
