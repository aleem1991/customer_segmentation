import os
import sys

# Add project root to path if needed
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.config_loader import CONFIG
from src.logger_config import logger

MODEL_PATH = CONFIG["paths"]["model"]
CHALLENGER_PATH = os.path.join(project_root, "models", "churn_rf_model.pkl")
HISTORY_PATH = os.path.join(project_root, "logs", "inference_history.csv")
DB_PATH = os.path.join(project_root, "logs", "predictions.db")
CLEANED_DATA_PATH = os.path.join(project_root, "data", "processed", "cleaned_customer_data.csv")
CUSTOMERS_JSON_PATH = os.path.join(project_root, "dashboard", "public", "data", "customers.json")
