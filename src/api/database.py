import os
import sqlite3
import datetime
from src.api.config import DB_PATH, HISTORY_PATH, logger

def init_db() -> None:
    """Initializes the CSV inference log and SQLite shadow database schemas."""
    # Initialize CSV file
    try:
        os.makedirs(os.path.dirname(HISTORY_PATH), exist_ok=True)
        if not os.path.exists(HISTORY_PATH):
            with open(HISTORY_PATH, "w") as f:
                f.write("Timestamp,Recency,Frequency,Monetary,BasketSize\n")
            logger.info(f"Initialized inference history log at {HISTORY_PATH}")
    except Exception as e:
        logger.error(f"Failed to initialize inference log file: {str(e)}")

    # Initialize SQLite database
    try:
        os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS shadow_predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                recency REAL,
                frequency REAL,
                monetary REAL,
                basket_size REAL,
                champion_prob REAL,
                challenger_prob REAL
            )
        """)
        conn.commit()
        conn.close()
        logger.info(f"Successfully initialized shadow prediction database at {DB_PATH}")
    except Exception as e:
        logger.error(f"Failed to initialize SQLite shadow DB: {str(e)}")

def log_inference(recency: float, frequency: float, monetary: float, basket_size: float) -> None:
    """Appends live inference inputs to the CSV history log."""
    try:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(HISTORY_PATH, "a") as f:
            f.write(f"{timestamp},{recency},{frequency},{monetary},{basket_size}\n")
    except Exception as e:
        logger.error(f"Failed to log inference request to CSV: {str(e)}")

def log_shadow_prediction(recency: float, frequency: float, monetary: float, basket_size: float, champion_prob: float, challenger_prob: float) -> None:
    """Logs prediction inputs and outputs of Champion and Challenger models to SQLite."""
    try:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO shadow_predictions (timestamp, recency, frequency, monetary, basket_size, champion_prob, challenger_prob) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (timestamp, recency, frequency, monetary, basket_size, champion_prob, challenger_prob)
        )
        conn.commit()
        conn.close()
    except Exception as e:
        logger.error(f"Failed to log shadow prediction to SQLite: {str(e)}")
