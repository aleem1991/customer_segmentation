from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from src.api.config import logger
from src.api.database import init_db
from src.api.ml_services import load_models_and_data
from src.api.routes import router

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Load databases and models
    logger.info("Starting up Customer Churn Prediction API...")
    init_db()
    load_models_and_data()
    yield
    # Shutdown
    logger.info("Shutting down Customer Churn Prediction API...")

app = FastAPI(
    title="Customer Churn Prediction API",
    description="Production-grade REST API serving real-time customer churn forecasts based on XGBoost & Random Forest features.",
    version="2.0.0",
    lifespan=lifespan
)

# Enable CORS for frontend dashboard access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict origins in production environments
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routes APIRouter
app.include_router(router)
