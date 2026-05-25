import os
import time
import uuid
import logging
from contextlib import asynccontextmanager
from typing import List, Literal, Optional

import joblib
import numpy as np
import pandas as pd

from fastapi import (
    FastAPI,
    HTTPException,
    Request,
    status,
    Depends,
)
from fastapi.responses import JSONResponse
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field, ConfigDict


# =========================================================
# LOGGING CONFIG
# =========================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)


# =========================================================
# CONFIG
# =========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_DIR = os.path.abspath(
    os.path.join(BASE_DIR, "..", "..", "models")
)


XGB_MODEL_PATH = os.path.join(MODEL_DIR, "xgboost_model.pkl")
ISO_MODEL_PATH = os.path.join(MODEL_DIR, "isolation_forest.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "preprocess_scaler.pkl")

API_KEY = os.getenv("API_KEY", "HiddenKey123")

MODEL_VERSION = "v1.0.0"


# =========================================================
# GLOBAL MODELS
# =========================================================

xgb_model = None
iso_model = None
scaler = None


# =========================================================
# LIFESPAN STARTUP
# =========================================================

@asynccontextmanager
async def lifespan(app: FastAPI):

    global xgb_model, iso_model, scaler

    try:
        logger.info("Loading ML models...")

        xgb_model = joblib.load(XGB_MODEL_PATH)
        iso_model = joblib.load(ISO_MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)

        logger.info("Models loaded successfully")

    except Exception as e:
        logger.exception("Failed to load models")
        raise RuntimeError(f"Startup failed: {str(e)}")

    yield

    logger.info("Shutting down Fraud Detection API")


# =========================================================
# FASTAPI APP
# =========================================================

app = FastAPI(
    title="Fraud Detection API",
    description="Production Grade Fraud Detection System",
    version=MODEL_VERSION,
    lifespan=lifespan
)


# =========================================================
# SECURITY
# =========================================================

api_key_header = APIKeyHeader(name="X-API-KEY")


async def validate_api_key(api_key: str = Depends(api_key_header)):

    if api_key != API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API Key"
        )


# =========================================================
# REQUEST / RESPONSE SCHEMAS
# =========================================================

class TransactionRequest(BaseModel):

    model_config = ConfigDict(extra="forbid")

    amount: float = Field(..., ge=0)

    transaction_hour: int = Field(..., ge=0, le=23)

    foreign_transaction: Literal[0, 1]

    location_mismatch: Literal[0, 1]

    device_trust_score: int = Field(..., ge=0, le=100)

    velocity_last_24h: int = Field(..., ge=0)

    cardholder_age: int = Field(..., ge=18, le=100)

    merchant_category_food: Literal[0, 1] = 0

    merchant_category_grocery: Literal[0, 1] = 0

    merchant_category_retail: Literal[0, 1] = 0

    merchant_category_travel: Literal[0, 1] = 0


class BatchTransactionRequest(BaseModel):

    transactions: List[TransactionRequest]


class PredictionResponse(BaseModel):

    request_id: str

    model_version: str

    xgboost_probability: float

    anomaly_detected: bool

    anomaly_score: float

    ensemble_score: float

    fraud_prediction: bool

    risk_level: str

    latency_ms: float


# =========================================================
# GLOBAL ERROR HANDLER
# =========================================================

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):

    logger.exception(f"Unhandled exception: {str(exc)}")

    return JSONResponse(
        status_code=500,
        content={
            "message": "Internal Server Error"
        }
    )


# =========================================================
# HEALTH ROUTES
# =========================================================

@app.get("/")
async def home():

    return {
        "message": "Fraud Detection API Running",
        "model_version": MODEL_VERSION
    }


@app.get("/health")
async def health_check():

    models_loaded = (
        xgb_model is not None and
        iso_model is not None and
        scaler is not None
    )

    return {
        "status": "healthy" if models_loaded else "unhealthy",
        "models_loaded": models_loaded,
        "model_version": MODEL_VERSION
    }


# =========================================================
# FEATURE ENGINEERING
# =========================================================

def engineer_features(input_data: pd.DataFrame) -> pd.DataFrame:

    input_data = input_data.copy()

    input_data["is_night_transaction"] = (
        input_data["transaction_hour"].apply(
            lambda x: 1 if x >= 22 or x <= 5 else 0
        )
    )

    input_data["high_amount_flag"] = (
        input_data["amount"].apply(
            lambda x: 1 if x > 5000 else 0
        )
    )

    input_data["log_amount"] = np.log1p(input_data["amount"])

    input_data["high_velocity_flag"] = (
        input_data["velocity_last_24h"].apply(
            lambda x: 1 if x > 10 else 0
        )
    )

    def get_merchant_category(row):

        if row["merchant_category_food"] == 1:
            return "food"

        if row["merchant_category_grocery"] == 1:
            return "grocery"

        if row["merchant_category_retail"] == 1:
            return "retail"

        if row["merchant_category_travel"] == 1:
            return "travel"

        return "unknown"

    input_data["merchant_category"] = input_data.apply(
        get_merchant_category,
        axis=1
    )

    return input_data


# =========================================================
# RISK LEVEL
# =========================================================

def get_risk_level(score: float) -> str:

    if score > 0.8:
        return "HIGH"

    elif score > 0.5:
        return "MEDIUM"

    return "LOW"


# =========================================================
# SINGLE PREDICTION LOGIC
# =========================================================

def run_inference(transaction: TransactionRequest):

    if xgb_model is None or iso_model is None or scaler is None:

        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Models not available"
        )

    start_time = time.time()

    request_id = str(uuid.uuid4())

    input_df = pd.DataFrame([transaction.model_dump()])

    logger.info(f"[{request_id}] Incoming transaction")

    # Feature Engineering
    input_df = engineer_features(input_df)

    # Scaling
    try:

        scaled_data = scaler.transform(input_df)

    except Exception as e:

        logger.exception(f"[{request_id}] Scaling failed")

        raise HTTPException(
            status_code=400,
            detail=f"Scaling failed: {str(e)}"
        )

    # XGBoost Prediction
    xgb_probability = float(
        xgb_model.predict_proba(scaled_data)[0][1]
    )

    # Isolation Forest Prediction
    iso_raw_prediction = iso_model.predict(scaled_data)[0]

    anomaly_detected = bool(iso_raw_prediction == -1)

    anomaly_score = abs(
        float(
            iso_model.decision_function(scaled_data)[0]
        )
    )

    # Normalize anomaly score
    anomaly_score = min(anomaly_score, 1.0)

    # Ensemble Score
    ensemble_score = (
        (0.7 * xgb_probability) +
        (0.3 * anomaly_score)
    )

    fraud_prediction = bool(ensemble_score > 0.5)

    latency_ms = round(
        (time.time() - start_time) * 1000,
        2
    )

    logger.info(
        f"[{request_id}] "
        f"Fraud={fraud_prediction} "
        f"Score={ensemble_score:.4f} "
        f"Latency={latency_ms}ms"
    )

    return PredictionResponse(
        request_id=request_id,
        model_version=MODEL_VERSION,
        xgboost_probability=round(xgb_probability, 4),
        anomaly_detected=anomaly_detected,
        anomaly_score=round(anomaly_score, 4),
        ensemble_score=round(ensemble_score, 4),
        fraud_prediction=fraud_prediction,
        risk_level=get_risk_level(ensemble_score),
        latency_ms=latency_ms
    )


# =========================================================
# PREDICT ENDPOINT
# =========================================================

@app.post(
    "/predict",
    response_model=PredictionResponse,
    dependencies=[Depends(validate_api_key)]
)
async def predict_fraud(data: TransactionRequest):

    return run_inference(data)


# =========================================================
# BATCH PREDICTION
# =========================================================

@app.post(
    "/predict/batch",
    dependencies=[Depends(validate_api_key)]
)
async def batch_predict(
    batch_data: BatchTransactionRequest
):

    predictions = []

    for transaction in batch_data.transactions:

        result = run_inference(transaction)

        predictions.append(result.model_dump())

    return {
        "total_transactions": len(predictions),
        "predictions": predictions
    }


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )