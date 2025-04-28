from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd
from config import settings
from schemas import Transaction, PredictionResponse

app = FastAPI(
    title="Fraud Detection API",
    description="API for detecting fraudulent transactions",
    version="1.0.0"
)

# Load models at startup
try:
    supervised_model = joblib.load(settings.SUPERVISED_MODEL_PATH)
    anomaly_detector = joblib.load(settings.ANOMALY_MODEL_PATH)
    preprocessor = joblib.load(settings.PREPROCESSOR_PATH)
except Exception as e:
    raise RuntimeError(f"Failed to load models: {str(e)}")

@app.post("/predict", response_model=PredictionResponse)
async def predict_fraud(transaction: Transaction):
    """
    Predict whether a transaction is fraudulent
    
    Returns:
    - supervised_prediction: Probability from supervised model (0-1)
    - anomaly_score: Anomaly score from isolation forest (-1 to 1)
    - is_fraud: Final determination based on thresholds
    """
    try:
        # Convert to DataFrame for preprocessing
        input_data = pd.DataFrame([transaction.dict()])
        
        # Preprocess features
        processed_data = preprocessor.transform(input_data)
        
        # Get predictions
        supervised_prob = supervised_model.predict_proba(processed_data)[0][1]
        anomaly_score = anomaly_detector.decision_function(processed_data)[0]
        
        # Determine final prediction
        is_fraud = (supervised_prob > settings.SUPERVISED_THRESHOLD) or \
                   (anomaly_score < settings.ANOMALY_THRESHOLD)
        
        return {
            "supervised_prediction": float(supervised_prob),
            "anomaly_score": float(anomaly_score),
            "is_fraud": bool(is_fraud),
            "status": "success"
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/health")
async def health_check():
    """Service health endpoint"""
    return {"status": "healthy"}
