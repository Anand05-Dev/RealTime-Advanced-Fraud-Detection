from pydantic import BaseModel
from typing import Optional

class Transaction(BaseModel):
    transaction_id: int
    amount: float
    merchant_id: int
    transaction_type: int
    location: int
    transaction_hour: int
    transaction_day: int
    transaction_month: int
    
    class Config:
        schema_extra = {
            "example": {
                "transaction_id": 123456,
                "amount": 150.75,
                "merchant_id": 284,
                "transaction_type": 1,
                "location": 5,
                "transaction_hour": 14,
                "transaction_day": 15,
                "transaction_month": 7
            }
        }

class PredictionResponse(BaseModel):
    supervised_prediction: float
    anomaly_score: float
    is_fraud: bool
    status: str
