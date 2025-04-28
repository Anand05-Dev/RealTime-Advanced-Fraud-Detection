from pydantic import BaseSettings
import os

class Settings(BaseSettings):
    SUPERVISED_MODEL_PATH: str = os.getenv(
        "SUPERVISED_MODEL_PATH", 
        "../models/supervised_model.pkl"
    )
    ANOMALY_MODEL_PATH: str = os.getenv(
        "ANOMALY_MODEL_PATH",
        "../models/anomaly_detector.pkl"
    )
    PREPROCESSOR_PATH: str = os.getenv(
        "PREPROCESSOR_PATH",
        "../models/preprocessor.pkl"
    )
    SUPERVISED_THRESHOLD: float = 0.5
    ANOMALY_THRESHOLD: float = -0.2
    
    class Config:
        env_file = ".env"

settings = Settings()
