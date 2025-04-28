import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score
import joblib
import json
from pathlib import Path

def train_anomaly_detector():
    # Load processed data
    train_df = pd.read_csv('../data/processed/train.csv')
    X_train = train_df[train_df['IsFraud'] == 0].drop('IsFraud', axis=1)  # Train on normal transactions
    
    # Load preprocessing artifacts
    scaler = joblib.load('../models/scaler.pkl')
    
    # Train Isolation Forest
    model = IsolationForest(
        n_estimators=150,
        contamination=0.01,  # Expected fraud rate
        random_state=42
    )
    
    model.fit(X_train)
    
    # Evaluate
    test_df = pd.read_csv('../data/processed/test.csv')
    X_test = test_df.drop('IsFraud', axis=1)
    y_test = test_df['IsFraud']
    
    anomaly_scores = model.decision_function(X_test)
    roc_auc = roc_auc_score(y_test, -anomaly_scores)  # Convert to "fraud probability"
    
    metrics = {
        'roc_auc': roc_auc,
        'contamination': 0.01
    }
    
    # Save artifacts
    Path('../models').mkdir(exist_ok=True)
    joblib.dump(model, '../models/anomaly_detector.pkl')
    with open('../models/anomaly_metrics.json', 'w') as f:
        json.dump(metrics, f)
    
    print("Anomaly detection training complete.")
    print(f"ROC AUC: {roc_auc:.4f}")

if __name__ == '__main__':
    train_anomaly_detector()
