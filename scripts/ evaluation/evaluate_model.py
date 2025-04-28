import pandas as pd
import joblib
from sklearn.metrics import (precision_recall_curve, 
                            average_precision_score,
                            confusion_matrix)
import matplotlib.pyplot as plt
import seaborn as sns
import json

def evaluate_models():
    # Load data
    test_df = pd.read_csv('../data/processed/test.csv')
    X_test = test_df.drop('IsFraud', axis=1)
    y_test = test_df['IsFraud']
    
    # Load models
    supervised_model = joblib.load('../models/supervised_model.pkl')
    anomaly_detector = joblib.load('../models/anomaly_detector.pkl')
    
    # Supervised evaluation
    y_pred_prob = supervised_model.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_prob > 0.5).astype(int)
    
    print("Supervised Model Performance:")
    print(classification_report(y_test, y_pred))
    
    # Plot precision-recall curve
    precision, recall, _ = precision_recall_curve(y_test, y_pred_prob)
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.savefig('../models/supervised_pr_curve.png')
    
    # Anomaly detection evaluation
    anomaly_scores = anomaly_detector.decision_function(X_test)
    anomaly_roc_auc = roc_auc_score(y_test, -anomaly_scores)
    
    print("\nAnomaly Detection Performance:")
    print(f"ROC AUC: {anomaly_roc_auc:.4f}")
    
    # Combined evaluation
    combined_score = (y_pred_prob + (-anomaly_scores))/2
    combined_roc_auc = roc_auc_score(y_test, combined_score)
    
    print("\nCombined Approach Performance:")
    print(f"ROC AUC: {combined_roc_auc:.4f}")

if __name__ == '__main__':
    evaluate_models()
