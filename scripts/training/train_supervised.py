import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
import joblib
import json
from pathlib import Path

def train_supervised_model():
    # Load processed data
    train_df = pd.read_csv('../data/processed/train.csv')
    X_train = train_df.drop('IsFraud', axis=1)
    y_train = train_df['IsFraud']
    
    # Load preprocessing artifacts
    label_encoders = joblib.load('../models/label_encoders.pkl')
    scaler = joblib.load('../models/scaler.pkl')
    
    # Define preprocessing pipeline
    numeric_features = ['Amount', 'MerchantID', 'TransactionHour', 'TransactionDay', 'TransactionMonth']
    categorical_features = ['TransactionType', 'Location']
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', 'passthrough', categorical_features)
        ])
    
    # Build model pipeline
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            class_weight='balanced',
            random_state=42
        ))
    ])
    
    # Train model
    model.fit(X_train, y_train)
    
    # Evaluate
    test_df = pd.read_csv('../data/processed/test.csv')
    X_test = test_df.drop('IsFraud', axis=1)
    y_test = test_df['IsFraud']
    
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    metrics = {
        'classification_report': classification_report(y_test, y_pred, output_dict=True),
        'roc_auc': roc_auc_score(y_test, y_proba)
    }
    
    # Save artifacts
    Path('../models').mkdir(exist_ok=True)
    joblib.dump(model, '../models/supervised_model.pkl')
    with open('../models/supervised_metrics.json', 'w') as f:
        json.dump(metrics, f)
    
    print("Supervised model training complete.")
    print(f"ROC AUC: {metrics['roc_auc']:.4f}")

if __name__ == '__main__':
    train_supervised_model()
