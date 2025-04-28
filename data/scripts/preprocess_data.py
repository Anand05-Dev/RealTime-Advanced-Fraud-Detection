import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import joblib
from pathlib import Path

def preprocess_data():
    # Load raw data
    df = pd.read_csv('../data/raw/credit_card_fraud_dataset.csv')
    
    # Convert TransactionDate to datetime and extract features
    df['TransactionDate'] = pd.to_datetime(df['TransactionDate'])
    df['TransactionHour'] = df['TransactionDate'].dt.hour
    df['TransactionDay'] = df['TransactionDate'].dt.day
    df['TransactionMonth'] = df['TransactionDate'].dt.month
    df = df.drop(columns=['TransactionDate'])
    
    # Encode categorical features
    label_encoders = {}
    categorical_cols = ['TransactionType', 'Location']
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le
    
    # Save encoders for inference
    Path('../models').mkdir(exist_ok=True)
    joblib.dump(label_encoders, '../models/label_encoders.pkl')
    
    # Split data
    X = df.drop('IsFraud', axis=1)
    y = df['IsFraud']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Scale numerical features
    scaler = StandardScaler()
    numerical_cols = ['Amount', 'MerchantID', 'TransactionHour', 'TransactionDay', 'TransactionMonth']
    X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
    X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])
    
    joblib.dump(scaler, '../models/scaler.pkl')
    
    # Save processed data
    pd.concat([X_train, y_train], axis=1).to_csv('../data/processed/train.csv', index=False)
    pd.concat([X_test, y_test], axis=1).to_csv('../data/processed/test.csv', index=False)

if __name__ == '__main__':
    preprocess_data()
