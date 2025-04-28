from flask import Flask, request, jsonify
import pandas as pd
import joblib
from preprocess import preprocess_data  # Your preprocessing function

app = Flask(__name__)

# Load your trained model
model = joblib.load('fraud_detection_model.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    # Get transaction data
    transaction_data = request.json
    
    # Preprocess the data
    processed_data = preprocess_data(transaction_data)
    
    # Make prediction
    prediction = model.predict(processed_data)
    probability = model.predict_proba(processed_data)[:,1]
    
    return jsonify({
        'is_fraud': int(prediction[0]),
        'fraud_probability': float(probability[0])
    })

if __name__ == '__main__':
    app.run(debug=True)
