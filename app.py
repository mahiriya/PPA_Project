
from flask import Flask, jsonify, request
import joblib
import pandas as pd
import json
import numpy as np

app = Flask(__name__)

# Load the trained model and feature columns
model = joblib.load('model.pkl')

with open('feature_columns.json', 'r') as f:
    feature_columns = json.load(f)

@app.route('/')
def home():
    return "Churn Prediction API"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)
        
        # Convert input data to DataFrame
        # Ensure the order of columns matches the training data
        input_df = pd.DataFrame([data])
        
        # The model expects certain feature names in the preprocessor. 
        # This example assumes input 'data' dictionary keys match the 'feature_cols' 
        # used in X_train. The preprocessing pipeline will handle one-hot encoding.
        
        # Reorder columns to match the order used during training (X.columns)
        # Note: If your input 'data' does not contain all original features or has different names,
        # you'll need more robust handling here, possibly creating a dummy dataframe 
        # and filling it with provided values.
        
        # For this example, we'll assume `data` contains keys matching the `feature_cols`
        # from the notebook's X dataframe before preprocessing.
        
        # Filter input_df to only include the features expected by the model's first step (preprocessor)
        # This requires the input `data` to have the keys corresponding to `feature_cols`
        # from the notebook's Cell 6.
        
        # Let's reconstruct the expected input DataFrame structure based on the 'feature_cols' from the notebook
        # If the incoming JSON doesn't exactly match the structure of X, this part needs careful handling.
        # A simpler approach for the Flask app is to accept the raw features that went into `X`.
        
        # Creating a DataFrame from the input data, ensuring the order of columns from `feature_cols`
        # Note: The `feature_columns` saved in JSON are the *processed* feature names (numeric + one-hot encoded).
        # The model's `preprocessor` expects the *original* `feature_cols` from Cell 6.
        
        # To correctly handle this, we need the original `feature_cols` list here.
        # Since `all_feature_names` contains the post-preprocessing names, we need to ensure
        # that the input JSON has the raw features (from `feature_cols` in Cell 6).
        
        # For simplicity, assume the incoming `data` keys match the `feature_cols` from original `X`.
        # This is a critical point for a real-world deployment; the input structure must match.
        
        # The `feature_columns` list loaded from `feature_columns.json` are the names *after* preprocessing.
        # The `model.predict` or `model.predict_proba` call on a pipeline expects the *raw* dataframe `X`.
        
        # Let's assume the JSON input `data` has the keys corresponding to the `feature_cols`.
        expected_raw_features = [
            'total_orders', 'total_amount', 'avg_order_value', 'avg_quantity',
            'recency_days', 'tenure_days', 'orders_per_month', 'avg_rating',
            'num_restaurants', 'most_common_payment', 'city', 'gender'
        ]
        
        processed_input = pd.DataFrame([data], columns=expected_raw_features)
        
        prediction = model.predict(processed_input)
        probability = model.predict_proba(processed_input)[:, 1]
        
        return jsonify({
            'prediction': int(prediction[0]), # Convert numpy int to Python int
            'probability_churn': float(probability[0]) # Convert numpy float to Python float
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5000)
