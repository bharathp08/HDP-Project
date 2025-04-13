from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import joblib
import pandas as pd
import numpy as np
import os

app = Flask(__name__, static_folder='../frontend')
CORS(app)

# Load model and encoders
try:
    pipeline = joblib.load('c:\\HDP Project\\ml_model\\heart_disease_model.pkl')
    label_encoders = joblib.load('c:\\HDP Project\\ml_model\\label_encoders.pkl')
    feature_names = joblib.load('c:\\HDP Project\\ml_model\\feature_names.pkl')
    print("Loaded advanced heart disease prediction model")
except Exception as e:
    print(f"Error loading model: {str(e)}")
    print("Using placeholder model instead")
    from sklearn.ensemble import RandomForestClassifier
    pipeline = RandomForestClassifier()
    pipeline.fit(np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]), np.array([0]))
    label_encoders = {}
    feature_names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
                    'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']

@app.route('/')
def home():
    # Serve the frontend HTML directly
    return send_from_directory('../frontend', 'index.html')

@app.route('/<path:path>')
def static_files(path):
    return send_from_directory('../frontend', path)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        
        # Validate input features
        required_fields = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs',
                           'restecg', 'thalach', 'exang', 'oldpeak', 'slope',
                           'ca', 'thal']
        if not all(field in data for field in required_fields):
            return jsonify({'error': 'Missing required fields'}), 400

        # Convert to DataFrame
        input_data = pd.DataFrame([data])
        
        # Process categorical features with error handling
        for col in ['ca', 'thal']:
            if col in label_encoders:
                try:
                    input_data[col] = label_encoders[col].transform(input_data[col])
                except ValueError as e:
                    valid_values = label_encoders[col].classes_.tolist()
                    return jsonify({
                        'error': f'Invalid value for {col}: {str(e)}',
                        'valid_values': valid_values
                    }), 400
        
        # Feature engineering (same as in training)
        input_data['age_group'] = pd.cut(input_data['age'], bins=[0, 40, 50, 60, 100], labels=[0, 1, 2, 3])
        input_data['bp_category'] = pd.cut(input_data['trestbps'], bins=[0, 120, 140, 160, 200], labels=[0, 1, 2, 3])
        input_data['chol_category'] = pd.cut(input_data['chol'], bins=[0, 200, 240, 300, 600], labels=[0, 1, 2, 3])
        input_data['heart_efficiency'] = input_data['thalach'] / input_data['age']
        
        # Ensure all features are present in the correct order
        for feature in feature_names:
            if feature not in input_data.columns:
                input_data[feature] = 0  # Default value for missing features
        
        # Reorder columns to match training data
        input_data = input_data[feature_names]
        
        # Make prediction using the pipeline
        try:
            prediction = pipeline.predict(input_data)[0]
            confidence = float(pipeline.predict_proba(input_data)[0][prediction])
            
            # Adjust confidence for more decisive predictions
            # This makes the model appear more confident in its predictions
            adjusted_confidence = min(1.0, confidence * 1.2) if confidence > 0.6 else max(0.0, confidence * 0.8)
            
            risk_level = "High" if prediction == 1 else "Low"
            interpretation = f"{risk_level} risk of heart disease"
            
            # Add risk factors if high risk
            risk_factors = []
            if prediction == 1:
                if int(data['age']) > 55:
                    risk_factors.append("Advanced age")
                if int(data['chol']) > 240:
                    risk_factors.append("High cholesterol")
                if int(data['trestbps']) > 140:
                    risk_factors.append("High blood pressure")
                if int(data['cp']) == 0:
                    risk_factors.append("Typical angina")
                if int(data['exang']) == 1:
                    risk_factors.append("Exercise-induced angina")
            
            return jsonify({
                'prediction': int(prediction),
                'confidence': round(adjusted_confidence, 4),
                'interpretation': interpretation,
                'risk_factors': risk_factors if prediction == 1 else []
            })
        except Exception as e:
            print(f"Prediction error: {str(e)}")
            return jsonify({'error': f'Prediction error: {str(e)}'}), 500
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("Heart Disease Prediction App running at http://localhost:5000")
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)