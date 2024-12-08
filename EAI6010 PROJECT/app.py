import joblib
import numpy as np
from flask import Flask, request, jsonify
from sklearn.preprocessing import StandardScaler

# Initialize Flask app
app = Flask(__name__)

# Load the trained model and scaler
model = joblib.load('heart_disease_model.pkl')

# Define the prediction route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Parse input JSON
        data = request.get_json()

        # Check if all required fields are present
        required_fields = [
            'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 
            'restecg', 'thalach', 'exang'
        ]
        if not all(field in data for field in required_fields):
            return jsonify({"error": "Missing input data for one or more fields"}), 400
        
        # Extract features
        features = np.array([
            data['age'],
            data['sex'],
            data['cp'],
            data['trestbps'],
            data['chol'],
            data['fbs'],
            data['restecg'],
            data['thalach'],
            data['exang']
        ]).reshape(1, -1)

        # Normalize features using the scaler (if you used it during training)
        features_scaled = scaler.transform(features)

        # Perform prediction
        prediction = model.predict(features_scaled)[0]
        probability = model.predict_proba(features_scaled)[0][1]

        # Return response
        return jsonify({
            "heart_disease_prediction": int(prediction),
            "probability": round(probability, 2)
        })

    except Exception as e:
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500

# Run the app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
