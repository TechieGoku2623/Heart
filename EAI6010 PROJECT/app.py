import joblib
from flask import Flask, request, jsonify

# Load the trained model
model = joblib.load('heart_disease_model.pkl')

# Initialize Flask app
app = Flask(__name__)

# Define the prediction route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Parse input JSON
        data = request.get_json()

        # Ensure all necessary features are provided
        required_features = [
            'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
            'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'
        ]

        # Check if all required features are in the input data
        if not all(feature in data for feature in required_features):
            return jsonify({"error": "Missing one or more required features"}), 400

        # Extract features from input data
        features = [
            data['age'],
            data['sex'],
            data['cp'],
            data['trestbps'],
            data['chol'],
            data['fbs'],
            data['restecg'],
            data['thalach'],
            data['exang'],
            data['oldpeak'],
            data['slope'],
            data['ca'],
            data['thal']
        ]
        
        # Perform prediction
        prediction = model.predict([features])[0]
        probability = model.predict_proba([features])[0][1]

        # Return the prediction and probability as a JSON response
        return jsonify({
            "heart_disease_prediction": int(prediction),
            "probability": round(probability, 2)
        })
    
    except Exception as e:
        # Return error message if something goes wrong
        return jsonify({"error": str(e)}), 500

# Run the app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
