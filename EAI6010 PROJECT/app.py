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
        
        # Extract features
        features = [
            data['age'],
            data['sex'],
            data['cp'],
            data['trestbps'],
            data['chol'],
            data['fbs'],
            data['restecg'],
            data['thalach'],
            data['exang']
        ]
        
        # Perform prediction
        prediction = model.predict([features])[0]
        probability = model.predict_proba([features])[0][1]

        # Return response
        return jsonify({
            "heart_disease_prediction": int(prediction),
            "probability": round(probability, 2)
        })
    except Exception as e:
        return jsonify({"error": str(e)})

# Run the app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)


