from flask import Flask, render_template, request, jsonify
from sklearn.preprocessing import StandardScaler
import numpy as np
import joblib

# Load your model (make sure it's available)
model = joblib.load('heart_disease_model.pkl')  # Replace with your model file path

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get form data
        age = int(request.form['age'])
        sex = int(request.form['sex'])
        cp = int(request.form['cp'])
        trestbps = int(request.form['trestbps'])
        chol = int(request.form['chol'])
        fbs = int(request.form['fbs'])
        restecg = int(request.form['restecg'])
        thalach = int(request.form['thalach'])
        exang = int(request.form['exang'])
        
        # Create an array of features
        features = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang]])
        
        # Standardize features (if needed)
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        # Make prediction
        prediction = model.predict(features_scaled)
        probability = model.predict_proba(features_scaled)[0][1]
        
        # Send results to the HTML page
        return render_template('index.html', prediction=prediction[0], probability=probability)
    
    return render_template('index.html', prediction=None, probability=None)

if __name__ == '__main__':
    app.run(debug=True)
