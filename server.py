from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import os

app = Flask(__name__)

# Load ML models and scalers
MODEL_DIR = os.path.dirname(os.path.abspath(__file__))

diabetes_model = pickle.load(open(os.path.join(MODEL_DIR, 'random_forest_diabetes.pkl'), 'rb'))
heart_model = pickle.load(open(os.path.join(MODEL_DIR, 'svm_heart.pkl'), 'rb'))
diabetes_scaler = pickle.load(open(os.path.join(MODEL_DIR, 'scaler_diabetes.pkl'), 'rb'))
heart_scaler = pickle.load(open(os.path.join(MODEL_DIR, 'scaler_heart.pkl'), 'rb'))

@app.route('/')
def home():
    return render_template('f1.html')  # Ensure f1.html is inside 'templates' folder

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.form
        disease_type = data.get("disease")

        if disease_type == "diabetes":
            features = np.array([[
                float(data.get("pregnancies")),
                float(data.get("glucose")),
                float(data.get("blood_pressure")),
                float(data.get("skin_thickness")),
                float(data.get("insulin")),
                float(data.get("bmi")),
                float(data.get("dpf")),
                float(data.get("age"))
            ]])
            scaled_features = diabetes_scaler.transform(features)
            prediction = diabetes_model.predict(scaled_features)
            result = "Diabetes Detected" if prediction[0] == 1 else "No Diabetes"
        
        elif disease_type == "heart":
            features = np.array([[
                float(data.get("age")),
                float(data.get("sex")),
                float(data.get("cp")),
                float(data.get("trestbps")),
                float(data.get("chol")),
                float(data.get("fbs")),
                float(data.get("restecg")),
                float(data.get("thalach")),
                float(data.get("exang")),
                float(data.get("oldpeak")),
                float(data.get("slope")),
                float(data.get("ca")),
                float(data.get("thal"))
            ]])
            scaled_features = heart_scaler.transform(features)
            prediction = heart_model.predict(scaled_features)
            result = "Heart Disease Detected" if prediction[0] == 1 else "No Heart Disease"
        
        else:
            return jsonify({"error": "Invalid disease type"})
        
        return render_template("f1.html", prediction=result)
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True)
