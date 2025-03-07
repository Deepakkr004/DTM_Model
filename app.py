from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
import numpy as np

# Initialize Flask app
app = Flask(__name__, template_folder='template')

# Load pre-trained ML model and vectorizer (adjust model path as needed)
mlp_clf = joblib.load('models/model.joblib')  # Load your trained model
vectorizer = joblib.load('models/vectorizer.joblib')  # Load your vectorizer
df = pd.read_csv('models/disease_data.csv')  # Load disease-related data

@app.route('/')
def index():
    return render_template('index.html')  # Render the index.html template

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    symptom_input = data.get('symptoms', '').lower().strip()

    # Split the symptoms by commas and remove extra spaces
    symptoms_list = [symptom.strip() for symptom in symptom_input.split(',') if symptom.strip()]

    # Check the number of symptoms
    if len(symptoms_list) <= 2:
        return jsonify({
            "message": "Please provide more than two symptoms for an accurate prediction."
        }), 400

    symptom_vector = vectorizer.transform([' '.join(symptoms_list)])

    try:
        # Predict the disease using the model
        predicted_disease = mlp_clf.predict(symptom_vector)[0]

        # Get probabilities of all diseases
        probabilities = mlp_clf.predict_proba(symptom_vector)[0]

        # Create a dictionary of diseases and their corresponding probabilities
        disease_probabilities = dict(zip(mlp_clf.classes_, probabilities))

        # Sort the diseases by their probabilities in descending order and get the top 5
        top_diseases = sorted(disease_probabilities.items(), key=lambda x: x[1], reverse=True)[:5]
        top_diseases_dict = {disease: prob for disease, prob in top_diseases}

        # Get treatment and doctor for the predicted disease
        treatment = df.loc[df['disease'] == predicted_disease, 'cures'].values[0] if predicted_disease in df['disease'].values else "Not available"
        doctor = df.loc[df['disease'] == predicted_disease, 'doctor'].values[0] if predicted_disease in df['disease'].values else "Not available"

        return jsonify({
            "disease": predicted_disease,
            "treatment": treatment,
            "doctor": doctor,
            "probabilities": top_diseases_dict  # Add top 5 disease probabilities
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)  # Run the application
