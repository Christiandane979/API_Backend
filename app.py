from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load models and encoders
total_score_model = joblib.load('trained_data/total_score_model.pkl')
grade_model = joblib.load('trained_data/grade_model.pkl')
asp_model = joblib.load('trained_data/career_aspiration_model.pkl')
grade_encoder = joblib.load('trained_data/grade_label_encoder.pkl')
aspiration_encoder = joblib.load('trained_data/career_aspiration_label_encoder.pkl')
features = joblib.load('trained_data/model_features.pkl')

@app.route('/')
def home():
    return "Student Score Prediction API is running!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        # Ensure all required features are in the input
        input_data = [data[feature] for feature in features]
        input_array = np.array([input_data])

        # Make predictions
        predicted_score = total_score_model.predict(input_array)[0]
        predicted_grade_encoded = grade_model.predict(input_array)[0]
        predicted_asp_encoded = asp_model.predict(input_array)[0]

        predicted_grade = grade_encoder.inverse_transform([predicted_grade_encoded])[0]
        predicted_asp = aspiration_encoder.inverse_transform([predicted_asp_encoded])[0]

        return jsonify({
            'predicted_total_score': round(predicted_score, 2),
            'predicted_grade': predicted_grade,
            'predicted_career_aspiration': predicted_asp
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
