import pandas as pd
from joblib import load
from flask import Flask, request, jsonify
from flask_cors import CORS
from category_encoders import BinaryEncoder

# Load the trained model
try:
    model = load('decision_tree_model.joblib')
except FileNotFoundError:
    raise FileNotFoundError("The model file 'decision_tree_model.joblib' was not found. Ensure it is in the same directory as this script.")

# Load the dataset to fit the encoder
try:
    data = pd.read_csv("cbb.csv")
except FileNotFoundError:
    raise FileNotFoundError("The dataset file 'cbb.csv' was not found. Ensure it is in the same directory as this script.")

categorical_features = ['CONF', 'POSTSEASON']

# Initialize the encoder and fit it on the dataset
encoder = BinaryEncoder()
encoder.fit(data[categorical_features])

# Get the feature names used during training, excluding 'TEAM'
training_features = data.drop(columns=['W', 'TEAM']).columns  # Drop the target column and 'TEAM'

# Initialize Flask app
api = Flask(__name__)
CORS(api)

@api.route('/api/cbb_prediction', methods=['POST'])
def predict_cbb():
    try:
        # Get input data from the client
        input_data = request.json.get('inputs', None)
        if input_data is None:
            return jsonify({"error": "No input data provided. Please include 'inputs' in the JSON payload."}), 400

        # Convert input data to a DataFrame
        input_df = pd.DataFrame(input_data)

        # Drop the target column if it exists in the input
        if 'W' in input_df.columns:
            input_df = input_df.drop(columns=['W'])

        # Drop the 'TEAM' column if it exists in the input
        if 'TEAM' in input_df.columns:
            input_df = input_df.drop(columns=['TEAM'])

        # Reorder the input features to match the training order
        input_df = input_df[training_features]

        # Encode categorical features
        input_encoded = encoder.transform(input_df[categorical_features])

        # Drop original categorical features
        input_df = input_df.drop(categorical_features, axis=1)

        # Merge encoded features back into the input DataFrame
        input_encoded = input_encoded.reset_index(drop=True)
        final_input = pd.concat([input_df, input_encoded], axis=1)

        # Make predictions
        predictions = model.predict_proba(final_input)
        class_labels = model.classes_

        # Map predictions to "Low Wins" and "High Wins"
        response = []
        for prob in predictions:
            # Calculate probabilities for Low Wins and High Wins
            low_wins_prob = sum(prob[class_labels < 20])  # Sum probabilities for wins < 20
            high_wins_prob = sum(prob[class_labels >= 20])  # Sum probabilities for wins >= 20

            # Normalize probabilities to ensure they sum to 100%
            total_prob = low_wins_prob + high_wins_prob
            low_wins_percent = (low_wins_prob / total_prob) * 100
            high_wins_percent = (high_wins_prob / total_prob) * 100

            # Append the result
            response.append({
                "Low Wins": round(low_wins_percent, 2),
                "High Wins": round(high_wins_percent, 2)
            })

        return jsonify(response)

    except Exception as e:
        # Log the error and return a 500 response
        print("Error:", e)
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    api.run(port=8001, debug=True)