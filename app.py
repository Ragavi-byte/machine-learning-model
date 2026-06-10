from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load the trained scaled model and the scaler
scaler = joblib.load('scaler.joblib')
model_scaled = joblib.load('logistic_regression_model_scaled.joblib')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400

        try:
            # Convert input data to DataFrame, ensuring column order matches training data
            # Assuming the order of features is the same as X_train.columns
            input_df = pd.DataFrame([data])

            # Ensure columns are in the correct order as during training
            # This assumes X_train.columns are known or passed. For robustness, it's best to pass them explicitly.
            # For now, let's assume the input_df has the correct column names and order.
            # You might retrieve X_train.columns from your training notebook if needed.
            
            # Scale the input data using the loaded scaler
            input_scaled = scaler.transform(input_df)
            input_scaled_df = pd.DataFrame(input_scaled, columns=input_df.columns)

            # Make predictions using the scaled model
            prediction = model_scaled.predict(input_scaled_df)
            prediction_proba = model_scaled.predict_proba(input_scaled_df)

            return jsonify({
                'prediction': int(prediction[0]),
                'prediction_probability_class_0': float(prediction_proba[0][0]),
                'prediction_probability_class_1': float(prediction_proba[0][1])
            })
        except Exception as e:
            return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # To run this Flask app in Colab, you'll typically use ngrok or a similar tool
    # to expose your local Flask server to the internet.
    # For direct local execution (e.g., if you download this file and run locally):
    # app.run(debug=True, host='0.0.0.0', port=5000)

    # In Colab, you would usually run a separate cell to set up ngrok.
    print("Flask app is ready. To run it, you'll need to set up ngrok in a separate cell or run this script locally.")
    print("Example ngrok setup for Colab:")
    print("from flask_ngrok import run_with_ngrok")
    print("run_with_ngrok(app)")
    print("app.run()")
