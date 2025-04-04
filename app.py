from flask import Flask, request, jsonify
import numpy as np
from sklearn.linear_model import LogisticRegression
import pickle

app = Flask(__name__)

# Synthetic data generation and model training (for demo purposes)
def train_model():
    # Features: [credit.policy, int.rate, installment, log.annual.inc, dti, fico, revol.util, inq.last.6mths]
    X = np.array([
        [1, 0.11, 300, 11.0, 20.0, 700, 30.0, 2],
        [0, 0.15, 500, 10.5, 35.0, 650, 50.0, 5],
        [1, 0.09, 200, 12.0, 15.0, 750, 20.0, 1],
        [0, 0.13, 400, 11.5, 25.0, 680, 40.0, 3],
        # Add more data in a real scenario
    ])
    # Target: not.fully.paid (0 = fully paid, 1 = not fully paid)
    y = np.array([0, 1, 0, 1])
    
    # Train a simple Logistic Regression model
    model = LogisticRegression()
    model.fit(X, y)
    
    # Save the model to a file
    with open("loan_model.pkl", "wb") as f:
        pickle.dump(model, f)

# Load the pre-trained model
try:
    with open("loan_model.pkl", "rb") as f:
        model = pickle.load(f)
except FileNotFoundError:
    train_model()
    with open("loan_model.pkl", "rb") as f:
        model = pickle.load(f)

# Define the prediction endpoint
@app.route("/predict", methods=["POST"])
def predict_loan_approval():
    try:
        # Get JSON data from the request
        data = request.get_json()
        
        # Extract features from the request
        credit_policy = float(data["credit_policy"])
        int_rate = float(data["int_rate"])
        installment = float(data["installment"])
        log_annual_inc = float(data["log_annual_inc"])
        dti = float(data["dti"])
        fico = float(data["fico"])
        revol_util = float(data["revol_util"])
        inq_last_6mths = float(data["inq_last_6mths"])
        
        # Prepare input data for prediction
        input_data = np.array([[credit_policy, int_rate, installment, log_annual_inc, 
                                dti, fico, revol_util, inq_last_6mths]])
        
        # Make prediction
        prediction = model.predict(input_data)[0]
        
        # Interpret the result
        result = "Approved" if prediction == 0 else "Rejected"
        
        # Return the response as JSON
        return jsonify({
            "loan_approval_status": result,
            "message": "Prediction successful"
        })
    
    except KeyError as e:
        return jsonify({
            "error": f"Missing field: {str(e)}",
            "message": "Please provide all required fields"
        }), 400
    except Exception as e:
        return jsonify({
            "error": str(e),
            "message": "An error occurred during prediction"
        }), 500

# Health check endpoint
@app.route("/ping", methods=["GET"])
def ping():
    return jsonify({"message": "Flask backend is running"}), 200

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)