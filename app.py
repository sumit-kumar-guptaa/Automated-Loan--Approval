# D:\Automated Loan Approval System\app.py
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))  # Add current directory to path

from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from training_pipeline import TrainingPipeline
from prediction_pipeline import PredictionPipeline
import logging

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
ARTIFACTS_FOLDER = 'artifacts'
ALLOWED_EXTENSIONS = {'csv'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(handler)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/train', methods=['POST'])
def train_model():
    try:
        if 'file' not in request.files:
            logger.error("No file part in request")
            return jsonify({'error': 'No file part'}), 400
        
        file = request.files['file']
        if file.filename == '':
            logger.error("No selected file")
            return jsonify({'error': 'No selected file'}), 400
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            logger.info(f"File uploaded: {file_path}")
            
            os.makedirs(ARTIFACTS_FOLDER, exist_ok=True)
            
            trainer = TrainingPipeline(file_path)
            best_model, best_model_name = trainer.run_pipeline()
            logger.info(f"Training completed. Best model: {best_model_name}")
            
            return jsonify({
                'message': 'Training completed successfully',
                'best_model': best_model_name
            }), 200
        else:
            logger.error("Invalid file type")
            return jsonify({'error': 'Invalid file type. Only CSV allowed'}), 400
    except Exception as e:
        logger.error(f"Error in training: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/predict', methods=['POST'])
def predict_loan_approval():
    try:
        if not request.is_json:
            logger.error("Request must be JSON")
            return jsonify({'error': 'Request must be JSON'}), 400
        
        features = request.get_json()
        logger.info(f"Received features: {features}")
        
        if 'credit.policy' in features:
            del features['credit.policy']
        
        if not os.path.exists('artifacts/best_loan_model.pkl') or not os.path.exists('artifacts/transformer.pkl'):
            logger.error("Model or transformer not found. Please train the model first.")
            return jsonify({'error': 'Model not trained. Please run /train first.'}), 400
        
        predictor = PredictionPipeline()
        probability = predictor.run_pipeline(features)
        
        if probability is not None:
            status = "Approved" if probability >= 50 else "Rejected"
            logger.info(f"Prediction successful: {probability:.2f}% - {status}")
            return jsonify({
                'probability': round(probability, 2),
                'loan_approval_status': status,
                'message': 'Prediction successful'
            }), 200
        else:
            logger.error("Prediction failed")
            return jsonify({'error': 'Prediction failed'}), 500
    except Exception as e:
        logger.error(f"Error in prediction: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy'}), 200

if __name__ == '__main__':
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    app.run(debug=True, host='0.0.0.0', port=5000)