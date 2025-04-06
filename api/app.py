from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import joblib
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
# Enable CORS for all routes and all origins
CORS(app, resources={r"/api/*": {"origins": "*"}})

# Directory to save trained models
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'models')
os.makedirs(MODEL_DIR, exist_ok=True)

# Model definitions with their sklearn implementations
models = {
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "SVM": SVC(probability=True, random_state=42),
    "Logistic Regression": LogisticRegression(random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Neural Network": MLPClassifier(hidden_layer_sizes=(100,), max_iter=300, random_state=42)
}

@app.route('/api/train', methods=['POST'])
def train_model():
    logger.info("Received training request")
    data = request.json
    model_type = data.get('model')
    training_data = data.get('data')
    
    logger.info(f"Training request for model: {model_type} with {len(training_data) if training_data else 0} data points")
    
    if not model_type or not training_data or model_type not in models:
        logger.error(f"Invalid request parameters: model={model_type}, data_length={len(training_data) if training_data else 0}")
        return jsonify({"error": "Invalid request parameters"}), 400
    
    try:
        # Convert training data to pandas DataFrame
        df = pd.DataFrame(training_data)
        logger.info(f"Data converted to DataFrame with shape: {df.shape}")
        
        # Extract features and target
        X = df[['lat', 'lon', 'hour']].values
        y = df['illegal'].values
        logger.info(f"Features shape: {X.shape}, Target shape: {y.shape}")
        
        # Get the model
        model = models[model_type]
        
        # Train the model
        logger.info(f"Starting model training for {model_type}")
        model.fit(X, y)
        logger.info(f"Model training completed for {model_type}")
        
        # Save the trained model
        model_path = os.path.join(MODEL_DIR, f"{model_type.lower().replace(' ', '_')}.joblib")
        joblib.dump(model, model_path)
        logger.info(f"Model saved to {model_path}")
        
        # Calculate accuracy
        y_pred = model.predict(X)
        accuracy = accuracy_score(y, y_pred)
        
        # Calculate confusion matrix
        cm = confusion_matrix(y, y_pred).tolist()
        
        response = {
            "success": True,
            "model": model_type,
            "accuracy": float(accuracy),
            "confusionMatrix": cm
        }
        logger.info(f"Training successful: {response}")
        return jsonify(response)
    
    except Exception as e:
        logger.exception(f"Error during model training: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/predict', methods=['POST'])
def predict():
    logger.info("Received prediction request")
    data = request.json
    model_type = data.get('model')
    lat = data.get('lat')
    lon = data.get('lon')
    hour = data.get('hour')
    
    logger.info(f"Prediction request: model={model_type}, lat={lat}, lon={lon}, hour={hour}")
    
    if not model_type or lat is None or lon is None or hour is None:
        logger.error("Missing required parameters")
        return jsonify({"error": "Missing required parameters"}), 400
    
    try:
        # Check if model exists
        model_path = os.path.join(MODEL_DIR, f"{model_type.lower().replace(' ', '_')}.joblib")
        if not os.path.exists(model_path):
            logger.error(f"Model file not found: {model_path}")
            return jsonify({"error": f"Model {model_type} not trained yet"}), 404
        
        # Load the model
        logger.info(f"Loading model from {model_path}")
        model = joblib.load(model_path)
        
        # Make prediction
        X = np.array([[lat, lon, hour]])
        prediction = model.predict(X)[0]
        
        # Get probability if the model supports it
        probability = 0.5  # Default
        if hasattr(model, "predict_proba"):
            probability = float(model.predict_proba(X)[0][1])
        
        response = {
            "result": bool(prediction),
            "probability": probability,
            "location": [lat, lon],
            "hour": hour
        }
        logger.info(f"Prediction result: {response}")
        return jsonify(response)
    
    except Exception as e:
        logger.exception(f"Error during prediction: {str(e)}")
        return jsonify({"error": str(e)}), 500

# Add a health check endpoint for debugging
@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({
        "status": "healthy",
        "models_available": os.listdir(MODEL_DIR) if os.path.exists(MODEL_DIR) else []
    })

if __name__ == '__main__':
    logger.info("Starting Flask server")
    app.run(debug=True, host='0.0.0.0', port=5001) 