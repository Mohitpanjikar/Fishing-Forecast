# Fishing Forecast ML Backend

This Python backend provides real machine learning capabilities for the Fishing Forecast Guardian application, replacing the mock prediction functions with actual trained models.

## Features

- Trains machine learning models on fishing data
- Makes real-time predictions for illegal fishing activity
- Supports multiple ML algorithms:
  - Random Forest
  - Support Vector Machine (SVM)
  - Logistic Regression
  - Decision Tree
  - K-Nearest Neighbors (KNN)
  - Neural Network

## Setup

### Prerequisites

- Python 3.7+
- pip (Python package manager)

### Installation

1. Navigate to the api directory:
   ```
   cd api
   ```

2. Create a virtual environment (optional but recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

### Running the Server

Start the Flask development server:
```
python app.py
```

The API will be available at http://localhost:5000/api

### Production Deployment

For production, use Gunicorn as a WSGI server:
```
gunicorn app:app
```

## API Endpoints

### Train Model

```
POST /api/train
```

Trains a machine learning model on fishing data.

**Request Body:**
```json
{
  "model": "Random Forest",
  "data": [
    {"lat": 40.7128, "lon": -74.0060, "hour": 14, "illegal": 0},
    {"lat": 34.0522, "lon": -118.2437, "hour": 2, "illegal": 1},
    ...
  ]
}
```

**Response:**
```json
{
  "success": true,
  "model": "Random Forest",
  "accuracy": 0.87,
  "confusionMatrix": [[120, 15], [10, 55]]
}
```

### Predict Illegal Fishing

```
POST /api/predict
```

Makes a prediction about illegal fishing activity.

**Request Body:**
```json
{
  "model": "Random Forest",
  "lat": 40.7128,
  "lon": -74.0060,
  "hour": 14
}
```

**Response:**
```json
{
  "result": false,
  "probability": 0.23,
  "location": [40.7128, -74.0060],
  "hour": 14
}
```

## Integration with Frontend

The frontend automatically uses this API if it's running. If the API is not available, it falls back to the mock implementation for demonstration purposes.

## Data Format

- **lat**: Latitude (-90 to 90)
- **lon**: Longitude (-180 to 180)
- **hour**: Hour of the day (0-23)
- **illegal**: Binary target variable (0 = legal, 1 = illegal) 