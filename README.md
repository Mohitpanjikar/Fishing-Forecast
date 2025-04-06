# Fishing Forecast Guardian

An AI-powered application to predict and help prevent illegal fishing activities based on location and time data.

## Features

- Multiple ML model support (Random Forest, SVM, Logistic Regression, Decision Tree, KNN, Neural Network)
- Real-time model training via API
- Prediction endpoint for illegal fishing likelihood
- Performance metrics tracking (accuracy, confusion matrix)
- Cross-origin resource sharing enabled

## Data Visualization

![Data Visualization](docs/images/data_visualization.png)

*Heatmap showing illegal fishing hotspots based on geospatial data.*

## Model Performance

![Model Comparison](docs/images/model_comparison.png)

*Comparison of different machine learning models for illegal fishing prediction.*

## Prediction Results

![Prediction Results](docs/images/prediction_results.png)

*Sample prediction results for illegal fishing likelihood across various locations and times.*

## Tech Stack

- **Backend**: Flask, Python
- **ML Libraries**: scikit-learn, pandas, numpy, joblib
- **Frontend**: [Your frontend technology here]

## API Endpoints

### Train a Model
```
POST /api/train
```
Request body:
```json
{
  "model": "Random Forest",
  "data": [
    {"lat": 12.345, "lon": 67.890, "hour": 14, "illegal": 1},
    {"lat": 11.345, "lon": 68.890, "hour": 8, "illegal": 0},
    ...
  ]
}
```

### Make a Prediction
```
POST /api/predict
```
Request body:
```json
{
  "model": "Random Forest",
  "lat": 12.345,
  "lon": 67.890,
  "hour": 14
}
```

### Health Check
```
GET /api/health
```

## Setup and Installation

1. Clone the repository
   ```
   git clone https://github.com/yourusername/fishing-forecast-guardian.git
   cd fishing-forecast-guardian
   ```

2. Create a virtual environment
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies
   ```
   pip install -r requirements.txt
   ```

4. Run the API server
   ```
   python api/app.py
   ```

## Development

[Include development guidelines here]

## License

[Specify your license]

## Contributing

[Include contribution guidelines]
