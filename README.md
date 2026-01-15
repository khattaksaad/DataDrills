# ML Exam Score Predictor

A machine learning project that predicts exam scores based on hours studied using linear regression, with a FastAPI web API for making predictions.

## Features

- **Model Training**: Generates synthetic data and trains a linear regression model
- **Model Persistence**: Saves trained model and scaler using joblib
- **Prediction API**: FastAPI endpoint for real-time score predictions
- **Data Visualization**: Scatter plot of training data

## Project Structure

- `train_model.py`: Trains the linear regression model on synthetic data
- `load_model.py`: Loads the trained model and demonstrates prediction
- `fapi.py`: FastAPI application with prediction endpoint
- `linear_regression_model.pkl`: Saved trained model
- `scaler.pkl`: Saved data scaler

## Installation

1. Clone the repository:
```bash
git clone https://github.com/khattaksaad/ml_pr.git
cd ml_pr
```

2. Create and activate virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install fastapi pydantic joblib numpy pandas scikit-learn matplotlib uvicorn
```

## Usage

### Training the Model

Run the training script to generate data and train the model:

```bash
python train_model.py
```

This will:
- Generate synthetic data (1000 samples)
- Train a linear regression model
- Save the model and scaler to disk
- Display evaluation metrics

### Running the API

Start the FastAPI server:

```bash
uvicorn fapi:app --reload
```

The API will be available at `http://127.0.0.1:8000`

### API Endpoints

- `GET /`: Welcome message
- `POST /predict`: Predict exam score
  - Request body: `{"hours_studied": float}`
  - Response: `{"predicted_exam_score": float}`

### Example API Usage

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"hours_studied": 6.5}'
```

### Testing the Model

Run the load model script to test predictions:

```bash
python load_model.py
```

## Dependencies

- fastapi: Web framework for API
- pydantic: Data validation
- joblib: Model serialization
- numpy: Numerical computing
- pandas: Data manipulation
- scikit-learn: Machine learning algorithms
- matplotlib: Data visualization
- uvicorn: ASGI server

## Model Details

- **Algorithm**: Linear Regression
- **Features**: Hours studied
- **Target**: Exam score
- **Preprocessing**: Standard scaling
- **Evaluation**: MSE and R² score