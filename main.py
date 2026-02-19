from fastapi import HTTPException
import annotated_types
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd
import logging
from dotenv import load_dotenv
import os

from fastapi.responses import FileResponse

load_dotenv()
model = None
scaler = None


from fastapi.staticfiles import StaticFiles

app = FastAPI()

# app.mount("/", StaticFiles(directory="static", html=True), name="static") 


@app.on_event("startup")
async def load_model():
    global model, scaler
    try:
        model_path = os.getenv("MODEL_PATH", "best_model.pkl")
        scaler_path = os.getenv("SCALER_PATH", "scaler.pkl")
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        logging.info("Model loaded successfully")
    except Exception as e:
        logging.error(f"Error loading model: {e}")
        raise HTTPException(status_code=500, detail="Error loading model")


class Wine(BaseModel):
    fixed_acidity: float
    volatile_acidity: float
    citric_acid: float
    residual_sugar: float
    chlorides: float
    free_sulfur_dioxide: float
    total_sulfur_dioxide: float
    density: float
    pH: float
    sulphates: float
    alcohol: float
@app.get("/")
def home():
    return FileResponse('static/index.html')

@app.get("/predict")
async def predict_page():
    return FileResponse('static/predict.html')

    
@app.post('/predict')
def predict(wine: Wine):
    if model is None or scaler is None:
        logging.error("Model is not loaded")
        raise HTTPException(status_code=500, detail="Model is not loaded, please try again later")

    features = np.array([[wine.fixed_acidity, wine.volatile_acidity, 
    wine.citric_acid, wine.residual_sugar, 
    wine.chlorides, wine.free_sulfur_dioxide, 
    wine.total_sulfur_dioxide, wine.density, 
    wine.pH, wine.sulphates, wine.alcohol]])
    scaled_features = scaler.transform(features)
    prediction = model.predict(scaled_features)
    return {'predicted_quality': str(prediction[0])}