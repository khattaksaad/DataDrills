from fastapi import FastAPI
import joblib
import pandas as pd
import numpy as np
from pydantic import BaseModel
import logging

logger = logging.getLogger("FastAPI Iris Prediction")

import os
try:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(BASE_DIR, 'svm_model.pkl')
    model = joblib.load(model_path)
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error("Error loading model: %s", e)
    raise RuntimeError("Model not found") from e


class Iris(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

species_mapping = {0: 'setosa', 1: 'versicolor', 2: 'virginica'}

from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

app = FastAPI()

# Mount the static directory specific to Iris
static_dir = os.path.join(BASE_DIR, "static")
app.mount("/static", StaticFiles(directory=static_dir), name="static")

@app.get("/health")
async def health_check():
    return {"status": "ok"}


@app.get('/')
async def home():
    return FileResponse(os.path.join(static_dir, 'index.html'))

@app.get('/predict')
async def predict_page():
    return FileResponse(os.path.join(static_dir, 'predict.html'))


@app.post('/api/predict')
async def predict(input_data: Iris):
    """
    Predict the species of an Iris flower based on its measurements.
    """
    feature_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
    try:
        input_df = pd.DataFrame([input_data.dict()], columns=feature_names)
        logger.info("Input data: %s", input_df)
        
        prediction = model.predict(input_df)[0]
        species = species_mapping.get(prediction, 'Unknown')
        return {'prediction': species}
    except Exception as e:
        logger.error("Error predicting species: %s", e)
        return JSONResponse(status_code=500, content={"error": "Prediction failed"})    

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='[IP_ADDRESS]', port=8000)