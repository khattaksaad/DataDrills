from fastapi import FastAPI
import joblib
import pandas as pd
import numpy as np
from pydantic import BaseModel

app = FastAPI()

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, 'svm_model.pkl')
model = joblib.load(model_path)

class Iris(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

species_mapping = {0: 'setosa', 1: 'versicolor', 2: 'virginica'}

from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

app = FastAPI()

# Mount the static directory specific to Iris
static_dir = os.path.join(BASE_DIR, "static")
app.mount("/static", StaticFiles(directory=static_dir), name="static")


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
    input_df = pd.DataFrame([input_data.dict()], columns=feature_names)
    prediction = model.predict(input_df)[0]
    species = species_mapping.get(prediction, 'Unknown')
    return {'prediction': species}

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='[IP_ADDRESS]', port=8000)