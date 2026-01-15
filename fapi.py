from fastapi import FastAPI
from pydantic import BaseModel  
import joblib
import numpy as np
import pandas as pd     


loaded_model = joblib.load('linear_regression_model.pkl')
loaded_scaler = joblib.load('scaler.pkl')

app=FastAPI()
class ExamScoreRequest(BaseModel):
    hours_studied: float    

@app.get("/")
def home():
    return {"message": "Welcome to the Exam Score Prediction API!"}


@app.post("/predict")
def predict_exam_score(request: ExamScoreRequest):
    hours_studied = np.array([[request.hours_studied]])
    input_df = pd.DataFrame(hours_studied, columns=['Hours_Studied'])
    input_scaled = loaded_scaler.transform(input_df)
    prediction = loaded_model.predict(input_scaled)
    return {"predicted_exam_score": prediction[0]}
