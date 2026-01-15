import numpy as np 
import pandas as pd
import joblib
loaded_model = joblib.load('linear_regression_model.pkl')
loaded_scaler = joblib.load('scaler.pkl')
sample_data = np.array([[8]])   
sample_data_df = pd.DataFrame(sample_data, columns=['Hours_Studied'])
sample_data_scaled = loaded_scaler.transform(sample_data_df)
sample_prediction = loaded_model.predict(sample_data_scaled)
print(f'Predicted Exam Score for 8 hours studied (using loaded model): {sample_prediction[0]}')