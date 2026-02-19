import numpy as np 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import joblib
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

np.random.seed(42)
n_samples = 1000

X1 = np.random.uniform(0, 10, n_samples)
noise = np.random.normal(0, 5, n_samples)
y = 10 * X1 + noise

data = pd.DataFrame({'Hours_Studied': X1, 'Exam_Score': y})
print(data.head())


plt.scatter(data['Hours_Studied'], data['Exam_Score'])
plt.xlabel('Hours Studied')
plt.ylabel('Exam Score')
plt.title('Hours Studied vs Exam Score')
plt.show()  

X = data[['Hours_Studied']]
y = data['Exam_Score']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)   

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)    

model = LinearRegression()
model.fit(X_train_scaled, y_train)

print(f'Model Coefficient: {model.coef_}')
print(f'Model Intercept: {model.intercept_}')

new_data = np.array([[6]])
new_data_df = pd.DataFrame(new_data, columns=['Hours_Studied'])
new_data_scaled = scaler.transform(new_data_df)

s_prediction = model.predict(new_data_scaled)
print(f'Predicted Exam Score for 6 hours studied: {s_prediction[0]}')
y_pred = model.predict(X_test_scaled)
mean_squared_error_value = mean_squared_error(y_test, y_pred)
r2_score_value = r2_score(y_test, y_pred)
print(f'Mean Squared Error on Test Set: {mean_squared_error_value}')
print(f'R^2 Score on Test Set: {r2_score_value}')
joblib.dump(model, 'linear_regression_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
plt.scatter(y_test, y_pred)
plt.xlabel('Actual Exam Scores')
plt.ylabel('Predicted Exam Scores')
plt.title('Actual vs Predicted Exam Scores')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.show()      


