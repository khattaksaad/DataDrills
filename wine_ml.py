import matplotlib.quiver
import sklearn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV, train_test_split, cross_val_score
import warnings
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression 
import joblib 
from sklearn.metrics import classification_report

data = pd.read_csv('winequality-red.csv', sep=',')
print(data.head())


data.info()
#data.describe().T.style.background_gradient(axis=0)

data.isnull().sum()

data = data.replace({'quality' : {8: 'Good', 7: 'Good', 6: 'Average', 5: 'Average', 4: 'Bad', 3: 'Bad', 2: 'Bad'}})
data.head()

X = data.drop('quality', axis=1)
y = data['quality']
scaler = MinMaxScaler(feature_range=(0, 1))
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

models = {
    'RandomForestClassifier': RandomForestClassifier(),
    'SVC': SVC(),
    'DecisionTreeClassifier': DecisionTreeClassifier(),
    'GaussianNB': GaussianNB(),
    'KNeighborsClassifier': KNeighborsClassifier(),
    'LogisticRegression': LogisticRegression(),
}

results = {}
for mdle_name, model in models.items():
    print(f"Training {mdle_name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    results[mdle_name] = accuracy
    print(f"{mdle_name} accuracy: {accuracy}")

    
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
}

grid_search = GridSearchCV(RandomForestClassifier(), param_grid, cv=5, verbose=1)
grid_search.fit(X_train, y_train)
print(f"Best parameters: {grid_search.best_params_}")
print(f"Best accuracy: {grid_search.best_score_}")


best_model = grid_search.best_estimator_
cv_scores = cross_val_score(best_model, X, y, cv=5)
print(f"Cross-validation scores: {cv_scores}")
print(f"Cross-validation accuracy: {cv_scores.mean()}")

y_pred = best_model.predict(X_test)
print(f"\nClassification report:\n {classification_report(y_test, y_pred, zero_division=1)}")


joblib.dump(best_model, 'best_model.pkl')
print("Model saved to best_model.pkl")

joblib.dump(scaler, 'scaler.pkl')
print("MinMaxScaler saved to scaler.pkl")