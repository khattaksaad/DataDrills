import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.svm import SVC
import joblib 
import os

# Get directory of current script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

iris = pd.read_csv(os.path.join(BASE_DIR, 'iris.csv'))
print(iris.head())

print('\nSummary of the dataset')
print(iris.describe())

print('Missing values in the dataset')
print(iris.isnull().sum())

label_encoder = LabelEncoder()
iris['species'] = label_encoder.fit_transform(iris['species'])
print(iris.head())

print('\nEncoded Classes:')
print(label_encoder.classes_)

print('\nDataset shape:')
print(iris.shape)

print('\nGrouped Mean by Species:')
print(iris.groupby('species').mean())

sns.scatterplot(x='sepal_length', y='sepal_width', hue='species', data=iris)
plt.title('Sepal Length vs Sepal Width')
#plt.show()


numeric_iris = iris.drop('species', axis=1)

sns.heatmap(numeric_iris.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix of Iris Dataset')
#plt.show()


x = iris.drop('species', axis=1)
y = iris['species']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=42)
print(f'Training set shape: {x_train.shape}')
print(f'Testing set shape: {x_test.shape}')

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

svm = SVC(kernel='linear', random_state=42)
svm.fit(x_train, y_train)

y_pred = svm.predict(x_test)

print('SVM Accuracy Score:', accuracy_score(y_test, y_pred))
print('\nSVM Classification Report:')
print(classification_report(y_test, y_pred))
print('\nSVM Confusion Matrix:')
print(confusion_matrix(y_test, y_pred))

joblib.dump(svm, os.path.join(BASE_DIR, 'svm_model.pkl'))
print('\nModel saved to svm_model.pkl')