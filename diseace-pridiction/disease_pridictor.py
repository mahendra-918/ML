import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


df = pd.read_csv('../data/disease_prediction_data.csv')


X = df[['age','blood_pressure','cholesterol','bmi']]
Y = df['disease']

X_train,X_test,y_train,y_test = train_test_split(X,Y,test_size=0.2,random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LogisticRegression(random_state=42,max_iter=1000)
model.fit(X_train_scaled,y_train)

y_train_pred = model.predict(X_train_scaled)
y_test_pred = model.predict(X_test_scaled)


train_accuracy = accuracy_score(y_train,y_train_pred)
print(f"Accuracy: {train_accuracy:.4f} ({train_accuracy*100:.2f}%)")

print(classification_report(y_train, y_train_pred, target_names=['Healthy', 'Disease']))