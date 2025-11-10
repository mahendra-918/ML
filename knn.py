
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt

df = pd.read_csv("data/diabetes.csv")

X = df[['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','Pedigree','Age']]

Y = df['Outcome']

X_train,X_test,y_train,y_test = train_test_split( X, Y, test_size=0.2, random_state=42, stratify=Y)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

k = 5
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(X_train_scaled,y_train)

# scores = []
# for i in range(1,21):
#     knn = KNeighborsClassifier(n_neighbors=i)
#     knn.fit(X_train,y_train)
#     y_pred = knn.predict(X_test)
#     scores.append(accuracy_score(y_test, y_pred))

# plt.plot(range(1,21),scores)
y_train_pred = knn.predict(X_train_scaled)
y_test_pred = knn.predict(X_test_scaled)

train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)


print(train_accuracy)
print(test_accuracy)