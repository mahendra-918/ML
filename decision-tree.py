import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv('data/tennis_weather_data.csv')

# le_outlook = LabelEncoder()
# le_temperature = LabelEncoder()
# le_humidity = LabelEncoder()
# le_wind = LabelEncoder()
# le_play = LabelEncoder()

# df['outlook_encoded'] = le_outlook.fit_transform(df['outlook'])
# df['temperature_encoded'] = le_temperature.fit_transform(df['temperature'])
# df['humidity_encoded'] = le_humidity.fit_transform(df['humidity'])
# df['wind_encoded'] = le_wind.fit_transform(df['wind'])
# df['play_tennis_encoded'] = le_play.fit_transform(df['play_tennis'])


label_encoders = {}

for column in ['outlook','temperature','humidity','wind']:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    label_encoders[column] = le


le_target = LabelEncoder()
df['play_tennis'] = le_target.fit_transform(df['play_tennis'])


X = df[['outlook', 'temperature', 'humidity', 'wind']]
Y = df['play_tennis']

X_train,X_test,y_train,y_test = train_test_split(X,Y,test_size=0.30)

classifier = DecisionTreeClassifier(criterion='entropy')
classifier.fit(X_train,y_train)
y_pred = classifier.predict(X_test)

y_train_pred = classifier.predict(X_train)
train_accuracy = accuracy_score(y_train, y_train_pred)

accuracy = accuracy_score(y_test,y_pred)
print(accuracy)

print(f"Training Accuracy: {train_accuracy*100:.2f}%")
print(f"Test Accuracy: {y_test_accuracy*100:.2f}%")
