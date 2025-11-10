import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

data = pd.read_csv("head_brain_data.csv")


X = data['head_size'].values
Y = data['brain_weight'].values

X = X.reshape(-1,1)


X_train,X_test,y_train,y_test = train_test_split(X,Y,test_size=0.20,random_state=42)


lr = LinearRegression()
lr = lr.fit(X_train,y_train)

y_pred = lr.predict(X_test)

# Calculate metrics
R2 = metrics.r2_score(y_test, y_pred)
# mae = metrics.mean_absolute_error(y_test, y_pred)
# mse = metrics.mean_squared_error(y_test, y_pred)
# rmse = np.sqrt(mse)

print(R2)

plt.scatter(X_test,y_test,label='actual data',color="black")
plt.plot(X_test,y_pred,color='red',label='Prediction')
plt.grid(True)
plt.legend()

plt.show()