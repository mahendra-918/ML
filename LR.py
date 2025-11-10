import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

weeks_data = [1,2,3,4,5]
sales_data = [2,4,5,4,5]

pf = pd.DataFrame({"weeks":weeks_data,"sales":sales_data})

X = pf["weeks"].values
Y = pf["sales"].values

mean_x = np.mean(X)
mean_y = np.mean(Y)
n = len(X)
numer = 0
denom = 0

for i in range(n):
       numer += (X[i] - mean_x)*(Y[i]-mean_y)
       denom += (X[i] - mean_x)**2

m = numer/denom
c = mean_y - (m*mean_x)

max_x = np.max(X) + 1
min_x = np.min(X) - 1
x = np.linspace(min_x, max_x)
y = c + m*x

# print(len(x))

plt.plot(x,y,color="#8c250f",label="Regression label",linestyle="--")
plt.scatter(X,Y,color="#f45b69",label="Data points")
plt.xlabel("Weeks")
plt.ylabel("Sales")
plt.title("Weeks vs Sales")
plt.grid(True,alpha=0.5)
plt.legend(loc='best')

X = X.reshape((n,1))
reg = LinearRegression()

reg = reg.fit(X,Y)
print(reg.score(X,Y))


# plt.show()