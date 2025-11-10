import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Load data
data = pd.read_csv('head_brain_data.csv')

X = data['head_size'].values.reshape(-1, 1)
Y = data['brain_weight'].values

# Train model
model = LinearRegression()
model.fit(X, Y)
y_pred = model.predict(X)

# Create visualization
plt.figure(figsize=(10, 6))
plt.scatter(X, Y, alpha=0.5, color='blue', s=50, label='Actual Data')
plt.plot(X, y_pred, color='red', linewidth=2, label='Regression Line')

plt.xlabel('Head Size (cm³)', fontsize=12)
plt.ylabel('Brain Weight (grams)', fontsize=12)
plt.title('Head Size vs Brain Weight - Showing Weak Correlation', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)

# Add text annotation
r2 = model.score(X, Y)
plt.text(0.05, 0.95, f'R² = {r2:.4f}\nCorrelation = {np.sqrt(r2):.4f}',
         transform=plt.gca().transAxes, fontsize=12,
         verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig('head_brain_scatter.png', dpi=150)
print("Plot saved as 'head_brain_scatter.png'")
plt.show()
