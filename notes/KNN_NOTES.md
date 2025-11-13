# K-NEAREST NEIGHBORS (KNN) - SHORT NOTES

## 1. WHAT IS KNN?

**Definition**: KNN is a **supervised machine learning algorithm** used for **classification** and **regression** that predicts based on the k closest training examples.

**Key Idea**: "You are the average of your k nearest neighbors"

**Type**:
- Lazy learner (no training phase)
- Instance-based learning
- Non-parametric algorithm

---

## 2. HOW KNN WORKS

### Step-by-Step Process:

1. **Choose k** (number of neighbors)
2. **Calculate distance** from test point to all training points
3. **Find k nearest neighbors** (smallest distances)
4. **Classification**: Take majority vote
   **Regression**: Take average of k neighbors
5. **Predict** the class/value

### Visual Example:

```
         Class A (○)        Class B (×)

    ○           ×
       ○     ×
          ?      ×           ? = Test point
       ○     ×
    ○           ×

If k=3, nearest neighbors are: ○, ○, ×
Majority vote: Class A (○) wins!
```

---

## 3. DISTANCE METRICS

### A. Euclidean Distance (Most Common)

```
d = √[(x₂-x₁)² + (y₂-y₁)²]

For n dimensions:
d = √[Σ(xᵢ - yᵢ)²]
```

**Example:**
```
Point A: (1, 2)
Point B: (4, 6)

d = √[(4-1)² + (6-2)²]
  = √[9 + 16]
  = √25
  = 5
```

### B. Manhattan Distance

```
d = |x₂-x₁| + |y₂-y₁|
```

### C. Minkowski Distance (Generalized)

```
d = (Σ|xᵢ - yᵢ|ᵖ)^(1/p)

p=1 → Manhattan
p=2 → Euclidean
```

---

## 4. CHOOSING K VALUE

### Rules:

- **k=1**: Most complex, can overfit, sensitive to noise
- **k=3 to 7**: Often good starting point
- **k too small**: Overfitting, sensitive to outliers
- **k too large**: Underfitting, slow predictions
- **Odd k**: Avoids ties in binary classification

### Finding Optimal k:

```python
# Test different k values
k_values = range(1, 21)
accuracies = []

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    accuracies.append(knn.score(X_test, y_test))

# Plot k vs accuracy
plt.plot(k_values, accuracies)
best_k = k_values[np.argmax(accuracies)]
```

**Graph Pattern:**

```
Accuracy
   ^
   |    *
   |   * *
   |  *   *
   | *     *
   |*       * *
   +-----------> k
   1  3  5  7  9

Usually peaks around k=3-7
```

---

## 5. MAIN FORMULAS

### Classification (Majority Vote)

```
y_pred = mode(y₁, y₂, ..., yₖ)

Example: If 3 neighbors are [1, 1, 0]
y_pred = 1 (majority)
```

### Regression (Average)

```
y_pred = (1/k) × Σyᵢ

Example: If 3 neighbors are [5, 7, 6]
y_pred = (5+7+6)/3 = 6
```

### Distance-Weighted KNN

```
weight = 1 / distance

Weighted vote:
y_pred = Σ(weightᵢ × yᵢ) / Σweightᵢ
```

---

## 6. PYTHON CODE (BASIC)

### Simple KNN Implementation

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Load data
df = pd.read_csv('data.csv')
X = df.drop('target', axis=1)
y = df['target']

# Split data (80-20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# IMPORTANT: Scale features (KNN is distance-based!)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create KNN classifier
knn = KNeighborsClassifier(
    n_neighbors=5,        # k value
    weights='uniform',    # 'uniform' or 'distance'
    metric='euclidean'    # distance metric
)

# Train (actually just stores data)
knn.fit(X_train_scaled, y_train)

# Predict
y_pred = knn.predict(X_test_scaled)

# Evaluate
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
```

---

## 7. KEY PARAMETERS

### KNeighborsClassifier Parameters

```python
KNeighborsClassifier(
    n_neighbors=5,          # k value
    weights='uniform',      # or 'distance' (weight by inverse distance)
    algorithm='auto',       # 'ball_tree', 'kd_tree', 'brute'
    metric='minkowski',     # distance metric
    p=2                     # p=1 (Manhattan), p=2 (Euclidean)
)
```

| Parameter | Values | Default | Purpose |
|-----------|--------|---------|---------|
| n_neighbors | 1, 3, 5, 7... | 5 | Number of neighbors |
| weights | uniform, distance | uniform | All equal or weighted |
| algorithm | auto, brute, kd_tree | auto | Search algorithm |
| metric | euclidean, manhattan | minkowski | Distance measure |
| p | 1, 2, etc. | 2 | Power for Minkowski |

---

## 8. ADVANTAGES OF KNN

✅ **Simple**: Easy to understand and implement
✅ **No Training**: No model to train, just store data
✅ **No Assumptions**: Works with any data distribution
✅ **Multi-class**: Naturally handles multiple classes
✅ **Flexible**: Works for classification and regression
✅ **Adaptable**: Updates easily with new data

---

## 9. DISADVANTAGES OF KNN

❌ **Slow Predictions**: Must calculate distance to all training points
❌ **Memory Intensive**: Stores entire training dataset
❌ **Curse of Dimensionality**: Poor performance with many features
❌ **Sensitive to Scale**: Requires feature normalization
❌ **Sensitive to Outliers**: Noisy data affects predictions
❌ **Choosing k is Tricky**: Optimal k depends on dataset

---

## 10. WHEN TO USE KNN

### ✅ Use KNN When:

- Small to medium dataset (<10,000 samples)
- Low dimensionality (<20 features)
- Non-linear decision boundaries
- Need simple baseline model
- Data is well-scaled
- No time constraint for predictions

### ❌ Avoid KNN When:

- Very large dataset (>100,000 samples)
- High dimensionality (>50 features)
- Real-time predictions needed
- Features on different scales (unless you scale)
- Lots of irrelevant features

---

## 11. FEATURE SCALING IS CRITICAL!

### Why?

```
Without Scaling:
Feature 1: Age (20-80)
Feature 2: Income (20000-200000)

Income dominates distance calculation!
```

### Solution: StandardScaler

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Now all features have mean=0, std=1
```

### Impact:

```
Before:  distance = √[(60-50)² + (150000-100000)²] ≈ 50000
After:   distance = √[(0.5-0.2)² + (0.8-0.3)²] ≈ 0.6

Much better!
```

---

## 12. EVALUATION METRICS

### For Classification:

```python
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

# Detailed Report
print(classification_report(y_test, y_pred))
```

### Metrics Explained:

```
Accuracy = (TP + TN) / Total

Precision = TP / (TP + FP)  # Of predicted positives, how many correct?

Recall = TP / (TP + FN)     # Of actual positives, how many found?

F1-Score = 2 × (Precision × Recall) / (Precision + Recall)
```

---

## 13. COMMON ISSUES & SOLUTIONS

### Issue 1: Poor Accuracy

**Causes:**
- Wrong k value
- Features not scaled
- Too many irrelevant features
- Imbalanced classes

**Solutions:**
```python
# Try different k values
for k in range(1, 20, 2):
    knn = KNeighborsClassifier(n_neighbors=k)
    # ... test accuracy

# Always scale!
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Feature selection
from sklearn.feature_selection import SelectKBest
selector = SelectKBest(k=10)
X_selected = selector.fit_transform(X, y)
```

### Issue 2: Slow Predictions

**Solutions:**
```python
# Use KD-Tree for faster search
knn = KNeighborsClassifier(algorithm='kd_tree')

# Reduce training data
from sklearn.model_selection import StratifiedShuffleSplit
# Use subset of data

# Reduce dimensionality
from sklearn.decomposition import PCA
pca = PCA(n_components=10)
X_reduced = pca.fit_transform(X)
```

### Issue 3: Curse of Dimensionality

**Problem**: With many features, all points seem far apart

**Solutions:**
```python
# PCA for dimensionality reduction
from sklearn.decomposition import PCA
pca = PCA(n_components=0.95)  # Keep 95% variance
X_reduced = pca.fit_transform(X)

# Feature selection
from sklearn.feature_selection import SelectKBest, f_classif
selector = SelectKBest(f_classif, k=15)
X_selected = selector.fit_transform(X, y)
```

---

## 14. COMPLETE WORKFLOW

```python
# 1. Import libraries
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd

# 2. Load and explore data
df = pd.read_csv('data.csv')
print(df.head())
print(df.info())

# 3. Prepare features and target
X = df.drop('target', axis=1)
y = df['target']

# 4. Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 5. Scale features (IMPORTANT!)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 6. Find optimal k
best_k = 1
best_score = 0
for k in range(1, 21, 2):
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X_train_scaled, y_train, cv=5)
    if scores.mean() > best_score:
        best_score = scores.mean()
        best_k = k

print(f"Best k: {best_k}")

# 7. Train final model
knn = KNeighborsClassifier(n_neighbors=best_k)
knn.fit(X_train_scaled, y_train)

# 8. Predict and evaluate
y_pred = knn.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)

print(f"Test Accuracy: {accuracy:.2f}")
print(classification_report(y_test, y_pred))

# 9. Predict new data
new_data = [[5.1, 3.5, 1.4, 0.2]]  # New sample
new_data_scaled = scaler.transform(new_data)
prediction = knn.predict(new_data_scaled)
probability = knn.predict_proba(new_data_scaled)

print(f"Prediction: {prediction}")
print(f"Probabilities: {probability}")
```

---

## 15. KNN VS OTHER ALGORITHMS

| Algorithm | Training | Prediction | Accuracy | Memory |
|-----------|----------|------------|----------|--------|
| **KNN** | Instant | Slow | Good | High |
| Logistic Regression | Fast | Fast | Good | Low |
| Decision Tree | Fast | Fast | Good | Medium |
| Random Forest | Medium | Medium | Better | High |
| SVM | Slow | Fast | Better | Medium |
| Neural Network | Very Slow | Fast | Best | Medium |

---

## 16. PRACTICAL TIPS

### Do's:

✅ **Always scale features** with StandardScaler
✅ **Start with k=3 or k=5** and tune from there
✅ **Use odd k** for binary classification
✅ **Cross-validate** to find optimal k
✅ **Remove irrelevant features** before training
✅ **Handle missing values** and outliers

### Don'ts:

❌ **Don't use without scaling** (will fail!)
❌ **Don't use on very large datasets** (too slow)
❌ **Don't use with high dimensionality** (curse of dimensionality)
❌ **Don't forget train/test split**
❌ **Don't use k=2 or even numbers** (can cause ties)

---

## 17. REAL-WORLD APPLICATIONS

1. **Recommendation Systems**: Netflix, Amazon
2. **Image Recognition**: Digit recognition, face detection
3. **Medical Diagnosis**: Disease prediction
4. **Credit Scoring**: Loan approval
5. **Pattern Recognition**: Handwriting, speech
6. **Anomaly Detection**: Fraud detection
7. **Customer Segmentation**: Marketing
8. **Stock Market Prediction**: Financial forecasting

---

## 18. KEY TAKEAWAYS

1. **KNN is simple but powerful** - Great baseline algorithm
2. **No training phase** - Just stores data
3. **k is crucial** - Usually 3-7 works well
4. **Scaling is mandatory** - Use StandardScaler always
5. **Distance-based** - Euclidean distance most common
6. **Lazy learner** - Slow at prediction time
7. **Memory hungry** - Stores all training data
8. **Works for classification & regression**
9. **Good for small-medium datasets**
10. **Curse of dimensionality** - Struggles with many features

---

## 19. QUICK REFERENCE CODE

### Basic Template:

```python
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

# Scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train)

# Predict
y_pred = knn.predict(X_test_scaled)
```

### With Hyperparameter Tuning:

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_neighbors': [3, 5, 7, 9, 11],
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan']
}

grid_search = GridSearchCV(
    KNeighborsClassifier(),
    param_grid,
    cv=5,
    scoring='accuracy'
)

grid_search.fit(X_train_scaled, y_train)
print(f"Best params: {grid_search.best_params_}")
best_knn = grid_search.best_estimator_
```

---

## 20. SUMMARY TABLE

| Aspect | Details |
|--------|---------|
| **Type** | Supervised, Instance-based |
| **Use Case** | Classification, Regression |
| **Training** | None (lazy learner) |
| **Prediction** | Slow (calculates distances) |
| **Best k** | Usually 3-7 |
| **Scaling** | Mandatory (StandardScaler) |
| **Pros** | Simple, No assumptions, Multi-class |
| **Cons** | Slow, Memory intensive, Sensitive to scale |
| **Complexity** | O(n × d) for each prediction |
| **When to Use** | Small datasets, <20 features |

---

**END OF NOTES**

*For more: sklearn KNN documentation and ML textbooks*
*Practice: Iris, MNIST, Diabetes datasets*
