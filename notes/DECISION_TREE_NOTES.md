# DECISION TREE - SHORT NOTES

## 1. WHAT IS DECISION TREE?

**Definition**: A **supervised machine learning algorithm** that makes decisions by asking a series of questions, splitting data into branches like a tree.

**Key Idea**: "Ask questions to split data until you get pure groups"

**Type**:
- Supervised learning
- Works for classification and regression
- Non-parametric
- White-box model (interpretable)

**Visual Structure:**
```
                [Root Node]
                 outlook?
               /    |    \
           Sunny  Over  Rain
             /     cast    \
          [Split] [Leaf]  [Split]
         humidity?  YES   wind?
         /     \          /    \
       High  Normal   Weak  Strong
        /       \      /        \
      NO       YES   YES        NO
     [Leaf]  [Leaf] [Leaf]    [Leaf]
```

---

## 2. HOW DECISION TREE WORKS

### Building Process:

1. **Start at root** with all data
2. **Choose best feature** to split (using entropy/gini)
3. **Split data** based on feature values
4. **Repeat** for each subset (recursive)
5. **Stop** when:
   - All samples in node have same class (pure)
   - Maximum depth reached
   - Minimum samples reached
   - No more features to split

### Decision Process (Prediction):

```
Start at Root
    ↓
Is outlook Sunny?
    ↓ Yes
Is humidity High?
    ↓ No
Predict: Play Tennis = YES
```

---

## 3. KEY CONCEPTS

### A. Entropy (Impurity Measure)

**Formula:**
```
H(S) = -Σ pᵢ × log₂(pᵢ)

where:
- pᵢ = proportion of class i
- H(S) = entropy of dataset S
```

**Range:** 0 (pure) to 1 (max impurity for binary)

**Example:**
```
Dataset: [Yes, Yes, Yes, Yes, No, No]
p(Yes) = 4/6 = 0.67
p(No) = 2/6 = 0.33

H = -(0.67×log₂(0.67) + 0.33×log₂(0.33))
H = -(0.67×(-0.585) + 0.33×(-1.585))
H = 0.918
```

### B. Information Gain

**Formula:**
```
IG = H(parent) - Σ[(nᵢ/n) × H(childᵢ)]

where:
- IG = Information Gain
- H(parent) = entropy before split
- nᵢ = samples in child i
- n = total samples
```

**Goal:** Maximize information gain (choose feature with highest IG)

**Example:**
```
Parent entropy = 0.94
After split on "outlook":
  Sunny: 5 samples, entropy = 0.97
  Overcast: 4 samples, entropy = 0
  Rain: 5 samples, entropy = 0.97

IG = 0.94 - [(5/14)×0.97 + (4/14)×0 + (5/14)×0.97]
IG = 0.94 - 0.69
IG = 0.25
```

### C. Gini Impurity (Alternative to Entropy)

**Formula:**
```
Gini = 1 - Σ pᵢ²

where pᵢ = proportion of class i
```

**Range:** 0 (pure) to 0.5 (max impurity for binary)

**Example:**
```
Dataset: [Yes, Yes, Yes, No]
p(Yes) = 3/4 = 0.75
p(No) = 1/4 = 0.25

Gini = 1 - (0.75² + 0.25²)
Gini = 1 - (0.5625 + 0.0625)
Gini = 0.375
```

---

## 4. ENTROPY VS GINI

| Metric | Range | Calculation | Speed | Use Case |
|--------|-------|-------------|-------|----------|
| **Entropy** | 0 to 1 | Logarithmic | Slower | Better splits, interpretable |
| **Gini** | 0 to 0.5 | Polynomial | Faster | Default in sklearn, faster |

**When to use:**
- **Entropy**: When you want to understand splits (educational)
- **Gini**: When you want speed (production)

**Result**: Usually similar performance

---

## 5. MAIN FORMULAS SUMMARY

### Classification Tree:

```
1. Entropy:
   H(S) = -Σ pᵢ × log₂(pᵢ)

2. Information Gain:
   IG = H(parent) - Weighted_Average(H(children))

3. Gini Impurity:
   Gini = 1 - Σ pᵢ²

4. Split Criterion:
   Choose feature with maximum IG (or minimum Gini)
```

### Regression Tree:

```
1. MSE (Mean Squared Error):
   MSE = (1/n) × Σ(yᵢ - ŷ)²

2. Split Criterion:
   Minimize MSE after split
```

---

## 6. PYTHON CODE (BASIC)

### Simple Decision Tree

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

# Load data
df = pd.read_csv('data.csv')

# Encode categorical features
label_encoders = {}
for column in ['outlook', 'temperature', 'humidity', 'wind']:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    label_encoders[column] = le

# Encode target
le_target = LabelEncoder()
df['play_tennis'] = le_target.fit_transform(df['play_tennis'])

# Features and target
X = df[['outlook', 'temperature', 'humidity', 'wind']]
y = df['play_tennis']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create classifier
classifier = DecisionTreeClassifier(
    criterion='entropy',      # or 'gini'
    max_depth=5,             # Prevent overfitting
    min_samples_split=2,
    min_samples_leaf=1,
    random_state=42
)

# Train
classifier.fit(X_train, y_train)

# Predict
y_pred = classifier.predict(X_test)

# Evaluate
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy*100:.2f}%")
print(classification_report(y_test, y_pred))
```

---

## 7. KEY PARAMETERS

### DecisionTreeClassifier Parameters

```python
DecisionTreeClassifier(
    criterion='gini',           # 'gini' or 'entropy'
    max_depth=None,            # Maximum tree depth
    min_samples_split=2,       # Min samples to split node
    min_samples_leaf=1,        # Min samples in leaf
    max_features=None,         # Features to consider for split
    random_state=None          # Reproducibility
)
```

| Parameter | Values | Default | Purpose |
|-----------|--------|---------|---------|
| **criterion** | gini, entropy | gini | Split quality measure |
| **max_depth** | int or None | None | Max tree depth |
| **min_samples_split** | int | 2 | Min samples to split |
| **min_samples_leaf** | int | 1 | Min samples in leaf |
| **max_features** | int, float, None | None | Features per split |

---

## 8. HYPERPARAMETERS FOR OVERFITTING CONTROL

### Preventing Overfitting:

```python
# Restrictive parameters (prevents overfitting)
DecisionTreeClassifier(
    max_depth=5,              # Limit depth
    min_samples_split=10,     # Need more samples to split
    min_samples_leaf=5,       # Leaves must have 5+ samples
    max_features='sqrt'       # Use subset of features
)
```

### Impact:

| Parameter ↑ | Tree Complexity | Overfitting Risk |
|-------------|-----------------|------------------|
| max_depth ↑ | More complex | Higher |
| min_samples_split ↑ | Less complex | Lower |
| min_samples_leaf ↑ | Less complex | Lower |

---

## 9. ADVANTAGES OF DECISION TREES

✅ **Easy to Understand**: Visual, interpretable
✅ **No Feature Scaling**: Works with raw data (no StandardScaler needed!)
✅ **Handles Both**: Numerical and categorical data
✅ **Non-linear Relationships**: Captures complex patterns
✅ **Feature Importance**: Shows which features matter
✅ **Multi-output**: Can predict multiple targets
✅ **Fast Predictions**: O(log n) traversal
✅ **White Box**: Can see decision logic

---

## 10. DISADVANTAGES OF DECISION TREES

❌ **Overfitting**: Easily memorizes training data
❌ **Unstable**: Small data changes = different tree
❌ **Biased**: Favors features with more levels
❌ **Not Optimal**: Greedy algorithm (local optimum)
❌ **Poor with Linear Data**: Works better with non-linear
❌ **High Variance**: Sensitive to training data

---

## 11. WHEN TO USE DECISION TREES

### ✅ Use Decision Trees When:

- Need interpretable model (explain decisions)
- Data has categorical features
- Mixed data types (numerical + categorical)
- Non-linear relationships
- Don't want to scale features
- Need quick baseline model
- Feature importance analysis needed

### ❌ Avoid Decision Trees When:

- Need high accuracy (use Random Forest/XGBoost instead)
- Data is perfectly linear
- Very high-dimensional data
- Need stable model (small data changes)
- Can't afford overfitting

---

## 12. FEATURE SCALING NOT NEEDED!

### Unlike KNN/Logistic Regression:

```python
# Decision Tree: NO scaling needed ✅
X = df[['age', 'salary', 'years_experience']]
classifier = DecisionTreeClassifier()
classifier.fit(X, y)  # Works fine!

# KNN: MUST scale ❌ won't work without scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
knn = KNeighborsClassifier()
knn.fit(X_scaled, y)
```

**Why?** Decision trees use splitting thresholds, not distances!

---

## 13. HANDLING CATEGORICAL DATA

### Method 1: LabelEncoder (For Tree)

```python
from sklearn.preprocessing import LabelEncoder

# Encode each column
le_outlook = LabelEncoder()
df['outlook'] = le_outlook.fit_transform(df['outlook'])
# Sunny=2, Overcast=0, Rain=1

# Decision tree works fine with this!
```

### Method 2: One-Hot Encoding (For Other Models)

```python
# NOT needed for decision trees, but for reference:
df_encoded = pd.get_dummies(df, columns=['outlook'])
# Creates: outlook_Sunny, outlook_Overcast, outlook_Rain
```

**For Decision Trees: LabelEncoder is enough!**

---

## 14. VISUALIZING DECISION TREE

### Method 1: Text Representation

```python
from sklearn.tree import export_text

tree_rules = export_text(classifier,
                         feature_names=['outlook', 'temperature', 'humidity', 'wind'])
print(tree_rules)
```

**Output:**
```
|--- outlook <= 1.50
|   |--- humidity <= 0.50
|   |   |--- class: 1
|   |--- humidity >  0.50
|   |   |--- class: 0
|--- outlook >  1.50
|   |--- wind <= 0.50
|   |   |--- class: 1
```

### Method 2: Visual Tree

```python
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

plt.figure(figsize=(20, 10))
plot_tree(classifier,
          feature_names=['outlook', 'temperature', 'humidity', 'wind'],
          class_names=['No', 'Yes'],
          filled=True,
          rounded=True)
plt.savefig('decision_tree.png', dpi=150, bbox_inches='tight')
plt.show()
```

### Method 3: Graphviz (Best Quality)

```python
from sklearn.tree import export_graphviz
import graphviz

dot_data = export_graphviz(classifier,
                           out_file=None,
                           feature_names=['outlook', 'temperature', 'humidity', 'wind'],
                           class_names=['No', 'Yes'],
                           filled=True,
                           rounded=True)

graph = graphviz.Source(dot_data)
graph.render('decision_tree')  # Saves as PDF
```

---

## 15. FEATURE IMPORTANCE

### Get Feature Importance:

```python
# Train model
classifier.fit(X_train, y_train)

# Get importances
importances = classifier.feature_importances_
features = ['outlook', 'temperature', 'humidity', 'wind']

# Print
for feature, importance in zip(features, importances):
    print(f"{feature:15} : {importance:.4f}")

# Plot
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.bar(features, importances)
plt.xlabel('Features')
plt.ylabel('Importance')
plt.title('Feature Importance')
plt.show()
```

**Output Example:**
```
outlook         : 0.4523
humidity        : 0.3214
temperature     : 0.1523
wind            : 0.0740
```

**Interpretation:** Outlook is most important for decision

---

## 16. OVERFITTING DETECTION & PREVENTION

### Detection:

```python
# Train predictions
y_train_pred = classifier.predict(X_train)
train_accuracy = accuracy_score(y_train, y_train_pred)

# Test predictions
y_test_pred = classifier.predict(X_test)
test_accuracy = accuracy_score(y_test, y_test_pred)

print(f"Training Accuracy: {train_accuracy*100:.2f}%")
print(f"Test Accuracy: {test_accuracy*100:.2f}%")

# If train >> test → OVERFITTING!
if train_accuracy - test_accuracy > 0.1:
    print("⚠️  Model is OVERFITTING!")
```

### Prevention Methods:

#### 1. Limit Tree Depth
```python
classifier = DecisionTreeClassifier(max_depth=5)
```

#### 2. Increase Min Samples
```python
classifier = DecisionTreeClassifier(
    min_samples_split=20,    # Need 20 samples to split
    min_samples_leaf=10      # Leaves must have 10+ samples
)
```

#### 3. Pruning (Post-Pruning)
```python
# Train full tree
classifier = DecisionTreeClassifier()
classifier.fit(X_train, y_train)

# Use ccp_alpha for pruning
path = classifier.cost_complexity_pruning_path(X_train, y_train)
ccp_alphas = path.ccp_alphas

# Try different alphas
for alpha in ccp_alphas:
    clf = DecisionTreeClassifier(ccp_alpha=alpha)
    clf.fit(X_train, y_train)
    # Evaluate...
```

#### 4. Use Random Forest Instead
```python
from sklearn.ensemble import RandomForestClassifier

# More robust, less overfitting
rf = RandomForestClassifier(n_estimators=100)
rf.fit(X_train, y_train)
```

---

## 17. CROSS-VALIDATION

### Check Generalization:

```python
from sklearn.model_selection import cross_val_score

classifier = DecisionTreeClassifier(max_depth=5)

# 5-fold cross-validation
cv_scores = cross_val_score(classifier, X, y, cv=5)

print(f"CV Scores: {cv_scores}")
print(f"Mean: {cv_scores.mean():.2f}")
print(f"Std: {cv_scores.std():.2f}")
```

**Good Model:**
```
CV Scores: [0.85, 0.87, 0.86, 0.88, 0.85]
Mean: 0.86 (±0.01)
```

**Overfitting Model:**
```
CV Scores: [0.65, 0.72, 0.58, 0.81, 0.63]
Mean: 0.68 (±0.09)  ← High variance!
```

---

## 18. COMPLETE WORKFLOW

```python
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

# 1. Load data
df = pd.read_csv('tennis_weather_data.csv')

# 2. Encode categorical features
label_encoders = {}
for column in ['outlook', 'temperature', 'humidity', 'wind']:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    label_encoders[column] = le

# Encode target
le_target = LabelEncoder()
df['play_tennis'] = le_target.fit_transform(df['play_tennis'])

# 3. Prepare features and target
X = df[['outlook', 'temperature', 'humidity', 'wind']]
y = df['play_tennis']

# 4. Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 5. Train with overfitting prevention
classifier = DecisionTreeClassifier(
    criterion='entropy',
    max_depth=5,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42
)
classifier.fit(X_train, y_train)

# 6. Check for overfitting
y_train_pred = classifier.predict(X_train)
y_test_pred = classifier.predict(X_test)

train_acc = accuracy_score(y_train, y_train_pred)
test_acc = accuracy_score(y_test, y_test_pred)

print("="*50)
print("PERFORMANCE")
print("="*50)
print(f"Training Accuracy: {train_acc*100:.2f}%")
print(f"Test Accuracy: {test_acc*100:.2f}%")
print(f"Difference: {(train_acc - test_acc)*100:.2f}%")

if train_acc - test_acc > 0.1:
    print("\n⚠️  OVERFITTING DETECTED!")
else:
    print("\n✓ Good generalization")

# 7. Cross-validation
cv_scores = cross_val_score(classifier, X, y, cv=5)
print(f"\nCV Scores: {cv_scores}")
print(f"Mean CV: {cv_scores.mean():.2f} (±{cv_scores.std():.2f})")

# 8. Feature importance
print("\n" + "="*50)
print("FEATURE IMPORTANCE")
print("="*50)
features = ['outlook', 'temperature', 'humidity', 'wind']
for feature, importance in zip(features, classifier.feature_importances_):
    print(f"{feature:15} : {importance:.4f}")

# 9. Visualize tree
plt.figure(figsize=(20, 10))
plot_tree(classifier,
          feature_names=features,
          class_names=['No', 'Yes'],
          filled=True,
          rounded=True)
plt.savefig('tree.png', dpi=150, bbox_inches='tight')
print("\n✓ Tree visualization saved to tree.png")

# 10. Classification report
print("\n" + "="*50)
print("CLASSIFICATION REPORT")
print("="*50)
print(classification_report(y_test, y_test_pred,
                           target_names=['No', 'Yes']))
```

---

## 19. DECISION TREE VS OTHER ALGORITHMS

| Algorithm | Interpretability | Accuracy | Speed | Overfitting | Scaling Needed |
|-----------|------------------|----------|-------|-------------|----------------|
| **Decision Tree** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ | High | ❌ No |
| Random Forest | ⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | Low | ❌ No |
| Logistic Regression | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | Low | ✅ Yes |
| KNN | ⭐⭐ | ⭐⭐⭐ | ⭐⭐ | Low | ✅ Yes |
| SVM | ⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | Medium | ✅ Yes |
| Neural Network | ⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐ | Medium | ✅ Yes |

---

## 20. PRACTICAL TIPS

### Do's:

✅ **Use for interpretable models** - Easy to explain
✅ **Handle categorical data** with LabelEncoder
✅ **Set max_depth** to prevent overfitting (3-10)
✅ **Check train vs test accuracy** - Detect overfitting
✅ **Use cross-validation** for robust evaluation
✅ **Check feature importance** - Understand decisions
✅ **Visualize the tree** - See decision logic
✅ **For production: Use Random Forest** - More robust

### Don'ts:

❌ **Don't use default settings** - Will overfit!
❌ **Don't expect perfect accuracy** - Single trees are weak
❌ **Don't ignore overfitting** - Always check train vs test
❌ **Don't scale features** - Not needed (waste of time)
❌ **Don't use for perfectly linear data** - Use linear models
❌ **Don't rely on single tree** - Use ensemble (Random Forest)

---

## 21. REAL-WORLD APPLICATIONS

1. **Medical Diagnosis**: Disease prediction based on symptoms
2. **Credit Scoring**: Loan approval decisions
3. **Customer Segmentation**: Marketing strategies
4. **Fraud Detection**: Suspicious transaction detection
5. **Recommendation Systems**: Product suggestions
6. **Manufacturing**: Quality control, defect detection
7. **HR**: Employee attrition prediction
8. **Finance**: Stock market analysis

---

## 22. KEY TAKEAWAYS

1. **Visual & Interpretable** - Can see decision logic
2. **No scaling needed** - Works with raw data
3. **Handles mixed data** - Numerical + categorical
4. **Entropy or Gini** - Both work, Gini is faster
5. **Prone to overfitting** - Use hyperparameters to control
6. **Check train vs test** - Detect overfitting early
7. **Feature importance** - Know what matters
8. **Single tree is weak** - Use Random Forest for better accuracy
9. **Fast predictions** - O(log n) tree traversal
10. **Greedy algorithm** - Not globally optimal

---

## 23. ENTROPY CALCULATION EXAMPLE

### Step-by-Step:

**Dataset:** 9 Yes, 5 No (14 total)

**Step 1: Calculate proportions**
```
p(Yes) = 9/14 = 0.643
p(No) = 5/14 = 0.357
```

**Step 2: Calculate log values**
```
log₂(0.643) = -0.637
log₂(0.357) = -1.486
```

**Step 3: Apply entropy formula**
```
H = -(p(Yes) × log₂(p(Yes)) + p(No) × log₂(p(No)))
H = -(0.643 × (-0.637) + 0.357 × (-1.486))
H = -(−0.410 − 0.530)
H = 0.940
```

**Interpretation:** Entropy = 0.940 (quite impure, close to 1)

---

## 24. INFORMATION GAIN CALCULATION EXAMPLE

**Parent Node:** 14 samples (9 Yes, 5 No)
```
H(parent) = 0.940
```

**Split on "Outlook":**

**Sunny:** 5 samples (2 Yes, 3 No)
```
H(Sunny) = -(2/5 × log₂(2/5) + 3/5 × log₂(3/5))
H(Sunny) = 0.971
```

**Overcast:** 4 samples (4 Yes, 0 No)
```
H(Overcast) = -(1 × log₂(1) + 0 × log₂(0))
H(Overcast) = 0  (pure!)
```

**Rain:** 5 samples (3 Yes, 2 No)
```
H(Rain) = -(3/5 × log₂(3/5) + 2/5 × log₂(2/5))
H(Rain) = 0.971
```

**Information Gain:**
```
IG = H(parent) - Σ[(nᵢ/n) × H(childᵢ)]
IG = 0.940 - [(5/14)×0.971 + (4/14)×0 + (5/14)×0.971]
IG = 0.940 - [0.347 + 0 + 0.347]
IG = 0.940 - 0.694
IG = 0.246
```

**Interpretation:** Splitting on "Outlook" reduces entropy by 0.246

---

## 25. QUICK REFERENCE CODE

### Basic Template:

```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder

# Encode categorical data
le = LabelEncoder()
df['feature'] = le.fit_transform(df['feature'])

# Train
classifier = DecisionTreeClassifier(criterion='entropy', max_depth=5)
classifier.fit(X_train, y_train)

# Predict
y_pred = classifier.predict(X_test)
```

### With Grid Search:

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'max_depth': [3, 5, 7, 10],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'criterion': ['gini', 'entropy']
}

grid_search = GridSearchCV(
    DecisionTreeClassifier(random_state=42),
    param_grid,
    cv=5,
    scoring='accuracy'
)

grid_search.fit(X_train, y_train)
print(f"Best params: {grid_search.best_params_}")
best_tree = grid_search.best_estimator_
```

---

## 26. SUMMARY TABLE

| Aspect | Details |
|--------|---------|
| **Type** | Supervised, Tree-based |
| **Use Case** | Classification, Regression |
| **Training** | Fast (greedy algorithm) |
| **Prediction** | Very fast (O(log n)) |
| **Interpretability** | Excellent (white-box) |
| **Scaling** | NOT needed |
| **Handles Categorical** | Yes (with encoding) |
| **Pros** | Interpretable, Fast, No scaling |
| **Cons** | Overfitting, Unstable, Biased |
| **Best Parameter** | max_depth=3-10 |
| **Criterion** | Gini (fast) or Entropy (interpretable) |
| **Overfitting Control** | max_depth, min_samples_split/leaf |

---

**END OF NOTES**

*For more: sklearn Decision Tree documentation*
*Practice: Tennis, Iris, Titanic datasets*
*Next: Learn Random Forest (ensemble of Decision Trees)*
