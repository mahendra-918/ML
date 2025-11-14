import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import os

np.random.seed(42)

iris = datasets.load_iris()
X = iris.data
y = iris.target

df = pd.DataFrame(X,columns=iris.feature_names)
df['species'] = iris.target

df['species_name'] = df['species'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})

df.to_csv('../data/iris.csv',index = False)

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

joblib.dump(scaler,'../models/scaler.pkl')

results = {}

print("\n" + "=" * 60)
print("TRAINING SVM WITH LINEAR KERNEL")
print("=" * 60)
svm_linear = SVC(kernel='linear',random_state=42)
svm_linear.fit(X_train_scaled,y_train)
y_pred_linear = svm_linear.predict(X_test_scaled)
accuracy_linear = accuracy_score(y_test,y_pred_linear)
print(f"✓ Accuracy: {accuracy_linear:.4f} ({accuracy_linear*100:.2f}%)")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_linear, target_names=iris.target_names))
joblib.dump(svm_linear,'../models/svm_linear.pkl')
results['Linear'] = accuracy_linear

print("\n" + "=" * 60)
print("TRAINING SVM WITH RBF KERNEL")
print("=" * 60)
svm_rbf = SVC(kernel='rbf',random_state=42)
svm_rbf.fit(X_train_scaled,y_train)
y_pred_rbf = svm_rbf.predict(X_test_scaled)
accuracy_rbf = accuracy_score(y_test,y_pred_rbf)
print(f"✓ Accuracy: {accuracy_rbf:.4f} ({accuracy_rbf*100:.2f}%)")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_rbf, target_names=iris.target_names))
joblib.dump(svm_rbf,'../models/svm_rbf.pkl')
results['RBF'] = accuracy_rbf

print("\n" + "=" * 60)
print("TRAINING SVM WITH POLYNOMIAL KERNEL (degree=3)")
print("=" * 60)
svm_poly = SVC(kernel='poly',random_state=42)
svm_poly.fit(X_train_scaled,y_train)
y_pred_poly = svm_poly.predict(X_test_scaled)
accuracy_poly = accuracy_score(y_test,y_pred_poly)
print(f"✓ Accuracy: {accuracy_poly:.4f} ({accuracy_poly*100:.2f}%)")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_poly, target_names=iris.target_names))
joblib.dump(svm_poly,'../models/svm_poly.pkl')
results['poly'] = accuracy_poly

print("\n" + "=" * 60)
print("TRAINING SVM WITH SIGMOID KERNEL")
print("=" * 60)
svm_sigmoid = SVC(kernel='sigmoid',random_state=42)
svm_sigmoid.fit(X_train_scaled,y_train)
y_pred_sigmoid = svm_sigmoid.predict(X_test_scaled)
accuracy_sigmoid = accuracy_score(y_test,y_pred_sigmoid)
print(f"✓ Accuracy: {accuracy_sigmoid:.4f} ({accuracy_sigmoid*100:.2f}%)")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_sigmoid, target_names=iris.target_names))
joblib.dump(svm_sigmoid,'../models/svm_sigmoid.pkl')
results['sigmoid'] = accuracy_sigmoid

print("\n" + "=" * 60)
print("KERNEL COMPARISON SUMMARY")
print("=" * 60)
results_df = pd.DataFrame(list(results.items()),columns=['Kernel','Accuracy'])
results_df = results_df.sort_values('Accuracy',ascending=False).reset_index(drop=True)
print(results_df.to_string(index=False))

best_kernel = results_df.iloc[0]['Kernel']
best_accuracy = results_df.iloc[0]['Accuracy']
print(f"\n🏆 Best Kernel: {best_kernel} with accuracy {best_accuracy:.4f}")
# Save results
results_df.to_csv('../results/kernel_comparison.csv', index=False)

print("\n" + "=" * 60)
print("HYPERPARAMETER TUNING (GridSearchCV on RBF Kernel)")
print("=" * 60)
param_grid = {
    'C':[0.1,1,10,100],
    'gamma': ['scala','auto',0.001,0.01,0.1,1],
    'kernel':['rbf']
}
print("Parameter grid:", param_grid)
print("\nSearching for best parameters...")
grid_search = GridSearchCV(SVC(random_state=42),param_grid,scoring='accuracy',verbose=1,n_jobs=1)
grid_search.fit(X_train_scaled,y_train)
print(f"\n✓ Best parameters: {grid_search.best_params_}")
print(f"✓ Best cross-validation accuracy: {grid_search.best_score_:.4f}")

#Training final model with best parameters
best_svm = grid_search.best_estimator_
y_pred_best = best_svm.predict(X_test_scaled)
accuracy_best = accuracy_score(y_test,y_pred_best)
print(f"✓ Test accuracy with best model: {accuracy_best:.4f}")

joblib.dump(best_svm,'../models/best_svm.pkl')

print("\n" + "=" * 60)
print("TRAINING COMPLETE!")
print("=" * 60)
print("✓ All models saved in 'models/' directory")
print("✓ Results saved in 'results/' directory")
print("✓ Dataset saved in 'data/' directory")
print("\nNext steps:")
print("1. Run 'python src/visualize.py' to generate visualizations")
print("2. Run 'python src/predict.py' for interactive predictions")