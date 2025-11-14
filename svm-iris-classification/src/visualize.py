import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
import joblib

sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12,8)

print("=" * 60)
print("SVM VISUALIZATION GENERATOR")
print("=" * 60)

# Load dataset
print("\n[1] Loading dataset...")
iris = datasets.load_iris()
X = iris.data
y = iris.target

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)

scaler = joblib.load('../models/scaler.pkl')
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Load models
print("[2] Loading trained models...")
models = {
    'Linear': joblib.load('../models/svm_linear.pkl'),
    'RBF': joblib.load('../models/svm_rbf.pkl'),
    'Polynomial': joblib.load('../models/svm_poly.pkl'),
    'Sigmoid': joblib.load('../models/svm_sigmoid.pkl')
}

print("\n[3] Creating decision boundary plots...")
# Use only first 2 features for 2D visualization
X_2d = X[:, :2]  # Sepal length and Sepal width
X_train_2d, X_test_2d, y_train_2d, y_test_2d = train_test_split(
    X_2d, y, test_size=0.2, random_state=42, stratify=y
)
scaler_2d = StandardScaler()
X_train_2d_scaled = scaler_2d.fit_transform(X_train_2d)
X_test_2d_scaled = scaler_2d.transform(X_test_2d)

# Helper function to plot decision boundaries
def plot_decision_boundary(model,X,y,title,filename):
    h = 0.02 # step sixe in the mesh

    #create mesh
    x_min,x_max = X[:,0].min()-1,X[:,0].max()+1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    # Predict on mesh
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.figure(figsize=(10, 7))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap='viridis')
    # Plot training points
    scatter = plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', 
                         edgecolors='black', s=100, alpha=0.8)
    plt.xlabel('Sepal Length (scaled)', fontsize=12)
    plt.ylabel('Sepal Width (scaled)', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.colorbar(scatter, label='Species')
    plt.grid(True, alpha=0.3)
    plt.savefig(f'../results/plots/{filename}', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {filename}")
    plt.close()

# Train 2D models and plot
kernels = ['linear', 'rbf', 'poly', 'sigmoid']
kernel_names = ['Linear', 'RBF', 'Polynomial', 'Sigmoid']
for kernel,name in zip(kernels,kernel_names):
    model_2d = SVC(kernel=kernel,degree=3 if kernel == 'poly' else 3,random_state=42)
    model_2d.fit(X_train_2d_scaled,y_train_2d)
    plot_decision_boundary(
        model_2d,X_train_2d_scaled,y_train_2d,f'SVM Decision Boundary - {name} Kernel',
        f'decision_boundary_{kernel}.png'
    )

print("\n[4] Creating confusion matrix...")
# Use best model (RBF)
best_model = models['RBF']
y_pred = best_model.predict(X_test_scaled)
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=iris.target_names,
            yticklabels=iris.target_names,
            cbar_kws={'label': 'Count'})
plt.xlabel('Predicted', fontsize=12)
plt.ylabel('Actual', fontsize=12)
plt.title('Confusion Matrix - RBF Kernel SVM', fontsize=14, fontweight='bold')
plt.savefig('../results/plots/confusion_matrix.png', dpi=300, bbox_inches='tight')
print("✓ Saved: confusion_matrix.png")
plt.close()

print("\n[5] Creating kernel comparison chart...")
results_df = pd.read_csv('../results/kernel_comparison.csv')
plt.figure(figsize=(10, 6))
bars = plt.bar(results_df['Kernel'], results_df['Accuracy'], 
               color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A'],
               edgecolor='black', linewidth=1.5)

# Add value labels on bars
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.4f}',
            ha='center', va='bottom', fontsize=12, fontweight='bold')

plt.xlabel('Kernel Type', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)
plt.title('SVM Kernel Comparison on Iris Dataset', fontsize=14, fontweight='bold')
plt.ylim([0, 1.1])
plt.grid(axis='y', alpha=0.3)
plt.savefig('../results/plots/kernel_comparison.png', dpi=300, bbox_inches='tight')
print("✓ Saved: kernel_comparison.png")
plt.close()

print("\n[6] Creating support vectors visualization...")
# Train model on 2D data to visualize support vectors
model_sv = SVC(kernel='rbf', random_state=42)
model_sv.fit(X_train_2d_scaled, y_train_2d)

plt.figure(figsize=(10, 7))
h = 0.02
x_min, x_max = X_train_2d_scaled[:, 0].min() - 1, X_train_2d_scaled[:, 0].max() + 1
y_min, y_max = X_train_2d_scaled[:, 1].min() - 1, X_train_2d_scaled[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

Z = model_sv.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.3, cmap='viridis')
plt.scatter(X_train_2d_scaled[:, 0], X_train_2d_scaled[:, 1], 
           c=y_train_2d, cmap='viridis', edgecolors='black', s=100, alpha=0.8)

# Highlight support vectors
plt.scatter(model_sv.support_vectors_[:, 0], model_sv.support_vectors_[:, 1],
           s=200, facecolors='none', edgecolors='red', linewidths=2, 
           label=f'Support Vectors (n={len(model_sv.support_vectors_)})')

plt.xlabel('Sepal Length (scaled)', fontsize=12)
plt.ylabel('Sepal Width (scaled)', fontsize=12)
plt.title('Support Vectors Highlighted - RBF Kernel', fontsize=14, fontweight='bold')
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.savefig('../results/plots/support_vectors.png', dpi=300, bbox_inches='tight')
print("✓ Saved: support_vectors.png")
plt.close()

print("\n" + "=" * 60)
print("VISUALIZATION COMPLETE!")
print("=" * 60)
print("✓ All plots saved in 'results/plots/' directory")
print("\nGenerated visualizations:")
print("  1. Decision boundaries for all kernels")
print("  2. Confusion matrix")
print("  3. Kernel comparison chart")
print("  4. Support vectors visualization")


# plt.show()