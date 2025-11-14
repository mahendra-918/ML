# 🌸 SVM Iris Flower Classification

A comprehensive machine learning project demonstrating Support Vector Machine (SVM) classification with multiple kernels on the classic Iris dataset.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3.0-orange.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Dataset](#dataset)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [SVM Theory](#svm-theory)
- [Kernel Trick](#kernel-trick)
- [Results](#results)
- [Visualizations](#visualizations)
- [Technologies Used](#technologies-used)
- [Future Improvements](#future-improvements)
- [Contributing](#contributing)
- [License](#license)
- [Author](#author)

---

## 🎯 Overview

This project implements **Support Vector Machine (SVM)** classification to predict Iris flower species based on sepal and petal measurements. It demonstrates:

- Multiple SVM kernels (Linear, RBF, Polynomial, Sigmoid)
- The kernel trick for handling non-linear data
- Hyperparameter tuning with GridSearchCV
- Decision boundary visualization
- Interactive prediction tool

---

## ✨ Features

- ✅ **Multiple Kernel Comparison**: Linear, RBF, Polynomial, Sigmoid
- ✅ **Hyperparameter Optimization**: GridSearchCV for C and gamma
- ✅ **Decision Boundary Plots**: 2D visualization of classification regions
- ✅ **Support Vector Visualization**: Highlights key training samples
- ✅ **Model Persistence**: Save/load trained models
- ✅ **Interactive Predictions**: CLI tool for real-time species prediction
- ✅ **Comprehensive Metrics**: Accuracy, confusion matrix, classification reports
- ✅ **Beautiful Visualizations**: Publication-ready plots

---

## 🌺 Dataset

**Iris Flower Dataset** (Fisher's Iris dataset)

- **Source**: Built-in scikit-learn dataset
- **Samples**: 150 (50 per species)
- **Features**: 4 numeric features
  - Sepal Length (cm)
  - Sepal Width (cm)
  - Petal Length (cm)
  - Petal Width (cm)
- **Target Classes**: 3 species
  - Setosa (0)
  - Versicolor (1)
  - Virginica (2)

### Feature Statistics

| Feature | Min | Max | Mean | Std Dev |
|---------|-----|-----|------|---------|
| Sepal Length | 4.3 | 7.9 | 5.84 | 0.83 |
| Sepal Width | 2.0 | 4.4 | 3.05 | 0.43 |
| Petal Length | 1.0 | 6.9 | 3.76 | 1.76 |
| Petal Width | 0.1 | 2.5 | 1.20 | 0.76 |

---

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup Steps

1. **Clone or navigate to the project directory**
   ```bash
   cd svm-iris-classification
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify installation**
   ```bash
   python3 -c "import sklearn; print('✓ scikit-learn installed')"
   ```

---

## 📁 Project Structure

```
svm-iris-classification/
│
├── data/
│   └── iris.csv                      # Iris dataset (generated)
│
├── src/
│   ├── __init__.py                   # Package initializer
│   ├── train_svm.py                  # Main training script
│   ├── visualize.py                  # Visualization generator
│   └── predict.py                    # Interactive prediction tool
│
├── models/
│   ├── svm_linear.pkl                # Linear kernel model
│   ├── svm_rbf.pkl                   # RBF kernel model
│   ├── svm_poly.pkl                  # Polynomial kernel model
│   ├── svm_sigmoid.pkl               # Sigmoid kernel model
│   ├── svm_best.pkl                  # Best tuned model
│   └── scaler.pkl                    # Feature scaler
│
├── results/
│   ├── kernel_comparison.csv         # Kernel performance metrics
│   └── plots/
│       ├── decision_boundary_linear.png
│       ├── decision_boundary_rbf.png
│       ├── decision_boundary_poly.png
│       ├── decision_boundary_sigmoid.png
│       ├── confusion_matrix.png
│       ├── kernel_comparison.png
│       └── support_vectors.png
│
├── notebooks/
│   └── 01_SVM_Analysis.ipynb         # Jupyter notebook analysis
│
├── requirements.txt                  # Python dependencies
├── .gitignore                        # Git ignore rules
└── README.md                         # Project documentation
```

---

## 💻 Usage

### 1. Train SVM Models

Run the training script to train all kernel variants:

```bash
cd src
python3 train_svm.py
```

**Output:**
- Trains 4 SVM models (Linear, RBF, Polynomial, Sigmoid)
- Performs hyperparameter tuning with GridSearchCV
- Saves trained models to `models/` directory
- Generates performance comparison
- Displays accuracy and classification reports

### 2. Generate Visualizations

Create decision boundary plots and performance charts:

```bash
python3 visualize.py
```

**Output:**
- Decision boundary plots for each kernel
- Confusion matrix heatmap
- Kernel comparison bar chart
- Support vectors visualization
- All plots saved to `results/plots/`

### 3. Interactive Predictions

Use the prediction tool to classify new flowers:

```bash
python3 predict.py
```

**Example interaction:**
```
Enter Flower Measurements:
Sepal Length (cm): 5.1
Sepal Width (cm): 3.5
Petal Length (cm): 1.4
Petal Width (cm): 0.2

PREDICTION RESULT
🌸 Predicted Species: SETOSA
   Class: 0
```

---

## 📚 SVM Theory

### What is SVM?

**Support Vector Machine (SVM)** is a supervised machine learning algorithm used for classification and regression tasks. It works by:

1. **Finding the optimal hyperplane** that separates different classes
2. **Maximizing the margin** between classes
3. **Using support vectors** (critical data points closest to the decision boundary)

### Key Concepts

#### 1. **Hyperplane**
- Decision boundary that separates classes
- In 2D: a line
- In 3D: a plane
- In higher dimensions: a hyperplane

#### 2. **Margin**
- Distance between hyperplane and nearest data points (support vectors)
- SVM maximizes this margin for better generalization

#### 3. **Support Vectors**
- Data points closest to the hyperplane
- Only these points influence the decision boundary
- Removing other points doesn't change the model

#### 4. **Hard vs Soft Margin**
- **Hard Margin**: Perfect separation (only for linearly separable data)
- **Soft Margin**: Allows some misclassification (controlled by C parameter)

### SVM Parameters

#### **C (Regularization)**
- Controls trade-off between margin maximization and classification error
- **High C**: Small margin, fewer misclassifications (risk of overfitting)
- **Low C**: Large margin, more misclassifications (better generalization)

#### **Gamma (for RBF kernel)**
- Defines influence of a single training example
- **High gamma**: Close points have high influence (risk of overfitting)
- **Low gamma**: Far points have influence (smoother decision boundary)

---

## 🎭 Kernel Trick

### The Problem

Real-world data is often **not linearly separable** in its original feature space.

### The Solution: Kernel Trick

The kernel trick allows SVM to:
1. **Implicitly map** data to a higher-dimensional space
2. **Find linear separation** in the transformed space
3. **Avoid expensive computation** of explicit transformation

### How It Works

Instead of computing φ(x) explicitly, the kernel function K(x, y) directly computes:

```
K(x, y) = φ(x) · φ(y)
```

This is computationally efficient because we never actually compute φ(x)!

### Kernel Functions

#### 1. **Linear Kernel**
```
K(x, y) = x · y
```
- No transformation
- Best for linearly separable data
- Fastest computation

#### 2. **RBF (Radial Basis Function) Kernel** ⭐ Most Popular
```
K(x, y) = exp(-γ||x - y||²)
```
- Maps to infinite-dimensional space
- Handles complex non-linear boundaries
- Controlled by gamma parameter

#### 3. **Polynomial Kernel**
```
K(x, y) = (x · y + c)^d
```
- Polynomial degree d
- Good for curved boundaries
- Computationally expensive for high degrees

#### 4. **Sigmoid Kernel**
```
K(x, y) = tanh(αx · y + c)
```
- Similar to neural network activation
- Less commonly used

### Visual Explanation

```
Original 2D Space          →    Transformed 3D Space
(Not separable)                 (Linearly separable)

    •  •                            •
  •  ○  •          Kernel               •
    •  •           =====>          ○         ○
  ○      ○                     •         •
    ○  ○                           •

Red: Class 0                    Now a hyperplane can
Blue: Class 1                   separate the classes!
```

---

## 📊 Results

### Kernel Performance Comparison

| Kernel | Accuracy | Best Use Case |
|--------|----------|---------------|
| RBF | 100.00% | Non-linear, general purpose |
| Linear | 100.00% | Linearly separable data |
| Polynomial | 96.67% | Polynomial relationships |
| Sigmoid | 96.67% | Specific non-linear cases |

### Best Model (After Hyperparameter Tuning)

- **Kernel**: RBF
- **Best Parameters**:
  - C: 10
  - Gamma: 0.1
- **Cross-validation Accuracy**: ~98%
- **Test Accuracy**: 100%

### Classification Report (RBF Kernel)

```
              precision    recall  f1-score   support

      setosa       1.00      1.00      1.00        10
  versicolor       1.00      1.00      1.00         9
   virginica       1.00      1.00      1.00        11

    accuracy                           1.00        30
```

### Confusion Matrix

```
              Predicted
           Setosa  Versicolor  Virginica
Actual
Setosa        10         0          0
Versicolor     0         9          0
Virginica      0         0         11
```

Perfect classification! 🎯

---

## 📈 Visualizations

### 1. Decision Boundaries

Shows how each kernel separates the three Iris species in 2D feature space (Sepal Length vs Sepal Width).

- **Linear**: Straight line boundaries
- **RBF**: Smooth curved boundaries
- **Polynomial**: Curved boundaries with specific degree
- **Sigmoid**: S-shaped boundaries

### 2. Support Vectors

Highlights the critical data points that define the decision boundary. RBF kernel typically uses fewer support vectors for simpler boundaries.

### 3. Kernel Comparison

Bar chart showing accuracy comparison across all kernel types, helping identify the best kernel for this dataset.

### 4. Confusion Matrix

Heatmap visualization showing actual vs predicted classifications, revealing any misclassification patterns.

---

## 🛠 Technologies Used

- **Python 3.8+**: Programming language
- **NumPy**: Numerical computations
- **Pandas**: Data manipulation
- **scikit-learn**: Machine learning library
  - SVM implementation
  - Model selection (GridSearchCV)
  - Preprocessing (StandardScaler)
  - Metrics
- **Matplotlib**: Plotting library
- **Seaborn**: Statistical visualization
- **Joblib**: Model serialization

---

## 🔮 Future Improvements

- [ ] Add multi-class classification with more datasets
- [ ] Implement custom kernel functions
- [ ] Add ROC curve and AUC metrics
- [ ] Create web interface with Flask/Streamlit
- [ ] Add cross-validation visualization
- [ ] Implement SVM from scratch (educational)
- [ ] Add feature importance analysis
- [ ] Compare with other classifiers (KNN, Decision Trees, Random Forest)
- [ ] Add real-time prediction API
- [ ] Implement one-class SVM for anomaly detection

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the project
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 👨‍💻 Author

**Your Name**

- GitHub: [@yourusername](https://github.com/yourusername)
- Email: your.email@example.com

---

## 🙏 Acknowledgments

- Iris dataset: R.A. Fisher (1936)
- scikit-learn documentation and community
- UC Irvine Machine Learning Repository

---

## 📖 References

1. [Support Vector Machines - scikit-learn](https://scikit-learn.org/stable/modules/svm.html)
2. [The Elements of Statistical Learning - Hastie, Tibshirani, Friedman](https://web.stanford.edu/~hastie/ElemStatLearn/)
3. [A Tutorial on Support Vector Machines for Pattern Recognition - Burges (1998)](https://www.microsoft.com/en-us/research/publication/a-tutorial-on-support-vector-machines-for-pattern-recognition/)

---

<div align="center">

### ⭐ If you found this project helpful, please give it a star!

Made with ❤️ and Python

</div>
