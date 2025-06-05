# Task 7 - Support Vector Machines (SVM) Classification

This project involves using Support Vector Machines (SVM) to classify breast cancer tumors as **Malignant** or **Benign** based on selected features from the Breast Cancer Wisconsin dataset.

## 🔍 Dataset Overview

- Source: `data/breast_cancer.csv`
- Instances: 569
- Features: 30 numeric features (originally)
- Target: `diagnosis` (0 = Malignant, 1 = Benign)

## Features Used

We randomly selected **2 highly relevant features** to enable 2D visualization of decision boundaries:

- `radius_mean`: Mean of distances from center to points on the tumor perimeter.
- `concave points_mean`: Mean number of concave portions of the contour.

These were chosen because they significantly influence tumor classification based on prior studies and data correlation.

## ⚙️ Workflow Steps

### 1. Load and Preprocess Data
- Loaded breast cancer dataset
- Converted to Pandas DataFrame
- Selected 2 features + diagnosis label
- Dropped null values
- Encoded `diagnosis` (M → 0, B → 1)

### 2. Exploratory Data Analysis
- Checked value counts of target labels
- Plotted pairplots and feature distributions (optional step)

### 3. Feature Scaling
- Standardized the 2 selected features using `StandardScaler`

### 4. SVM Classification
- Trained 3 different SVM classifiers:
  - Linear Kernel
  - RBF (Gaussian) Kernel
  - Polynomial Kernel

### 5. Model Evaluation
- Evaluated using:
  - Accuracy
  - Confusion Matrix
  - Classification Report
- Visualized Decision Boundaries for each model

### 6. Cross-Validation & Hyperparameter Tuning
- Used `GridSearchCV` to tune parameters like `C`, `gamma`, `kernel`
- Performed 5-fold Cross Validation

---

## Visualizations

- 2D decision boundary plots for each kernel
- Comparison of margins and separation lines

---

## 📂 Project Structure

├── data/  
│ └── breast_cancer.csv  
├── support_vector_machine.ipynb  
└── README.md  


---

## Requirements

- Python 3.8+
- `scikit-learn`
- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`  
