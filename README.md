# 🧠 Dementia Prediction Using Machine Learning

> A machine learning system that predicts dementia risk using clinical assessments and MRI scan data to support early diagnosis and intervention.

---


## 🎯 Overview

Early detection of dementia is critical for effective treatment and care planning. This project leverages machine learning to analyze clinical and neuroimaging data, identifying patterns that may indicate dementia risk before severe symptoms appear.

The system processes three key types of information:
- **Demographic data** (age, education, socioeconomic status)
- **Cognitive test scores** (MMSE, CDR)
- **Brain volumetric measurements** from MRI scans (eTIV, nWBV, ASF)

By combining these diverse data sources, the model provides healthcare professionals with a data-driven diagnostic support tool.

---

## 🔍 Problem Statement

Dementia affects millions worldwide, and by the time symptoms become obvious, significant brain changes have already occurred. This project addresses the challenge of:

- **Early detection**: Identifying dementia risk before severe cognitive decline
- **Data-driven diagnosis**: Supporting clinical decisions with objective measurements
- **Accessible prediction**: Using non-invasive, routinely collected medical data

---

## 📊 Dataset

### Dataset Features

| Feature       | Description | Type |
|---------------|-------------|------|
| `Visit`       | Clinical visit number | Numeric |
| `MR Delay`    | Days between clinical visit and MRI scan | Numeric |
| `M/F`         | Gender (M = Male, F = Female) | Categorical |
| `Hand`        | Dominant hand (R = Right, L = Left) | Categorical |
| `Age`         | Patient age in years | Numeric |
| `EDUC`        | Years of formal education | Numeric |
| `SES`         | Socioeconomic status (1-5 scale) | Numeric |
| `MMSE`        | Mini-Mental State Examination score (0-30) | Numeric |
| `CDR`         | Clinical Dementia Rating (0, 0.5, 1, 2, 3) | Numeric |
| `eTIV`        | Estimated Total Intracranial Volume (cm³) | Numeric |
| `nWBV`        | Normalized Whole Brain Volume | Numeric |
| `ASF`         | Atlas Scaling Factor | Numeric |

### Understanding Key Features

**Cognitive Assessments:**
- **MMSE**: Scores range from 0-30; scores below 24 suggest cognitive impairment
- **CDR**: Rates dementia severity (0 = normal, 0.5 = very mild, 1 = mild, 2 = moderate, 3 = severe)

**Neuroimaging Metrics:**
- **eTIV**: Measures overall brain cavity size
- **nWBV**: Brain volume adjusted for head size; lower values may indicate atrophy
- **ASF**: Volumetric scaling parameter for brain normalization

---

## ✨ Features

- ✅ Binary classification for dementia prediction
- ✅ Support Vector Machine (SVM) classifier
- ✅ Comprehensive data preprocessing and feature engineering
- ✅ Missing value handling
- ✅ Feature scaling and normalization
- ✅ Model evaluation with multiple metrics
- ✅ Confusion matrix and classification reports
- ✅ Data visualization capabilities

---

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup


## 💻 Usage

### Basic Usage

```python
# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix

# Load the dataset
data = pd.read_csv('dementia_dataset.csv')

# Preprocess the data
# (Add your preprocessing steps here)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = SVC(kernel='rbf', random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
print(classification_report(y_test, y_pred))
```


## 🤖 Model Details

### Algorithm: Support Vector Machine (SVM)

**Why SVM?**
- Effective in high-dimensional spaces
- Memory efficient
- Versatile with different kernel functions
- Robust to overfitting, especially in high-dimensional space

### Model Pipeline

1. **Data Preprocessing**
   - Handle missing values
   - Encode categorical variables
   - Feature scaling/normalization

2. **Feature Engineering**
   - Select relevant features
   - Create interaction terms if needed
   - Remove highly correlated features

3. **Model Training**
   - Train-test split (80-20)
   - Cross-validation
   - Hyperparameter tuning

4. **Evaluation**
   - Accuracy score
   - Precision, Recall, F1-score
   - Confusion matrix
   - ROC-AUC curve

---

## 📈 Results

### Model Performance

| Metric    | Score |
|-----------|-------|
| Accuracy  | 77%   |


*Note: Update these values with your actual model results*

---

## 👥 Contact

**Your Name**
- GitHub: [Farhanbwn](https://github.com/Farhanbwn)
- Email: farhanbwn2003@gmail.com

---

Made with ❤️ from Bardhaman and Python

</div>
