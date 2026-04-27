# 🫀 Heart Failure Prediction with Machine Learning

**Author:** Pedro Kauan Silveira Silva

A compact, reproducible project that predicts **death events** in patients with heart failure using clinical records. We compare multiple machine learning classifiers (KNN, Decision Tree, Random Forest, SVM, MLP, Logistic Regression, XGBoost, Gaussian Naive Bayes, Nearest Centroid) using cross-validation with NearMiss undersampling pipeline, and statistical validation via **Wilcoxon signed-rank test**.

---

## 🚀 Quick Highlights

- Predicts mortality from heart failure clinical records (**DEATH_EVENT**: 0 = Survived, 1 = Deceased)
- **299 patient records** with **13 clinical features**
- Uses **10-fold Cross-Validation** with **NearMiss undersampling** inside pipeline (no data leakage)
- Evaluated using **Accuracy, Precision, Recall** and **F1-Score**
- **Wilcoxon Test** applied for statistical validation between all model pairs
- Full exploratory and ML analysis inside: `src/heart_failure.ipynb`
- Dataset included: `src/heart_failure.csv`

---

## 🔬 Project Summary

Heart failure is a critical clinical condition where the heart cannot pump enough blood to meet the body's needs. Early prediction of mortality in heart failure patients can enable timely interventions and improve outcomes. This project investigates which clinical and demographic factors influence patient survival, and provides a reproducible ML benchmark for academic and clinical decision support.

---

## 🧾 Dataset

**Source:** Heart Failure Clinical Records Dataset

**Samples:** 299

**Features:** 13 clinical and demographic variables

**Target:** `DEATH_EVENT` (0 = Survived, 1 = Deceased)

**Class Distribution (imbalanced):**
- 0 (Survived): **203** samples (67.9%)
- 1 (Deceased): **96** samples (32.1%)

### Feature Description

| Feature | Type | Description |
|---|---|---|
| `age` | Numerical | Age of the patient (years) |
| `anaemia` | Categorical | Whether the patient has anaemia (0: No, 1: Yes) |
| `creatinine_phosphokinase` | Numerical | Level of CPK enzyme in blood (mcg/L) |
| `diabetes` | Categorical | Whether the patient has diabetes (0: No, 1: Yes) |
| `ejection_fraction` | Numerical | Percentage of blood leaving the heart per contraction (%) |
| `high_blood_pressure` | Categorical | Whether the patient has high blood pressure (0: No, 1: Yes) |
| `platelets` | Numerical | Platelet count in blood (kiloplatelets/mL) |
| `serum_creatinine` | Numerical | Level of serum creatinine in blood (mg/dL) |
| `serum_sodium` | Numerical | Level of serum sodium in blood (mEq/L) |
| `sex` | Categorical | Sex of the patient (0: Female, 1: Male) — *removed during preprocessing* |
| `smoking` | Categorical | Whether the patient smokes (0: No, 1: Yes) |
| `time` | Numerical | Follow-up period (days) — *removed during preprocessing* |
| `DEATH_EVENT` | Target | Whether the patient died (0: Survived, 1: Deceased) |

> **Note:** Features `time` and `sex` were dropped during preprocessing.

---

## ⚙️ Methodology Overview

- **Data Cleaning & Feature Selection:** Removal of `time` and `sex` features
- **Scaling:** StandardScaler / MinMaxScaler inside the CV pipeline
- **Imbalanced Data Handling:** NearMiss (version 3) undersampling applied **inside** the pipeline to avoid data leakage
- **Splitting:** 10-fold Cross-Validation
- **Models Tested:**
  - K-Nearest Neighbors (KNN)
  - Decision Tree
  - Random Forest
  - Support Vector Machine (SVM)
  - Multi-Layer Perceptron (MLP)
  - Logistic Regression
  - XGBoost
  - Gaussian Naive Bayes
  - Nearest Centroid
- **Metrics:** Accuracy, Precision, Recall, F1-Score
- **Statistical Test:** Wilcoxon signed-rank test (α = 0.05)

---

## 🤖 Machine Learning Models Evaluated

| Model | Type | Evaluation Metrics |
|---|---|---|
| **KNeighborsClassifier** | Instance-based | Accuracy, Precision, Recall, F1-Score |
| **DecisionTreeClassifier** | Tree-based | Accuracy, Precision, Recall, F1-Score |
| **RandomForestClassifier** | Ensemble (Bagging) | Accuracy, Precision, Recall, F1-Score |
| **SVM** | Kernel-based | Accuracy, Precision, Recall, F1-Score |
| **MLPClassifier** | Neural Network | Accuracy, Precision, Recall, F1-Score |
| **LogisticRegression** | Linear Model | Accuracy, Precision, Recall, F1-Score |
| **XGBClassifier** | Ensemble (Boosting) | Accuracy, Precision, Recall, F1-Score |
| **GaussianNB** | Probabilistic | Accuracy, Precision, Recall, F1-Score |
| **NearestCentroid** | Prototype-based | Accuracy, Precision, Recall, F1-Score |

All models were evaluated using **10-fold cross-validation** with NearMiss undersampling integrated in the pipeline.

---

## 📊 Results

### Mean Scores across 10-fold CV

#### Accuracy

| Model | Mean Accuracy |
|---|---|
| **DecisionTreeClassifier** | **0.7529** |
| **RandomForestClassifier** | **0.7429** |
| XGBClassifier | 0.7362 |
| LogisticRegression | 0.7295 |
| MLPClassifier | 0.7129 |
| GaussianNB | 0.7129 |
| SVM | 0.6763 |
| KNeighborsClassifier | 0.6559 |
| NearestCentroid | 0.6394 |

### Statistical Validation — Wilcoxon Signed-Rank Test (α = 0.05)

The Wilcoxon test was applied to compare all model pairs across all metrics. Key findings:

#### Accuracy Comparisons (Significant Results)

| Comparison | p-value | Result |
|---|---|---|
| KNN vs DecisionTree | 0.0273 | **Significant** — DecisionTree better |
| KNN vs RandomForest | 0.0371 | **Significant** — RandomForest better |
| KNN vs XGBClassifier | 0.0371 | **Significant** — XGBClassifier better |
| DecisionTree vs NearestCentroid | 0.0273 | **Significant** — DecisionTree better |
| RandomForest vs NearestCentroid | 0.0371 | **Significant** — RandomForest better |
| SVM vs XGBClassifier | 0.0371 | **Significant** — XGBClassifier better |

#### F1-Score Comparisons

| Comparison | p-value | Result |
|---|---|---|
| LogisticRegression vs XGBClassifier | 0.4922 | Not significant |
| LogisticRegression vs GaussianNB | 0.1172 | Not significant |
| LogisticRegression vs NearestCentroid | 0.1055 | Not significant |
| XGBClassifier vs GaussianNB | 0.3750 | Not significant |
| XGBClassifier vs NearestCentroid | 0.2324 | Not significant |
| MLPClassifier vs GaussianNB | 0.9609 | Not significant |
| GaussianNB vs NearestCentroid | 0.8457 | Not significant |

> **Note:** For F1-Score, no statistically significant differences were found between models (p ≥ 0.05).

---

## 🏆 Main Findings

- The dataset is **imbalanced** (67.9% survived vs 32.1% deceased), addressed with **NearMiss undersampling** inside the CV pipeline
- **DecisionTreeClassifier** achieved the highest accuracy (**0.7529**), followed by **RandomForestClassifier** (**0.7429**)
- **LogisticRegression** achieved the highest F1-Score (**0.7083**), followed by **XGBClassifier** (**0.6945**)
- For **accuracy**, some statistically significant differences were found (e.g., DecisionTree vs KNN)
- For **F1-Score**, **no statistically significant differences** were found between any model pairs
- Features `time` and `sex` were removed during preprocessing — `time` due to potential data leakage, `sex` due to low predictive power

---

## 🗂️ Project Structure

```
heart_failure/
├── src/
│   ├── heart_failure.ipynb   # Main notebook with full analysis
│   └── heart_failure.csv     # Dataset
├── requirements.txt          # Project dependencies
├── venv/                     # Python virtual environment
├── .gitignore
└── README.md
```

---

## 🧰 Requirements

```txt
contourpy==1.3.3
cycler==0.12.1
fonttools==4.62.1
imbalanced-learn==0.14.1
joblib==1.5.3
kiwisolver==1.5.0
matplotlib==3.10.8
numpy==2.4.4
packaging==26.0
pandas==3.0.2
pillow==12.2.0
pyparsing==3.3.2
python-dateutil==2.9.0.post0
scikit-learn==1.8.0
scipy==1.17.1
seaborn==0.13.2
six==1.17.0
sklearn-compat==0.1.5
threadpoolctl==3.6.0
xgboost==3.2.0
```

---

## 🚀 How to Run

### 1. Clone the repository
```bash
git clone <repository-url>
cd heart_failure
```

### 2. Create and activate virtual environment
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the notebook
```bash
cd src
jupyter notebook heart_failure.ipynb
```

---

## 📈 Visual Outputs

`heart_failure.ipynb` includes:
- Boxplots per feature by `DEATH_EVENT`
- Class distribution bar chart and pie chart
- Confusion matrices for all 9 models
- Wilcoxon test comparison outputs

---

## 🔭 Future Work

- Apply **SMOTE** or **cost-sensitive learning** for better handling of class imbalance
- Explore **hyperparameter tuning** with GridSearchCV for all models
- Add **SHAP/LIME** explainability for model interpretation
- Validate models on **external datasets**
- Explore **deep learning** approaches
- Implement **feature importance analysis** with Permutation Importance

---

## 📚 References

- Chicco, D., Jurman, G. "Machine learning can predict survival of patients with heart failure from serum creatinine and ejection fraction alone." *BMC Medical Informatics and Decision Making* 20, 16 (2020)
- UCI Machine Learning Repository — Heart Failure Clinical Records Dataset
- scikit-learn documentation
- imbalanced-learn documentation
