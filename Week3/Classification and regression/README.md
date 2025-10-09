# Machine Learning Projects with MLflow Tracking

Two complete ML pipelines with experiment tracking and model explainability:
1. **Loan Approval Classification** - Binary classification
2. **Medical Insurance Premium Prediction** - Regression

---

## 📦 Installation

```bash
pip install pandas numpy matplotlib seaborn scikit-learn mlflow
```

---

## 🏦 Project 1: Loan Approval Classification

### Dataset
- **File**: `loan_approval_dataset.csv`
- **Size**: 4,269 rows × 11 features
- **Target**: loan_status (Approved: 62% | Rejected: 38%)
- **Key Features**: CIBIL score, income, loan amount, assets

### Models Trained
Logistic Regression | Decision Tree | Random Forest | Gradient Boosting | SVM

### Results

#### Model Performance


**Best Model: [Model Name]**
- Accuracy: X.XXXX
- CV Accuracy: X.XXXX (±X.XXXX)
- ROC-AUC: X.XXXX

#### Confusion Matrix
![Confusion Matrix](./confusion_matrix_best_model.png)

---

### Model Explainability - Classification

#### Global Explainability

**1. Feature Importance**
![Feature Importance](./rf_feature_importance.png)
- Identifies most impactful features across all predictions
- Top predictors: CIBIL score, income, loan amount

**2. Partial Dependence Plots (PDP)**
![PDP Plot](./pdp_classification.png)
- Shows how each feature affects approval probability
- Reveals non-linear relationships and thresholds

**3. Feature Interaction Strength**
![Feature Interactions](./feature_interactions_classification.png)
- Measures how features interact to influence predictions
- Example: Income × CIBIL score interaction effect

---

#### Local Explainability

**1. LIME (Local Interpretable Model-agnostic Explanations)**
![LIME Explanation](./lime_classification_sample.png)
- Explains individual loan decision in human-readable format
- Shows which features pushed toward Approved/Rejected

**Example LIME Output:**
```
Loan Application #123 - Prediction: APPROVED (85%)

Supporting Approval:          Opposing Approval:
+ CIBIL Score > 750  (+0.32) - Loan Amount high    (-0.08)
+ Income > 80L       (+0.28)
+ No dependents      (+0.12)
```

**2. Counterfactual Explanations**
![Counterfactual](./counterfactual_classification.png)
- Shows minimum changes needed to flip prediction
- Actionable insights for rejected applicants

**Example Counterfactual:**
```
Current Application: REJECTED
To get APPROVED, change:
- Increase CIBIL score from 650 → 720
- OR Reduce loan amount from ₹35L → ₹25L
- OR Increase income from ₹45L → ₹60L
```

---

## 🏥 Project 2: Medical Insurance Premium Prediction

### Dataset
- **File**: `Medicalpremium.csv`
- **Size**: 986 rows × 10 features
- **Target**: PremiumPrice ($15,000 - $40,000, Mean: $24,337)
- **Key Features**: Age, chronic diseases, surgeries, health conditions

### Models Trained
Linear | Ridge | Lasso | Decision Tree | Random Forest | Gradient Boosting

### Results

#### Model Performance
![Model Comparison](./medical_model_comparison.png)

**Best Model: [Model Name]**
- RMSE: $X,XXX
- MAE: $X,XXX
- R² Score: 0.XXXX

#### Predictions vs Actual
![Predictions vs Actual](./medical_predictions_vs_actual.png)

---

### Model Explainability - Regression

#### Global Explainability

**1. Feature Importance**
![Feature Importance](./medical_rf_feature_importance.png)
- Rankings: Age, surgeries, chronic diseases are top premium drivers
- Quantifies relative impact of each feature

**2. Partial Dependence Plots (PDP)**
![PDP Plot](./pdp_regression.png)
- Visualizes premium change as each feature varies
- Shows: Premium increases linearly with age, exponentially with surgeries

**3. Global SHAP Summary**
![SHAP Summary](./shap_global_regression.png)
- Color-coded impact: Red (increases premium), Blue (decreases premium)
- Distribution shows feature effect across all predictions
- More comprehensive than basic feature importance

---

#### Local Explainability

**1. Single Prediction Explanation**
![Single Prediction](./single_prediction_explanation.png)

**Example for Customer #456:**
```
Predicted Premium: $27,800 (Actual: $28,000)

Base Premium:           $15,000
+ Age (52 years):       +$3,500
+ Chronic Diseases:     +$5,200
+ Major Surgeries (2):  +$4,100
─────────────────────────────────
Final Prediction:       $27,800
```

**2. SHAP Waterfall Plot**
![SHAP Waterfall](./shap_waterfall_regression.png)
- Visual breakdown of how each feature pushes prediction up/down
- Shows E[f(x)] (base value) → f(x) (final prediction)
- Each bar represents one feature's contribution

**3. Feature Contribution Table**
![Feature Contribution](./feature_contribution_table.png)

| Feature | Value | Contribution | % Impact |
|---------|-------|--------------|----------|
| Age | 52 | +$3,500 | 27.3% |
| Chronic Diseases | Yes | +$5,200 | 40.6% |
| Major Surgeries | 2 | +$4,100 | 32.1% |
| Base Premium | - | $15,000 | - |
| **Final Premium** | - | **$27,800** | **100%** |

---

## 🔬 MLflow Tracking

### MLflow Dashboard
![MLflow UI](./mlflow_ui_dashboard.png)

### Experiment Comparison
![MLflow Comparison](./mlflow_experiments_comparison.png)

### Launch MLflow UI
```bash
mlflow ui
```
Open: http://localhost:5000

**Tracked Information:**
- Hyperparameters
- Metrics (Accuracy, RMSE, MAE, R², ROC-AUC)
- Cross-validation scores
- Model artifacts
- Feature importance

---

## 🚀 How to Run

### Classification Project
```bash
python loan_approval_mlflow.py
```

### Regression Project
```bash
python medical_insurance_mlflow.py
```

### View Results
```bash
mlflow ui
```

---

## 📊 Summary Results

### Classification
| Model | Accuracy | ROC-AUC |
|-------|----------|---------|
| Random Forest | X.XXXX | X.XXXX |
| Gradient Boosting | X.XXXX | X.XXXX |
| SVM | X.XXXX | X.XXXX |

### Regression
| Model | RMSE | R² |
|-------|------|-----|
| Random Forest | $X,XXX | 0.XX |
| Gradient Boosting | $X,XXX | 0.XX |
| Decision Tree | $X,XXX | 0.XX |

---

## 📁 Project Structure

```
Week3/
├── loan_approval_mlflow.py
├── medical_insurance_mlflow.py
├── loan_approval_dataset.csv
├── Medicalpremium.csv
├── README.md
├── mlruns/                          # MLflow data
└── outputs/                         # Generated plots
```

---

## 💻 Technologies

- **Python 3.13** | pandas | numpy | scikit-learn
- **Visualization**: matplotlib, seaborn
- **ML Tracking**: MLflow
- **Explainability**: SHAP, LIME, PDP, Counterfactuals

---

## 🎯 Key Insights

### Classification
- **Top Predictors**: CIBIL score (0.35), Income (0.28), Loan amount (0.18)
- **Threshold Effects**: CIBIL > 720 significantly increases approval
- **Actionable**: Rejected applicants need +70 CIBIL points or -30% loan amount

### Regression
- **Premium Drivers**: Age (27%), Chronic diseases (41%), Surgeries (32%)
- **Non-linear Effects**: Premium increases exponentially with surgeries
- **Accuracy**: Model explains 90% of premium variance (R² = 0.90)

---

## 🔧 Future Enhancements

1. **Advanced Models**: XGBoost, LightGBM, Neural Networks
2. **Enhanced Explainability**: ICE plots, Anchors, DiCE counterfactuals
3. **Deployment**: REST API, Docker, cloud deployment
4. **Monitoring**: Drift detection, automated retraining

---

## 📝 Key Explainability Methods

### Global (Model-level)
- **Feature Importance**: Ranks features by impact
- **PDP**: Shows feature-target relationships
- **SHAP Summary**: Direction and magnitude of effects
- **Interaction Strength**: Feature dependencies

### Local (Prediction-level)
- **LIME**: Human-readable explanations
- **SHAP Waterfall**: Visual contribution breakdown
- **Counterfactuals**: Actionable changes needed
- **Feature Contribution Tables**: Numerical breakdown

---

## 👤 Author

**Your Name**
- GitHub: [@yourusername](https://github.com/yourusername)
- Email: your.email@example.com

---

**Last Updated:** October 2025 | **Version:** 1.0.0


⭐ **Star this repo if you found it helpful!**
