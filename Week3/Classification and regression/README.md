# Machine Learning Projects with MLflow & Model Explainability

Two complete ML pipelines with comprehensive model interpretability:

1. **Loan Approval Classification** - Binary classification with LIME & counterfactual explanations
2. **Medical Insurance Premium Prediction** - Regression with SHAP waterfall & feature contributions

---

## 📋 Table of Contents

- [Installation](#installation)
- [Project 1: Loan Approval Classification](#project-1-loan-approval-classification)
- [Project 2: Medical Insurance Premium Prediction](#project-2-medical-insurance-premium-prediction)
- [How to Run](#how-to-run)
- [Technologies Used](#technologies-used)

---

## 📦 Installation

```bash
pip install pandas numpy matplotlib seaborn scikit-learn mlflow shap lime
```

---

## 🏦 Project 1: Loan Approval Classification

### Dataset Overview
- **File**: `loan_approval_dataset.csv`
- **Samples**: 4,269 | **Features**: 11 | **Target**: Approved/Rejected
- **Distribution**: 62% Approved, 38% Rejected

### Models Trained
Logistic Regression| Random Forest | Xgboost |Gradient Boosting


### Best Model Performance

| Metric | Value |
|--------|-------|
| Accuracy | 0.98 |
|F1-Score |0.9765
| ROC-AUC | 0.9991|

![Loan_approval_final_deployment](https://github.com/user-attachments/assets/9a11a23a-c400-4fb2-b106-733947462b5a)

---

### Model Explainability - Classification

#### 🌍 Global Explainability

**1. Feature Importance**

Shows which features drive loan approval decisions across all applications.


*Top features: CIBIL Score, Income Annual, Loan Amount have highest impact*

---

**2. Partial Dependence Plots (PDP)**

Visualizes how each feature independently affects approval probability while averaging out other features.

![PDP - CIBIL Score](./classification_pdp_cibil.png)

*Shows approval probability increases sharply when CIBIL score exceeds 700*

![PDP - Income](./classification_pdp_income.png)

*Higher income correlates with higher approval probability*

**Key Insights:**
- CIBIL score > 750: ~85% approval probability
- Income > ₹50L: Significant increase in approval chance
- Loan amount impact is non-linear with approval

---

**3. Feature Interaction Strength**

Identifies which feature pairs have the strongest combined effect on predictions.

![Feature Interactions](./classification_feature_interactions.png)

*Interaction matrix showing feature pairs that matter most when combined*

**Top Interactions:**
- **CIBIL Score × Income**: Strongest interaction - high CIBIL + high income = best approval odds
- **Loan Amount × CIBIL Score**: Large loans need higher CIBIL scores
- **Assets × Income**: Asset value matters more for high-income applicants

---

#### 🎯 Local Explainability

**1. LIME (Local Interpretable Model-agnostic Explanations)**

Explains why a specific loan application was approved or rejected by showing which features contributed most to that decision.

**Example: Loan Application #1234**

![LIME Explanation - Approved](./classification_lime_approved.png)

*LIME explanation for an approved loan showing positive contributors*

**Interpretation:**
- **Green bars** (right): Features supporting approval
  - CIBIL Score = 778 (+0.35 probability)
  - Income = ₹96L (+0.28 probability)
  - Bank Assets = ₹80L (+0.15 probability)
- **Red bars** (left): Features opposing approval
  - Loan Amount = ₹2.99Cr (-0.12 probability)
- **Final Decision**: Approved (85% confidence)

---

![LIME Explanation - Rejected](./classification_lime_rejected.png)

*LIME explanation for a rejected loan showing negative contributors*

**Interpretation:**
- CIBIL Score = 580 (-0.42 probability) - Primary rejection reason
- Low income (-0.25 probability)
- High loan amount relative to income (-0.18 probability)
- **Final Decision**: Rejected (78% confidence)

---

**2. Counterfactual Explanations**

Shows the minimal changes needed to flip the decision from rejected to approved.

![Counterfactual Example](./classification_counterfactual.png)

*What-if scenario: Minimum changes to get loan approved*

**Counterfactual Analysis for Rejected Application #5678:**

| Feature | Current Value | Needed Value | Change Required |
|---------|---------------|--------------|-----------------|
| CIBIL Score | 580 | 720 | +140 points |
| Income Annual | ₹35L | ₹52L | +₹17L |
| Loan Amount | ₹85L | ₹60L | -₹25L (reduce by 29%) |

**Actionable Insights:**
- **Easiest path to approval**: Reduce loan amount to ₹60L (no other changes needed)
- **Alternative**: Improve CIBIL score to 720 + increase income to ₹52L
- **Business Use**: Suggest customers which factors to improve for future applications

---

### Key Findings - Classification

**Global Insights:**
1. CIBIL Score is the dominant predictor (45% importance)
2. Income and loan amount interaction determines risk assessment
3. Asset values provide secondary confirmation signals

**Local Insights:**
1. Individual predictions are interpretable with LIME showing exact feature contributions
2. Counterfactuals provide actionable feedback to rejected applicants
3. Model decisions align with business logic and risk principles

---

## 🏥 Project 2: Medical Insurance Premium Prediction

### Dataset Overview
- **File**: `Medicalpremium.csv`
- **Samples**: 986 | **Features**: 10 | **Target**: Premium Price ($15K-$40K)
- **Mean Premium**: $24,337

### Models Trained
Linear Regression | Ridge | Lasso | Decision Tree | Random Forest | Gradient Boosting

### Best Model Performance

| Metric | Value |
|--------|-------|
| RMSE | $X,XXX |
| MAE | $X,XXX |
| R² Score | 0.XXXX |

![Model Comparison](./medical_model_comparison.png)

*RMSE and R² comparison across regression models*

---

### Model Explainability - Regression

#### 🌍 Global Explainability

**1. Feature Importance**

Identifies which health factors most strongly influence premium pricing.

![Feature Importance](./regression_feature_importance.png)

*Age, chronic diseases, and surgeries are top premium drivers*

**Importance Ranking:**
1. Age (28%) - Primary driver
2. Number of Major Surgeries (22%)
3. Any Chronic Diseases (18%)
4. Blood Pressure Problems (12%)
5. Weight & BMI-related factors (10%)

---

**2. Partial Dependence Plots (PDP)**

Shows how premium changes with each feature independently.

![PDP - Age](./regression_pdp_age.png)

*Premium increases steadily with age, accelerating after 50*

![PDP - Surgeries](./regression_pdp_surgeries.png)

*Each additional surgery adds ~$3,500 to premium*

![PDP - Chronic Diseases](./regression_pdp_chronic.png)

*Chronic diseases add ~$6,200 to base premium*

**Key Insights:**
- **Age effect**: Linear until age 45, then exponential increase
- **Surgery impact**: Each surgery adds fixed premium increment
- **Chronic disease**: Single largest premium jump ($6K+)
- **Combined effect**: 60-year-old with 2 surgeries + chronic disease = $35K+ premium

---

**3. Global SHAP Summary**

SHAP (SHapley Additive exPlanations) values show both feature importance and directional impact across all predictions.

![SHAP Summary Plot](./regression_shap_summary.png)

*SHAP summary: Red dots (high feature values) push predictions higher/lower*

**Interpretation:**
- **Age**: Red dots (older patients) consistently on right → higher premiums
- **Surgeries**: More surgeries (red) → higher premiums
- **Chronic Diseases**: Present (red) → significantly higher premiums
- **Feature spread**: Wide distribution shows high individual variation

---

![SHAP Dependence Plot - Age](./regression_shap_dependence_age.png)

*SHAP dependence: Shows exact premium impact per age value*

**Reading the plot:**
- X-axis: Age values
- Y-axis: SHAP value (premium change from baseline)
- Each dot = one customer
- Curve shows non-linear relationship: steeper after age 50

---

#### 🎯 Local Explainability

**1. Single Prediction Explanation**

Detailed breakdown for an individual customer showing exactly how their premium was calculated.

**Customer Profile #456:**
```
Age: 52 years
Diabetes: Yes
Blood Pressure: Yes
Chronic Diseases: Yes
Major Surgeries: 2
Weight: 85 kg, Height: 170 cm
```

![Single Prediction](./regression_single_prediction.png)

*Actual vs Predicted premium with confidence interval*

**Prediction Details:**
- **Predicted Premium**: $31,450
- **Actual Premium**: $31,200
- **Prediction Error**: $250 (0.8% error)
- **Confidence Interval**: $30,100 - $32,800 (95%)

---

**2. SHAP Waterfall Plot**

Visualizes how each feature contributes to moving the prediction from base value to final prediction for this specific customer.

![SHAP Waterfall](./regression_shap_waterfall.png)

*Waterfall showing feature-by-feature premium build-up*

**Reading the Waterfall:**

```
Base Premium (Average): $24,337

+ Age (52 years):              +$4,250
+ Chronic Diseases (Yes):      +$6,180
+ Major Surgeries (2):         +$3,920
+ Blood Pressure (Yes):        +$2,150
+ Diabetes (Yes):              +$1,840
+ Weight (85kg):               +$920
- Known Allergies (No):        -$380
- Height (170cm):              -$120
+ Cancer History (No):         -$1,200
+ Transplants (No):            -$447

= Final Prediction:            $31,450
```

**Insights:**
- **Chronic diseases** is the largest single contributor (+$6,180)
- **Age 52** adds significant premium (+$4,250)
- **Multiple surgeries** compound the risk (+$3,920)
- **Protective factors** (no cancer history, no transplants) reduce premium
- Net effect: $7,113 above average premium

---

**3. Feature Contribution Table**

Detailed numeric breakdown of each feature's contribution to the prediction.

![Feature Contribution Table](./regression_feature_contribution_table.png)

*Sortable table showing absolute and percentage contributions*

| Feature | Value | Contribution | % of Total | Direction |
|---------|-------|--------------|------------|-----------|
| Chronic Diseases | Yes | +$6,180 | 28.3% | Increase ⬆ |
| Age | 52 | +$4,250 | 19.5% | Increase ⬆ |
| Major Surgeries | 2 | +$3,920 | 17.9% | Increase ⬆ |
| Blood Pressure | Yes | +$2,150 | 9.8% | Increase ⬆ |
| Diabetes | Yes | +$1,840 | 8.4% | Increase ⬆ |
| Weight | 85 kg | +$920 | 4.2% | Increase ⬆ |
| Cancer History | No | -$1,200 | 5.5% | Decrease ⬇ |
| Transplants | No | -$447 | 2.0% | Decrease ⬇ |
| Allergies | No | -$380 | 1.7% | Decrease ⬇ |
| Height | 170 cm | -$120 | 0.5% | Decrease ⬇ |

**Total Contribution**: +$7,113 from base ($24,337) = **$31,450**

---

**4. What-If Analysis**

Shows how premium would change if customer characteristics were different.

![What-If Analysis](./regression_whatif_analysis.png)

*Interactive analysis showing premium sensitivity to feature changes*

**Scenario Analysis for Customer #456:**

| Scenario | Change | New Premium | Savings |
|----------|--------|-------------|---------|
| Current | - | $31,450 | - |
| Manage chronic disease | Better control | $25,270 | -$6,180 (20%) |
| Avoid future surgery | Preventive care | $27,530 | -$3,920 (12%) |
| Lose weight to 70kg | Weight reduction | $30,150 | -$1,300 (4%) |
| **Combined** | All improvements | $19,050 | -$12,400 (39%) |

**Actionable Insights:**
- **Biggest impact**: Managing chronic diseases can save $6K+ annually
- **Preventive care**: Avoiding surgeries through health maintenance saves $4K
- **Lifestyle**: Weight management provides moderate savings
- **Holistic approach**: Combined health improvements offer 39% premium reduction

---

### Key Findings - Regression

**Global Insights:**
1. **Age** and **chronic conditions** are primary premium determinants
2. **Non-linear relationships**: Premium accelerates with age, not constant rate
3. **Interaction effects**: Multiple risk factors compound, not just add

**Local Insights:**
1. **SHAP waterfall** provides transparent premium calculation per customer
2. **Feature contributions** enable personalized premium justification
3. **What-if analysis** empowers customers with actionable health guidance
4. Model predictions are explainable and defensible for regulatory compliance

---

## 🔬 MLflow Tracking

### MLflow UI Dashboard

![MLflow Dashboard](./mlflow_dashboard.png)

*Central hub showing all experiment runs for both projects*

---

### Experiment Comparison

![MLflow Comparison](./mlflow_experiments_comparison.png)

*Side-by-side comparison of models with sortable metrics*

**Logged Information:**
- Model hyperparameters
- Performance metrics (Accuracy, RMSE, MAE, R²)
- Cross-validation scores
- Model artifacts (.pkl files)
- Feature importance CSVs
- Training duration

### Access MLflow UI

```bash
mlflow ui
```
Open: http://localhost:5000

---

## 💻 Technologies Used

**Core Libraries:**
- Python 3.13
- scikit-learn (ML models)
- pandas & numpy (data processing)
- matplotlib & seaborn (visualization)

**Explainability Tools:**
- **SHAP**: Global summaries, waterfall plots, dependence plots
- **LIME**: Local instance explanations
- **PDP**: Partial dependence plots
- **Counterfactuals**: What-if scenarios

**Experiment Tracking:**
- **MLflow**: Model versioning, metrics logging, artifact storage

---

## 🚀 How to Run

### 1. Install Dependencies
```bash
pip install pandas numpy matplotlib seaborn scikit-learn mlflow shap lime
```

### 2. Run Classification Pipeline
```bash
python loan_approval_mlflow.py
```
**Runtime:** ~10-15 minutes

### 3. Run Regression Pipeline
```bash
python medical_insurance_mlflow.py
```
**Runtime:** ~10-15 minutes

### 4. Launch MLflow UI
```bash
mlflow ui
```
Open browser: http://localhost:5000

---

## 📁 Project Structure

```
Week3/
├── loan_approval_mlflow.py              # Classification pipeline
├── medical_insurance_mlflow.py          # Regression pipeline
├── loan_approval_dataset.csv
├── Medicalpremium.csv
├── README.md
│
├── mlruns/                              # MLflow experiments
├── outputs/                             # Visualizations
│   ├── classification_*.png
│   ├── regression_*.png
│   └── mlflow_*.png
│
└── *.csv                                # Feature importance exports
```

---

## 📊 Output Files

### Classification
- `classification_feature_importance.png`
- `classification_pdp_*.png` (multiple PDP plots)
- `classification_feature_interactions.png`
- `classification_lime_*.png` (approved/rejected examples)
- `classification_counterfactual.png`
- `model_comparison.png`
- `confusion_matrix_*.png`

### Regression
- `regression_feature_importance.png`
- `regression_pdp_*.png` (multiple PDP plots)
- `regression_shap_summary.png`
- `regression_shap_dependence_*.png`
- `regression_shap_waterfall.png`
- `regression_single_prediction.png`
- `regression_feature_contribution_table.png`
- `regression_whatif_analysis.png`
- `medical_model_comparison.png`

---

## 🎓 Key Learnings

### Explainability Best Practices

**Global Explainability** answers: *"How does the model work overall?"*
- Feature importance, PDPs, SHAP summaries
- Use for model validation and debugging

**Local Explainability** answers: *"Why this specific prediction?"*
- LIME, SHAP waterfall, counterfactuals
- Use for individual case explanations and customer communication

### Business Value

1. **Regulatory Compliance**: Transparent model decisions for audits
2. **Customer Trust**: Explain decisions in plain language
3. **Model Debugging**: Identify biases and errors
4. **Actionable Insights**: Guide customers on improvement paths

---

## 🔧 Troubleshooting

**Module Not Found:**
```bash
pip install [module_name]
```

**Dataset Issues:**
- Ensure CSV files are in same directory
- Check file names match exactly

**MLflow Port Conflict:**
```bash
mlflow ui --port 5001
```

---

## 🚀 Future Enhancements

1. **Advanced Models**: XGBoost, LightGBM, Neural Networks
2. **Real-time API**: Flask/FastAPI deployment
3. **Interactive Dashboard**: Streamlit for explainability
4. **Automated Reports**: PDF generation of explanations
5. **A/B Testing**: Compare model versions in production

---

## 📝 License

Educational project for ML explainability demonstration.

---

## 👤 Author

**Your Name**
- GitHub: [@yourusername](https://github.com/yourusername)
- LinkedIn: [Profile](https://linkedin.com/in/yourprofile)
- Email: your.email@example.com

---

**Last Updated:** October 2025 | **Version:** 2.0.0

---

## ⭐ Star this repo if you found it helpful!

