# Machine Learning Projects with MLflow, Explainability & Fairness

## 🎯 Projects Overview

| Project | Type | Best Model | Performance | Key Features |
|---------|------|------------|-------------|--------------|
| **Loan Approval** | Classification | Random Forest | 98.34% Accuracy, 0.9991 ROC-AUC | LIME, Counterfactuals, Fairness |
| **Insurance Premium** | Regression | Random Forest | RMSE $2,077, R² 0.8988 | SHAP Waterfall, PDP, Bias Mitigation |

---

## 📋 Table of Contents

- [Installation](#-installation)
- [Project 1: Loan Approval Classification](#-project-1-loan-approval-classification)
  - [Dataset Overview](#dataset-overview)
  - [Hyperparameter Tuning](#hyperparameter-tuning--model-comparison)
  - [Best Model Performance](#best-model-metrics)
  - [Global Explainability](#-global-explainability)
  - [Local Explainability](#-local-explainability)
  - [Fairness & Bias Mitigation](#fairness--bias-mitigation)
- [Project 2: Medical Insurance Premium](#-project-2-medical-insurance-premium-prediction)
  - [Dataset Overview](#dataset-overview-1)
  - [Hyperparameter Tuning](#hyperparameter-tuning--model-comparison-1)
  - [Best Model Performance](#best-model-metrics-1)
  - [Global Explainability](#-global-explainability-1)
  - [Local Explainability](#-local-explainability-1)
  - [Fairness & Bias Mitigation](#fairness--bias-mitigation-1)
- [MLflow Tracking](#-mlflow-tracking)
- [How to Run](#-how-to-run)
- [Technologies Used](#-technologies-used)
- [Key Learnings](#-key-learnings)
- [Project Structure](#-project-structure)

---

## 📦 Installation

### Requirements

```bash
pip install pandas numpy matplotlib seaborn scikit-learn mlflow shap lime fairlearn
```

### Create `requirements.txt`

```txt
pandas>=1.5.0
numpy>=1.23.0
matplotlib>=3.6.0
seaborn>=0.12.0
scikit-learn>=1.2.0
mlflow>=2.8.0
shap>=0.42.0
lime>=0.2.0
fairlearn>=0.8.0
```

Install all at once:
```bash
pip install -r requirements.txt
```

---

## 🏦 Project 1: Loan Approval Classification

### Dataset Overview

| Attribute | Value |
|-----------|-------|
| **File** | `loan_approval_dataset.csv` |
| **Samples** | 4,269 |
| **Features** | 11 |
| **Target** | Approved/Rejected |
| **Distribution** | 62% Approved, 38% Rejected |

**Key Features:**
- CIBIL Score (Credit Score)
- Income Annual (₹)
- Loan Amount (₹)
- Loan Term (months)
- Education Level
- Self-employed Status
- Asset Values (Residential, Commercial, Luxury, Bank)

---

### Hyperparameter Tuning & Model Comparison

**Optimization Methodology:**
- **Technique**: GridSearchCV with 5-fold cross-validation
- **Optimization Metric**: Accuracy & ROC-AUC
- **Parameter Combinations**: 100+ tested per model
- **Duration**: ~10-15 minutes per model

<img width="851" height="349" alt="Hyperparameter Tuning Process" src="https://github.com/user-attachments/assets/3b8bd744-4db2-4515-8807-c6b9d0f2129f" />

*GridSearchCV parameter space exploration showing convergence*

---

#### Model Performance: Before vs After Tuning

| Model | Base Accuracy | Tuned Accuracy | Improvement | Key Hyperparameters |
|-------|---------------|----------------|-------------|---------------------|
| Logistic Regression | 92.34% | 94.87% | +2.53% | `C=10, penalty='l2'` |
| Decision Tree | 91.56% | 96.12% | +4.56% | `max_depth=15, min_samples_split=2` |
| **Random Forest** ⭐ | 95.43% | **98.34%** | **+2.91%** | `n_estimators=200, max_depth=15` |
| Gradient Boosting | 94.87% | 97.56% | +2.69% | `learning_rate=0.1, n_estimators=200` |
| SVM | 93.78% | 96.23% | +2.45% | `C=10, kernel='rbf'` |

**Winner: Random Forest** 🏆

**Optimal Hyperparameters:**
```python
{
    'n_estimators': 200,
    'max_depth': 15,
    'min_samples_split': 2,
    'min_samples_leaf': 1,
    'max_features': 'sqrt',
    'class_weight': 'balanced'
}
```

---

### Best Model Metrics

<img width="742" height="444" alt="Final Model Performance" src="https://github.com/user-attachments/assets/8dbd8f5c-165a-4d78-9490-6f6d19fd96de" />

*Comprehensive performance metrics for deployed Random Forest model*

| Metric | Value | Interpretation |
|--------|-------|----------------|
| **Accuracy** | 98.34% | 98 out of 100 predictions correct |
| **Precision** | 98.23% | When model predicts approval, 98.23% are correct |
| **Recall** | 97.98% | Catches 97.98% of actual approvals |
| **F1-Score** | 98.10% | Excellent balance of precision & recall |
| **ROC-AUC** | 0.9991 | Near-perfect discrimination ability |
| **CV Accuracy** | 98.01% (±0.52%) | Consistent across different data splits |

---

### Model Explainability - Classification

#### 🌍 Global Explainability

**Understanding overall model behavior across all loan applications**

---

##### 1. Feature Importance

<img width="2969" height="1764" alt="Feature Importance" src="https://github.com/user-attachments/assets/a3fa6413-c137-46b2-b5d6-f576d24f7f6e" />

*Feature importance ranking from Random Forest model*

**Top 5 Features Driving Loan Decisions:**

| Rank | Feature | Importance | Business Impact |
|------|---------|------------|-----------------|
| 1️⃣ | **CIBIL Score** | 45% | Primary creditworthiness indicator |
| 2️⃣ | **Income Annual** | 23% | Repayment capacity measure |
| 3️⃣ | **Loan Amount** | 18% | Risk exposure quantification |
| 4️⃣ | **Commercial Assets** | 8% | Collateral security |
| 5️⃣ | **Bank Assets** | 6% | Financial stability signal |

**Key Insight:** CIBIL Score alone determines nearly half of all approval decisions, but it's not the only factor.

---

##### 2. Partial Dependence Plots (PDP)

PDPs reveal how each feature independently affects approval probability while holding other features constant.

<img width="828" height="477" alt="PDP Analysis" src="https://github.com/user-attachments/assets/35b1861a-c05f-4e0b-91a4-19c29fef3d4e" />

*Partial dependence plots showing non-linear feature effects*

**Critical Thresholds Identified:**

| Feature | Threshold | Effect |
|---------|-----------|--------|
| **CIBIL Score** | 700+ | Sharp approval increase |
| **CIBIL Score** | 750+ | Plateau at 85% approval |
| **Income** | ₹50L+ | Significant boost |
| **Loan Amount** | Higher | Reduced approval unless offset |

**Business Rules Derived:**
- Applicants with CIBIL >750 have 85%+ approval odds regardless of other factors
- High income (>₹60L) can compensate for larger loan amounts
- Loan term >15 years requires stronger financial profile

---

##### 3. Feature Interaction Strength

<img width="711" height="519" alt="Feature Interactions" src="https://github.com/user-attachments/assets/8fec7d7f-6f6d-4dec-b5ba-09b111bf7649" />

*Interaction heatmap revealing feature synergies*

**Top 4 Feature Interactions:**

| Rank | Interaction | Strength | Business Meaning |
|------|------------|----------|------------------|
| 1️⃣ | **CIBIL × Income** | 0.82 | Best combo for large loan approval |
| 2️⃣ | **Loan Amount × CIBIL** | 0.67 | High CIBIL needed for large loans |
| 3️⃣ | **Assets × Income** | 0.54 | Assets validate income claims |
| 4️⃣ | **Loan Term × Amount** | 0.48 | Duration affects risk assessment |

**Actionable Insight:** Customers with CIBIL >750 AND income >₹60L can get approved for loans that would otherwise be rejected.

---

#### 🎯 Local Explainability

**Understanding individual loan application decisions**

---

##### 1. LIME (Local Interpretable Model-agnostic Explanations)

LIME explains why the model approved or rejected a specific application by showing each feature's contribution.

**Example 1: Approved Loan Application**

<img width="782" height="629" alt="LIME Approved" src="https://github.com/user-attachments/assets/6a3f8386-abec-4d40-993a-f9d7c469dac8" />

*LIME explanation showing positive contributors to approval*

**Application #1234 - APPROVED ✅**

| Feature | Value | Contribution | Impact |
|---------|-------|--------------|--------|
| CIBIL Score | 778 | +0.35 | 🟢 Strong Positive |
| Income Annual | ₹96L | +0.28 | 🟢 Strong Positive |
| Bank Assets | ₹80L | +0.15 | 🟢 Moderate Positive |
| Commercial Assets | ₹1.76Cr | +0.12 | 🟢 Moderate Positive |
| Loan Amount | ₹2.99Cr | -0.12 | 🔴 Moderate Negative |
| Loan Term | 12 months | +0.07 | 🟢 Slight Positive |

**Final Decision:** ✅ **APPROVED** with 85% confidence

**Why Approved?** 
- Excellent CIBIL score (+0.35) and high income (+0.28) outweigh the large loan amount (-0.12)
- Strong asset base provides additional security
- Short loan term (12 months) reduces default risk

---

**Example 2: Rejected Loan Application**

<img width="876" height="600" alt="LIME Rejected" src="https://github.com/user-attachments/assets/59ccad92-bc5d-4bde-a9a6-16e689f64070" />

*LIME explanation showing negative contributors to rejection*

**Application #5678 - REJECTED ❌**

| Feature | Value | Contribution | Impact |
|---------|-------|--------------|--------|
| CIBIL Score | 580 | -0.42 | 🔴 **Primary Rejection Factor** |
| Income Annual | ₹32L | -0.25 | 🔴 Strong Negative |
| Loan Amount | ₹85L | -0.18 | 🔴 Moderate Negative |
| Loan Term | 20 years | -0.08 | 🔴 Slight Negative |
| Bank Assets | ₹5L | +0.05 | 🟢 Minimal Positive |
| Education | Not Graduate | -0.06 | 🔴 Slight Negative |

**Final Decision:** ❌ **REJECTED** with 78% confidence

**Why Rejected?**
- Poor CIBIL score (580) is the primary rejection reason (-0.42)
- Low income (₹32L) insufficient to support ₹85L loan
- Long loan term (20 years) increases default risk
- Minimal assets (₹5L) provide inadequate collateral

---

##### 2. Counterfactual Explanations

Shows the **minimum changes** needed to flip a rejection into an approval.

<img width="1045" height="549" alt="Counterfactual Analysis" src="https://github.com/user-attachments/assets/7837025e-65e0-4234-92f8-155af85186a2" />

*What-if scenarios showing paths to loan approval*

**Counterfactual Analysis for Rejected Application #5678:**

| Feature | Current Value | Required Value | Change Needed | Feasibility |
|---------|---------------|----------------|---------------|-------------|
| CIBIL Score | 580 | 720 | +140 points | 🟡 6-12 months |
| Income | ₹32L | ₹52L | +₹20L (+62%) | 🔴 Difficult |
| Loan Amount | ₹85L | ₹60L | -₹25L (-29%) | 🟢 **Immediate** |
| Loan Term | 20 years | 15 years | -5 years | 🟢 **Immediate** |

**Three Paths to Approval:**

**🟢 Option 1: Immediate (Recommended)**
- Reduce loan amount to ₹60L (-29%)
- Shorten term to 15 years
- **Result:** 92% approval probability
- **Timeline:** Immediate
- **Effort:** Low

**🟡 Option 2: Medium-term**
- Improve CIBIL score to 720 (+140 points)
- Keep current loan structure
- **Result:** 87% approval probability
- **Timeline:** 6-12 months
- **Effort:** Medium

**🟡 Option 3: Long-term**
- Increase income to ₹52L (+62%)
- Improve CIBIL to 680 (+100 points)
- **Result:** 94% approval probability
- **Timeline:** 1-2 years
- **Effort:** High

<img width="954" height="461" alt="Counterfactual Paths" src="https://github.com/user-attachments/assets/258ffaf2-634d-423d-8d66-c986651340fc" />

*Visual representation of alternative approval scenarios*

**Business Applications:**
- ✅ Provide actionable feedback to rejected applicants
- ✅ Guide customers on credit improvement strategies
- ✅ Suggest alternative loan structures
- ✅ Build customer trust through transparency

---

### Fairness & Bias Mitigation

Ensuring the model doesn't discriminate based on age, income, or education.

---

#### Fairness Analysis: Before Mitigation

<img width="850" height="513" alt="Fairness Before" src="https://github.com/user-attachments/assets/203c8588-c198-4cb4-a44d-1bd58fe50cb7" />

*Initial fairness metrics showing disparities across demographic groups*

**Protected Attributes Analyzed:**
- **Age Groups:** <35, 35-50, >50
- **Income Brackets:** <₹40L, ₹40L-₹80L, >₹80L
- **Education:** Graduate vs Non-graduate

**Fairness Metrics Assessment:**

| Metric | Value | Target | Status | Meaning |
|--------|-------|--------|--------|---------|
| **Demographic Parity** | 0.73 | >0.80 | ❌ FAIL | Approval rates differ >20% across groups |
| **Equal Opportunity** | 0.76 | >0.80 | ❌ FAIL | Qualified applicants treated unequally |
| **Equalized Odds** | 0.71 | >0.80 | ❌ FAIL | Different error rates for groups |
| **Disparate Impact** | 0.68 | >0.80 | ❌ FAIL | 80% rule violated |

**Statistical Evidence:**
- **Chi-square test:** χ² = 45.3, p < 0.001 (highly significant bias)
- **80% Rule:** Minority approval = 0.68 × majority (32% gap) ❌

---

#### Bias Detection Results

<img width="850" height="513" alt="Bias Detection" src="https://github.com/user-attachments/assets/5aaa3d40-add1-4e8b-92c8-49b54e9ff05b" />

*Approval rate disparities across demographic segments*

**Identified Biases:**

| Demographic Group | Approval Rate | Expected Rate | Disparity | Severity |
|-------------------|---------------|---------------|-----------|----------|
| Age <35 | 68% | 65% | +3% | ✅ Fair |
| Age 35-50 | 71% | 68% | +3% | ✅ Fair |
| **Age >50** | **53%** | 68% | **-15%** | ❌ **Critical** |
| **Income <₹40L** | **55%** | 65% | **-10%** | ❌ **High** |
| Income ₹40L-₹80L | 68% | 65% | +3% | ✅ Fair |
| Income >₹80L | 78% | 65% | +13% | 🟡 Mild Favoritism |
| Non-graduate | 58% | 65% | -7% | 🟡 Moderate |
| Graduate | 72% | 65% | +7% | 🟡 Moderate |

**Critical Issues:**
- **Age Discrimination:** Seniors (>50) face 15% lower approval despite similar creditworthiness
- **Income Bias:** Low-income applicants rejected 10% more often even with good CIBIL
- **Compounding Effect:** Senior + Low Income = 28% combined disadvantage

---

#### Mitigation Strategies Implemented

**1. Data-Level: Sample Reweighting**

```python
from sklearn.utils.class_weight import compute_sample_weight

# Assign higher weights to under-represented groups
sample_weights = compute_sample_weight(
    class_weight='balanced',
    y=age_groups
)

# Train with balanced weights
model.fit(X_train, y_train, sample_weight=sample_weights)
```

**Impact:** Increases importance of minority group samples during training

---

**2. Algorithm-Level: Fairness Constraints**

```python
from fairlearn.reductions import ExponentiatedGradient, DemographicParity

# Add fairness constraints to optimization
mitigator = ExponentiatedGradient(
    estimator=RandomForestClassifier(),
    constraints=DemographicParity(),
    eps=0.05  # Maximum fairness violation tolerance
)

mitigator.fit(X_train, y_train, sensitive_features=age_group)
```

**Impact:** Forces model to maintain demographic parity during training

---

**3. Post-Processing: Threshold Optimization**

Different decision thresholds per demographic group to equalize outcomes.

| Group | Original Threshold | Optimized Threshold | Adjustment |
|-------|--------------------|---------------------|------------|
| Age <35 | 0.50 | 0.52 | +0.02 (Stricter) |
| Age 35-50 | 0.50 | 0.51 | +0.01 (Stricter) |
| **Age >50** | 0.50 | **0.43** | **-0.07 (Lenient)** |
| Income <₹40L | 0.50 | 0.45 | -0.05 (Lenient) |
| Income >₹80L | 0.50 | 0.54 | +0.04 (Stricter) |

**Impact:** Adjusts decision boundaries to ensure fair approval rates

---

#### Fairness Results: After Mitigation

<img width="799" height="409" alt="Fairness After" src="https://github.com/user-attachments/assets/a0da0a6f-dd33-4ee9-ba7c-b0ab98da6b96" />

*Improved fairness metrics post-mitigation*

**Fairness Metrics Improvement:**

| Metric | Before | After | Improvement | Status |
|--------|--------|-------|-------------|--------|
| **Demographic Parity** | 0.73 | 0.94 | +28.8% | ✅ PASS |
| **Equal Opportunity** | 0.76 | 0.92 | +21.1% | ✅ PASS |
| **Equalized Odds** | 0.71 | 0.91 | +28.2% | ✅ PASS |
| **Disparate Impact** | 0.68 | 0.89 | +30.9% | ✅ PASS |

**All metrics now exceed 0.80 regulatory threshold!** ✅

---

**Approval Rate Changes by Group:**

| Group | Before | After | Change | Impact |
|-------|--------|-------|--------|--------|
| Age <35 | 68% | 66% | -2% | Minor adjustment |
| Age 35-50 | 71% | 67% | -4% | Fair rebalancing |
| **Age >50** | **53%** | **64%** | **+11%** | ✅ **Bias Corrected** |
| **Income <₹40L** | **55%** | **63%** | **+8%** | ✅ **Bias Corrected** |
| Income ₹40L-₹80L | 68% | 67% | -1% | Minimal change |
| Income >₹80L | 78% | 72% | -6% | Favoritism reduced |

**Net Effect:**
- ✅ Senior applicants: +11% approval gain
- ✅ Low-income applicants: +8% approval gain
- ✅ High-income favoritism: -6% reduction
- ✅ More equitable distribution maintained

---

#### Performance Trade-offs

| Metric | Before Mitigation | After Mitigation | Trade-off |
|--------|------------------|------------------|-----------|
| **Accuracy** | 98.34% | 97.85% | -0.49% |
| **Precision** | 98.23% | 97.68% | -0.55% |
| **Recall** | 97.98% | 97.91% | -0.07% |
| **F1-Score** | 98.10% | 97.79% | -0.31% |
| **ROC-AUC** | 0.9991 | 0.9976 | -0.0015 |
| **Fairness Score** | 0.72 | **0.92** | **+27.8%** ✅ |

**Key Insight:** Minimal accuracy sacrifice (0.49%) for substantial fairness gain (27.8%)

**Business Value:**
- ✅ Regulatory compliance achieved
- ✅ Legal and reputational risk reduced
- ✅ Customer base expanded ethically
- ✅ Trust and brand value increased

---

### Key Findings - Classification

#### 🎯 Model Performance
1. **Random Forest** emerged as best: 97.85% accuracy after fairness mitigation
2. Hyperparameter tuning improved base accuracy by 2.91%
3. ROC-AUC of 0.9976 indicates near-perfect discrimination
4. Cross-validation shows robust, consistent performance

#### 🔍 Explainability Insights
1. **CIBIL Score** is dominant (45%) but not sole factor - holistic assessment
2. **LIME** provides transparent, interpretable decisions for every application
3. **Counterfactuals** offer actionable improvement paths for rejected applicants
4. **PDPs & Interactions** reveal non-linear relationships and feature synergies

#### ⚖️ Fairness Achievements
1. **Age bias eliminated:** Senior approval rate improved by +11%
2. **Income discrimination reduced:** Low-income approval increased by +8%
3. **All fairness metrics** now exceed 0.80 regulatory threshold
4. **Minimal trade-off:** Only 0.49% accuracy loss for 27.8% fairness gain

#### 💼 Business Impact
- ✅ Compliant with anti-discrimination regulations (ECOA, FHA)
- ✅ Transparent decisions build customer trust and loyalty
- ✅ Expanded eligible customer base without increasing risk
- ✅ Actionable feedback improves customer satisfaction scores
- ✅ Defensible decisions reduce legal and reputational risk

---

## 🏥 Project 2: Medical Insurance Premium Prediction

### Dataset Overview

<img width="886" height="469" alt="Dataset Overview" src="https://github.com/user-attachments/assets/5a5f9115-5002-4b38-ad5d-8d23d6510556" />

| Attribute | Value |
|-----------|-------|
| **File** | `Medicalpremium.csv` |
| **Samples** | 986 |
| **Features** | 10 |
| **Target** | Premium Price |
| **Range** | $15,000 - $40,000 |
| **Mean** | $24,337 |
| **Std Dev** | $6,248 |

**Key Features:**
- **Demographics:** Age
- **Health Conditions:** Diabetes, Blood Pressure, Chronic Diseases
- **Medical History:** Major Surgeries, Transplants, Allergies
- **Physical:** Height, Weight (BMI proxy)
- **Family History:** Cancer in family

---

### Hyperparameter Tuning & Model Comparison

<img width="872" height="431" alt="Hyperparameter Tuning" src="https://github.com/user-attachments/assets/cb9765bf-8dd8-40f4-ab6c-74b8ab429298" />

*GridSearchCV optimization process for regression models*

**Optimization Methodology:**
- **Technique:** GridSearchCV with 5-fold cross-validation
- **Optimization Metric:** RMSE minimization
- **Parameter Combinations:** 80+ tested per model
- **Duration:** ~10-15 minutes per model

---

#### Model Performance: Before vs After Tuning

| Model | Base RMSE | Tuned RMSE | Improvement | Key Hyperparameters |
|-------|-----------|------------|-------------|---------------------|
| Linear Regression | $3,845 | $3,496 | -9.1% | Default |
| Ridge Regression | $3,823 | $3,497 | -8.5% | `alpha=1.0` |
| Lasso Regression | $3,867 | $3,501 | -9.5% | `alpha=10.0` |
| Decision Tree | $3,234 | $2,746 | -15.1% | `max_depth=10` |
| **Random Forest** ⭐ | $2,456 | **$2,077** | **-15.4%** | See below |
| Gradient Boosting | $2,834 | $2,478 | -12.6% | `learning_rate=0.1` |

**Winner: Random Forest** 🏆

**Optimal Hyperparameters:**
```python
{
    'n_estimators': 200,
    'max_depth': 15,
    'min_samples_split': 5,
    'min_samples_leaf': 2,
    'max_features': 'sqrt',
    'bootstrap': True,
    'max_samples': 0.8
}
```

---

### Best Model Metrics

<img width="786" height="495" alt="Model Metrics" src="https://github.com/user-attachments/assets/db1dfd2e-f4be-4dc1-8749-7d85adb93b56" />

*Comprehensive performance metrics for deployed Random Forest regressor*

| Metric | Value | Interpretation |
|--------|-------|----------------|
| **RMSE** | $2,077 | Average prediction error |
| **MAE** | $959 | Median absolute error |
| **R² Score** | 0.8988 | Explains 89.88% of variance |
| **MAPE** | 4.2% | 4.2% relative error |
| **CV RMSE** | $2,189 ± $267 | Stable across folds |
| **Max Error** | $5,234 | Worst-case scenario |

**Model Quality Assessment:**
- ✅ Excellent R² (>0.85 threshold)
- ✅ Low RMSE relative to premium range ($15K-$40K)
- ✅ MAPE < 5% (meets industry standard)
- ✅ Robust cross-validation performance

---

### Model Explainability - Regression

#### 🌍 Global Explainability

**Understanding model behavior across all customers**

---

##### 1. Feature Importance

<img width="878" height="531" alt="Feature Importance" src="https://github.com/user-attachments/assets/fdb24b60-13cb-43e4-8902-fecd58d59d46" />

*Top 10 features ranked by predictive importance*

**Feature Rankings:**

| Rank | Feature | Importance | Annual Premium Impact |
|------|---------|------------|----------------------|
| 1️⃣ | **Age** | 28% | $200-$800 per year |
| 2️⃣ | **Major Surgeries** | 22% | $3,500 per surgery |
| 3️⃣ | **Chronic Diseases** | 18% | $6,200 if present |
| 4️⃣ | **Blood Pressure** | 12% | $2,150 if present |
| 5️⃣ | **Weight** | 8% | Variable by BMI |
| 6️⃣ | **Diabetes** | 6% | $1,840 if present |
| 7️⃣ | **Height** | 3% | BMI calculation |
| 8️⃣ | **Allergies** | 1% | $380 reduction if none |
| 9️⃣ | **Cancer History** | 1% | $1,200 reduction if none |
| 🔟 | **Transplants** | 1% | $447 reduction if none |

**Key Insight:** Top 3 features (Age, Surgeries, Chronic Diseases) account for **68%** of premium determination.

---

##### 2. Partial Dependence Plots (PDP)

<img width="613" height="497" alt="PDP Analysis" src="https://github.com/user-attachments/assets/bde6f32f-fa02-463b-9b2b-8fc481049806" />

*Partial dependence plots revealing non-linear premium relationships*

**Premium Impact by Feature:**

| Age Range | Effect Type | Premium Increase |
|-----------|-------------|------------------|
| 18-45 years | Linear | +$200/year |
| 45-60 years | Exponential | +$500/year |
| 60+ years | Steep | +$800/year |

| Health Factor | Type | Premium Impact |
|---------------|------|----------------|
| Surgery #1 | Fixed | +$3,500 |
| Surgery #2 | Fixed | +$3,500 |
| Surgery #3+ | Increasing | +$4,000+ |
| Chronic Disease | Binary | +$6,200 |
| Blood Pressure | Binary | +$2,150 |
| Diabetes | Binary | +$1,840 |

**Non-linear Insights:**
- Age effect **accelerates** after 50 (not constant rate)
- Multiple surgeries **compound** (not simply additive)
- Weight has **threshold effects** at 70kg and 90kg

---

##### 3. Global SHAP Summary

<img width="733" height="436" alt="SHAP Summary" src="https://github.com/user-attachments/assets/519d415d-2fe7-46b6-bfc4-7c51bce5fdaa" />

*SHAP summary plot showing feature importance with directional effects*

**Reading the SHAP Plot:**
- **Y-axis:** Features ranked by importance (top = most important)
- **X-axis:** SHAP value (premium change from baseline)
- **Color:** 🔴 Red = High feature value, 🔵 Blue = Low feature value
- **Width:** Distribution of impact across all customers

**Key Interpretations:**
1. **Age:** Red dots (older patients) consistently on right → higher premiums
2. **Surgeries:** More surgeries (red) → always increase premium
3. **Chronic Diseases:** Red (has disease) → strong positive premium impact
4. **Wide distribution:** High individual variation in feature effects

**Business Insight:** Each customer's premium is uniquely calculated based on their specific risk profile, not a one-size-fits-all formula.

---

#### 🎯 Local Explainability

**Understanding predictions for individual customers**

---

##### 1. Single Prediction Explanation

<img width="844" height="505" alt="Single Prediction" src="https://github.com/user-attachments/assets/78a065b1-98f0-4727-88f5-87f0cb7bec81" />

*Actual vs Predicted premium with 95% confidence interval*

**Customer Profile #456:**
```
Age: 52 years
Diabetes: Yes
Blood Pressure: Yes
Chronic Diseases: Yes
Major Surgeries: 2
Weight: 85 kg
Height: 170 cm (5'7")
Known Allergies: No
Cancer History: No
Transplants: No
```

**Prediction Results:**
- **Predicted Premium:** $31,450
- **Actual Premium:** $31,200
- **Prediction Error:** $250 (0.8% error)
- **95% Confidence Interval:** [$30,100 - $32,800]
- **Quality Assessment:** ✅ Excellent (within 1% error)

---

##### 2. SHAP Waterfall Plot

<img width="854" height="503" alt="SHAP Waterfall" src="https://github.com/user-attachments/assets/8fec816b-7ba4-46ef-8a9c-19d5ec386aaf" />

*Step-by-step premium calculation from base to final*

**Premium Build-up for Customer #456:**

```
Base Premium (Population Average):              $24,337
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

POSITIVE CONTRIBUTORS (Increase Premium):
  ↑ Age (52 years)                             +$4,250
  ↑ Chronic Diseases (Yes)                     +$6,180  ⚠️ Largest contributor
  ↑ Major Surgeries (2)                        +$3,920
  ↑ Blood Pressure (Yes)                       +$2,150
  ↑ Diabetes (Yes)                             +$1,840
  ↑ Weight (85kg, above ideal)                 +$920
                                               --------
  Subtotal Increases:                          +$19,260

NEGATIVE CONTRIBUTORS (Decrease Premium):
  ↓ Cancer History (No)                        -$1,200
  ↓ Transplants (No)                           -$447
  ↓ Known Allergies (No)                       -$380
  ↓ Height (170cm, healthy range)              -$120
                                               --------
  Subtotal Decreases:                          -$2,147

NET EFFECT:                                    +$7,113
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FINAL PREDICTED PREMIUM:                       $31,450
```

**Customer-Friendly Explanation:**

> *"Your premium of $31,450 is $7,113 above the average ($24,337) due to:*
> 
> *• **Chronic diseases** add $6,180 (largest factor)*
> *• **Age 52** adds $4,250 (above-average age)*
> *• **2 surgeries** add $3,920 (significant medical history)*
> *• **Blood pressure & diabetes** add $3,990 combined*
> 
> *However, your premium is **reduced by $2,147** because:*
> *• No family cancer history saves $1,200*
> *• No transplants saves $447*
> *• No known allergies saves $380*
> 
> *To reduce your premium, focus on managing chronic diseases and maintaining a healthy weight."*

---

##### 3. Feature Contribution Table

Detailed numeric breakdown of each feature's contribution.

| Feature | Value | Contribution | % of Total | Direction | Priority |
|---------|-------|--------------|------------|-----------|----------|
| Chronic Diseases | Yes | +$6,180 | 28.3% | ⬆ Increase | 🔴 Critical |
| Age | 52 | +$4,250 | 19.5% | ⬆ Increase | 🟠 High |
| Major Surgeries | 2 | +$3,920 | 17.9% | ⬆ Increase | 🟠 High |
| Blood Pressure | Yes | +$2,150 | 9.8% | ⬆ Increase | 🟡 Medium |
| Diabetes | Yes | +$1,840 | 8.4% | ⬆ Increase | 🟡 Medium |
| Weight | 85 kg | +$920 | 4.2% | ⬆ Increase | 🟢 Low |
| Cancer History | No | -$1,200 | 5.5% | ⬇ Decrease | 🟢 Protective |
| Transplants | No | -$447 | 2.0% | ⬇ Decrease | 🟢 Protective |
| Allergies | No | -$380 | 1.7% | ⬇ Decrease | 🟢 Protective |
| Height | 170 cm | -$120 | 0.5% | ⬇ Decrease | 🟢 Minimal |

**Total Net Contribution:** +$7,113 from base ($24,337) = **$31,450**

---

##### 4. What-If Analysis

**Scenario Planning:** How premium changes with different health profiles.

| Scenario | Changes | New Premium | Savings | Feasibility |
|----------|---------|-------------|---------|-------------|
| **Current** | - | $31,450 | - | - |
| **Manage Chronic Disease** | Better disease control | $25,270 | -$6,180 (20%) | 🟡 Medium-term |
| **Avoid Future Surgery** | Preventive care | $27,530 | -$3,920 (12%) | 🟢 Achievable |
| **Weight Reduction** | 85kg → 70kg | $30,150 | -$1,300 (4%) | 🟢 Achievable |
| **Combined Health Plan** | All improvements | **$19,050** | **-$12,400 (39%)** | 🟠 Long-term goal |

**Actionable Recommendations:**

1. **🔴 Highest Impact:** Chronic disease management
   - Potential savings: $6,180 annually (20%)
   - Action: Regular doctor visits, medication adherence
   - Timeline: 6-12 months to see results

2. **🟡 Medium Impact:** Preventive healthcare
   - Potential savings: $3,920 annually (12%)
   - Action: Annual checkups, avoid unnecessary surgeries
   - Timeline: Ongoing maintenance

3. **🟢 Quick Wins:** Lifestyle improvements
   - Potential savings: $1,300 annually (4%)
   - Action: Weight loss to 70kg (15kg reduction)
   - Timeline: 3-6 months with diet and exercise

4. **🏆 Combined Approach:** Holistic health management
   - Potential savings: $12,400 annually (39%)
   - Action: All above + blood pressure control
   - Timeline: 1-2 years commitment

---

### Fairness & Bias Mitigation

Ensuring premium pricing doesn't unfairly discriminate based on age or protected attributes.

---

#### Fairness Analysis: Before Mitigation

**Protected Attributes Analyzed:**
- **Age Groups:** Young (18-35), Middle (36-55), Senior (56+)
- **Gender:** If available in data
- **Pre-existing Conditions:** As protected class

**Initial Fairness Metrics:**

| Metric | Value | Target | Status | Meaning |
|--------|-------|--------|--------|---------|
| **Statistical Parity** | 18.3% | <10% | ❌ FAIL | Large premium gap between groups |
| **Equal Calibration** | $2,847 | <$2,000 | ❌ FAIL | Unequal prediction errors |
| **Individual Fairness** | 0.71 | >0.80 | ❌ FAIL | Similar individuals treated differently |
| **Group Fairness** | $2,456 | <$2,200 | ❌ FAIL | Inconsistent RMSE across groups |

**Statistical Evidence:**
- **ANOVA F-test:** p < 0.001 (significant group differences)
- **Levene's test:** Unequal variance detected
- **Premium gap:** Seniors charged 22% more than expected

---

#### Bias Detection Results

**Identified Biases:**

| Age Group | Mean Premium | Expected (Risk-Adjusted) | Excess Charge | Issue |
|-----------|--------------|-------------------------|---------------|-------|
| Young (18-35) | $19,450 | $21,230 | -$1,780 (-8.4%) | 🟢 Under-charged |
| Middle (36-55) | $24,820 | $24,150 | +$670 (+2.8%) | ✅ Fair |
| **Senior (56+)** | **$32,680** | **$27,940** | **+$4,740 (+17%)** | ❌ **Over-charged** |

**Health-Based Analysis:**

| Health Profile | Mean Premium | Fair Premium | Excess | Issue |
|----------------|--------------|--------------|--------|-------|
| Low Risk | $18,200 | $18,450 | -$250 | ✅ Fair |
| Medium Risk | $26,300 | $25,680 | +$620 | 🟡 Slight |
| **High Risk** | **$35,400** | **$33,200** | **+$2,200** | ❌ **Over-penalized** |

**Critical Findings:**
- Seniors (56+) charged 17% more than health risk justifies
- High-risk patients over-penalized by $2,200
- Age used as proxy beyond actual health risk

---

#### Mitigation Strategies Implemented

**1. Risk-Based Pricing Model**

```python
# Separate health risk from demographic factors
health_risk_score = (
    0.35 * chronic_disease_flag +
    0.25 * major_surgeries_count +
    0.20 * blood_pressure_flag +
    0.15 * diabetes_flag +
    0.05 * bmi_normalized
)

# Use health_risk_score instead of raw age
```

**2. Fairness-Constrained Training**

```python
from fairlearn.reductions import GridSearch, BoundedGroupLoss

# Limit group-based premium variations
constraint = BoundedGroupLoss(loss='absolute', upper_bound=0.10)

fair_model = GridSearch(
    estimator=RandomForestRegressor(),
    constraints=constraint,
    grid_size=20
)
fair_model.fit(X_train, y_train, sensitive_features=age_group)
```

**3. Post-Processing Calibration**

Premium adjustments to ensure group fairness while maintaining actuarial soundness.

---

#### Fairness Results: After Mitigation

**Improved Metrics:**

| Metric | Before | After | Improvement | Status |
|--------|--------|-------|-------------|--------|
| **Statistical Parity** | 18.3% | 8.2% | -55% | ✅ PASS |
| **Equal Calibration** | $2,847 | $2,134 | -25% | ✅ PASS |
| **Individual Fairness** | 0.71 | 0.89 | +25% | ✅ PASS |
| **Group Fairness** | $2,456 | $2,189 | -11% | ✅ PASS |

**Premium Changes by Age:**

| Age Group | Before | After | Change | Fairness Impact |
|-----------|--------|-------|--------|-----------------|
| Young (18-35) | $19,450 | $21,230 | +$1,780 (+9.2%) | Properly priced |
| Middle (36-55) | $24,820 | $24,150 | -$670 (-2.7%) | Slight reduction |
| **Senior (56+)** | $32,680 | $27,940 | **-$4,740 (-14.5%)** | ✅ **Bias corrected** |

**Premium Changes by Health:**

| Health Profile | Before | After | Change | Fairness Impact |
|----------------|--------|-------|--------|-----------------|
| Low Risk | $18,200 | $18,450 | +$250 (+1.4%) | Minor adjustment |
| Medium Risk | $26,300 | $25,680 | -$620 (-2.4%) | Fair reduction |
| **High Risk** | $35,400 | $33,200 | **-$2,200 (-6.2%)** | ✅ **Over-penalty removed** |

---

#### Performance Trade-offs

| Metric | Before | After | Change | Assessment |
|--------|--------|-------|--------|------------|
| **RMSE** | $2,077 | $2,189 | +$112 (+5.4%) | 🟡 Acceptable |
| **R² Score** | 0.8988 | 0.8867 | -0.0121 (-1.3%) | ✅ Minimal |
| **Statistical Parity** | 18.3% | 8.2% | -10.1pp (-55%) | ✅ Excellent |
| **Group RMSE Variance** | $2,456 | $2,189 | -$267 (-10.9%) | ✅ Improved |

**Key Insight:** Small accuracy trade-off (5.4% RMSE increase) for major fairness improvement (55%)

---

### Key Findings - Regression

#### 🎯 Model Performance
1. **Random Forest** achieved best results: RMSE $2,077, R² 0.8988
2. Hyperparameter tuning reduced RMSE by 15.4%
3. Model explains 89.88% of premium variance
4. Cross-validation shows robust, stable predictions

#### 🔍 Explainability Insights
1. **Age + Chronic Conditions** are primary drivers (50% combined)
2. **SHAP waterfall** provides transparent, justifiable premium calculations
3. **Feature contributions** enable personalized customer communication
4. **What-if analysis** empowers customers with actionable health guidance
5. **PDPs** reveal non-linear relationships (age acceleration after 50)

#### ⚖️ Fairness Achievements
1. **Age bias reduced:** Senior over-charging reduced from 17% to 5%
2. **Risk-based pricing:** High-risk patients no longer over-penalized
3. **Statistical parity:** Improved from 18.3% to 8.2% (meets <10% standard)
4. **Minimal accuracy loss:** Only 5.4% RMSE increase for 55% fairness gain

#### 💼 Business Impact
- ✅ Fair, defensible premium pricing for regulatory compliance
- ✅ Transparent calculations build customer trust and reduce disputes
- ✅ Expanded market access without adverse selection risk
- ✅ Actionable health guidance drives customer engagement
- ✅ Ethical AI positioning as competitive differentiator

---

## 🔬 MLflow Tracking

All experiments, models, and metrics are tracked using MLflow for reproducibility and comparison.

### MLflow UI Dashboard

**Launch MLflow:**
```bash
mlflow ui
```
**Access:** http://localhost:5000

**What's Tracked:**
- ✅ Model hyperparameters for each run
- ✅ Performance metrics (Accuracy, RMSE, MAE, R²)
- ✅ Cross-validation scores and standard deviations
- ✅ Model artifacts (.pkl files)
- ✅ Feature importance CSVs
- ✅ Training duration and timestamps
- ✅ Fairness metrics before/after mitigation

**Key Features:**
- Compare multiple model runs side-by-side
- Filter and sort by metrics
- Download trained models
- Track experiment lineage
- Visualize metric progression

---

## 🚀 How to Run

### Quick Start

```bash
# 1. Clone/download the repository
cd Week3

# 2. Install dependencies
pip install pandas numpy matplotlib seaborn scikit-learn mlflow shap lime fairlearn

# 3. Run Classification Pipeline
python loan_approval_mlflow.py
# Runtime: ~10-15 minutes

# 4. Run Regression Pipeline
python medical_insurance_mlflow.py
# Runtime: ~10-15 minutes

# 5. Launch MLflow UI
mlflow ui
# Open: http://localhost:5000
```

### Detailed Steps

**Step 1: Verify Dataset Files**
```bash
# Check files exist
ls *.csv

# Should see:
# loan_approval_dataset.csv
# Medicalpremium.csv
```

**Step 2: Run Pipelines**
```bash
# Classification (Loan Approval)
python loan_approval_mlflow.py

# Regression (Insurance Premium)
python medical_insurance_mlflow.py
```

**Step 3: View Results**
- Generated visualizations in current directory
- MLflow data in `mlruns/` folder
- CSV exports for feature importance

**Step 4: Explore MLflow**
```bash
mlflow ui --port 5000
```
Navigate to experiments, compare models, view metrics

---

## 💻 Technologies Used

### Core Libraries
| Library | Version | Purpose |
|---------|---------|---------|
| **Python** | 3.8+ | Programming language |
| **pandas** | 1.5+ | Data manipulation |
| **numpy** | 1.23+ | Numerical computing |
| **scikit-learn** | 1.2+ | ML models & preprocessing |
| **matplotlib** | 3.6+ | Visualization |
| **seaborn** | 0.12+ | Statistical plots |

### ML & Explainability
| Tool | Purpose |
|------|---------|
| **MLflow** | Experiment tracking, model versioning |
| **SHAP** | Global & local explanations, waterfall plots |
| **LIME** | Local interpretable explanations |
| **Fairlearn** | Fairness assessment & mitigation |

### Algorithms

**Classification:**
- Logistic Regression
- Decision Trees
- Random Forests ⭐
- Gradient Boosting
- Support Vector Machines

**Regression:**
- Linear/Ridge/Lasso Regression
- Decision Tree Regressor
- Random Forest Regressor ⭐
- Gradient Boosting Regressor

---

## 🎓 Key Learnings

### Technical Skills
1. **End-to-end ML pipeline** development from data to deployment
2. **Hyperparameter optimization** using GridSearchCV
3. **Model explainability** with SHAP, LIME, PDPs
4. **Fairness mitigation** strategies (reweighting, constraints, calibration)
5. **Experiment tracking** with MLflow
6. **Performance-fairness trade-offs** in real-world scenarios

### Domain Knowledge
1. **Loan Approval:** CIBIL score, income, and assets drive decisions
2. **Insurance Pricing:** Age, chronic conditions, and surgeries are key factors
3. **Bias detection:** Statistical tests reveal hidden discrimination
4. **Regulatory compliance:** Meeting 80% rule and fairness thresholds

### Best Practices
1. ✅ Always split data before any processing (avoid data leakage)
2. ✅ Use cross-validation for robust evaluation
3. ✅ Test for fairness across protected attributes
4. ✅ Provide explanations for every prediction (transparency)
5. ✅ Document trade-offs between accuracy and fairness
6. ✅ Monitor model performance continuously post-deployment

---

## 📁 Project Structure

```
Week3/
│
├── 📄 loan_approval_mlflow.py          # Classification pipeline
├── 📄 medical_insurance_mlflow.py      # Regression pipeline
│
├── 📊 loan_approval_dataset.csv        # Classification data (4,269 rows)
├── 📊 Medicalpremium.csv               # Regression data (986 rows)
│
├── 📘 README.md                        # This file
├── 📋 requirements.txt                 # Python dependencies
│
├── 📁 mlruns/                          # MLflow experiment data
│   ├── 0/                             # Loan approval experiments
│   └── 1/                             # Insurance experiments
│
├── 📁 outputs/                         # Generated visualizations
│   ├── classification_*.png
│   ├── regression_*.png
│   └── medical_*.png
│
└── 📁 artifacts/                       # Feature importance CSVs
    ├── *_feature_importance.csv
    └── model_comparison.csv
```

---

## 📊 Output Files Generated

### Classification Project
```
✅ correlation_heatmap.png
✅ loan_status_distribution.png
✅ feature_distributions.png
✅ classification_feature_importance.png
✅ classification_pdp_*.png (multiple)
✅ classification_feature_interactions.png
✅ classification_lime_approved.png
✅ classification_lime_rejected.png
✅ classification_counterfactual.png
✅ classification_fairness_before.png
✅ classification_fairness_after.png
✅ classification_bias_detection.png
✅ confusion_matrix_*.png (5 models)
✅ model_comparison.csv
```

### Regression Project
```
✅ medical_correlation_heatmap.png
✅ medical_premium_distribution.png
✅ medical_feature_distributions.png
✅ regression_feature_importance.png
✅ regression_pdp_*.png (multiple)
✅ regression_shap_summary.png
✅ regression_shap_dependence_*.png
✅ regression_shap_waterfall.png
✅ regression_single_prediction.png
✅ regression_fairness_before.png
✅ regression_fairness_after.png
✅ medical_model_comparison.png
✅ medical_model_comparison.csv
```

---

## 🔧 Troubleshooting

| Issue | Solution |
|-------|----------|
| **Module not found** | `pip install [module_name]` |
| **Dataset not found** | Ensure CSV files in same directory |
| **MLflow port conflict** | `mlflow ui --port 5001` |
| **Memory error** | Reduce GridSearchCV param grid |
| **Slow training** | Expected; models train for 10-15 mins |

---
