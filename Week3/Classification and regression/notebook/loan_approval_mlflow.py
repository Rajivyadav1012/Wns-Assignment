 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
import mlflow
import mlflow.sklearn
import warnings
warnings.filterwarnings('ignore')

# Initialize MLflow
mlflow.set_experiment("Loan_Approval_Classification")

# Step 1: Data Loading and Exploration
print("Loading and exploring data...")
# Load the dataset and strip whitespace from column names
df = pd.read_csv('loan_approval_dataset.csv')

# **FIX: Strip whitespace from column names**
df.columns = df.columns.str.strip()

# **FIX: Strip whitespace from string columns**
for col in df.select_dtypes(include=['object']).columns:
    df[col] = df[col].str.strip()

print("Dataset Shape:", df.shape)
print("\nColumn Names:", df.columns.tolist())
print("\nDataset Info:")
print(df.info())
print("\nFirst 5 rows:")
print(df.head())
print("\nMissing Values:")
print(df.isnull().sum())
print("\nDataset Description:")
print(df.describe())

# Check target variable distribution
print("\n" + "!"*80)
print("Target Variable Distribution:")
if 'loan_status' in df.columns:
    print(df['loan_status'].value_counts())
    print("\nTarget Variable Percentage:")
    print(df['loan_status'].value_counts(normalize=True) * 100)
else:
    print("ERROR: 'loan_status' column not found!")
    print("Available columns:", df.columns.tolist())

# Step 2: Data Preprocessing
def preprocess_data(df):
    """Preprocess the loan dataset for classification"""
    df_clean = df.copy()
    
    # Handle missing values
    print("\nHandling missing values...")
    for col in df_clean.columns:
        if df_clean[col].isnull().sum() > 0:
            if df_clean[col].dtype in ['int64', 'float64']:
                df_clean[col].fillna(df_clean[col].median(), inplace=True)
            else:
                df_clean[col].fillna(df_clean[col].mode()[0], inplace=True)
    
    # Convert categorical variables to numerical
    print("Encoding categorical variables...")
    
    # Education encoding
    if 'education' in df_clean.columns:
        education_map = {'Graduate': 1, 'Not Graduate': 0}
        df_clean['education'] = df_clean['education'].map(education_map)
        df_clean['education'].fillna(0, inplace=True)
    
    # Self-employed encoding
    if 'self_employed' in df_clean.columns:
        self_employed_map = {'Yes': 1, 'No': 0}
        df_clean['self_employed'] = df_clean['self_employed'].map(self_employed_map)
        df_clean['self_employed'].fillna(0, inplace=True)
    
    # Loan status encoding (target variable)
    if 'loan_status' in df_clean.columns:
        loan_status_map = {'Approved': 1, 'Rejected': 0}
        df_clean['loan_status'] = df_clean['loan_status'].map(loan_status_map)
        df_clean['loan_status'].fillna(0, inplace=True)
    
    # Convert all columns to numeric where possible
    for col in df_clean.columns:
        if df_clean[col].dtype == 'object':
            try:
                df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
                if df_clean[col].isnull().sum() > 0:
                    df_clean[col].fillna(df_clean[col].median(), inplace=True)
            except Exception as e:
                print(f"Could not convert column {col} to numeric: {e}")
    
    return df_clean

# Preprocess the data
df_processed = preprocess_data(df)
print("\nProcessed Dataset Shape:", df_processed.shape)
print("Processed Data Types:")
print(df_processed.dtypes)
print("\nProcessed Data Columns:")
print(df_processed.columns.tolist())
print("\nFirst 5 rows after preprocessing:")
print(df_processed.head())

# Step 3: Exploratory Data Analysis
def perform_eda(df):
    """Perform exploratory data analysis"""
    
    # Select only numeric columns
    df_numeric = df.select_dtypes(include=[np.number])
    
    if df_numeric.shape[1] < 2:
        print("Not enough numeric columns for EDA")
        return
    
    # Correlation heatmap
    plt.figure(figsize=(14, 10))
    correlation_matrix = df_numeric.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f', 
                square=True, linewidths=0.5)
    plt.title('Feature Correlation Heatmap', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('correlation_heatmap.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Target distribution
    if 'loan_status' in df.columns:
        plt.figure(figsize=(8, 6))
        loan_counts = df['loan_status'].value_counts()
        plt.bar(['Rejected', 'Approved'], loan_counts.values, color=['#FF6B6B', '#4ECDC4'])
        plt.title('Loan Status Distribution', fontsize=14, fontweight='bold')
        plt.xlabel('Loan Status', fontsize=12)
        plt.ylabel('Count', fontsize=12)
        plt.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for i, v in enumerate(loan_counts.values):
            plt.text(i, v + 5, str(v), ha='center', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('loan_status_distribution.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    # Box plots for important features
    important_features = ['cibil_score', 'income_annum', 'loan_amount', 'loan_term']
    available_features = [f for f in important_features if f in df_numeric.columns]
    
    if len(available_features) > 0 and 'loan_status' in df_numeric.columns:
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.ravel()
        
        for i, feature in enumerate(available_features[:4]):
            if i < len(axes):
                df_numeric.boxplot(column=feature, by='loan_status', ax=axes[i])
                axes[i].set_title(f'{feature.replace("_", " ").title()} by Loan Status')
                axes[i].set_xlabel('Loan Status (0: Rejected, 1: Approved)')
                axes[i].set_ylabel(feature.replace("_", " ").title())
        
        plt.suptitle('')  # Remove the default title
        plt.tight_layout()
        plt.savefig('feature_distributions.png', dpi=300, bbox_inches='tight')
        plt.show()

print("\n" + "="*80)
print("PERFORMING EXPLORATORY DATA ANALYSIS")
print("="*80)
perform_eda(df_processed)

# Step 4: Prepare Data for Modeling
def prepare_features_target(df):
    """Prepare features and target variable"""
    
    # Ensure all data is numeric
    df_numeric = df.select_dtypes(include=[np.number])
    
    # Check if loan_status exists
    if 'loan_status' not in df_numeric.columns:
        raise ValueError("loan_status column not found or not numeric. Available columns: " + str(df_numeric.columns.tolist()))
    
    # Define features and target
    X = df_numeric.drop('loan_status', axis=1)
    y = df_numeric['loan_status']
    
    # Remove loan_id if present
    if 'loan_id' in X.columns:
        X = X.drop('loan_id', axis=1)
    
    print(f"\nFinal feature set ({X.shape[1]} features): {X.columns.tolist()}")
    print(f"Target variable shape: {y.shape}")
    print(f"Target distribution:\n{y.value_counts()}")
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale numerical features
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns,
        index=X_test.index
    )
    
    return X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test, scaler

print("\n" + "="*80)
print("PREPARING DATA FOR MODELING")
print("="*80)
X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test, scaler = prepare_features_target(df_processed)
print(f"Training set size: {X_train.shape}")
print(f"Test set size: {X_test.shape}")

# Step 5: Model Building and Evaluation Functions
def evaluate_model(model, X_train, X_test, y_train, y_test, model_name, use_scaled=False):
    """Evaluate model performance"""
    
    # Use scaled or unscaled data
    X_test_eval = X_test_scaled if use_scaled else X_test
    X_train_eval = X_train_scaled if use_scaled else X_train
    
    # Make predictions
    y_pred = model.predict(X_test_eval)
    y_pred_proba = model.predict_proba(X_test_eval)[:, 1] if hasattr(model, 'predict_proba') else None
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    classification_rep = classification_report(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba) if y_pred_proba is not None else None
    
    # Cross-validation score
    cv_scores = cross_val_score(model, X_train_eval, y_train, cv=5, scoring='accuracy')
    
    print(f"\n{'='*50}")
    print(f"Model: {model_name}")
    print(f"{'='*50}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Cross-validation Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    if roc_auc:
        print(f"ROC-AUC Score: {roc_auc:.4f}")
    print(f"\nClassification Report:\n{classification_rep}")
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=True,
                xticklabels=['Rejected', 'Approved'],
                yticklabels=['Rejected', 'Approved'])
    plt.title(f'Confusion Matrix - {model_name}', fontsize=14, fontweight='bold')
    plt.ylabel('Actual', fontsize=12)
    plt.xlabel('Predicted', fontsize=12)
    plt.tight_layout()
    plt.savefig(f'confusion_matrix_{model_name.replace(" ", "_")}.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return {
        'model_name': model_name,
        'model': model,
        'accuracy': accuracy,
        'cv_accuracy_mean': cv_scores.mean(),
        'cv_accuracy_std': cv_scores.std(),
        'roc_auc': roc_auc,
        'classification_report': classification_rep,
        'confusion_matrix': conf_matrix,
        'use_scaled': use_scaled
    }

def train_and_log_model(model, model_name, X_train, y_train, params=None, use_scaled=False):
    """Train model and log to MLflow"""
    
    with mlflow.start_run(run_name=model_name):
        # Log parameters
        if params:
            mlflow.log_params(params)
        
        # Use scaled or unscaled data
        X_train_data = X_train_scaled if use_scaled else X_train
        
        # Train model
        model.fit(X_train_data, y_train)
        
        # Make predictions
        X_test_data = X_test_scaled if use_scaled else X_test
        y_pred = model.predict(X_test_data)
        y_pred_proba = model.predict_proba(X_test_data)[:, 1] if hasattr(model, 'predict_proba') else None
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_proba) if y_pred_proba is not None else None
        cv_scores = cross_val_score(model, X_train_data, y_train, cv=5, scoring='accuracy')
        
        # Log metrics
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("cv_accuracy_mean", cv_scores.mean())
        mlflow.log_metric("cv_accuracy_std", cv_scores.std())
        if roc_auc:
            mlflow.log_metric("roc_auc", roc_auc)
        
        # Log model
        mlflow.sklearn.log_model(model, model_name)
        
        # Log feature importance for tree-based models
        if hasattr(model, 'feature_importances_'):
            feature_importance = pd.DataFrame({
                'feature': X_train.columns,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            importance_file = f"{model_name}_feature_importance.csv"
            feature_importance.to_csv(importance_file, index=False)
            mlflow.log_artifact(importance_file)
        
        return model

# Step 6: Train Individual Models

print("\n" + "="*80)
print("TRAINING MODELS")
print("="*80)

# 6.1 Logistic Regression
print("\n[1/5] Training Logistic Regression...")
lr_params = {
    'C': [0.1, 1, 10],
    'penalty': ['l2'],
    'solver': ['liblinear', 'lbfgs'],
    'max_iter': [1000]
}
lr_grid = GridSearchCV(
    LogisticRegression(random_state=42),
    lr_params,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)
lr_grid.fit(X_train_scaled, y_train)
print("Best parameters for Logistic Regression:", lr_grid.best_params_)
lr_best = lr_grid.best_estimator_
lr_results = evaluate_model(lr_best, X_train, X_test, y_train, y_test, "Logistic Regression", use_scaled=True)
train_and_log_model(lr_best, "Logistic_Regression", X_train, y_train, lr_grid.best_params_, use_scaled=True)

# 6.2 Decision Tree
print("\n[2/5] Training Decision Tree...")
dt_params = {
    'max_depth': [5, 10, 15, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'criterion': ['gini', 'entropy']
}
dt_grid = GridSearchCV(
    DecisionTreeClassifier(random_state=42),
    dt_params,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)
dt_grid.fit(X_train, y_train)
print("Best parameters for Decision Tree:", dt_grid.best_params_)
dt_best = dt_grid.best_estimator_
dt_results = evaluate_model(dt_best, X_train, X_test, y_train, y_test, "Decision Tree", use_scaled=False)
train_and_log_model(dt_best, "Decision_Tree", X_train, y_train, dt_grid.best_params_, use_scaled=False)

# Plot feature importance for Decision Tree
plt.figure(figsize=(10, 6))
feature_importance_dt = pd.DataFrame({
    'feature': X_train.columns,
    'importance': dt_best.feature_importances_
}).sort_values('importance', ascending=True).tail(10)
plt.barh(feature_importance_dt['feature'], feature_importance_dt['importance'], color='skyblue')
plt.title('Decision Tree - Top 10 Feature Importance', fontsize=14, fontweight='bold')
plt.xlabel('Importance', fontsize=12)
plt.tight_layout()
plt.savefig('dt_feature_importance.png', dpi=300, bbox_inches='tight')
plt.show()

# 6.3 Random Forest
print("\n[3/5] Training Random Forest...")
rf_params = {
    'n_estimators': [100, 200],
    'max_depth': [10, 15, None],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
    'max_features': ['sqrt']
}
rf_grid = GridSearchCV(
    RandomForestClassifier(random_state=42, n_jobs=-1),
    rf_params,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)
rf_grid.fit(X_train, y_train)
print("Best parameters for Random Forest:", rf_grid.best_params_)
rf_best = rf_grid.best_estimator_
rf_results = evaluate_model(rf_best, X_train, X_test, y_train, y_test, "Random Forest", use_scaled=False)
train_and_log_model(rf_best, "Random_Forest", X_train, y_train, rf_grid.best_params_, use_scaled=False)

# Plot feature importance for Random Forest
plt.figure(figsize=(10, 6))
feature_importance_rf = pd.DataFrame({
    'feature': X_train.columns,
    'importance': rf_best.feature_importances_
}).sort_values('importance', ascending=True).tail(10)
plt.barh(feature_importance_rf['feature'], feature_importance_rf['importance'], color='lightgreen')
plt.title('Random Forest - Top 10 Feature Importance', fontsize=14, fontweight='bold')
plt.xlabel('Importance', fontsize=12)
plt.tight_layout()
plt.savefig('rf_feature_importance.png', dpi=300, bbox_inches='tight')
plt.show()

# 6.4 Gradient Boosting
print("\n[4/5] Training Gradient Boosting...")
gb_params = {
    'n_estimators': [100, 200],
    'learning_rate': [0.1, 0.2],
    'max_depth': [3, 5],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}
gb_grid = GridSearchCV(
    GradientBoostingClassifier(random_state=42),
    gb_params,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)
gb_grid.fit(X_train, y_train)
print("Best parameters for Gradient Boosting:", gb_grid.best_params_)
gb_best = gb_grid.best_estimator_
gb_results = evaluate_model(gb_best, X_train, X_test, y_train, y_test, "Gradient Boosting", use_scaled=False)
train_and_log_model(gb_best, "Gradient_Boosting", X_train, y_train, gb_grid.best_params_, use_scaled=False)

# 6.5 Support Vector Machine
print("\n[5/5] Training Support Vector Machine...")
svm_params = {
    'C': [0.1, 1, 10],
    'kernel': ['rbf', 'linear'],
    'gamma': ['scale']
}
svm_grid = GridSearchCV(
    SVC(probability=True, random_state=42),
    svm_params,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)
svm_grid.fit(X_train_scaled, y_train)
print("Best parameters for SVM:", svm_grid.best_params_)
svm_best = svm_grid.best_estimator_
svm_results = evaluate_model(svm_best, X_train, X_test, y_train, y_test, "Support Vector Machine", use_scaled=True)
train_and_log_model(svm_best, "Support_Vector_Machine", X_train, y_train, svm_grid.best_params_, use_scaled=True)

# Step 7: Model Comparison and Selection
def compare_models(results_list):
    """Compare all models and select the best one"""
    
    comparison_df = pd.DataFrame([{
        'Model': result['model_name'],
        'Accuracy': result['accuracy'],
        'CV_Accuracy_Mean': result['cv_accuracy_mean'],
        'CV_Accuracy_Std': result['cv_accuracy_std'],
        'ROC_AUC': result.get('roc_auc', 0)
    } for result in results_list])
    
    comparison_df = comparison_df.sort_values('Accuracy', ascending=False)
    
    print("\n" + "="*80)
    print("MODEL COMPARISON")
    print("="*80)
    print(comparison_df.to_string(index=False))
    
    # Save comparison to CSV
    comparison_df.to_csv('model_comparison.csv', index=False)
    
    # Visual comparison
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Accuracy comparison
    models = comparison_df['Model']
    accuracy = comparison_df['Accuracy']
    axes[0].bar(range(len(models)), accuracy, color='skyblue', alpha=0.7)
    axes[0].set_xticks(range(len(models)))
    axes[0].set_xticklabels(models, rotation=45, ha='right')
    axes[0].set_title('Model Accuracy Comparison', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('Accuracy', fontsize=12)
    axes[0].grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for i, v in enumerate(accuracy):
        axes[0].text(i, v + 0.005, f'{v:.4f}', ha='center', fontsize=9)
    
    # CV Accuracy comparison with error bars
    axes[1].bar(range(len(models)), comparison_df['CV_Accuracy_Mean'], 
                yerr=comparison_df['CV_Accuracy_Std'], 
                capsize=5, color='lightgreen', alpha=0.7)
    axes[1].set_xticks(range(len(models)))
    axes[1].set_xticklabels(models, rotation=45, ha='right')
    axes[1].set_title('Cross-Validation Accuracy Comparison', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('CV Accuracy', fontsize=12)
    axes[1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Select best model
    best_idx = comparison_df.index[0]
    best_model_info = results_list[best_idx]
    
    print(f"\n🎯 BEST MODEL: {best_model_info['model_name']}")
    print(f"   Accuracy: {best_model_info['accuracy']:.4f}")
    print(f"   CV Accuracy: {best_model_info['cv_accuracy_mean']:.4f} (+/- {best_model_info['cv_accuracy_std']:.4f})")
    if best_model_info.get('roc_auc'):
        print(f"   ROC-AUC: {best_model_info['roc_auc']:.4f}")
    
    return best_model_info

# Compare all models
all_results = [lr_results, dt_results, rf_results, gb_results, svm_results]
best_model_info = compare_models(all_results)

# Step 8: Deploy Final Model
def deploy_final_model(best_model_info, X_train, y_train, scaler):
    """Deploy the best model with MLflow"""
    
    print(f"\n🚀 Deploying final model: {best_model_info['model_name']}")
    
    with mlflow.start_run(run_name="FINAL_DEPLOYED_MODEL"):
        final_model = best_model_info['model']
        
        # Use appropriate data
        X_train_data = X_train_scaled if best_model_info['use_scaled'] else X_train
        final_model.fit(X_train_data, y_train)
        
        # Log model details
        mlflow.log_param("final_model_type", best_model_info['model_name'])
        mlflow.log_metric("final_accuracy", best_model_info['accuracy'])
        mlflow.log_metric("final_cv_accuracy", best_model_info['cv_accuracy_mean'])
        
        if best_model_info.get('roc_auc'):
            mlflow.log_metric("final_roc_auc", best_model_info['roc_auc'])
        
        # Log the final model
        mlflow.sklearn.log_model(final_model, "deployed_model")
        mlflow.sklearn.log_model(scaler, "scaler")
        
        # Log feature names
        mlflow.log_param("feature_names", list(X_train.columns))
        mlflow.log_param("used_scaled_data", best_model_info['use_scaled'])
        
        print("✅ Final model deployed successfully to MLflow!")
        
        return final_model

final_model = deploy_final_model(best_model_info, X_train, y_train, scaler)

# Step 9: Model Inference Function
def predict_loan_approval(model, scaler, input_data, feature_columns, use_scaled=False):
    """Make predictions using the deployed model"""
    
    # Create DataFrame from input
    input_df = pd.DataFrame([input_data])
    
    # Encode categorical variables
    if 'education' in input_df.columns:
        input_df['education'] = input_df['education'].map({'Graduate': 1, 'Not Graduate': 0})
    
    if 'self_employed' in input_df.columns:
        input_df['self_employed'] = input_df['self_employed'].map({'Yes': 1, 'No': 0})
    
    # Remove loan_id if present
    if 'loan_id' in input_df.columns:
        input_df = input_df.drop('loan_id', axis=1)
    
    # Ensure correct feature order
    input_df = input_df[feature_columns]
    
    # Scale if needed
    if use_scaled:
        input_processed = scaler.transform(input_df)
    else:
        input_processed = input_df.values
    
    # Make prediction
    prediction = model.predict(input_processed)
    probability = model.predict_proba(input_processed)
    
    return {
        'prediction': 'Approved' if prediction[0] == 1 else 'Rejected',
        'approval_probability': probability[0][1],
        'rejection_probability': probability[0][0]
    }

# Example usage
example_input = {
    'no_of_dependents': 2,
    'education': 'Graduate',
    'self_employed': 'No',
    'income_annum': 9600000,
    'loan_amount': 29900000,
    'loan_term': 12,
    'cibil_score': 778,
    'residential_assets_value': 2400000,
    'commercial_assets_value': 17600000,
    'luxury_assets_value': 22700000,
    'bank_asset_value': 8000000
}

prediction_result = predict_loan_approval(
    final_model, 
    scaler, 
    example_input, 
    X_train.columns,
    use_scaled=best_model_info['use_scaled']
)

print("\n" + "="*80)
print("EXAMPLE PREDICTION")
print("="*80)
print(f"Input Data: {example_input}")
print(f"\n📊 Prediction Results:")
print(f"   Loan Status: {prediction_result['prediction']}")
print(f"   Approval Probability: {prediction_result['approval_probability']:.2%}")
print(f"   Rejection Probability: {prediction_result['rejection_probability']:.2%}")

# Step 10: Summary
print("\n" + "="*80)
print("PROJECT SUMMARY")
print("="*80)
print(f"✅ Dataset processed: {df.shape[0]} rows, {df.shape[1]} columns")
print(f"✅ Models trained: Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, SVM")
print(f"✅ Best model: {best_model_info['model_name']}")
print(f"✅ Best accuracy: {best_model_info['accuracy']:.4f}")
print(f"✅ Best CV accuracy: {best_model_info['cv_accuracy_mean']:.4f}")
print(f"✅ Model deployed to MLflow")
print(f"✅ Results saved to CSV files")
print("\nNext steps:")
print("1. Run 'mlflow ui' to view experiment tracking")
print("2. Use predict_loan_approval() function for new predictions")
print("3. Deploy model to production environment")
print("4. Set up monitoring and periodic retraining")

print("\n🎉 Loan Approval Classification Pipeline Completed Successfully!")
print(f"\n📊 MLflow Tracking URI: {mlflow.get_tracking_uri()}")
