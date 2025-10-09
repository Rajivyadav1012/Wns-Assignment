import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import mlflow
import mlflow.sklearn
import warnings
warnings.filterwarnings('ignore')

# Initialize MLflow
mlflow.set_experiment("Medical_Insurance_Premium_Prediction")

# Step 1: Data Loading and Exploration
print("Loading and exploring data...")
# Load the dataset and strip whitespace from column names
df = pd.read_csv('Medicalpremium.csv')

# Strip whitespace from column names
df.columns = df.columns.str.strip()

# Strip whitespace from string columns
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

# Check target variable (PremiumPrice)
print("\n" + "!"*80)
print("Target Variable Statistics (PremiumPrice):")
if 'PremiumPrice' in df.columns:
    print(f"Mean: {df['PremiumPrice'].mean():.2f}")
    print(f"Median: {df['PremiumPrice'].median():.2f}")
    print(f"Std Dev: {df['PremiumPrice'].std():.2f}")
    print(f"Min: {df['PremiumPrice'].min():.2f}")
    print(f"Max: {df['PremiumPrice'].max():.2f}")
else:
    print("ERROR: 'PremiumPrice' column not found!")
    print("Available columns:", df.columns.tolist())

# Step 2: Data Preprocessing
def preprocess_data(df):
    """Preprocess the medical insurance dataset"""
    df_clean = df.copy()
    
    # Handle missing values
    print("\nHandling missing values...")
    for col in df_clean.columns:
        if df_clean[col].isnull().sum() > 0:
            if df_clean[col].dtype in ['int64', 'float64']:
                df_clean[col].fillna(df_clean[col].median(), inplace=True)
            else:
                df_clean[col].fillna(df_clean[col].mode()[0], inplace=True)
    
    # Encode categorical variables
    print("Encoding categorical variables...")
    
    label_encoders = {}
    categorical_columns = df_clean.select_dtypes(include=['object']).columns
    
    for col in categorical_columns:
        if col != 'PremiumPrice':
            le = LabelEncoder()
            df_clean[col] = le.fit_transform(df_clean[col])
            label_encoders[col] = le
            print(f"   Encoded: {col}")
    
    # Convert all columns to numeric
    for col in df_clean.columns:
        if df_clean[col].dtype == 'object':
            try:
                df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
                if df_clean[col].isnull().sum() > 0:
                    df_clean[col].fillna(df_clean[col].median(), inplace=True)
            except Exception as e:
                print(f"Could not convert column {col} to numeric: {e}")
    
    return df_clean, label_encoders

# Preprocess the data
df_processed, label_encoders = preprocess_data(df)
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
    plt.savefig('medical_correlation_heatmap.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Premium Price distribution
    if 'PremiumPrice' in df.columns:
        plt.figure(figsize=(10, 6))
        plt.hist(df['PremiumPrice'], bins=50, color='#4ECDC4', edgecolor='black', alpha=0.7)
        plt.title('Premium Price Distribution', fontsize=14, fontweight='bold')
        plt.xlabel('Premium Price', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig('medical_premium_distribution.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    # Feature distributions
    important_cols = [col for col in df_numeric.columns if col != 'PremiumPrice'][:4]
    
    if len(important_cols) > 0:
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.ravel()
        
        for i, col in enumerate(important_cols):
            if i < len(axes):
                axes[i].hist(df[col], bins=30, color='skyblue', edgecolor='black', alpha=0.7)
                axes[i].set_title(f'{col} Distribution', fontsize=12, fontweight='bold')
                axes[i].set_xlabel(col, fontsize=10)
                axes[i].set_ylabel('Frequency', fontsize=10)
                axes[i].grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('medical_feature_distributions.png', dpi=300, bbox_inches='tight')
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
    
    # Check if PremiumPrice exists
    if 'PremiumPrice' not in df_numeric.columns:
        raise ValueError("PremiumPrice column not found or not numeric. Available columns: " + str(df_numeric.columns.tolist()))
    
    # Define features and target
    X = df_numeric.drop('PremiumPrice', axis=1)
    y = df_numeric['PremiumPrice']
    
    # Remove ID column if present
    id_cols = [col for col in X.columns if 'id' in col.lower()]
    if id_cols:
        X = X.drop(id_cols, axis=1)
        print(f"Removed ID columns: {id_cols}")
    
    print(f"\nFinal feature set ({X.shape[1]} features): {X.columns.tolist()}")
    print(f"Target variable shape: {y.shape}")
    print(f"Target statistics:")
    print(f"   Mean: {y.mean():.2f}")
    print(f"   Median: {y.median():.2f}")
    print(f"   Std: {y.std():.2f}")
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
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
    """Evaluate regression model performance"""
    
    # Use scaled or unscaled data
    X_test_eval = X_test_scaled if use_scaled else X_test
    X_train_eval = X_train_scaled if use_scaled else X_train
    
    # Make predictions
    y_pred = model.predict(X_test_eval)
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Cross-validation score
    cv_scores = cross_val_score(model, X_train_eval, y_train, cv=5, scoring='neg_mean_squared_error')
    cv_rmse = np.sqrt(-cv_scores)
    
    print(f"\n{'='*50}")
    print(f"Model: {model_name}")
    print(f"{'='*50}")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAE: {mae:.2f}")
    print(f"R2 Score: {r2:.4f}")
    print(f"Cross-validation RMSE: {cv_rmse.mean():.2f} (+/- {cv_rmse.std() * 2:.2f})")
    
    # Prediction vs Actual plot
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5, color='blue')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
             'r--', lw=2, label='Perfect Prediction')
    plt.xlabel('Actual Premium Price', fontsize=12)
    plt.ylabel('Predicted Premium Price', fontsize=12)
    plt.title(f'{model_name} - Predictions vs Actual', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'medical_{model_name.replace(" ", "_")}_predictions.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return {
        'model_name': model_name,
        'model': model,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'cv_rmse_mean': cv_rmse.mean(),
        'cv_rmse_std': cv_rmse.std(),
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
        
        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        cv_scores = cross_val_score(model, X_train_data, y_train, cv=5, scoring='neg_mean_squared_error')
        cv_rmse = np.sqrt(-cv_scores)
        
        # Log metrics
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2_score", r2)
        mlflow.log_metric("cv_rmse_mean", cv_rmse.mean())
        mlflow.log_metric("cv_rmse_std", cv_rmse.std())
        
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

# 6.1 Linear Regression
print("\n[1/6] Training Linear Regression...")
lr_model = LinearRegression()
lr_model.fit(X_train_scaled, y_train)
lr_results = evaluate_model(lr_model, X_train, X_test, y_train, y_test, "Linear Regression", use_scaled=True)
train_and_log_model(lr_model, "Linear_Regression", X_train, y_train, use_scaled=True)

# 6.2 Ridge Regression
print("\n[2/6] Training Ridge Regression...")
ridge_params = {
    'alpha': [0.1, 1, 10, 100]
}
ridge_grid = GridSearchCV(
    Ridge(random_state=42),
    ridge_params,
    cv=5,
    scoring='neg_mean_squared_error',
    n_jobs=-1,
    verbose=1
)
ridge_grid.fit(X_train_scaled, y_train)
print("Best parameters for Ridge Regression:", ridge_grid.best_params_)
ridge_best = ridge_grid.best_estimator_
ridge_results = evaluate_model(ridge_best, X_train, X_test, y_train, y_test, "Ridge Regression", use_scaled=True)
train_and_log_model(ridge_best, "Ridge_Regression", X_train, y_train, ridge_grid.best_params_, use_scaled=True)

# 6.3 Lasso Regression
print("\n[3/6] Training Lasso Regression...")
lasso_params = {
    'alpha': [0.1, 1, 10, 100]
}
lasso_grid = GridSearchCV(
    Lasso(random_state=42),
    lasso_params,
    cv=5,
    scoring='neg_mean_squared_error',
    n_jobs=-1,
    verbose=1
)
lasso_grid.fit(X_train_scaled, y_train)
print("Best parameters for Lasso Regression:", lasso_grid.best_params_)
lasso_best = lasso_grid.best_estimator_
lasso_results = evaluate_model(lasso_best, X_train, X_test, y_train, y_test, "Lasso Regression", use_scaled=True)
train_and_log_model(lasso_best, "Lasso_Regression", X_train, y_train, lasso_grid.best_params_, use_scaled=True)

# 6.4 Decision Tree
print("\n[4/6] Training Decision Tree...")
dt_params = {
    'max_depth': [5, 10, 15, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
dt_grid = GridSearchCV(
    DecisionTreeRegressor(random_state=42),
    dt_params,
    cv=5,
    scoring='neg_mean_squared_error',
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
plt.savefig('medical_dt_feature_importance.png', dpi=300, bbox_inches='tight')
plt.show()

# 6.5 Random Forest
print("\n[5/6] Training Random Forest...")
rf_params = {
    'n_estimators': [100, 200],
    'max_depth': [10, 15, None],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}
rf_grid = GridSearchCV(
    RandomForestRegressor(random_state=42, n_jobs=-1),
    rf_params,
    cv=5,
    scoring='neg_mean_squared_error',
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
plt.savefig('medical_rf_feature_importance.png', dpi=300, bbox_inches='tight')
plt.show()

# 6.6 Gradient Boosting
print("\n[6/6] Training Gradient Boosting...")
gb_params = {
    'n_estimators': [100, 200],
    'learning_rate': [0.1, 0.2],
    'max_depth': [3, 5],
    'min_samples_split': [2, 5]
}
gb_grid = GridSearchCV(
    GradientBoostingRegressor(random_state=42),
    gb_params,
    cv=5,
    scoring='neg_mean_squared_error',
    n_jobs=-1,
    verbose=1
)
gb_grid.fit(X_train, y_train)
print("Best parameters for Gradient Boosting:", gb_grid.best_params_)
gb_best = gb_grid.best_estimator_
gb_results = evaluate_model(gb_best, X_train, X_test, y_train, y_test, "Gradient Boosting", use_scaled=False)
train_and_log_model(gb_best, "Gradient_Boosting", X_train, y_train, gb_grid.best_params_, use_scaled=False)

# Step 7: Model Comparison and Selection
def compare_models(results_list):
    """Compare all models and select the best one"""
    
    comparison_df = pd.DataFrame([{
        'Model': result['model_name'],
        'RMSE': result['rmse'],
        'MAE': result['mae'],
        'R2_Score': result['r2'],
        'CV_RMSE_Mean': result['cv_rmse_mean'],
        'CV_RMSE_Std': result['cv_rmse_std']
    } for result in results_list])
    
    comparison_df = comparison_df.sort_values('RMSE', ascending=True)
    
    print("\n" + "="*80)
    print("MODEL COMPARISON")
    print("="*80)
    print(comparison_df.to_string(index=False))
    
    # Save comparison to CSV
    comparison_df.to_csv('medical_model_comparison.csv', index=False)
    
    # Visual comparison
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # RMSE comparison
    models = comparison_df['Model']
    rmse = comparison_df['RMSE']
    axes[0].bar(range(len(models)), rmse, color='skyblue', alpha=0.7)
    axes[0].set_xticks(range(len(models)))
    axes[0].set_xticklabels(models, rotation=45, ha='right')
    axes[0].set_title('Model RMSE Comparison', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('RMSE', fontsize=12)
    axes[0].grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for i, v in enumerate(rmse):
        axes[0].text(i, v + 50, f'{v:.2f}', ha='center', fontsize=9)
    
    # R2 comparison
    r2_scores = comparison_df['R2_Score']
    axes[1].bar(range(len(models)), r2_scores, color='lightgreen', alpha=0.7)
    axes[1].set_xticks(range(len(models)))
    axes[1].set_xticklabels(models, rotation=45, ha='right')
    axes[1].set_title('Model R2 Score Comparison', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('R2 Score', fontsize=12)
    axes[1].grid(axis='y', alpha=0.3)
    
    for i, v in enumerate(r2_scores):
        axes[1].text(i, v + 0.01, f'{v:.4f}', ha='center', fontsize=9)
    
    # CV RMSE comparison with error bars
    axes[2].bar(range(len(models)), comparison_df['CV_RMSE_Mean'], 
                yerr=comparison_df['CV_RMSE_Std'], 
                capsize=5, color='coral', alpha=0.7)
    axes[2].set_xticks(range(len(models)))
    axes[2].set_xticklabels(models, rotation=45, ha='right')
    axes[2].set_title('Cross-Validation RMSE', fontsize=14, fontweight='bold')
    axes[2].set_ylabel('CV RMSE', fontsize=12)
    axes[2].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('medical_model_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Select best model (lowest RMSE)
    best_idx = comparison_df.index[0]
    best_model_info = results_list[best_idx]
    
    print(f"\n*** BEST MODEL: {best_model_info['model_name']} ***")
    print(f"   RMSE: {best_model_info['rmse']:.2f}")
    print(f"   MAE: {best_model_info['mae']:.2f}")
    print(f"   R2 Score: {best_model_info['r2']:.4f}")
    print(f"   CV RMSE: {best_model_info['cv_rmse_mean']:.2f} (+/- {best_model_info['cv_rmse_std']:.2f})")
    
    return best_model_info

# Compare all models
all_results = [lr_results, ridge_results, lasso_results, dt_results, rf_results, gb_results]
best_model_info = compare_models(all_results)

# Step 8: Deploy Final Model
def deploy_final_model(best_model_info, X_train, y_train, scaler):
    """Deploy the best model with MLflow"""
    
    print(f"\n[DEPLOYING] Final model: {best_model_info['model_name']}")
    
    with mlflow.start_run(run_name="FINAL_DEPLOYED_MODEL"):
        final_model = best_model_info['model']
        
        # Use appropriate data
        X_train_data = X_train_scaled if best_model_info['use_scaled'] else X_train
        final_model.fit(X_train_data, y_train)
        
        # Log model details
        mlflow.log_param("final_model_type", best_model_info['model_name'])
        mlflow.log_metric("final_rmse", best_model_info['rmse'])
        mlflow.log_metric("final_mae", best_model_info['mae'])
        mlflow.log_metric("final_r2", best_model_info['r2'])
        mlflow.log_metric("final_cv_rmse", best_model_info['cv_rmse_mean'])
        
        # Log the final model
        mlflow.sklearn.log_model(final_model, "deployed_model")
        mlflow.sklearn.log_model(scaler, "scaler")
        
        # Log feature names
        mlflow.log_param("feature_names", list(X_train.columns))
        mlflow.log_param("used_scaled_data", best_model_info['use_scaled'])
        
        print("[SUCCESS] Final model deployed successfully to MLflow!")
        
        return final_model

final_model = deploy_final_model(best_model_info, X_train, y_train, scaler)

# Step 9: Model Inference Function
def predict_premium(model, scaler, input_data, feature_columns, use_scaled=False):
    """Make predictions using the deployed model"""
    
    # Create DataFrame from input
    input_df = pd.DataFrame([input_data])
    
    # Ensure correct feature order
    input_df = input_df[feature_columns]
    
    # Scale if needed
    if use_scaled:
        input_processed = scaler.transform(input_df)
    else:
        input_processed = input_df.values
    
    # Make prediction
    prediction = model.predict(input_processed)
    
    return {
        'predicted_premium': prediction[0]
    }

# Example usage
example_input = {col: X_train[col].median() for col in X_train.columns}

prediction_result = predict_premium(
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
print(f"\nPrediction Results:")
print(f"   Predicted Premium Price: ${prediction_result['predicted_premium']:.2f}")

# Step 10: Summary
print("\n" + "="*80)
print("PROJECT SUMMARY")
print("="*80)
print(f"[OK] Dataset processed: {df.shape[0]} rows, {df.shape[1]} columns")
print(f"[OK] Models trained: Linear Regression, Ridge, Lasso, Decision Tree, Random Forest, Gradient Boosting")
print(f"[OK] Best model: {best_model_info['model_name']}")
print(f"[OK] Best RMSE: {best_model_info['rmse']:.2f}")
print(f"[OK] Best R2 Score: {best_model_info['r2']:.4f}")
print(f"[OK] Model deployed to MLflow")
print(f"[OK] Results saved to CSV files")
print("\nNext steps:")
print("1. Run 'mlflow ui' to view experiment tracking")
print("2. Use predict_premium() function for new predictions")
print("3. Deploy model to production environment")
print("4. Set up monitoring and periodic retraining")

print("\n*** Medical Insurance Premium Prediction Pipeline Completed Successfully! ***")
print(f"\nMLflow Tracking URI: {mlflow.get_tracking_uri()}")