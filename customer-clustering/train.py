import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
import joblib
import os

def train_model():
    print("🚀 Training Customer Segmentation Model...")
    
    # Create models directory if it doesn't exist
    os.makedirs("models", exist_ok=True)
    
    # Load data
    df = pd.read_csv("data/Wholesale customers data.csv")
    print(f"✅ Data loaded: {df.shape}")
    
    # Preprocess
    df_clean = df.drop(['Channel', 'Region'], axis=1)
    feature_cols = ['Fresh', 'Milk', 'Grocery', 'Frozen', 'Detergents_Paper', 'Delicassen']
    
    # Log transform skewed features
    df_processed = df_clean.copy()
    log_cols = []
    for col in feature_cols:
        if df_processed[col].skew() > 0.5:
            df_processed[col] = np.log1p(df_processed[col])
            log_cols.append(col)
    
    print(f"📊 Log-transformed columns: {log_cols}")
    
    # Scale features
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(df_processed[feature_cols])
    
    # Train KMeans (using K=2 based on your analysis)
    kmeans = KMeans(n_clusters=2, random_state=42)
    labels = kmeans.fit_predict(X_scaled)
    
    # Evaluate model
    silhouette = silhouette_score(X_scaled, labels)
    davies_bouldin = davies_bouldin_score(X_scaled, labels)
    
    print(f"📈 Model Performance:")
    print(f"   Silhouette Score: {silhouette:.3f}")
    print(f"   Davies-Bouldin Score: {davies_bouldin:.3f}")
    
    # Save models to models/ folder
    joblib.dump(kmeans, "models/kmeans_model.pkl")
    joblib.dump(scaler, "models/scaler.pkl")
    joblib.dump(log_cols, "models/log_cols.pkl")
    joblib.dump(feature_cols, "models/feature_cols.pkl")
    
    # Save cluster assignments
    df['Cluster'] = labels
    df.to_csv("models/training_data_with_clusters.csv", index=False)
    
    print("✅ Models saved to models/ folder!")
    print(f"📊 Cluster distribution:")
    cluster_counts = pd.Series(labels).value_counts().sort_index()
    for cluster, count in cluster_counts.items():
        percentage = (count / len(labels)) * 100
        print(f"   Cluster {cluster}: {count} customers ({percentage:.1f}%)")

if __name__ == "__main__":
    train_model()
