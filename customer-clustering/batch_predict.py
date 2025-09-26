import pandas as pd
import numpy as np
import joblib

def predict_batch(csv_file):
    """
    Predict clusters for multiple customers from CSV
    
    Args:
        csv_file: Path to CSV with customer data
        
    Returns:
        DataFrame with predictions added
    """
    try:
        # Load models from models/ folder
        kmeans = joblib.load("models/kmeans_model.pkl")
        scaler = joblib.load("models/scaler.pkl")
        log_cols = joblib.load("models/log_cols.pkl")
        feature_cols = joblib.load("models/feature_cols.pkl")
        
        # Load new customers
        df = pd.read_csv(csv_file)
        print(f"📊 Loaded {len(df)} customers from {csv_file}")
        
        # Validate columns
        missing_cols = set(feature_cols) - set(df.columns)
        if missing_cols:
            print(f"❌ Missing required columns: {missing_cols}")
            return None
        
        # Preprocess
        df_processed = df.copy()
        for col in log_cols:
            if col in df_processed.columns:
                df_processed[col] = np.log1p(df_processed[col])
        
        # Scale and predict
        X_scaled = scaler.transform(df_processed[feature_cols])
        clusters = kmeans.predict(X_scaled)
        
        # Calculate confidence scores
        distances = kmeans.transform(X_scaled)
        confidences = [1 / (1 + distances[i][clusters[i]]) for i in range(len(clusters))]
        
        # Add results to dataframe
        df['Predicted_Cluster'] = clusters
        df['Confidence'] = [round(c, 3) for c in confidences]
        df['Total_Spending'] = df[feature_cols].sum(axis=1)
        
        # Add segment names
        segment_names = {
            0: "High-Value Grocery Focused",
            1: "Fresh & Frozen Specialists"
        }
        df['Segment_Name'] = df['Predicted_Cluster'].map(segment_names)
        
        # Add business priorities
        priorities = {
            0: "High Priority",
            1: "Medium Priority"
        }
        df['Priority'] = df['Predicted_Cluster'].map(priorities)
        
        # Save results
        output_file = csv_file.replace('.csv', '_with_predictions.csv')
        df.to_csv(output_file, index=False)
        
        print(f"✅ Results saved to: {output_file}")
        
        # Show summary
        cluster_summary = df.groupby('Predicted_Cluster').agg({
            'Total_Spending': ['count', 'mean', 'std'],
            'Confidence': 'mean'
        }).round(2)
        
        print(f"\n📈 Prediction Summary:")
        print(cluster_summary)
        
        cluster_counts = df['Predicted_Cluster'].value_counts().sort_index()
        print(f"\n📊 Cluster Distribution:")
        for cluster, count in cluster_counts.items():
            percentage = (count / len(df)) * 100
            segment = segment_names[cluster]
            print(f"   Cluster {cluster} ({segment}): {count} customers ({percentage:.1f}%)")
        
        return df
        
    except FileNotFoundError as e:
        print(f"❌ Error: Model files not found. Please run train_simple.py first.")
        print(f"Details: {e}")
        return None
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

def create_sample_data():
    """Create sample new customers CSV for testing"""
    sample_data = {
        'Fresh': [5000, 15000, 2000, 8000, 12000],
        'Milk': [8000, 3000, 15000, 6000, 2000],
        'Grocery': [12000, 4000, 20000, 9000, 3500],
        'Frozen': [2000, 8000, 1000, 3000, 7000],
        'Detergents_Paper': [3000, 800, 8000, 2000, 600],
        'Delicassen': [1000, 2000, 500, 1200, 1800]
    }
    
    df = pd.DataFrame(sample_data)
    df.to_csv('new_customers_sample.csv', index=False)
    print("✅ Sample data created: new_customers_sample.csv")
    return df

if __name__ == "__main__":
    # Create sample data
    create_sample_data()
    
    # Test batch prediction
    results = predict_batch('new_customers_sample.csv')
    
    if results is not None:
        print(f"\n🎉 Batch prediction completed successfully!")