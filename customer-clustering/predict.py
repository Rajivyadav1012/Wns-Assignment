import pandas as pd
import numpy as np
import joblib
import os

def predict_customer(customer_data):
    """
    Predict cluster for new customer
    
    Args:
        customer_data: dict with keys ['Fresh', 'Milk', 'Grocery', 'Frozen', 'Detergents_Paper', 'Delicassen']
    
    Returns:
        dict with prediction results and business insights
    """
    try:
        # Load saved models from models/ folder
        kmeans = joblib.load("models/kmeans_model.pkl")
        scaler = joblib.load("models/scaler.pkl")
        log_cols = joblib.load("models/log_cols.pkl")
        feature_cols = joblib.load("models/feature_cols.pkl")
        
        # Validate input
        missing_cols = set(feature_cols) - set(customer_data.keys())
        if missing_cols:
            return {
                "success": False,
                "error": f"Missing required columns: {missing_cols}"
            }
        
        # Convert to DataFrame
        df = pd.DataFrame([customer_data])
        
        # Apply log transformation (same as training)
        for col in log_cols:
            if col in df.columns:
                df[col] = np.log1p(df[col])
        
        # Scale
        X_scaled = scaler.transform(df[feature_cols])
        
        # Predict
        cluster = kmeans.predict(X_scaled)[0]
        
        # Calculate confidence (distance to cluster center)
        distances = kmeans.transform(X_scaled)[0]
        confidence = 1 / (1 + distances[cluster])
        
        # Business segment profiles (based on your analysis)
        segment_info = {
            0: {
                "name": "High-Value Grocery Focused",
                "priority": "High Priority",
                "value_tier": "High Value",
                "characteristics": "High spending on Grocery, Milk, Detergents_Paper",
                "dominant_categories": ["Grocery", "Milk", "Detergents_Paper"],
                "growth_opportunities": ["Delicassen", "Frozen"],
                "marketing_strategies": [
                    "VIP loyalty program enrollment",
                    "Premium product recommendations",
                    "Cross-sell Delicassen products",
                    "Frozen food promotion campaigns"
                ]
            },
            1: {
                "name": "Fresh & Frozen Specialists",
                "priority": "Medium Priority",
                "value_tier": "Medium Value",
                "characteristics": "High Fresh and Frozen spending",
                "dominant_categories": ["Fresh", "Frozen", "Grocery"],
                "growth_opportunities": ["Detergents_Paper", "Delicassen"],
                "marketing_strategies": [
                    "Fresh product seasonal campaigns",
                    "Cross-sell household products",
                    "Loyalty program benefits",
                    "Bundle offers with cleaning products"
                ]
            }
        }
        
        # Calculate spending insights
        total_spending = sum(customer_data.values())
        avg_category_spending = total_spending / len(feature_cols)
        
        # Find top spending categories for this customer
        sorted_spending = sorted(customer_data.items(), key=lambda x: x[1], reverse=True)
        top_categories = [item[0] for item in sorted_spending[:3]]
        weak_categories = [item[0] for item in sorted_spending[-2:]]
        
        return {
            "success": True,
            "cluster": int(cluster),
            "confidence": round(confidence, 3),
            "segment_name": segment_info[cluster]["name"],
            "priority": segment_info[cluster]["priority"],
            "value_tier": segment_info[cluster]["value_tier"],
            "characteristics": segment_info[cluster]["characteristics"],
            "total_spending": round(total_spending, 2),
            "avg_category_spending": round(avg_category_spending, 2),
            "customer_top_categories": top_categories,
            "customer_weak_categories": weak_categories,
            "segment_strong_categories": segment_info[cluster]["dominant_categories"],
            "growth_opportunities": segment_info[cluster]["growth_opportunities"],
            "marketing_strategies": segment_info[cluster]["marketing_strategies"]
        }
        
    except FileNotFoundError as e:
        return {
            "success": False,
            "error": f"Model files not found. Please run train_simple.py first. Details: {e}"
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"Prediction error: {str(e)}"
        }

def test_predictions():
    """Test with sample customers"""
    
    test_customers = [
        {
            "name": "Customer A - High Grocery Spender",
            "data": {
                'Fresh': 3000,
                'Milk': 12000,
                'Grocery': 18000,
                'Frozen': 1500,
                'Detergents_Paper': 6000,
                'Delicassen': 800
            }
        },
        {
            "name": "Customer B - Fresh & Frozen Focused", 
            "data": {
                'Fresh': 20000,
                'Milk': 2000,
                'Grocery': 4000,
                'Frozen': 10000,
                'Detergents_Paper': 600,
                'Delicassen': 1500
            }
        },
        {
            "name": "Customer C - Balanced Spender",
            "data": {
                'Fresh': 8000,
                'Milk': 6000,
                'Grocery': 9000,
                'Frozen': 3000,
                'Detergents_Paper': 2000,
                'Delicassen': 1200
            }
        }
    ]
    
    print("🔮 Customer Segmentation Predictions")
    print("=" * 60)
    
    for customer in test_customers:
        print(f"\n👤 {customer['name']}")
        print(f"💰 Spending: {customer['data']}")
        print("-" * 50)
        
        result = predict_customer(customer['data'])
        
        if result['success']:
            print(f"🎯 Cluster: {result['cluster']}")
            print(f"📊 Segment: {result['segment_name']}")
            print(f"⭐ Priority: {result['priority']}")
            print(f"💎 Value Tier: {result['value_tier']}")
            print(f"📈 Confidence: {result['confidence']}")
            print(f"💵 Total Spending: ${result['total_spending']:,.2f}")
            print(f"📋 Characteristics: {result['characteristics']}")
            print(f"🔥 Customer's Top Categories: {', '.join(result['customer_top_categories'])}")
            print(f"📈 Growth Opportunities: {', '.join(result['growth_opportunities'])}")
            print(f"📢 Marketing Strategies:")
            for strategy in result['marketing_strategies']:
                print(f"   • {strategy}")
        else:
            print(f"❌ Error: {result['error']}")

if __name__ == "__main__":
    test_predictions()