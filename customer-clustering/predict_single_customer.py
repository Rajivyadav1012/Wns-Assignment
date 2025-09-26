from predict_new import predict_customer

def main():
    print("🔮 Single Customer Prediction")
    print("=" * 40)
    
    # Example customer
    customer = {
        'Fresh': 10000,
        'Milk': 8000,
        'Grocery': 15000,
        'Frozen': 2500,
        'Detergents_Paper': 4000,
        'Delicassen': 1500
    }
    
    print(f"Customer spending: {customer}")
    print("-" * 40)
    
    result = predict_customer(customer)
    
    if result['success']:
        print(f"🎯 Prediction: Cluster {result['cluster']}")
        print(f"📊 Segment: {result['segment_name']}")
        print(f"⭐ Priority: {result['priority']}")
        print(f"💰 Total Spending: ${result['total_spending']:,.2f}")
        print(f"📈 Confidence: {result['confidence']}")
        
        print(f"\n📢 Recommended Marketing Strategies:")
        for strategy in result['marketing_strategies']:
            print(f"   • {strategy}")
    else:
        print(f"❌ Error: {result['error']}")

if __name__ == "__main__":
    main()
