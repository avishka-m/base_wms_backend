#!/usr/bin/env python3
"""
Database Prediction Test Script

This script demonstrates how to get item-to-item predictions
directly from your WMS database.
"""

import asyncio
import sys
from pathlib import Path
import pandas as pd
from datetime import datetime
import numpy as np

# Add the src directory to Python path
current_dir = Path(__file__).parent
src_dir = current_dir / "src"
sys.path.insert(0, str(src_dir))

async def test_item_predictions():
    """Test item-to-item predictions from database."""
    print("🔮 ITEM-TO-ITEM PREDICTION TEST")
    print("=" * 50)
    
    try:
        from item_analysis_service import ItemAnalysisService
        from models.prophet_forecaster import ProphetForecaster
        
        # Initialize services
        service = ItemAnalysisService()
        forecaster = ProphetForecaster()
        
        print("\n📊 Testing with sample data (simulating your database)...")
        
        # Create sample data representing your WMS database
        sample_data = create_sample_wms_data()
        print(f"✅ Created {len(sample_data)} sample records")
        
        # Get unique products
        products = sample_data['product_id'].unique()[:5]
        print(f"🎯 Testing with products: {list(products)}")
        
        print(f"\n🔄 Training models and generating predictions...")
        
        for product_id in products:
            print(f"\n📋 ANALYZING PRODUCT: {product_id}")
            print("-" * 40)
            
            # Train model for this product
            product_data = sample_data[sample_data['product_id'] == product_id]
            result = forecaster.train_model(product_data, product_id)
            
            if not result.get("success"):
                print(f"❌ Training failed: {result.get('error')}")
                continue
            
            print(f"✅ Model trained with {result['training_points']} data points")
            
            # Get detailed item analysis
            details = forecaster.get_item_details(product_id)
            
            if "error" in details:
                print(f"❌ Analysis failed: {details['error']}")
                continue
            
            # Display item details
            historical = details.get("historical_stats", {})
            forecast = details.get("forecast_insights", {})
            recommendations = details.get("recommendations", [])
            
            print(f"\n📊 HISTORICAL ANALYSIS:")
            print(f"   Average Daily Demand: {historical.get('average_daily_demand', 0):.1f} units")
            print(f"   Total Demand: {historical.get('total_demand', 0):.0f} units")
            print(f"   Demand Volatility: {historical.get('demand_volatility', 0):.1f}")
            print(f"   Data Points: {historical.get('data_points', 0)}")
            
            date_range = historical.get('date_range', {})
            if date_range:
                print(f"   Date Range: {date_range.get('start', '')[:10]} to {date_range.get('end', '')[:10]}")
            
            print(f"\n🔮 DEMAND FORECAST (Next 30 days):")
            print(f"   Predicted Total Demand: {forecast.get('predicted_total_demand', 0):.0f} units")
            print(f"   Average Daily Demand: {forecast.get('average_daily_demand', 0):.1f} units")
            print(f"   Demand Trend: {forecast.get('demand_trend', 'unknown').title()}")
            
            peak_date = forecast.get('peak_demand_date', '')
            if peak_date:
                print(f"   Peak Demand: {forecast.get('peak_demand_value', 0):.0f} units on {peak_date[:10]}")
            
            low_date = forecast.get('low_demand_date', '')
            if low_date:
                print(f"   Low Demand: {forecast.get('low_demand_value', 0):.0f} units on {low_date[:10]}")
            
            print(f"\n💡 BUSINESS RECOMMENDATIONS:")
            for i, rec in enumerate(recommendations[:3], 1):
                print(f"   {i}. {rec}")
            
            # Generate inventory recommendations
            avg_demand = forecast.get('average_daily_demand', 0)
            reorder_point = avg_demand * 7 + avg_demand * 3  # 7 days lead time + 3 days safety
            monthly_order = forecast.get('predicted_total_demand', 0) * 1.1  # 10% buffer
            
            print(f"\n📦 INVENTORY RECOMMENDATIONS:")
            print(f"   Recommended Reorder Point: {reorder_point:.0f} units")
            print(f"   Monthly Order Quantity: {monthly_order:.0f} units")
            
            if forecast.get('demand_trend') == 'increasing':
                print(f"   🔥 ALERT: Increasing trend - consider building inventory")
            elif forecast.get('demand_trend') == 'decreasing':
                print(f"   📉 ALERT: Decreasing trend - consider promotions")
        
        print(f"\n🔄 PRODUCT COMPARISON ANALYSIS...")
        
        # Compare all products
        comparison = forecaster.compare_products(list(products))
        
        if "error" not in comparison:
            summary = comparison.get("summary_stats", {})
            rankings = comparison.get("rankings", {})
            
            print(f"\n📊 CROSS-PRODUCT INSIGHTS:")
            print(f"   Total Forecast Demand: {summary.get('total_forecast_demand', 0):.0f} units")
            print(f"   Highest Demand Product: {summary.get('highest_demand_product', 'N/A')}")
            print(f"   Most Volatile Product: {summary.get('most_volatile_product', 'N/A')}")
            
            if rankings.get("highest_demand"):
                print(f"   Top 3 Demand Products: {rankings['highest_demand'][:3]}")
            
            if rankings.get("increasing_trend"):
                print(f"   Products with Growth: {len(rankings['increasing_trend'])}")
            
            if rankings.get("decreasing_trend"):
                print(f"   Products Declining: {len(rankings['decreasing_trend'])}")
        
        print(f"\n✅ ITEM-TO-ITEM ANALYSIS COMPLETE!")
        print(f"🎯 Successfully analyzed {len(products)} products")
        
        await service.close()
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure you have installed Prophet: pip install prophet")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

def create_sample_wms_data():
    """Create sample WMS data for testing."""
    
    # Product catalog
    products = [
        {"id": "ELEC-001", "category": "Electronics", "base_demand": 45},
        {"id": "ELEC-002", "category": "Electronics", "base_demand": 32},
        {"id": "CLOTH-001", "category": "Clothing", "base_demand": 28},
        {"id": "HOME-001", "category": "Home", "base_demand": 38},
        {"id": "BOOK-001", "category": "Books", "base_demand": 15}
    ]
    
    # Generate 1 year of daily data
    date_range = pd.date_range(start='2024-01-01', end='2024-12-31', freq='D')
    
    data = []
    
    for product in products:
        product_id = product["id"]
        category = product["category"]
        base_demand = product["base_demand"]
        
        for date in date_range:
            # Seasonal patterns
            day_of_year = date.dayofyear
            seasonal_factor = 1.0 + 0.3 * np.sin(2 * np.pi * day_of_year / 365)
            
            # Weekly patterns (higher demand on weekends for some categories)
            if category in ["Electronics", "Clothing"]:
                weekly_factor = 1.3 if date.weekday() >= 5 else 1.0
            else:
                weekly_factor = 1.0
            
            # Holiday effects
            holiday_factor = 1.0
            if date.month == 12:  # December boost
                holiday_factor = 1.5
            elif date.month in [6, 7]:  # Summer boost for some items
                if category in ["Clothing", "Home"]:
                    holiday_factor = 1.2
            
            # Random variation
            noise = np.random.normal(1.0, 0.15)
            
            # Calculate final demand
            demand = max(1, int(base_demand * seasonal_factor * weekly_factor * holiday_factor * noise))
            
            data.append({
                'ds': date,
                'product_id': product_id,
                'y': demand,
                'category': category,
                'warehouse_id': 'MAIN-WH',
                'unit_price': 10.0 + (hash(product_id) % 50),
                'total_value': demand * (10.0 + (hash(product_id) % 50))
            })
    
    return pd.DataFrame(data)

def show_database_integration():
    """Show how to integrate with real database."""
    
    print(f"\n🔗 INTEGRATING WITH YOUR DATABASE")
    print("=" * 40)
    
    print("📝 To use your actual WMS database:")
    print()
    print("1. 🔧 Update config.py with your MongoDB settings:")
    print("""
MONGODB_URL = "mongodb://your-server:27017"
DATABASE_NAME = "your_wms_database"
""")
    
    print("2. 📊 Ensure your data has the required schema:")
    print("""
Required columns:
- ds (datetime): Date of the transaction
- product_id (string): Unique product identifier  
- y (numeric): Demand quantity
- category (string): Product category (optional)
- warehouse_id (string): Warehouse location (optional)
""")
    
    print("3. 🚀 Run predictions on your data:")
    print("""
from src.item_analysis_service import ItemAnalysisService

service = ItemAnalysisService()

# Analyze specific product
analysis = await service.analyze_item("YOUR-SKU-001")

# Compare similar products
comparison = await service.compare_similar_items("YOUR-SKU-001")

# Batch analyze multiple products
batch_results = await service.batch_analyze_products([
    "SKU-001", "SKU-002", "SKU-003"
])
""")
    
    print("4. 📈 Integration with your frontend:")
    print("""
// React component example
const SeasonalForecast = ({ productId }) => {
    const [forecast, setForecast] = useState(null);
    
    useEffect(() => {
        fetch(`/api/seasonal/analyze`, {
            method: 'POST',
            body: JSON.stringify({ product_id: productId })
        })
        .then(response => response.json())
        .then(data => setForecast(data));
    }, [productId]);
    
    return (
        <div>
            <h3>Demand Forecast for {productId}</h3>
            <p>Predicted demand: {forecast?.forecast_insights?.predicted_total_demand}</p>
            <p>Trend: {forecast?.forecast_insights?.demand_trend}</p>
        </div>
    );
};
""")

async def main():
    """Main test function."""
    print("🚀 WAREHOUSE SEASONAL PREDICTION - ITEM DETAILS TEST")
    print("=" * 60)
    print("Testing item-to-item demand prediction capabilities")
    print()
    
    # Test item predictions
    await test_item_predictions()
    
    # Show database integration
    show_database_integration()
    
    print("\n" + "=" * 60)
    print("🎉 ITEM PREDICTION TEST COMPLETED!")
    print()
    print("📋 What you can now do:")
    print("• Predict demand for any individual product")
    print("• Get detailed historical analysis")
    print("• Receive inventory recommendations")
    print("• Compare products and categories")
    print("• Identify trends and seasonality patterns")
    print()
    print("🔗 Next steps:")
    print("• Connect to your actual MongoDB database")
    print("• Train models with your historical data") 
    print("• Integrate with your existing dashboards")
    print("• Set up automated forecasting")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n👋 Test interrupted. Goodbye!")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("Make sure you have installed all dependencies: pip install -r requirements.txt")
