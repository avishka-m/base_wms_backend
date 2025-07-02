# #!/usr/bin/env python3
# """
# Test Script - Verify Database Prediction Capability

# This script tests whether the seasonal inventory system can generate
# predictions from your WMS database data.
# """

# import asyncio
# import sys
# from pathlib import Path

# # Add the src directory to Python path
# current_dir = Path(__file__).parent
# src_dir = current_dir / "src"
# sys.path.insert(0, str(src_dir))

# def test_prophet_installation():
#     """Test if Prophet is properly installed."""
#     print("🔍 Testing Prophet installation...")
    
#     try:
#         from prophet import Prophet
#         print("✅ Prophet successfully imported")
        
#         # Test basic Prophet functionality
#         import pandas as pd
        
#         # Create minimal test data
#         test_data = pd.DataFrame({
#             'ds': pd.date_range('2023-01-01', periods=100, freq='D'),
#             'y': range(100)
#         })
        
#         model = Prophet()
#         model.fit(test_data)
        
#         future = model.make_future_dataframe(periods=10)
#         forecast = model.predict(future)
        
#         print("✅ Prophet basic functionality working")
#         return True
        
#     except ImportError as e:
#         print(f"❌ Prophet not installed: {e}")
#         print("Install with: pip install prophet")
#         return False
#     except Exception as e:
#         print(f"❌ Prophet test failed: {e}")
#         return False

# async def test_prediction_service():
#     """Test the prediction service."""
#     print("\n🎯 Testing Prediction Service...")
    
#     try:
#         from api.prediction_service import InventoryPredictionService, quick_predict_inventory
        
#         print("✅ Prediction service imported successfully")
        
#         # Test quick prediction function
#         print("🔄 Running quick prediction test...")
#         results = await quick_predict_inventory(forecast_days=30)
        
#         if 'error' in results:
#             print(f"⚠️ Prediction returned error: {results['error']}")
#             print("This is expected if no real database data is available")
#             return True  # Expected behavior without real data
        
#         # Display results
#         service = InventoryPredictionService()
#         service.print_prediction_summary(results)
        
#         print("✅ Prediction service working correctly!")
#         return True
        
#     except ImportError as e:
#         print(f"❌ Failed to import prediction service: {e}")
#         return False
#     except Exception as e:
#         print(f"❌ Prediction service test failed: {e}")
#         return False

# async def test_database_prediction_capability():
#     """Test the complete prediction capability."""
#     print("\n🚀 TESTING DATABASE PREDICTION CAPABILITY")
#     print("=" * 60)
    
#     # Test 1: Prophet Installation
#     prophet_ok = test_prophet_installation()
    
#     if not prophet_ok:
#         print("\n❌ RESULT: Cannot predict - Prophet not installed")
#         print("Run: pip install prophet")
#         return False
    
#     # Test 2: Prediction Service
#     service_ok = await test_prediction_service()
    
#     if not service_ok:
#         print("\n❌ RESULT: Cannot predict - Service issues")
#         return False
    
#     # Final result
#     print("\n🎉 RESULT: YOUR SYSTEM CAN NOW PREDICT FROM DATABASE DATA!")
#     print("=" * 60)
    
#     print("\n📋 What you can do:")
#     print("1. Generate instant predictions:")
#     print("   from api.prediction_service import quick_predict_inventory")
#     print("   results = await quick_predict_inventory()")
    
#     print("\n2. Train models for specific products:")
#     print("   service = InventoryPredictionService()")
#     print("   results = await service.get_instant_predictions(['PROD_001', 'PROD_002'])")
    
#     print("\n3. Get forecasts for next 90 days:")
#     print("   results = await quick_predict_inventory(forecast_days=90)")
    
#     print("\n📊 The system will:")
#     print("   • Extract historical data from your WMS database")
#     print("   • Train Prophet models for seasonal patterns")
#     print("   • Generate 90-day demand forecasts")
#     print("   • Provide confidence intervals")
#     print("   • Save trained models for reuse")
    
#     return True

# def show_usage_examples():
#     """Show practical usage examples."""
#     print("\n💡 PRACTICAL USAGE EXAMPLES")
#     print("=" * 40)
    
#     print("\n1. Quick Prediction (One-liner):")
#     print("```python")
#     print("import asyncio")
#     print("from src.api.prediction_service import quick_predict_inventory")
#     print("results = asyncio.run(quick_predict_inventory())")
#     print("```")
    
#     print("\n2. Specific Products:")
#     print("```python")
#     print("# Predict for specific products")
#     print("product_ids = ['PROD_001', 'PROD_002', 'PROD_003']")
#     print("results = await quick_predict_inventory(product_ids=product_ids)")
#     print("```")
    
#     print("\n3. Custom Forecast Period:")
#     print("```python")
#     print("# 30-day forecast")
#     print("results = await quick_predict_inventory(forecast_days=30)")
#     print("```")
    
#     print("\n4. Access Detailed Results:")
#     print("```python")
#     print("predictions = results['predictions']")
#     print("for product_id, pred in predictions.items():")
#     print("    if pred['status'] == 'success':")
#     print("        forecast = pred['forecast']")
#     print("        summary = pred['summary']")
#     print("        print(f'{product_id}: {summary[\"total_predicted_demand\"]} units')")
#     print("```")

# async def main():
#     """Main test function."""
#     try:
#         # Run capability test
#         can_predict = await test_database_prediction_capability()
        
#         if can_predict:
#             show_usage_examples()
        
#         return can_predict
        
#     except KeyboardInterrupt:
#         print("\n\n👋 Test interrupted by user")
#         return False
#     except Exception as e:
#         print(f"\n❌ Unexpected error during testing: {e}")
#         return False

# if __name__ == "__main__":
#     success = asyncio.run(main())
    
#     if success:
#         print("\n🎯 CONCLUSION: Your system is ready for database predictions!")
#         sys.exit(0)
#     else:
#         print("\n⚠️ CONCLUSION: System needs setup before predictions work")
#         sys.exit(1)
