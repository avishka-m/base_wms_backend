# #!/usr/bin/env python3
# """
# Basic Setup Test Script

# This script tests if the core dependencies are installed and working.
# """

# import sys
# from pathlib import Path
# import pandas as pd
# import numpy as np
# from datetime import datetime, timedelta

# def test_core_imports():
#     """Test basic imports"""
#     print("🧪 Testing Core Dependencies")
#     print("=" * 40)
    
#     try:
#         import pandas as pd
#         print("✅ pandas:", pd.__version__)
#     except ImportError as e:
#         print("❌ pandas import failed:", e)
#         return False
    
#     try:
#         import numpy as np
#         print("✅ numpy:", np.__version__)
#     except ImportError as e:
#         print("❌ numpy import failed:", e)
#         return False
    
#     try:
#         from prophet import Prophet
#         print("✅ prophet: Available")
#     except ImportError as e:
#         print("❌ prophet import failed:", e)
#         return False
    
#     try:
#         import sklearn
#         print("✅ scikit-learn:", sklearn.__version__)
#     except ImportError as e:
#         print("❌ scikit-learn import failed:", e)
#         return False
    
#     try:
#         import matplotlib.pyplot as plt
#         print("✅ matplotlib: Available")
#     except ImportError as e:
#         print("❌ matplotlib import failed:", e)
#         return False
    
#     print("\n🎉 All core dependencies are working!")
#     return True

# def test_prophet_basic():
#     """Test basic Prophet functionality"""
#     print("\n🔮 Testing Prophet Basic Functionality")
#     print("=" * 40)
    
#     try:
#         from prophet import Prophet
        
#         # Create sample data
#         dates = pd.date_range('2023-01-01', '2024-12-31', freq='D')
#         # Simple trend with some seasonality
#         trend = np.linspace(100, 200, len(dates))
#         seasonal = 10 * np.sin(2 * np.pi * np.arange(len(dates)) / 365.25)
#         noise = np.random.normal(0, 5, len(dates))
        
#         df = pd.DataFrame({
#             'ds': dates,
#             'y': trend + seasonal + noise
#         })
        
#         print(f"✅ Sample data created: {len(df)} records")
#         print(f"   Date range: {df['ds'].min()} to {df['ds'].max()}")
#         print(f"   Value range: {df['y'].min():.2f} to {df['y'].max():.2f}")
        
#         # Train Prophet model
#         model = Prophet(daily_seasonality=False)
#         model.fit(df)
#         print("✅ Prophet model trained successfully")
        
#         # Make future predictions
#         future = model.make_future_dataframe(periods=30)
#         forecast = model.predict(future)
#         print(f"✅ Forecast generated: {len(forecast)} predictions")
        
#         # Show sample predictions
#         last_predictions = forecast.tail(5)[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
#         print("\n📊 Sample Predictions (last 5 days):")
#         for _, row in last_predictions.iterrows():
#             print(f"   {row['ds'].strftime('%Y-%m-%d')}: {row['yhat']:.2f} "
#                   f"[{row['yhat_lower']:.2f}, {row['yhat_upper']:.2f}]")
        
#         return True
        
#     except Exception as e:
#         print(f"❌ Prophet test failed: {e}")
#         return False

# def test_custom_modules():
#     """Test custom module imports"""
#     print("\n🛠️ Testing Custom Modules")
#     print("=" * 40)
    
#     # Add the src directory to Python path
#     current_dir = Path(__file__).parent
#     src_dir = current_dir / "src"
#     sys.path.insert(0, str(src_dir))
    
#     try:
#         from models.prophet_forecaster import ProphetForecaster
#         print("✅ ProphetForecaster import successful")
#     except ImportError as e:
#         print(f"❌ ProphetForecaster import failed: {e}")
#         return False
    
#     try:
#         from item_analysis_service import ItemAnalysisService
#         print("✅ ItemAnalysisService import successful")
#     except ImportError as e:
#         print(f"❌ ItemAnalysisService import failed: {e}")
#         return False
    
#     try:
#         from data_orchestrator import SeasonalDataOrchestrator
#         print("✅ SeasonalDataOrchestrator import successful")
#     except ImportError as e:
#         print(f"❌ SeasonalDataOrchestrator import failed: {e}")
#         return False
    
#     try:
#         import config
#         print("✅ Config module import successful")
#     except ImportError as e:
#         print(f"❌ Config module import failed: {e}")
#         return False
    
#     return True

# def test_kaggle_connection():
#     """Test Kaggle API connection"""
#     print("\n📦 Testing Kaggle Connection")
#     print("=" * 40)
    
#     try:
#         from kaggle.api.kaggle_api_extended import KaggleApi
        
#         api = KaggleApi()
#         api.authenticate()
#         print("✅ Kaggle API authentication successful")
        
#         # Test basic API call (list datasets instead of competitions)
#         datasets = api.dataset_list(search="ecommerce", page=1)
#         print(f"✅ Kaggle API connection working - found {len(datasets)} datasets")
#         return True
        
#     except Exception as e:
#         print(f"❌ Kaggle connection failed: {e}")
#         print("   Make sure you've configured your Kaggle credentials")
#         return False

# def main():
#     """Run all tests"""
#     print("🚀 SEASONAL INVENTORY SETUP TEST")
#     print("=" * 50)
    
#     # Test results
#     results = {
#         "core_imports": test_core_imports(),
#         "prophet_basic": test_prophet_basic(),
#         "custom_modules": test_custom_modules(),
#         "kaggle_connection": test_kaggle_connection()
#     }
    
#     print("\n" + "=" * 50)
#     print("📋 TEST SUMMARY")
#     print("=" * 50)
    
#     for test_name, passed in results.items():
#         status = "✅ PASS" if passed else "❌ FAIL"
#         print(f"{test_name:<20}: {status}")
    
#     all_passed = all(results.values())
#     if all_passed:
#         print("\n🎉 All tests passed! You're ready to start using the system.")
#         print("\nNext steps:")
#         print("1. Run: python quickstart.py")
#         print("2. Or run: python test_item_predictions.py")
#     else:
#         print("\n⚠️ Some tests failed. Please fix the issues before proceeding.")
    
#     return all_passed

# if __name__ == "__main__":
#     success = main()
#     sys.exit(0 if success else 1)
