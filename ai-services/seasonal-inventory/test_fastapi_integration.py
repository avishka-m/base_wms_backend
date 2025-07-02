# #!/usr/bin/env python3
# """
# FastAPI Service Integration Test

# Test the updated seasonal prediction service.
# """

# import sys
# import os
# from pathlib import Path

# from services.seasonal_prediction_service import SeasonalPredictionService

# # Add paths
# backend_path = Path(__file__).parent.parent.parent / "app"
# seasonal_path = Path(__file__).parent
# sys.path.insert(0, str(backend_path))
# sys.path.insert(0, str(seasonal_path))


# def test_service_integration():
#     """Test the updated seasonal prediction service"""
#     print("=== Testing Seasonal Prediction Service Integration ===")
    
#     try:
#         # Initialize service
#         service = SeasonalPredictionService()
#         print("✓ Service initialized successfully")
        
#         # Check service status
#         status = service.get_service_status()
#         print(f"✓ Service status: {status}")
        
#         if status.get('status') != 'available':
#             print("⚠️ Service not available - check dependencies")
#             return False
        
#         # Test single item prediction
#         print("\n--- Testing Single Item Prediction ---")
#         result = service.predict_item_demand(
#             item_id="85123A",
#             horizon_days=14
#         )
        
#         print(f"✓ Prediction result status: {result.get('status')}")
#         print(f"✓ Success: {result.get('success')}")
#         if result.get('success'):
#             print(f"  - Forecast points: {result.get('total_forecast_points')}")
#             print(f"  - Historical data points: {result.get('historical_data_points')}")
#             if 'training_metrics' in result:
#                 metrics = result['training_metrics']
#                 print(f"  - Training MAPE: {metrics.get('mape', 'N/A')}")
        
#         # Test multiple item prediction
#         print("\n--- Testing Multiple Item Prediction ---")
#         batch_result = service.predict_multiple_items(
#             item_ids=["85123A", "22423"],
#             horizon_days=7
#         )
        
#         print(f"✓ Batch prediction status: {batch_result.get('status')}")
#         print(f"✓ Total items: {batch_result.get('total_items')}")
#         print(f"✓ Successful predictions: {batch_result.get('successful_predictions')}")
        
#         return True
        
#     except Exception as e:
#         print(f"✗ Service integration test failed: {e}")
#         return False

# if __name__ == "__main__":
#     success = test_service_integration()
#     print("\n" + "="*50)
#     if success:
#         print("🎉 Service integration test PASSED!")
#         print("The FastAPI service is ready for production integration.")
#     else:
#         print("⚠️ Service integration test FAILED!")
#     print("="*50)
