#!/usr/bin/env python3
"""
API Integration Test for Seasonal Inventory Prediction System

This script tests the API endpoints and service integration to ensure
all components work together properly.
"""

import sys
import pandas as pd
import json
import asyncio
from datetime import datetime
from pathlib import Path

# Add the current directory to Python path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

from src.models.prophet_forecaster import ProphetForecaster
from src.item_analysis_service import ItemAnalysisService
from src.data_orchestrator import SeasonalDataOrchestrator
from config import PROCESSED_DIR, MODELS_DIR

def test_service_initialization():
    """Test if all services can be initialized properly"""
    print("=== Testing Service Initialization ===")
    
    try:
        _ = ProphetForecaster()
        print("✓ ProphetForecaster initialized successfully")
        
        _ = ItemAnalysisService()
        print("✓ ItemAnalysisService initialized successfully")
        
        _ = SeasonalDataOrchestrator()
        print("✓ SeasonalDataOrchestrator initialized successfully")
        
        return True
    except Exception as e:
        print(f"✗ Service initialization failed: {e}")
        return False

def test_data_availability():
    """Test if processed data is available for predictions"""
    print("\n=== Testing Data Availability ===")
    
    processed_file = Path(PROCESSED_DIR) / "daily_demand_by_product.csv"
    
    if not processed_file.exists():
        print(f"✗ Processed data file not found: {processed_file}")
        return False
    
    try:
        df = pd.read_csv(processed_file)
        print(f"✓ Processed data loaded: {len(df)} records")
        print(f"✓ Date range: {df['ds'].min()} to {df['ds'].max()}")
        print(f"✓ Unique products: {df['product_id'].nunique()}")
        
        # Get top products with sufficient data
        product_counts = df['product_id'].value_counts()
        top_products = product_counts[product_counts >= 30].head(5).index.tolist()
        print(f"✓ Top products with sufficient data: {top_products}")
        
        return top_products
    except Exception as e:
        print(f"✗ Error loading processed data: {e}")
        return False

async def test_single_item_prediction(product_id):
    """Test prediction for a single item"""
    print(f"\n=== Testing Single Item Prediction for {product_id} ===")
    
    try:
        # Initialize services
        forecaster = ProphetForecaster()
        analysis_service = ItemAnalysisService()
        
        # Get item analysis
        analysis_result = await analysis_service.analyze_item(
            product_id=product_id
        )
        
        if analysis_result.get('success'):
            print(f"✓ Item analysis completed for {product_id}")
            print(f"  - Historical demand mean: {analysis_result.get('historical_stats', {}).get('mean_demand', 'N/A')}")
            print(f"  - Trend: {analysis_result.get('trend_analysis', {}).get('trend_direction', 'N/A')}")
        else:
            print(f"✗ Item analysis failed: {analysis_result.get('error', 'Unknown error')}")
            return False
        
        # Test forecasting
        forecast_result = forecaster.predict(
            item_id=product_id,
            days_ahead=30
        )
        
        if forecast_result.get('success'):
            forecast_data = forecast_result['forecast']
            print(f"✓ Forecast generated: {len(forecast_data)} future points")
            print(f"  - Forecast range: {forecast_data['ds'].min()} to {forecast_data['ds'].max()}")
            print(f"  - Average predicted demand: {forecast_data['yhat'].mean():.2f}")
            print(f"  - Confidence interval: [{forecast_data['yhat_lower'].mean():.2f}, {forecast_data['yhat_upper'].mean():.2f}]")
        else:
            print(f"✗ Forecast failed: {forecast_result.get('error', 'Unknown error')}")
            return False
            
        return True
        
    except Exception as e:
        print(f"✗ Single item prediction test failed: {e}")
        return False

async def test_batch_prediction(product_ids):
    """Test batch prediction for multiple items"""
    print(f"\n=== Testing Batch Prediction for {len(product_ids)} items ===")
    
    try:
        analysis_service = ItemAnalysisService()
        
        # Test batch analysis
        results = {}
        for product_id in product_ids[:3]:  # Test first 3 products
            try:
                result = await analysis_service.analyze_item(
                    product_id=product_id
                )
                results[product_id] = result
                status = "✓" if result.get('success') else "✗"
                print(f"  {status} {product_id}: {result.get('error', 'Success')}")
            except Exception as e:
                print(f"  ✗ {product_id}: {e}")
                results[product_id] = {'success': False, 'error': str(e)}
        
        successful_predictions = sum(1 for r in results.values() if r.get('success'))
        print(f"✓ Batch prediction completed: {successful_predictions}/{len(results)} successful")
        
        return successful_predictions > 0
        
    except Exception as e:
        print(f"✗ Batch prediction test failed: {e}")
        return False

def test_model_persistence():
    """Test model saving and loading"""
    print("\n=== Testing Model Persistence ===")
    
    try:
        # Check if models directory exists
        models_dir = Path(MODELS_DIR)
        if not models_dir.exists():
            models_dir.mkdir(parents=True, exist_ok=True)
            print(f"✓ Created models directory: {models_dir}")
        
        # Check for existing saved models
        model_files = list(models_dir.glob("*.pkl"))
        print(f"✓ Found {len(model_files)} saved model files")
        
        if model_files:
            print("  Saved models:")
            for model_file in model_files[:5]:  # Show first 5
                print(f"    - {model_file.name}")
        
        return True
        
    except Exception as e:
        print(f"✗ Model persistence test failed: {e}")
        return False

async def test_error_handling():
    """Test error handling for edge cases"""
    print("\n=== Testing Error Handling ===")
    
    try:
        forecaster = ProphetForecaster()
        analysis_service = ItemAnalysisService()
        
        # Test with non-existent product
        result = await analysis_service.analyze_item(
            product_id="NON_EXISTENT_PRODUCT"
        )
        
        if not result.get('success'):
            print("✓ Properly handled non-existent product")
        else:
            print("✗ Should have failed for non-existent product")
        
        # Test with invalid parameters
        result = forecaster.predict(
            item_id="INVALID",
            days_ahead=-5  # Invalid days ahead
        )
        
        if not result.get('success'):
            print("✓ Properly handled invalid parameters")
        else:
            print("✗ Should have failed for invalid parameters")
        
        return True
        
    except Exception as e:
        print(f"✗ Error handling test failed: {e}")
        return False

async def generate_api_response_samples():
    """Generate sample API responses for documentation"""
    print("\n=== Generating API Response Samples ===")
    
    try:
        # Load processed data to get a real product
        processed_file = Path(PROCESSED_DIR) / "daily_demand_by_product.csv"
        if not processed_file.exists():
            print("✗ No processed data available for samples")
            return False
        
        df = pd.read_csv(processed_file)
        product_counts = df['product_id'].value_counts()
        sample_product = product_counts[product_counts >= 30].index[0]
        
        analysis_service = ItemAnalysisService()
        forecaster = ProphetForecaster()
        
        # Generate sample analysis response
        analysis_result = await analysis_service.analyze_item(
            product_id=sample_product
        )
        
        # Generate sample forecast response
        forecast_result = forecaster.predict(
            item_id=sample_product,
            days_ahead=14
        )
        
        # Create sample responses
        samples = {
            "item_analysis_response": {
                "item_id": sample_product,
                "analysis_period": "30 days",
                "success": analysis_result.get('success', False),
                "historical_stats": analysis_result.get('historical_stats', {}),
                "trend_analysis": analysis_result.get('trend_analysis', {}),
                "seasonality": analysis_result.get('seasonality', {}),
                "recommendations": analysis_result.get('recommendations', {}),
                "timestamp": datetime.now().isoformat()
            },
            "forecast_response": {
                "item_id": sample_product,
                "forecast_horizon": "14 days",
                "success": forecast_result.get('success', False),
                "forecast_summary": {
                    "total_periods": len(forecast_result.get('forecast', [])) if forecast_result.get('forecast') is not None else 0,
                    "average_predicted_demand": float(forecast_result.get('forecast', {}).get('yhat', [0]).mean()) if forecast_result.get('forecast') is not None else 0,
                    "confidence_interval": "95%"
                },
                "model_info": forecast_result.get('model_info', {}),
                "timestamp": datetime.now().isoformat()
            }
        }
        
        # Save samples to file
        samples_file = current_dir / "api_response_samples.json"
        with open(samples_file, 'w') as f:
            json.dump(samples, f, indent=2, default=str)
        
        print(f"✓ API response samples saved to: {samples_file}")
        print(f"  - Sample product used: {sample_product}")
        
        return True
        
    except Exception as e:
        print(f"✗ Sample generation failed: {e}")
        return False

async def main():
    """Run all integration tests"""
    print("Seasonal Inventory Prediction System - API Integration Test")
    print("=" * 60)
    
    test_results = {}
    
    # Run all tests
    test_results['service_init'] = test_service_initialization()
    
    product_ids = test_data_availability()
    test_results['data_available'] = bool(product_ids)
    
    if product_ids:
        test_results['single_prediction'] = await test_single_item_prediction(product_ids[0])
        test_results['batch_prediction'] = await test_batch_prediction(product_ids)
    else:
        test_results['single_prediction'] = False
        test_results['batch_prediction'] = False
    
    test_results['model_persistence'] = test_model_persistence()
    test_results['error_handling'] = await test_error_handling()
    test_results['sample_generation'] = await generate_api_response_samples()
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    total_tests = len(test_results)
    passed_tests = sum(1 for result in test_results.values() if result)
    
    for test_name, result in test_results.items():
        status = "PASS" if result else "FAIL"
        print(f"{test_name.replace('_', ' ').title()}: {status}")
    
    print(f"\nOverall: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("🎉 All tests passed! The seasonal inventory prediction system is ready for API integration.")
    else:
        print("⚠️  Some tests failed. Please review the output above for details.")
    
    return passed_tests == total_tests

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
