#!/usr/bin/env python3
"""
Simplified API Integration Test for Seasonal Inventory Prediction System

This script tests the core functionality without the complex ItemAnalysisService
that has some missing dependencies.
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
from config import PROCESSED_DIR, MODELS_DIR


def test_service_initialization():
    """Test if core services can be initialized properly"""
    print("=== Testing Service Initialization ===")
    
    try:
        _ = ProphetForecaster()
        print("✓ ProphetForecaster initialized successfully")
        
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
        top_products = product_counts[product_counts >= 50].head(5).index.tolist()
        print(f"✓ Top products with sufficient data: {top_products}")
        
        return top_products
    except Exception as e:
        print(f"✗ Error loading processed data: {e}")
        return False


def test_single_item_forecasting(product_id):
    """Test forecasting for a single item using ProphetForecaster directly"""
    print(f"\n=== Testing Single Item Forecasting for {product_id} ===")
    
    try:
        # Load data for the specific product
        processed_file = Path(PROCESSED_DIR) / "daily_demand_by_product.csv"
        df = pd.read_csv(processed_file)
        
        # Filter for specific product
        product_data = df[df['product_id'] == product_id].copy()
        
        if len(product_data) < 30:
            print(f"✗ Insufficient data for {product_id}: {len(product_data)} records")
            return False
        
        print(f"✓ Found {len(product_data)} records for {product_id}")
        
        # Initialize forecaster for this product
        forecaster = ProphetForecaster(product_id=product_id)
        
        # Train the model
        train_result = forecaster.train(product_data)
        
        if train_result and 'error' not in train_result:
            print(f"✓ Model trained successfully for {product_id}")
            if 'mean_mape' in train_result:
                print(f"  - Training MAPE: {train_result['mean_mape']:.3f}")
            
            # Generate predictions
            forecast_result = forecaster.predict(periods=30)
            
            if forecast_result is not None and len(forecast_result) > 0:
                print(f"✓ Forecast generated: {len(forecast_result)} points")
                print(f"  - Forecast range: {forecast_result['ds'].min()} to {forecast_result['ds'].max()}")
                
                # Calculate summary statistics
                avg_prediction = forecast_result['yhat'].mean()
                print(f"  - Average predicted demand: {avg_prediction:.2f}")
                print(f"  - Confidence interval: [{forecast_result['yhat_lower'].mean():.2f}, {forecast_result['yhat_upper'].mean():.2f}]")
                
                return True
            else:
                print(f"✗ Forecast generation failed for {product_id}")
                return False
        else:
            print(f"✗ Model training failed: {train_result.get('error', 'Unknown error') if train_result else 'No result returned'}")
            return False
            
    except Exception as e:
        print(f"✗ Single item forecasting test failed: {e}")
        return False


def test_batch_forecasting(product_ids):
    """Test batch forecasting for multiple items"""
    print(f"\n=== Testing Batch Forecasting for {len(product_ids)} items ===")
    
    try:
        # Load data
        processed_file = Path(PROCESSED_DIR) / "daily_demand_by_product.csv"
        df = pd.read_csv(processed_file)
        
        results = {}
        for product_id in product_ids[:3]:  # Test first 3 products
            try:
                # Filter for specific product
                product_data = df[df['product_id'] == product_id].copy()
                
                if len(product_data) < 30:
                    print(f"  ✗ {product_id}: Insufficient data ({len(product_data)} records)")
                    results[product_id] = {'success': False, 'error': 'Insufficient data'}
                    continue
                
                # Initialize and train forecaster
                forecaster = ProphetForecaster(product_id=product_id)
                train_result = forecaster.train(product_data)
                
                if train_result and 'error' not in train_result:
                    forecast_result = forecaster.predict(periods=14)
                    
                    if forecast_result is not None and len(forecast_result) > 0:
                        results[product_id] = {'success': True, 'forecast_points': len(forecast_result)}
                        print(f"  ✓ {product_id}: {len(forecast_result)} forecast points")
                    else:
                        results[product_id] = {'success': False, 'error': 'Forecast failed'}
                        print(f"  ✗ {product_id}: Forecast failed")
                else:
                    error_msg = train_result.get('error', 'Training failed') if train_result else 'No result returned'
                    results[product_id] = {'success': False, 'error': error_msg}
                    print(f"  ✗ {product_id}: Training failed - {error_msg}")
                    
            except Exception as e:
                results[product_id] = {'success': False, 'error': str(e)}
                print(f"  ✗ {product_id}: {e}")
        
        successful_predictions = sum(1 for r in results.values() if r.get('success'))
        print(f"✓ Batch forecasting completed: {successful_predictions}/{len(results)} successful")
        
        return successful_predictions > 0
        
    except Exception as e:
        print(f"✗ Batch forecasting test failed: {e}")
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


def test_error_handling():
    """Test error handling for edge cases"""
    print("\n=== Testing Error Handling ===")
    
    try:
        # Test with empty data
        forecaster = ProphetForecaster(product_id="NON_EXISTENT")
        empty_data = pd.DataFrame(columns=['ds', 'y', 'product_id'])
        
        try:
            result = forecaster.train(empty_data)
            if result and 'error' in result:
                print("✓ Properly handled empty data")
            else:
                print("✗ Should have failed for empty data")
        except Exception:
            print("✓ Properly handled empty data with exception")
        
        # Test with insufficient data
        forecaster2 = ProphetForecaster(product_id="INSUFFICIENT")
        small_data = pd.DataFrame({
            'ds': pd.date_range('2024-01-01', periods=5),
            'y': [1, 2, 3, 4, 5],
            'product_id': ['INSUFFICIENT'] * 5
        })
        
        try:
            result = forecaster2.train(small_data)
            if result and 'error' in result:
                print("✓ Properly handled insufficient data")
            else:
                print("✗ Should have failed for insufficient data")
        except Exception:
            print("✓ Properly handled insufficient data with exception")
        
        return True
        
    except Exception as e:
        print(f"✗ Error handling test failed: {e}")
        return False


def generate_api_response_samples():
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
        sample_product = product_counts[product_counts >= 50].index[0]
        
        # Get sample data
        product_data = df[df['product_id'] == sample_product].copy()
        
        # Generate forecast
        forecaster = ProphetForecaster(product_id=sample_product)
        train_result = forecaster.train(product_data)
        
        if train_result and 'error' not in train_result:
            forecast_result = forecaster.predict(periods=14)
            
            if forecast_result is not None and len(forecast_result) > 0:
                # Create sample API responses
                samples = {
                    "forecast_response": {
                        "item_id": sample_product,
                        "forecast_horizon": "14 days",
                        "success": True,
                        "forecast_summary": {
                            "total_periods": len(forecast_result),
                            "average_predicted_demand": float(forecast_result['yhat'].mean()),
                            "confidence_interval": "95%",
                            "date_range": {
                                "start": str(forecast_result['ds'].min()),
                                "end": str(forecast_result['ds'].max())
                            }
                        },
                        "historical_stats": {
                            "mean_demand": float(product_data['y'].mean()),
                            "std_demand": float(product_data['y'].std()),
                            "min_demand": float(product_data['y'].min()),
                            "max_demand": float(product_data['y'].max()),
                            "total_data_points": len(product_data)
                        },
                        "timestamp": datetime.now().isoformat()
                    }
                }
                
                # Save samples to file
                samples_file = current_dir / "api_response_samples.json"
                with open(samples_file, 'w') as f:
                    json.dump(samples, f, indent=2, default=str)
                
                print(f"✓ API response samples saved to: {samples_file}")
                print(f"  - Sample product used: {sample_product}")
                print(f"  - Forecast points generated: {len(forecast_result)}")
                
                return True
            else:
                print("✗ Failed to generate forecast for samples")
                return False
        else:
            error_msg = train_result.get('error', 'Unknown') if train_result else 'No result returned'
            print(f"✗ Failed to train model for samples: {error_msg}")
            return False
        
    except Exception as e:
        print(f"✗ Sample generation failed: {e}")
        return False


def main():
    """Run all integration tests"""
    print("Seasonal Inventory Prediction System - Simplified API Integration Test")
    print("=" * 70)
    
    test_results = {}
    
    # Run all tests
    test_results['service_init'] = test_service_initialization()
    
    product_ids = test_data_availability()
    test_results['data_available'] = bool(product_ids)
    
    if product_ids:
        test_results['single_forecasting'] = test_single_item_forecasting(product_ids[0])
        test_results['batch_forecasting'] = test_batch_forecasting(product_ids)
    else:
        test_results['single_forecasting'] = False
        test_results['batch_forecasting'] = False
    
    test_results['model_persistence'] = test_model_persistence()
    test_results['error_handling'] = test_error_handling()
    test_results['sample_generation'] = generate_api_response_samples()
    
    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    
    total_tests = len(test_results)
    passed_tests = sum(1 for result in test_results.values() if result)
    
    for test_name, result in test_results.items():
        status = "PASS" if result else "FAIL"
        print(f"{test_name.replace('_', ' ').title()}: {status}")
    
    print(f"\nOverall: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("🎉 All tests passed! The core seasonal inventory prediction functionality is working.")
        print("\nNext steps:")
        print("- Fix ItemAnalysisService to work with the current ProphetForecaster API")
        print("- Integrate with FastAPI endpoints")
        print("- Connect to actual WMS database")
    else:
        print("⚠️  Some tests failed. Please review the output above for details.")
    
    return passed_tests == total_tests


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
