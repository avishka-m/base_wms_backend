#!/usr/bin/env python3
"""
Simple API test for Prophet forecasting system
"""
import sys
import os
from pathlib import Path

# Add the base_wms_backend to path
current_dir = Path(__file__).parent
base_dir = current_dir
sys.path.insert(0, str(base_dir))
os.chdir(str(base_dir))

print("🚀 Starting Prophet Forecasting API Test")
print(f"Working Directory: {os.getcwd()}")
print(f"Python Path includes: {base_dir}")

try:
    # Test the service directly without async
    print("\n📊 Testing Service Import...")
    from app.services.prophet_forecasting_service import ProphetForecastingService
    
    print("✅ Service imported successfully!")
    
    # Initialize service
    print("\n🔧 Initializing service...")
    service = ProphetForecastingService()
    
    # Check if service is available
    if service._available:
        print("✅ Service initialized successfully!")
        
        # Get basic status
        print("\n📈 Getting service status...")
        status = service.get_service_status()
        print(f"Status: {status}")
        
        # Try category status
        print("\n📊 Getting category status...")
        cat_status = service.get_category_status()
        print(f"Category Status: {cat_status.get('status', 'Unknown')}")
        print(f"Categories: {cat_status.get('categories', [])}")
        
        if cat_status.get('status') == 'ready':
            print("✅ System is ready for forecasting!")
            
            # List available categories and their models
            models = cat_status.get('category_models', {})
            for category, info in models.items():
                status_emoji = "✅" if info.get('status') == 'trained' else "❌"
                print(f"  {status_emoji} {category}: {info.get('status', 'unknown')}")
        else:
            print(f"⚠️ System status: {cat_status.get('status', 'Unknown')}")
            if 'error' in cat_status:
                print(f"Error: {cat_status['error']}")
    else:
        print(f"❌ Service initialization failed: {service._initialization_error}")

except ImportError as e:
    print(f"❌ Import Error: {e}")
    print("This might be due to missing dependencies or path issues")
except Exception as e:
    print(f"❌ Unexpected Error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*50)
print("Test Complete!")
