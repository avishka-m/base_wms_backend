"""
Debug script to understand what's happening with the allocation service
"""

import os
import sys
import asyncio

# Add the required paths
sys.path.append(os.path.join(os.path.dirname(__file__), 'ai_services'))

def debug_import():
    """Debug the import of allocation_service"""
    print("Trying to import allocation_service...")
    try:
        from ai_services.path_optimization.allocation_service import allocation_service
        print("✅ Successfully imported allocation_service!")
        print(f"allocation_service type: {type(allocation_service)}")
        print(f"allocation_service methods: {dir(allocation_service)}")
        return True, allocation_service
    except ImportError as e:
        print(f"❌ Import Error: {e}")
        return False, None
    except Exception as e:
        print(f"❌ Other Error: {e}")
        print(f"Error type: {type(e).__name__}")
        return False, None

async def debug_allocation_call():
    """Test the allocation service call"""
    success, allocation_svc = debug_import()
    if not success:
        return
    
    print("\nTesting allocation service call...")
    try:
        # Mock data for testing
        mock_data = {
            "item_id": 123,
            "category": "Electronics",
            "item_size": "M",
            "quantity": 1
        }
        
        # We'll skip the DB collections for initial testing
        result = await allocation_svc.allocate_location_for_item(
            item_id=mock_data["item_id"],
            category=mock_data["category"],
            item_size=mock_data["item_size"],
            quantity=mock_data["quantity"],
            db_collection_seasonal=None,
            db_collection_storage=None,
            db_collection_location_inventory=None
        )
        print("✅ Allocation call successful!")
        print(f"Result: {result}")
        return True, result
    except Exception as e:
        print(f"❌ Allocation call failed: {e}")
        import traceback
        print(traceback.format_exc())
        return False, None

def check_predictor():
    """Check the location_predictor module"""
    print("\nChecking location_predictor module...")
    try:
        from ai_services.path_optimization.location_predictor import LocationPredictor
        print("✅ Successfully imported LocationPredictor!")
        
        print("Attempting to create LocationPredictor instance...")
        predictor = LocationPredictor()
        print("✅ Successfully created LocationPredictor instance!")
        print(f"Methods: {dir(predictor)}")
        
        return True, predictor
    except Exception as e:
        print(f"❌ LocationPredictor error: {e}")
        import traceback
        print(traceback.format_exc())
        return False, None

def check_warehouse_mapper():
    """Check the warehouse_mapper module"""
    print("\nChecking warehouse_mapper module...")
    try:
        from ai_services.path_optimization.warehouse_mapper import warehouse_mapper
        print("✅ Successfully imported warehouse_mapper!")
        print(f"warehouse_mapper type: {type(warehouse_mapper)}")
        print(f"warehouse_mapper has attributes: {dir(warehouse_mapper)}")
        return True, warehouse_mapper
    except Exception as e:
        print(f"❌ warehouse_mapper error: {e}")
        import traceback
        print(traceback.format_exc())
        return False, None
    
def check_models():
    """Check the models directory and files"""
    print("\nChecking models directory...")
    models_dir = os.path.join(os.path.dirname(__file__), 'ai_services', 'path_optimization', 'models')
    if not os.path.exists(models_dir):
        print(f"❌ Models directory does not exist: {models_dir}")
        return False
    
    print(f"✅ Models directory exists: {models_dir}")
    model_files = os.listdir(models_dir)
    print(f"Files in models directory: {model_files}")
    
    required_files = ['location_model.pkl', 'category_encoder.pkl', 'size_encoder.pkl', 'fallback_model.pkl']
    missing_files = [f for f in required_files if f not in model_files]
    
    if missing_files:
        print(f"❌ Missing required model files: {missing_files}")
        return False
    else:
        print("✅ All required model files exist!")
        
    # Check file sizes
    for file in required_files:
        file_path = os.path.join(models_dir, file)
        file_size = os.path.getsize(file_path)
        print(f"  - {file}: {file_size} bytes")
    
    return True

if __name__ == "__main__":
    print("="*50)
    print(" AI ALLOCATION SERVICE DEBUGGER ")
    print("="*50)
    
    # Check Python version
    print(f"Python version: {sys.version}")
    
    # Check module imports
    debug_import()
    
    # Check model files
    check_models()
    
    # Check location predictor
    check_predictor()
    
    # Check warehouse mapper
    check_warehouse_mapper()
    
    # Test allocation call
    asyncio.run(debug_allocation_call())
    
    print("\nDebug completed!")
