# Fix for allocation_service.py

import os
import sys
import pickle
import logging
from datetime import datetime

# Add the base directory to the Python path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

def fix_ai_prediction_imports():
    """Fix imports in AI prediction modules"""
    print("\n🔧 Fixing AI Module Imports")
    print("======================================")
    
    # Fix allocation_service.py imports
    allocation_file = os.path.join(BASE_DIR, "ai_services", "path_optimization", "allocation_service.py")
    if os.path.exists(allocation_file):
        try:
            with open(allocation_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            if "from warehouse_mapper import warehouse_mapper" in content:
                content = content.replace(
                    "from warehouse_mapper import warehouse_mapper",
                    "from .warehouse_mapper import warehouse_mapper"
                )
                print("✅ Fixed allocation_service.py: warehouse_mapper import")
            
            if "from location_predictor import location_predictor" in content:
                content = content.replace(
                    "from location_predictor import location_predictor",
                    "from .location_predictor import location_predictor"
                )
                print("✅ Fixed allocation_service.py: location_predictor import")
            
            with open(allocation_file, 'w', encoding='utf-8') as f:
                f.write(content)
        except Exception as e:
            print(f"❌ Error fixing allocation_service.py: {e}")
    else:
        print(f"❌ allocation_service.py not found at {allocation_file}")
    
    # Fix location_predictor.py imports
    predictor_file = os.path.join(BASE_DIR, "ai_services", "path_optimization", "location_predictor.py")
    if os.path.exists(predictor_file):
        try:
            with open(predictor_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            if "from warehouse_mapper import warehouse_mapper" in content:
                content = content.replace(
                    "from warehouse_mapper import warehouse_mapper",
                    "from .warehouse_mapper import warehouse_mapper"
                )
                print("✅ Fixed location_predictor.py: warehouse_mapper import")
            
            # Update model loading with better error handling
            if "def load_model(self):" in content:
                start_idx = content.find("def load_model(self):")
                end_idx = content.find("def load_encoders(self):", start_idx)
                
                if start_idx > 0 and end_idx > 0:
                    old_method = content[start_idx:end_idx]
                    
                    new_method = """def load_model(self):
        \"\"\"Load the trained model from pickle file with improved error handling\"\"\"
        try:
            with open(self.model_path, 'rb') as f:
                self.model = pickle.load(f)
            print(f"✅ Model loaded successfully from {self.model_path}")
        except FileNotFoundError:
            print(f"⚠️ Model file not found at {self.model_path}")
            print("   Using fallback prediction logic")
            self.model = None
        except (pickle.UnpicklingError, ModuleNotFoundError, AttributeError, ImportError, ValueError) as e:
            print(f"⚠️ Error loading model (may be from different Python/sklearn version): {str(e)}")
            print("   Using fallback prediction logic")
            self.model = None
        except Exception as e:
            print(f"⚠️ Unexpected error loading model: {str(e)}")
            self.model = None
    
    """
                    
                    content = content[:start_idx] + new_method + content[end_idx:]
                    print("✅ Updated location_predictor.py: Added robust model loading")
            
            with open(predictor_file, 'w', encoding='utf-8') as f:
                f.write(content)
        except Exception as e:
            print(f"❌ Error fixing location_predictor.py: {e}")
    else:
        print(f"❌ location_predictor.py not found at {predictor_file}")
    
    print("\n✅ AI module import fixes complete")

if __name__ == "__main__":
    fix_ai_prediction_imports()
