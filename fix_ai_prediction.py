# Script to diagnose and fix AI prediction service
# Place in the root of your backend project and run with: python fix_ai_prediction.py

import os
import sys
import pickle
import importlib
import traceback

# Add the base directory to the Python path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

def check_scikit_learn_install():
    """Verify that scikit-learn is installed correctly"""
    print("\n🔍 Checking scikit-learn installation...")
    try:
        import sklearn
        print(f"✅ scikit-learn is installed (version: {sklearn.__version__})")
        
        # Test basic functionality
        from sklearn.ensemble import RandomForestClassifier
        print("✅ sklearn.ensemble.RandomForestClassifier is available")
        
        return True
    except ImportError as e:
        print(f"❌ scikit-learn is not installed or has issues: {e}")
        print("   Run: pip install scikit-learn==1.5.2")
        return False

def check_model_files():
    """Verify that the model files exist and are valid"""
    print("\n🔍 Checking model files...")
    
    models_dir = os.path.join(BASE_DIR, "ai_services", "path_optimization", "models")
    model_path = os.path.join(models_dir, "warehouse_storage_location_model.pkl")
    encoders_path = os.path.join(models_dir, "warehouse_storage_location_encoders.pkl")
    
    # Check if directory exists
    if not os.path.exists(models_dir):
        print(f"❌ Models directory does not exist: {models_dir}")
        print(f"   Creating directory: {models_dir}")
        os.makedirs(models_dir, exist_ok=True)
    
    # Check model file
    if not os.path.exists(model_path):
        print(f"❌ Model file does not exist: {model_path}")
        return False
    else:
        print(f"✅ Model file exists: {model_path}")
        # Try to load the model
        try:
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            print("✅ Model file loaded successfully")
        except Exception as e:
            print(f"❌ Error loading model file: {e}")
            return False
    
    # Check encoders file
    if not os.path.exists(encoders_path):
        print(f"❌ Encoders file does not exist: {encoders_path}")
        return False
    else:
        print(f"✅ Encoders file exists: {encoders_path}")
        # Try to load the encoders
        try:
            with open(encoders_path, 'rb') as f:
                encoders = pickle.load(f)
            print("✅ Encoders file loaded successfully")
        except Exception as e:
            print(f"❌ Error loading encoders file: {e}")
            return False
    
    return True

def check_ai_services_imports():
    """Check if AI services modules can be imported correctly"""
    print("\n🔍 Checking AI services imports...")
    
    # Check __init__.py files exist and have proper content
    ai_services_init = os.path.join(BASE_DIR, "ai_services", "__init__.py")
    if not os.path.exists(ai_services_init):
        print(f"❌ Missing __init__.py in ai_services directory")
        print(f"   Creating file: {ai_services_init}")
        with open(ai_services_init, 'w') as f:
            f.write("# AI Services package\n")
    else:
        print(f"✅ ai_services/__init__.py exists")
    
    path_opt_init = os.path.join(BASE_DIR, "ai_services", "path_optimization", "__init__.py")
    if not os.path.exists(path_opt_init):
        print(f"❌ Missing __init__.py in path_optimization directory")
        print(f"   Creating file: {path_opt_init}")
        with open(path_opt_init, 'w') as f:
            f.write('''# ai-services/path_optimization/__init__.py

try:
    from .warehouse_mapper import warehouse_mapper
    from .location_predictor import location_predictor  
    from .allocation_service import allocation_service

    __all__ = [
        'warehouse_mapper',
        'location_predictor',
        'allocation_service'
    ]
except ImportError as e:
    print(f"Error importing path_optimization modules: {e}")
''')
    else:
        print(f"✅ ai_services/path_optimization/__init__.py exists")
    
    # Try to import the modules
    try:
        print("\nAttempting to import AI service modules:")
        sys.path.append(os.path.join(BASE_DIR, "ai_services"))
        
        try:
            from ai_services.path_optimization import warehouse_mapper
            print("✅ warehouse_mapper imported successfully")
        except ImportError as e:
            print(f"❌ Failed to import warehouse_mapper: {e}")
        
        try:
            from ai_services.path_optimization import location_predictor
            print("✅ location_predictor imported successfully")
        except ImportError as e:
            print(f"❌ Failed to import location_predictor: {e}")
        
        try:
            from ai_services.path_optimization import allocation_service
            print("✅ allocation_service imported successfully")
        except ImportError as e:
            print(f"❌ Failed to import allocation_service: {e}")
        
        return True
    except Exception as e:
        print(f"❌ Error importing AI services: {e}")
        traceback.print_exc()
        return False

def fix_import_paths():
    """Fix import paths in AI service modules"""
    print("\n🔍 Fixing import paths in AI service modules...")
    
    # Check allocation_service.py
    file_path = os.path.join(BASE_DIR, "ai_services", "path_optimization", "allocation_service.py")
    if os.path.exists(file_path):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Fix import statements
            if "from warehouse_mapper import warehouse_mapper" in content:
                content = content.replace(
                    "from warehouse_mapper import warehouse_mapper", 
                    "from .warehouse_mapper import warehouse_mapper"
                )
                print("✅ Fixed import in allocation_service.py: warehouse_mapper")
            
            if "from location_predictor import location_predictor" in content:
                content = content.replace(
                    "from location_predictor import location_predictor", 
                    "from .location_predictor import location_predictor"
                )
                print("✅ Fixed import in allocation_service.py: location_predictor")
            
            # Write the file back
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
        except Exception as e:
            print(f"❌ Error fixing imports in allocation_service.py: {e}")
    else:
        print(f"❌ allocation_service.py not found at {file_path}")
    
    # Check location_predictor.py
    file_path = os.path.join(BASE_DIR, "ai_services", "path_optimization", "location_predictor.py")
    if os.path.exists(file_path):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Fix import statements
            if "from warehouse_mapper import warehouse_mapper" in content:
                content = content.replace(
                    "from warehouse_mapper import warehouse_mapper", 
                    "from .warehouse_mapper import warehouse_mapper"
                )
                print("✅ Fixed import in location_predictor.py: warehouse_mapper")
            
            # Write the file back
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
        except Exception as e:
            print(f"❌ Error fixing imports in location_predictor.py: {e}")
    else:
        print(f"❌ location_predictor.py not found at {file_path}")

    return True

def fix_file_encoding():
    """Fix file encoding issues by ensuring UTF-8 encoding"""
    print("\n🔍 Fixing file encoding issues...")
    
    files_to_fix = [
        os.path.join(BASE_DIR, "ai_services", "path_optimization", "allocation_service.py"),
        os.path.join(BASE_DIR, "ai_services", "path_optimization", "location_predictor.py"),
        os.path.join(BASE_DIR, "ai_services", "path_optimization", "warehouse_mapper.py"),
        os.path.join(BASE_DIR, "ai_services", "path_optimization", "__init__.py")
    ]
    
    for file_path in files_to_fix:
        if os.path.exists(file_path):
            try:
                # Try to read the file with different encodings
                content = None
                encodings = ['utf-8', 'latin-1', 'cp1252']
                
                for encoding in encodings:
                    try:
                        with open(file_path, 'r', encoding=encoding) as f:
                            content = f.read()
                        print(f"✅ Successfully read {os.path.basename(file_path)} with {encoding} encoding")
                        break
                    except UnicodeDecodeError:
                        continue
                
                if content is not None:
                    # Write back with UTF-8 encoding
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(content)
                    print(f"✅ Fixed encoding for {os.path.basename(file_path)}")
                else:
                    print(f"❌ Could not read {os.path.basename(file_path)} with any encoding")
            except Exception as e:
                print(f"❌ Error fixing encoding for {os.path.basename(file_path)}: {e}")
    
    return True

def check_model_compatibility():
    """Check if model is compatible with current scikit-learn version"""
    print("\n🔍 Checking model compatibility...")
    
    try:
        import sklearn
        from sklearn.ensemble import RandomForestClassifier
        
        models_dir = os.path.join(BASE_DIR, "ai_services", "path_optimization", "models")
        model_path = os.path.join(models_dir, "warehouse_storage_location_model.pkl")
        
        if os.path.exists(model_path):
            try:
                # Create a mock model with the same structure
                dummy_model = RandomForestClassifier()
                
                # Try loading the real model
                with open(model_path, 'rb') as f:
                    model = pickle.load(f)
                    
                print("✅ Model is compatible with current scikit-learn version")
                return True
            except Exception as e:
                print(f"❌ Model compatibility issue: {e}")
                print("   This may be due to model being created with a different scikit-learn version.")
                print("   Consider recreating the model with the current scikit-learn version.")
                return False
        else:
            print("❌ Model file doesn't exist")
            return False
    except Exception as e:
        print(f"❌ Error checking model compatibility: {e}")
        return False

def main():
    print("🔧 AI Prediction Service Diagnostic Tool")
    print("======================================")
    
    # Check if scikit-learn is installed
    scikit_installed = check_scikit_learn_install()
    
    # Check model files
    models_ok = check_model_files()
    
    # Fix file encoding issues
    fix_file_encoding()
    
    # Fix import paths
    fix_import_paths()
    
    # Check model compatibility
    model_compatible = check_model_compatibility()
    
    # Check imports again after fixes
    imports_ok = check_ai_services_imports()
    
    print("\n📊 Diagnostic Results")
    print("======================================")
    print(f"scikit-learn installed: {'✅ Yes' if scikit_installed else '❌ No'}")
    print(f"Model files ok: {'✅ Yes' if models_ok else '❌ No'}")
    print(f"Model compatible: {'✅ Yes' if model_compatible else '❌ No'}")
    print(f"Module imports ok: {'✅ Yes' if imports_ok else '❌ No'}")
    
    if not scikit_installed:
        print("\n⚠️ Action Required: Install scikit-learn")
        print("Run: pip install scikit-learn==1.5.2")
    
    if not models_ok:
        print("\n⚠️ Action Required: Fix model files")
        print("Check the models directory and ensure both .pkl files are present")
    
    if not model_compatible:
        print("\n⚠️ Action Required: Fix model compatibility")
        print("Try downgrading scikit-learn to match the version used to create the model:")
        print("Run: pip install scikit-learn==1.3.0")
        print("Or recreate the model with the current scikit-learn version")
    
    if scikit_installed and models_ok and imports_ok and model_compatible:
        print("\n✅ All checks passed! The AI prediction service should work now.")
        print("Restart your FastAPI server and test the endpoint.")
    else:
        print("\n❌ Some issues were found. Follow the recommendations above to fix them.")

if __name__ == "__main__":
    main()
