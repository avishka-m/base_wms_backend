#!/usr/bin/env python3
"""
Test script to verify the exact import issue
"""

import sys
import os
from pathlib import Path

print("=== Testing seasonal inventory imports ===")

# Add seasonal inventory path
backend_dir = Path(__file__).parent
seasonal_path = backend_dir / 'ai-services' / 'seasonal-inventory'
seasonal_path = seasonal_path.resolve()

print(f"Backend directory: {backend_dir}")
print(f"Seasonal path: {seasonal_path}")
print(f"Seasonal path exists: {seasonal_path.exists()}")

if str(seasonal_path) not in sys.path:
    sys.path.insert(0, str(seasonal_path))
    print("Added seasonal path to sys.path")

print("\n=== Step 1: Test config import ===")
try:
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "config", 
        str(seasonal_path / "config.py")
    )
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)
    PROCESSED_DIR = config_module.PROCESSED_DIR
    print(f"✅ Config imported successfully - PROCESSED_DIR: {PROCESSED_DIR}")
except Exception as e:
    print(f"❌ Config import failed: {e}")
    import traceback
    traceback.print_exc()

print("\n=== Step 2: Test ProphetForecaster import ===")
try:
    from src.models.prophet_forecaster import ProphetForecaster
    print("✅ ProphetForecaster imported successfully")
    
    # Test instantiation
    forecaster = ProphetForecaster()
    print("✅ ProphetForecaster instantiated successfully")
    
except Exception as e:
    print(f"❌ ProphetForecaster import/instantiation failed: {e}")
    import traceback
    traceback.print_exc()

print("\n=== Step 3: Test data loading ===")
try:
    import pandas as pd
    processed_file = Path(PROCESSED_DIR) / "daily_demand_by_product.csv"
    print(f"Looking for data file: {processed_file}")
    print(f"Data file exists: {processed_file.exists()}")
    
    if processed_file.exists():
        data = pd.read_csv(processed_file)
        print(f"✅ Loaded {len(data)} processed records")
    else:
        print("⚠️ No data file found")
        
except Exception as e:
    print(f"❌ Data loading failed: {e}")
    import traceback
    traceback.print_exc()
