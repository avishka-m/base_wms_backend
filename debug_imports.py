#!/usr/bin/env python3
"""
Debug script to test the import chain and identify the exact issue
"""

import sys
import os
from pathlib import Path

print("=== Python Environment Debug ===")
print(f"Python executable: {sys.executable}")
print(f"Python version: {sys.version}")
print(f"Current working directory: {os.getcwd()}")
print(f"Python path: {sys.path}")

print("\n=== Testing basic imports ===")
try:
    import kaggle
    print("✅ kaggle import successful")
except ImportError as e:
    print(f"❌ kaggle import failed: {e}")

try:
    import prophet
    print("✅ prophet import successful")
except ImportError as e:
    print(f"❌ prophet import failed: {e}")

print("\n=== Testing seasonal inventory path setup ===")
# Add seasonal inventory path
seasonal_path = os.path.join(
    os.path.dirname(__file__), 
    'ai-services', 
    'seasonal-inventory'
)
seasonal_path = os.path.abspath(seasonal_path)
print(f"Seasonal path: {seasonal_path}")
print(f"Seasonal path exists: {os.path.exists(seasonal_path)}")

if seasonal_path not in sys.path:
    sys.path.insert(0, seasonal_path)
    print("Added seasonal path to sys.path")

print("\n=== Testing seasonal imports ===")
try:
    from src.models.prophet_forecaster import ProphetForecaster
    print("✅ ProphetForecaster import successful")
except ImportError as e:
    print(f"❌ ProphetForecaster import failed: {e}")
    import traceback
    traceback.print_exc()

try:
    from config import PROCESSED_DIR
    print("✅ config import successful")
    print(f"PROCESSED_DIR: {PROCESSED_DIR}")
except ImportError as e:
    print(f"❌ config import failed: {e}")
    import traceback
    traceback.print_exc()
