#!/usr/bin/env python3
'''
Setup script for Warehouse Management System
Run this on a new system to set up the environment
'''

import os
import sys
import subprocess
import shutil
from pathlib import Path

def check_python_version():
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ required")
        return False
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor}")
    return True

def install_requirements():
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Requirements installed")
        return True
    except:
        print("❌ Failed to install requirements")
        return False

def setup_environment():
    # Copy .env template if .env doesn't exist
    if not os.path.exists('.env'):
        if os.path.exists('.env.template'):
            shutil.copy('.env.template', '.env')
            print("✅ Created .env from template")
            print("⚠️  Please edit .env to configure your MongoDB connection")
        else:
            print("❌ No .env template found")
            return False
    else:
        print("✅ .env already exists")
    return True

def check_mongodb():
    try:
        from pymongo import MongoClient
        from app.config import MONGODB_URL
        
        client = MongoClient(MONGODB_URL, serverSelectionTimeoutMS=5000)
        client.server_info()
        print("✅ MongoDB connection successful")
        client.close()
        return True
    except Exception as e:
        print(f"❌ MongoDB connection failed: {e}")
        print("Please check your MONGODB_URL in .env")
        return False

def main():
    print("🚀 Warehouse Management System Setup")
    print("=" * 40)
    
    if not check_python_version():
        return False
        
    if not install_requirements():
        return False
        
    if not setup_environment():
        return False
        
    if not check_mongodb():
        print("🔧 MongoDB setup required - see .env file")
        return False
    
    print("\n🎉 Setup complete! You can now run the system.")
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
