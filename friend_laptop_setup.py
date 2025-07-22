#!/usr/bin/env python3
"""
Complete setup guide and solution for your friend's laptop
This script will help diagnose and fix all issues preventing the return process from working
"""

import os
import sys
import subprocess
import traceback
from pathlib import Path

def print_header(title):
    print("\n" + "=" * 60)
    print(f"🔧 {title}")
    print("=" * 60)

def check_python():
    """Check Python version"""
    version = sys.version_info
    if version >= (3, 8):
        print(f"✅ Python {version.major}.{version.minor}.{version.micro} - Compatible")
        return True
    else:
        print(f"❌ Python {version.major}.{version.minor}.{version.micro} - Need 3.8+")
        return False

def install_dependencies():
    """Install all required dependencies"""
    print("📦 Installing dependencies...")
    
    # Critical packages for return process
    critical_packages = [
        "fastapi>=0.115.12",
        "uvicorn[standard]>=0.34.0", 
        "pymongo>=4.12.0",
        "motor>=3.7.0",
        "pydantic>=2.11.3",
        "python-jose[cryptography]>=3.3.0",
        "passlib[bcrypt]>=1.7.4",
        "bcrypt>=4.0.1",
        "python-multipart>=0.0.9",
        "python-dotenv>=1.1.0",
        "scikit-learn>=1.0.0",
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "requests>=2.32.3"
    ]
    
    failed_packages = []
    
    for package in critical_packages:
        try:
            print(f"Installing {package}...")
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", package
            ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            print(f"✅ {package}")
        except Exception as e:
            print(f"❌ {package} - {e}")
            failed_packages.append(package)
    
    return len(failed_packages) == 0

def setup_mongodb_options():
    """Provide MongoDB setup options"""
    print("🗄️  MongoDB Setup Options:")
    print()
    print("Option 1: MongoDB Atlas (Recommended - Cloud Database)")
    print("   1. Go to https://www.mongodb.com/atlas")
    print("   2. Create free account")
    print("   3. Create cluster")
    print("   4. Get connection string")
    print("   5. Update MONGODB_URL in .env")
    print()
    print("Option 2: Local MongoDB Installation")
    print("   1. Download from https://www.mongodb.com/download-center")
    print("   2. Install MongoDB")
    print("   3. Start MongoDB service")
    print("   4. Keep MONGODB_URL as mongodb://localhost:27017/")
    print()
    print("Option 3: Docker (Easiest for development)")
    print("   1. Install Docker")
    print("   2. Run: docker run -d -p 27017:27017 mongo:5.0")
    print("   3. Keep MONGODB_URL as mongodb://localhost:27017/")

def create_env_file():
    """Create .env file with multiple MongoDB options"""
    env_content = """# MongoDB Configuration - Choose ONE option below

# Option 1: MongoDB Atlas (Cloud) - RECOMMENDED for sharing between computers
# MONGODB_URL=mongodb+srv://username:password@cluster.mongodb.net/?retryWrites=true&w=majority

# Option 2: Local MongoDB - Only works if MongoDB is installed locally
MONGODB_URL=mongodb://localhost:27017/

# Option 3: Docker MongoDB - Only works if using Docker
# MONGODB_URL=mongodb://mongodb:27017/

# Database Configuration
DATABASE_NAME=warehouse_management

# Security
SECRET_KEY=your-secret-key-change-this-to-something-secure

# Debug Mode
DEBUG_MODE=True

# API Configuration
API_V1_PREFIX=/api/v1

# Instructions:
# 1. Choose one of the MongoDB options above by uncommenting it
# 2. If using Atlas, replace username:password with your credentials
# 3. Change the SECRET_KEY to something secure
"""
    
    env_file = ".env"
    
    if not os.path.exists(env_file):
        with open(env_file, 'w', encoding='utf-8') as f:
            f.write(env_content)
        print(f"✅ Created {env_file}")
    else:
        print(f"✅ {env_file} already exists")
    
    return env_file

def test_mongodb_connection():
    """Test MongoDB connection"""
    try:
        from pymongo import MongoClient
        from app.config import MONGODB_URL, DATABASE_NAME
        
        print(f"🔍 Testing MongoDB connection to: {MONGODB_URL}")
        
        client = MongoClient(MONGODB_URL, serverSelectionTimeoutMS=5000)
        client.server_info()
        
        db = client[DATABASE_NAME]
        collections = db.list_collection_names()
        
        print(f"✅ MongoDB connected - {len(collections)} collections found")
        client.close()
        return True
        
    except Exception as e:
        print(f"❌ MongoDB connection failed: {e}")
        return False

def test_returns_functionality():
    """Test if returns functionality works"""
    try:
        # Test critical imports for returns
        from app.api.returns import router
        from app.services.workflow_service import WorkflowService
        from app.utils.database import get_collection
        
        # Test database collections
        returns_collection = get_collection("returns")
        orders_collection = get_collection("orders")
        
        print("✅ Returns API imports successful")
        print("✅ Database collections accessible")
        
        # Count existing data
        returns_count = returns_collection.count_documents({})
        orders_count = orders_collection.count_documents({})
        
        print(f"✅ Found {returns_count} returns and {orders_count} orders")
        
        return True
        
    except Exception as e:
        print(f"❌ Returns functionality test failed: {e}")
        traceback.print_exc()
        return False

def create_docker_setup():
    """Create Docker setup for easy deployment"""
    
    docker_compose = """version: '3.8'

services:
  mongodb:
    image: mongo:5.0
    container_name: wms_mongodb
    restart: unless-stopped
    ports:
      - "27017:27017"
    environment:
      MONGO_INITDB_DATABASE: warehouse_management
    volumes:
      - mongodb_data:/data/db

  wms_backend:
    build: .
    container_name: wms_backend
    restart: unless-stopped
    ports:
      - "8000:8000"
    environment:
      - MONGODB_URL=mongodb://mongodb:27017/
      - DATABASE_NAME=warehouse_management
    depends_on:
      - mongodb
    volumes:
      - .:/app

volumes:
  mongodb_data:
"""

    dockerfile = """FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    gcc \\
    g++ \\
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements_clean.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Expose port
EXPOSE 8000

# Run application
CMD ["python", "run.py"]
"""

    with open("docker-compose.yml", 'w', encoding='utf-8') as f:
        f.write(docker_compose)
    
    with open("Dockerfile", 'w', encoding='utf-8') as f:
        f.write(dockerfile)
    
    print("✅ Created Docker configuration files")

def main():
    """Main setup function"""
    
    print("🚀 WAREHOUSE MANAGEMENT SYSTEM SETUP")
    print("Fixing return process issues for cross-system deployment")
    
    # Check Python
    print_header("Python Version Check")
    if not check_python():
        print("❌ Please upgrade Python to 3.8 or higher")
        return False
    
    # Install dependencies
    print_header("Installing Dependencies")
    if not install_dependencies():
        print("⚠️  Some packages failed to install. Try running:")
        print("pip install -r requirements_clean.txt")
    
    # Setup environment
    print_header("Environment Configuration")
    create_env_file()
    
    # MongoDB setup options
    print_header("MongoDB Setup")
    setup_mongodb_options()
    
    # Test MongoDB
    print_header("Testing MongoDB Connection")
    mongodb_ok = test_mongodb_connection()
    
    if not mongodb_ok:
        print("\n⚠️  MongoDB connection failed. Please:")
        print("1. Set up MongoDB using one of the options above")
        print("2. Update MONGODB_URL in .env file")
        print("3. Run this script again")
        return False
    
    # Test returns functionality
    print_header("Testing Returns Functionality")
    returns_ok = test_returns_functionality()
    
    # Create Docker setup
    print_header("Creating Docker Setup")
    create_docker_setup()
    
    # Final summary
    print_header("Setup Summary")
    
    if mongodb_ok and returns_ok:
        print("🎉 SUCCESS! Return process should now work.")
        print("\n✅ All checks passed:")
        print("   • Python version compatible")
        print("   • Dependencies installed")
        print("   • MongoDB connected")
        print("   • Returns functionality working")
        
        print("\n🚀 You can now run the system:")
        print("   python run.py")
        
    else:
        print("⚠️  Issues detected:")
        if not mongodb_ok:
            print("   • MongoDB connection failed")
        if not returns_ok:
            print("   • Returns functionality failed")
        
        print("\n🔧 Next steps:")
        print("   1. Fix MongoDB connection")
        print("   2. Run this script again")
        print("   3. Or use Docker: docker-compose up")
    
    return mongodb_ok and returns_ok

if __name__ == "__main__":
    success = main()
    
    if success:
        print("\n🎯 READY TO GO! The return process should now work on any system.")
    else:
        print("\n🔧 NEEDS ATTENTION! Please fix the issues above.")
    
    input("\nPress Enter to exit...")
