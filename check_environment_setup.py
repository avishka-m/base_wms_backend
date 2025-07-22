#!/usr/bin/env python3
"""
Script to check and fix environment configuration for cross-system compatibility
"""

import os
import sys
import shutil
from pathlib import Path

def check_environment_setup():
    """Check and suggest fixes for environment setup"""
    
    print("🔍 Checking environment setup for cross-system compatibility...")
    
    project_root = os.path.dirname(os.path.abspath(__file__))
    env_file = os.path.join(project_root, '.env')
    example_env = os.path.join(project_root, '.env.example')
    
    issues = []
    suggestions = []
    
    # Check .env file
    if not os.path.exists(env_file):
        issues.append("❌ .env file missing")
        suggestions.append("Create .env file from .env.example")
    else:
        print("✅ .env file exists")
        
        # Check .env content
        with open(env_file, 'r') as f:
            env_content = f.read()
            
        if "localhost:27017" in env_content:
            issues.append("⚠️  Using localhost MongoDB (won't work on other systems)")
            suggestions.append("Consider using MongoDB Atlas or Docker")
            
        if "SECRET_KEY" not in env_content or "your-secret-key-here" in env_content:
            issues.append("⚠️  Default/missing SECRET_KEY")
            suggestions.append("Generate a secure SECRET_KEY")
    
    # Check .env.example
    if not os.path.exists(example_env):
        issues.append("❌ .env.example missing")
        suggestions.append("Create .env.example for team setup")
    
    # Check requirements.txt
    requirements_file = os.path.join(project_root, 'requirements.txt')
    if not os.path.exists(requirements_file):
        issues.append("❌ requirements.txt missing")
        suggestions.append("Create requirements.txt for dependency management")
    
    return issues, suggestions

def create_portable_env_template():
    """Create a portable .env template"""
    
    template_content = """# MongoDB Configuration
# Option 1: Local MongoDB (requires MongoDB installed)
MONGODB_URL=mongodb://localhost:27017/

# Option 2: MongoDB Atlas (shared cloud database)
# MONGODB_URL=mongodb+srv://username:password@cluster.mongodb.net/?retryWrites=true&w=majority

# Option 3: Docker MongoDB (requires Docker)
# MONGODB_URL=mongodb://mongodb:27017/

# Database Name
DATABASE_NAME=warehouse_management

# Security
SECRET_KEY=your-secret-key-here-generate-a-secure-one

# Debug Mode
DEBUG_MODE=True

# API Configuration
API_V1_PREFIX=/api/v1

# Instructions for team setup:
# 1. Copy this file to .env
# 2. Choose one of the MongoDB options above
# 3. Generate a secure SECRET_KEY
# 4. Update any other settings as needed
"""
    
    env_template_file = os.path.join(os.path.dirname(__file__), '.env.template')
    
    with open(env_template_file, 'w', encoding='utf-8') as f:
        f.write(template_content)
    
    print(f"✅ Created portable environment template: {env_template_file}")
    return env_template_file

def create_setup_script():
    """Create a setup script for easy deployment"""
    
    setup_script = """#!/usr/bin/env python3
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
    
    print("\\n🎉 Setup complete! You can now run the system.")
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
"""
    
    setup_file = os.path.join(os.path.dirname(__file__), 'setup.py')
    
    with open(setup_file, 'w', encoding='utf-8') as f:
        f.write(setup_script)
    
    # Make executable on Unix systems
    if os.name != 'nt':
        os.chmod(setup_file, 0o755)
    
    print(f"✅ Created setup script: {setup_file}")
    return setup_file

def create_docker_setup():
    """Create Docker configuration for consistent environment"""
    
    dockerfile_content = """FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    gcc \\
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port
EXPOSE 8000

# Run the application
CMD ["python", "run.py"]
"""
    
    docker_compose_content = """version: '3.8'

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
    
    dockerfile_path = os.path.join(os.path.dirname(__file__), 'Dockerfile')
    compose_path = os.path.join(os.path.dirname(__file__), 'docker-compose.yml')
    
    with open(dockerfile_path, 'w', encoding='utf-8') as f:
        f.write(dockerfile_content)
    
    with open(compose_path, 'w', encoding='utf-8') as f:
        f.write(docker_compose_content)
    
    print(f"✅ Created Docker configuration: {dockerfile_path}, {compose_path}")
    return dockerfile_path, compose_path

def main():
    """Main function to check and fix environment issues"""
    
    print("🔧 Environment Configuration Checker")
    print("=" * 50)
    
    # Check current environment
    issues, suggestions = check_environment_setup()
    
    if issues:
        print("\\n⚠️  Issues found:")
        for issue in issues:
            print(f"   {issue}")
        
        print("\\n💡 Suggestions:")
        for suggestion in suggestions:
            print(f"   • {suggestion}")
    else:
        print("✅ Environment configuration looks good!")
    
    # Create helpful files
    print("\\n🛠️  Creating setup files...")
    
    template_file = create_portable_env_template()
    setup_file = create_setup_script()
    dockerfile, compose_file = create_docker_setup()
    
    print("\\n📋 Team Setup Instructions:")
    print("1. Share these files with your friend:")
    print(f"   • {template_file}")
    print(f"   • {setup_file}")
    print(f"   • requirements.txt")
    
    print("\\n2. Your friend should run:")
    print("   python setup.py")
    
    print("\\n3. For Docker setup (recommended):")
    print("   docker-compose up -d")
    
    print("\\n4. Or use MongoDB Atlas (cloud database):")
    print("   • Create free account at mongodb.com/atlas")
    print("   • Get connection string")
    print("   • Update MONGODB_URL in .env")

if __name__ == "__main__":
    main()
