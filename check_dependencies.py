#!/usr/bin/env python3
"""
Script to check Python dependencies and environment consistency
"""

import sys
import subprocess
import pkg_resources
import os
from pathlib import Path

def check_python_version():
    """Check Python version compatibility"""
    
    print("🐍 Checking Python version...")
    
    current_version = sys.version_info
    required_version = (3, 8)
    
    print(f"Current Python: {current_version.major}.{current_version.minor}.{current_version.micro}")
    print(f"Required Python: {required_version[0]}.{required_version[1]}+")
    
    if current_version >= required_version:
        print("✅ Python version compatible")
        return True
    else:
        print("❌ Python version too old")
        return False

def check_virtual_environment():
    """Check if running in virtual environment"""
    
    print("\\n📦 Checking virtual environment...")
    
    in_venv = hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix)
    
    if in_venv:
        print("✅ Running in virtual environment")
        print(f"Virtual env path: {sys.prefix}")
        return True
    else:
        print("⚠️  Not running in virtual environment")
        print("Recommendation: Use virtual environment to avoid conflicts")
        return False

def check_requirements():
    """Check if all requirements are installed"""
    
    print("\\n📋 Checking requirements...")
    
    requirements_file = os.path.join(os.path.dirname(__file__), 'requirements.txt')
    
    if not os.path.exists(requirements_file):
        print("❌ requirements.txt not found")
        return False
    
    with open(requirements_file, 'r') as f:
        requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]
    
    missing_packages = []
    outdated_packages = []
    
    for requirement in requirements:
        try:
            # Parse requirement (handle version specifiers)
            if '>=' in requirement:
                package_name = requirement.split('>=')[0].strip()
                min_version = requirement.split('>=')[1].strip()
            elif '==' in requirement:
                package_name = requirement.split('==')[0].strip()
                min_version = requirement.split('==')[1].strip()
            else:
                package_name = requirement.strip()
                min_version = None
            
            # Check if package is installed
            try:
                installed_package = pkg_resources.get_distribution(package_name)
                print(f"✅ {package_name}: {installed_package.version}")
                
                # Check version if specified
                if min_version and pkg_resources.parse_version(installed_package.version) < pkg_resources.parse_version(min_version):
                    outdated_packages.append(f"{package_name} (installed: {installed_package.version}, required: {min_version})")
                    
            except pkg_resources.DistributionNotFound:
                missing_packages.append(requirement)
                print(f"❌ {package_name}: Not installed")
                
        except Exception as e:
            print(f"⚠️  Could not check {requirement}: {e}")
    
    if missing_packages:
        print(f"\\n📥 Missing packages: {missing_packages}")
        
    if outdated_packages:
        print(f"\\n📤 Outdated packages: {outdated_packages}")
    
    return len(missing_packages) == 0 and len(outdated_packages) == 0

def check_specific_imports():
    """Check specific imports that might cause issues"""
    
    print("\\n🔍 Checking critical imports...")
    
    critical_imports = [
        ('fastapi', 'FastAPI framework'),
        ('pymongo', 'MongoDB driver'),
        ('pydantic', 'Data validation'),
        ('uvicorn', 'ASGI server'),
        ('motor', 'Async MongoDB driver'),
        ('bcrypt', 'Password hashing'),
        ('jose', 'JWT tokens'),
        ('python_multipart', 'File uploads'),
        ('scikit-learn', 'Machine learning'),
        ('numpy', 'Numerical computing'),
        ('pandas', 'Data manipulation')
    ]
    
    failed_imports = []
    
    for module_name, description in critical_imports:
        try:
            __import__(module_name)
            print(f"✅ {module_name}: OK ({description})")
        except ImportError as e:
            failed_imports.append((module_name, description, str(e)))
            print(f"❌ {module_name}: FAILED ({description}) - {e}")
    
    return len(failed_imports) == 0

def generate_fix_script():
    """Generate a script to fix common issues"""
    
    fix_script_content = """#!/usr/bin/env python3
'''
Auto-fix script for common dependency issues
Run this to automatically install/fix dependencies
'''

import subprocess
import sys
import os

def install_package(package):
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        return True
    except:
        return False

def upgrade_pip():
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
        print("✅ pip upgraded")
        return True
    except:
        print("❌ Failed to upgrade pip")
        return False

def install_requirements():
    if os.path.exists('requirements.txt'):
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
            print("✅ Requirements installed")
            return True
        except:
            print("❌ Failed to install requirements")
            return False
    else:
        print("❌ requirements.txt not found")
        return False

def main():
    print("🔧 Auto-fixing dependencies...")
    
    print("1. Upgrading pip...")
    upgrade_pip()
    
    print("2. Installing requirements...")
    if not install_requirements():
        print("3. Trying individual package installation...")
        critical_packages = [
            "fastapi>=0.68.0",
            "uvicorn[standard]>=0.15.0",
            "pymongo>=3.12.0",
            "motor>=2.5.1",
            "pydantic>=1.8.0",
            "python-jose[cryptography]>=3.3.0",
            "passlib[bcrypt]>=1.7.4",
            "python-multipart>=0.0.5",
            "scikit-learn>=1.0.0",
            "numpy>=1.21.0",
            "pandas>=1.3.0"
        ]
        
        for package in critical_packages:
            print(f"Installing {package}...")
            install_package(package)
    
    print("\\n🎉 Fix attempt complete! Run check_dependencies.py again to verify.")

if __name__ == "__main__":
    main()
"""
    
    fix_script_path = os.path.join(os.path.dirname(__file__), 'fix_dependencies.py')
    
    with open(fix_script_path, 'w') as f:
        f.write(fix_script_content)
    
    print(f"✅ Created fix script: {fix_script_path}")
    return fix_script_path

def main():
    """Main dependency checker"""
    
    print("🔍 Python Dependencies Checker")
    print("=" * 50)
    
    # Check Python version
    python_ok = check_python_version()
    
    # Check virtual environment
    venv_ok = check_virtual_environment()
    
    # Check requirements
    requirements_ok = check_requirements()
    
    # Check specific imports
    imports_ok = check_specific_imports()
    
    # Generate fix script
    fix_script = generate_fix_script()
    
    print("\\n" + "=" * 50)
    print("📊 Summary:")
    
    all_ok = python_ok and requirements_ok and imports_ok
    
    if all_ok:
        print("🎉 All dependency checks passed!")
    else:
        print("⚠️  Issues detected:")
        
        if not python_ok:
            print("   • Python version needs upgrade")
        if not venv_ok:
            print("   • Consider using virtual environment")
        if not requirements_ok:
            print("   • Missing or outdated packages")
        if not imports_ok:
            print("   • Import failures detected")
    
    print("\\n🔧 To fix issues on your friend's laptop:")
    print(f"1. Run: python {fix_script}")
    print("2. Or manually: pip install -r requirements.txt")
    print("3. Check Python version (3.8+ required)")
    print("4. Use virtual environment for isolation")

if __name__ == "__main__":
    main()
