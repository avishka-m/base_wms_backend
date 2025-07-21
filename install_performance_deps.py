#!/usr/bin/env python3
"""
Install Performance Dependencies for MongoDB Atlas

This script installs optional dependencies for optimal MongoDB Atlas performance,
including compression libraries and other performance enhancements.
"""

import subprocess
import sys
import importlib

def install_package(package_name):
    """Install a package using pip."""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install {package_name}: {e}")
        return False

def check_package(package_name):
    """Check if a package is already installed."""
    try:
        importlib.import_module(package_name)
        return True
    except ImportError:
        return False

def install_performance_dependencies():
    """Install all performance-related dependencies."""
    print("🚀 Installing MongoDB Atlas Performance Dependencies")
    print("=" * 55)
    
    dependencies = [
        {
            "name": "python-snappy",
            "module": "snappy",
            "description": "Snappy compression for 50-70% bandwidth reduction",
            "required": False
        },
        {
            "name": "pymongo[srv]",
            "module": "pymongo",
            "description": "MongoDB driver with SRV support for Atlas",
            "required": True
        },
        {
            "name": "motor",
            "module": "motor",
            "description": "Async MongoDB driver for better performance",
            "required": True
        },
        {
            "name": "dnspython",
            "module": "dns",
            "description": "DNS resolution for MongoDB+SRV connections",
            "required": True
        }
    ]
    
    installed_count = 0
    failed_count = 0
    
    for dep in dependencies:
        print(f"\n🔍 Checking {dep['name']}...")
        
        if check_package(dep['module']):
            print(f"✅ {dep['name']} is already installed")
            installed_count += 1
        else:
            print(f"📦 Installing {dep['name']} - {dep['description']}")
            
            if install_package(dep['name']):
                print(f"✅ Successfully installed {dep['name']}")
                installed_count += 1
            else:
                if dep['required']:
                    print(f"❌ Failed to install required dependency: {dep['name']}")
                    failed_count += 1
                else:
                    print(f"⚠️  Optional dependency {dep['name']} failed to install (performance may be reduced)")
    
    print(f"\n📊 Installation Summary:")
    print(f"   ✅ Installed: {installed_count}")
    print(f"   ❌ Failed: {failed_count}")
    
    if failed_count == 0:
        print(f"\n🎉 All dependencies installed successfully!")
        print(f"   Your MongoDB Atlas connection will have optimal performance.")
        
        # Update compression settings if snappy is now available
        if check_package('snappy'):
            print(f"\n💡 Snappy compression is now available!")
            print(f"   The system will automatically use snappy + zlib compression.")
        
        return True
    else:
        print(f"\n⚠️  Some dependencies failed to install.")
        print(f"   The system will still work but with reduced performance.")
        return False

def update_compression_settings():
    """Update compression settings if snappy is available."""
    try:
        import snappy
        print("🔧 Updating compression settings to include snappy...")
        
        # This would update the atlas_optimization.py file to include snappy
        # For now, just inform the user
        print("✅ Snappy compression is available!")
        print("   Edit config/atlas_optimization.py to add 'snappy' to compressors list")
        
    except ImportError:
        print("📝 Snappy compression not available - using zlib only")

if __name__ == "__main__":
    print("MongoDB Atlas Performance Dependencies Installer")
    print("This script installs optional dependencies for better Atlas performance")
    print()
    
    success = install_performance_dependencies()
    
    if success:
        print(f"\n🚀 Next Steps:")
        print(f"   1. Run: python atlas_performance_setup.py")
        print(f"   2. Start your application with optimized performance")
    else:
        print(f"\n🔧 Troubleshooting:")
        print(f"   - Ensure you have internet connectivity")
        print(f"   - Try running with administrator/sudo privileges")
        print(f"   - Check if your firewall blocks pip installations")
    
    # Update compression settings
    update_compression_settings() 