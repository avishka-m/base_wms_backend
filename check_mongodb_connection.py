#!/usr/bin/env python3
"""
Script to check MongoDB connection and diagnose connection issues
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from pymongo import MongoClient
from app.config import MONGODB_URL, DATABASE_NAME
import traceback

def check_mongodb_connection():
    """Check MongoDB connection and database access"""
    
    print("🔍 Checking MongoDB connection...")
    print(f"📍 MongoDB URL: {MONGODB_URL}")
    print(f"📁 Database Name: {DATABASE_NAME}")
    
    try:
        # Test connection
        client = MongoClient(MONGODB_URL, serverSelectionTimeoutMS=5000)
        
        # Test server info
        server_info = client.server_info()
        print(f"✅ MongoDB Server Connected - Version: {server_info.get('version', 'Unknown')}")
        
        # Test database access
        db = client[DATABASE_NAME]
        
        # Test collections access
        collections = db.list_collection_names()
        print(f"✅ Database '{DATABASE_NAME}' accessible")
        print(f"📝 Available collections: {len(collections)}")
        
        # Check key collections for returns process
        required_collections = ["returns", "orders", "customers", "inventory", "location_inventory", "receiving"]
        missing_collections = []
        
        for collection_name in required_collections:
            if collection_name in collections:
                count = db[collection_name].count_documents({})
                print(f"  ✅ {collection_name}: {count} documents")
            else:
                missing_collections.append(collection_name)
                print(f"  ❌ {collection_name}: Missing")
        
        if missing_collections:
            print(f"\n⚠️  Missing collections: {missing_collections}")
            print("This could cause return process failures!")
        
        # Test a simple write operation
        test_collection = db["connection_test"]
        test_doc = {"test": True, "timestamp": "2025-01-21"}
        result = test_collection.insert_one(test_doc)
        test_collection.delete_one({"_id": result.inserted_id})
        print("✅ Write/Delete operations working")
        
        client.close()
        return True
        
    except Exception as e:
        print(f"❌ MongoDB Connection Failed: {str(e)}")
        print("\n🔧 Troubleshooting steps:")
        print("1. Check if MongoDB is installed and running")
        print("2. Verify the MONGODB_URL in .env file")
        print("3. Check firewall/network settings")
        print("4. Try connecting with MongoDB Compass")
        
        traceback.print_exc()
        return False

def check_environment():
    """Check environment variables and configuration"""
    
    print("\n🔍 Checking environment configuration...")
    
    # Check .env file
    env_file = os.path.join(os.path.dirname(__file__), '.env')
    if os.path.exists(env_file):
        print(f"✅ .env file exists: {env_file}")
        
        with open(env_file, 'r') as f:
            env_content = f.read()
            if "MONGODB_URL" in env_content:
                print("✅ MONGODB_URL found in .env")
            else:
                print("❌ MONGODB_URL not found in .env")
                
            if "DATABASE_NAME" in env_content:
                print("✅ DATABASE_NAME found in .env")
            else:
                print("❌ DATABASE_NAME not found in .env")
    else:
        print(f"❌ .env file not found: {env_file}")
        print("This could cause configuration issues!")
    
    # Check environment variables
    mongodb_url_env = os.getenv("MONGODB_URL")
    database_name_env = os.getenv("DATABASE_NAME")
    
    print(f"📍 MONGODB_URL from env: {mongodb_url_env}")
    print(f"📁 DATABASE_NAME from env: {database_name_env}")

def main():
    """Main diagnostic function"""
    
    print("🚀 MongoDB Connection Diagnostic Tool")
    print("=" * 50)
    
    # Check environment first
    check_environment()
    
    # Check MongoDB connection
    connection_ok = check_mongodb_connection()
    
    print("\n" + "=" * 50)
    if connection_ok:
        print("🎉 All checks passed! MongoDB connection is working.")
    else:
        print("💥 Connection issues detected. Please fix the issues above.")
        
    print("\n💡 Common fixes for friend's laptop:")
    print("1. Install MongoDB locally OR")
    print("2. Update .env with shared MongoDB Atlas URL OR")
    print("3. Use Docker for consistent environment")

if __name__ == "__main__":
    main()
