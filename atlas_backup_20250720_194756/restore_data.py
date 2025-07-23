#!/usr/bin/env python3
"""
MongoDB Atlas Data Restore Script
Auto-generated on 2025-07-20T19:48:25.672685

This script restores data from the backup to your new Singapore Atlas cluster.
"""

import os
import sys
import json
import subprocess
from pymongo import MongoClient
from bson import json_util

def restore_using_mongorestore(new_mongodb_url, database_name):
    """Restore using mongorestore (recommended method)."""
    print("🔄 Restoring data using mongorestore...")
    
    dump_dir = os.path.join("atlas_backup_20250720_194756", "mongodump")
    if not os.path.exists(dump_dir):
        print("❌ mongodump backup not found")
        return False
    
    try:
        cmd = [
            "mongorestore",
            "--uri", new_mongodb_url,
            "--dir", dump_dir
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            print("✅ mongorestore completed successfully")
            return True
        else:
            print(f"❌ mongorestore failed: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ mongorestore error: {e}")
        return False

def restore_using_pymongo(new_mongodb_url, database_name):
    """Restore using PyMongo as fallback."""
    print("🔄 Restoring data using PyMongo...")
    
    pymongo_dir = os.path.join("atlas_backup_20250720_194756", "pymongo_backup")
    if not os.path.exists(pymongo_dir):
        print("❌ PyMongo backup not found")
        return False
    
    try:
        client = MongoClient(new_mongodb_url)
        db = client[database_name]
        
        json_files = [f for f in os.listdir(pymongo_dir) if f.endswith('.json')]
        restored_collections = 0
        
        for json_file in json_files:
            collection_name = json_file.replace('.json', '')
            
            with open(os.path.join(pymongo_dir, json_file), 'r', encoding='utf-8') as f:
                documents = json.load(f, object_hook=json_util.object_hook)
            
            if documents:
                # Clear existing collection (optional)
                db[collection_name].delete_many({})
                
                # Insert documents
                db[collection_name].insert_many(documents)
                print(f"✅ {collection_name}: {len(documents)} documents restored")
                restored_collections += 1
            else:
                print(f"📝 {collection_name}: Empty collection")
        
        client.close()
        
        if restored_collections > 0:
            print(f"✅ PyMongo restore completed: {restored_collections} collections")
            return True
        else:
            print("❌ No collections were restored")
            return False
            
    except Exception as e:
        print(f"❌ PyMongo restore error: {e}")
        return False

if __name__ == "__main__":
    print("MongoDB Atlas Data Restore")
    print("=" * 40)
    
    # Get new cluster connection details
    new_mongodb_url = input("Enter your NEW Singapore cluster connection string: ")
    database_name = "warehouse_management"
    
    print(f"Database: {database_name}")
    print(f"Backup date: 2025-07-20T19:48:10.455161")
    print(f"Collections to restore: 29")
    print(f"Total documents: 4434")
    print()
    
    # Try mongorestore first (most reliable)
    if restore_using_mongorestore(new_mongodb_url, database_name):
        print("🎉 Data restoration completed successfully!")
    else:
        print("⚠️  mongorestore failed, trying PyMongo method...")
        if restore_using_pymongo(new_mongodb_url, database_name):
            print("🎉 Data restoration completed using PyMongo!")
        else:
            print("❌ All restore methods failed. Check the CSV exports for manual import.")
