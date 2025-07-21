#!/usr/bin/env python3
"""
Restore data to Singapore Atlas cluster
"""

import os
import json
from pymongo import MongoClient
from bson import json_util

# Singapore cluster connection string
singapore_url = "mongodb+srv://new_user:9jwFJKq7Yb1zce9A@cluster0.99chyus.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
database_name = "warehouse_management"

print("🚀 Restoring data to Singapore Atlas cluster...")
print(f"🌏 Target: Singapore (ap-southeast-1)")
print()

# Test connection
print("🔍 Testing connection...")
try:
    client = MongoClient(singapore_url)
    db = client[database_name]
    db.command('ping')
    print("✅ Connected to Singapore cluster successfully")
except Exception as e:
    print(f"❌ Connection failed: {e}")
    exit(1)

# Restore data
print("\n🔄 Restoring collections...")
pymongo_dir = "pymongo_backup"
json_files = [f for f in os.listdir(pymongo_dir) if f.endswith('.json')]

restored_collections = 0
total_documents = 0

for json_file in sorted(json_files):
    collection_name = json_file.replace('.json', '')
    print(f"📦 {collection_name}", end="... ")
    
    try:
        # Load documents
        with open(os.path.join(pymongo_dir, json_file), 'r', encoding='utf-8') as f:
            documents = json.load(f, object_hook=json_util.object_hook)
        
        if documents:
            # Clear and insert
            db[collection_name].delete_many({})
            result = db[collection_name].insert_many(documents, ordered=False)
            count = len(result.inserted_ids)
            print(f"✅ {count} docs")
            total_documents += count
        else:
            print("📝 empty")
        
        restored_collections += 1
        
    except Exception as e:
        print(f"❌ failed: {e}")

client.close()

print(f"\n🎉 MIGRATION COMPLETE!")
print(f"✅ Collections: {restored_collections}")
print(f"✅ Documents: {total_documents:,}")
print(f"🌏 Location: Singapore")
print()
print("🔄 Update your .env file with these settings:")
print("ATLAS_USERNAME=new_user")
print("ATLAS_PASSWORD=9jwFJKq7Yb1zce9A")
print("ATLAS_CLUSTER_HOST=cluster0.99chyus.mongodb.net")
print("DATABASE_NAME=warehouse_management") 