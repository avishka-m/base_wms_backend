# update_worker_locations.py - Standalone script

import pymongo
from datetime import datetime
import os

def update_worker_locations():
    """Add worker_location field to all existing workers"""
    try:
        # Connect directly to MongoDB
        # Update these connection details if needed
        MONGO_URL = "mongodb://localhost:27017/"  # Change if your MongoDB is elsewhere
        DATABASE_NAME = "warehouse_management"  # Change to your database name
        
        print("Connecting to MongoDB...")
        client = pymongo.MongoClient(MONGO_URL)
        db = client[DATABASE_NAME]
        workers_collection = db["workers"]
        
        # Test connection
        try:
            db.command('ping')
            print("✅ Connected to MongoDB successfully")
        except Exception as e:
            print(f"❌ Failed to connect to MongoDB: {e}")
            return
        
        # Get all workers
        workers = list(workers_collection.find({}))
        print(f"Found {len(workers)} workers to update")
        
        if len(workers) == 0:
            print("⚠️ No workers found in database")
            return
        
        updated_count = 0
        
        # Update each worker with default location
        for worker in workers:
            worker_id = worker.get("workerID")
            username = worker.get("username", "Unknown")
            
            # Check if worker already has location data
            if "worker_location" in worker:
                print(f"⏭️ Worker {username} already has location data, skipping...")
                continue
            
            update_data = {
                "worker_location": {
                    "x": 0,  # Default to receiving point
                    "y": 0,
                    "floor": 1,
                    "last_updated": datetime.utcnow().isoformat(),
                    "updated_by": "system_init",
                    "status": "offline"  # online, offline, working
                },
                "location_history": [],  # Track location movements
                "last_location_update": datetime.utcnow().isoformat()
            }
            
            result = workers_collection.update_one(
                {"workerID": worker_id},
                {"$set": update_data}
            )
            
            if result.modified_count > 0:
                print(f"✅ Updated worker {username} (ID: {worker_id})")
                updated_count += 1
            else:
                print(f"❌ Failed to update worker {username} (ID: {worker_id})")
        
        print(f"\n🎉 Successfully updated {updated_count} workers with location data")
        
        # Verify the updates
        print("\n🔍 Verifying updates...")
        workers_with_location = workers_collection.count_documents({"worker_location": {"$exists": True}})
        print(f"✅ {workers_with_location} workers now have location data")
        
        client.close()
        
    except Exception as e:
        print(f"❌ Error updating workers: {str(e)}")

if __name__ == "__main__":
    print("🚀 Starting worker location field update...")
    print("=" * 50)
    update_worker_locations()
    print("=" * 50)
    print("✅ Script completed!")
    
    input("\nPress Enter to exit...")  # Keep window open to see results