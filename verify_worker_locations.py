# verify_worker_locations.py - Check current worker locations

import pymongo

def verify_worker_locations():
    """Check current worker locations"""
    try:
        # Connect directly to MongoDB
        MONGO_URL = "mongodb://localhost:27017/"
        DATABASE_NAME = "warehouse_management"
        
        print("🔍 Checking current worker locations...")
        client = pymongo.MongoClient(MONGO_URL)
        db = client[DATABASE_NAME]
        workers_collection = db["workers"]
        
        # Test connection
        try:
            db.command('ping')
            print("✅ Connected to MongoDB successfully\n")
        except Exception as e:
            print(f"❌ Failed to connect to MongoDB: {e}")
            return
        
        # Get all workers with their locations
        workers = list(workers_collection.find({}))
        
        print("📍 Current Worker Locations:")
        print("=" * 60)
        
        receiving_count = 0
        free_path_count = 0
        
        for worker in workers:
            name = worker.get("name", worker.get("username", "Unknown"))
            role = worker.get("role", "Unknown")
            
            if "worker_location" in worker:
                location = worker["worker_location"]
                x = location.get("x", "?")
                y = location.get("y", "?")
                floor = location.get("floor", "?")
                status = location.get("status", "unknown")
                last_updated = location.get("last_updated", "never")
                
                # Check if at receiving counter
                if x == 0 and y == 0:
                    receiving_count += 1
                    location_type = "🏢 RECEIVING COUNTER"
                else:
                    free_path_count += 1
                    location_type = "🛤️  FREE PATH"
                
                print(f"{name:20} | {role:10} | ({x:2}, {y:2}, F{floor}) | {status:8} | {location_type}")
            else:
                print(f"{name:20} | {role:10} | No location data")
        
        print("=" * 60)
        print(f"📊 Summary:")
        print(f"   🏢 At Receiving Counter (0,0): {receiving_count} workers")
        print(f"   🛤️  In Free Paths: {free_path_count} workers")
        print(f"   📈 Total Workers: {len(workers)} workers")
        
        if receiving_count == 0:
            print("\n🎉 SUCCESS: No workers are stuck at the receiving counter!")
        else:
            print(f"\n⚠️  WARNING: {receiving_count} workers still at receiving counter")
        
    except Exception as e:
        print(f"❌ Error checking worker locations: {str(e)}")

if __name__ == "__main__":
    verify_worker_locations()
