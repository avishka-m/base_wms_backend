# redistribute_workers_locations.py - Move workers from receiving counter to free paths

import pymongo
from datetime import datetime
import random

def redistribute_workers():
    """Move all workers away from receiving counter (0,0) to free path locations"""
    try:
        # Connect directly to MongoDB
        MONGO_URL = "mongodb://localhost:27017/"
        DATABASE_NAME = "warehouse_management"
        
        print("🔄 Connecting to MongoDB...")
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
        
        # Get workers currently at receiving counter (0,0) or without location
        workers_at_receiving = list(workers_collection.find({
            "$or": [
                {"worker_location.x": 0, "worker_location.y": 0},
                {"worker_location": {"$exists": False}}
            ]
        }))
        
        print(f"Found {len(workers_at_receiving)} workers at receiving counter or without location")
        
        if len(workers_at_receiving) == 0:
            print("✅ No workers found at receiving counter - all good!")
            return
        
        # ✨ FREE PATH COORDINATES (avoiding rack locations)
        # Based on 10x12 warehouse grid layout
        free_path_locations = [
            {"x": 2, "y": 1, "floor": 1, "description": "Aisle between B Rack 1 & 2"},
            {"x": 4, "y": 1, "floor": 1, "description": "Aisle between B Rack 2 & 3"},
            {"x": 6, "y": 1, "floor": 1, "description": "Aisle between B Rack 3 & P Rack 1"},
            {"x": 8, "y": 1, "floor": 1, "description": "Aisle between P Rack 1 & 2"},
            {"x": 2, "y": 9, "floor": 1, "description": "Main pathway (north of D racks)"},
            {"x": 4, "y": 9, "floor": 1, "description": "Main pathway (north of D racks)"},
            {"x": 6, "y": 9, "floor": 1, "description": "Main pathway (north of D racks)"},
            {"x": 8, "y": 9, "floor": 1, "description": "Main pathway (north of D racks)"},
            {"x": 0, "y": 5, "floor": 1, "description": "West pathway (warehouse edge)"},
            {"x": 9, "y": 5, "floor": 1, "description": "East pathway (warehouse edge)"},
            {"x": 2, "y": 5, "floor": 1, "description": "Central aisle"},
            {"x": 4, "y": 5, "floor": 1, "description": "Central aisle"},
            {"x": 6, "y": 5, "floor": 1, "description": "Central aisle"},
            {"x": 8, "y": 5, "floor": 1, "description": "Central aisle"},
        ]
        
        current_time = datetime.utcnow().isoformat()
        moved_count = 0
        
        print("\n🚶 Moving workers to free path locations...")
        print("=" * 60)
        
        for i, worker in enumerate(workers_at_receiving):
            # Assign free path location (cycle through available locations)
            location = free_path_locations[i % len(free_path_locations)]
            
            update_data = {
                "worker_location": {
                    "x": location["x"],
                    "y": location["y"], 
                    "floor": location["floor"],
                    "last_updated": current_time,
                    "status": "online",
                    "redistributed_from_receiving": True
                },
                "last_location_update": current_time
            }
            
            workers_collection.update_one(
                {"_id": worker["_id"]},
                {"$set": update_data}
            )
            moved_count += 1
            
            worker_name = worker.get("name", worker.get("username", "Unknown"))
            print(f"✅ {worker_name:20} → ({location['x']:2}, {location['y']:2}) - {location['description']}")
        
        print("=" * 60)
        print(f"🎉 Successfully moved {moved_count} workers to free path locations!")
        print("\nNew worker distribution:")
        
        # Show final distribution
        location_counts = {}
        for i in range(moved_count):
            loc = free_path_locations[i % len(free_path_locations)]
            coord = f"({loc['x']}, {loc['y']})"
            location_counts[coord] = location_counts.get(coord, 0) + 1
        
        for coord, count in location_counts.items():
            print(f"  📍 {coord}: {count} worker(s)")
        
    except Exception as e:
        print(f"❌ Error redistributing workers: {str(e)}")

if __name__ == "__main__":
    print("🏭 Worker Location Redistribution Script")
    print("Moving workers from receiving counter to free path locations...\n")
    redistribute_workers()
