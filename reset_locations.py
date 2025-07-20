# reset_locations.py - Reset all locations to available

import pymongo
from datetime import datetime

def reset_all_locations():
    """Reset all locations in location_inventory to available"""
    try:
        # Connect to MongoDB
        MONGO_URL = "mongodb://localhost:27017/"  # Update if different
        DATABASE_NAME = "warehouse_management"  # Update to your database name
        
        print("Connecting to MongoDB...")
        client = pymongo.MongoClient(MONGO_URL)
        db = client[DATABASE_NAME]
        location_collection = db["location_inventory"]
        
        # Test connection
        try:
            db.command('ping')
            print("✅ Connected to MongoDB successfully")
        except Exception as e:
            print(f"❌ Failed to connect to MongoDB: {e}")
            return
        
        # Check current status
        total_locations = location_collection.count_documents({})
        occupied_locations = location_collection.count_documents({"available": False})
        available_locations = location_collection.count_documents({"available": True})
        
        print(f"\n📊 Current Status:")
        print(f"   Total locations: {total_locations}")
        print(f"   Available: {available_locations}")
        print(f"   Occupied: {occupied_locations}")
        
        if total_locations == 0:
            print("\n⚠️ No locations found! You need to run initialization first.")
            print("Run: python update_worker_locations.py")
            return
        
        if occupied_locations == 0:
            print("\n✅ All locations are already available!")
            return
        
        # Reset all locations to available
        print(f"\n🔄 Resetting {occupied_locations} occupied locations to available...")
        
        reset_data = {
            "available": True,
            "itemID": None,
            "itemName": None,
            "quantity": 0,
            "lastUpdated": datetime.utcnow().isoformat(),
            "resetAt": datetime.utcnow().isoformat(),
            "resetBy": "system_reset"
        }
        
        # Remove occupied-specific fields
        unset_data = {
            "storedAt": "",
            "storedBy": "",
            "receivingID": ""
        }
        
        result = location_collection.update_many(
            {"available": False},  # Only update occupied locations
            {
                "$set": reset_data,
                "$unset": unset_data
            }
        )
        
        print(f"✅ Reset {result.modified_count} locations to available")
        
        # Verify the reset
        final_available = location_collection.count_documents({"available": True})
        final_occupied = location_collection.count_documents({"available": False})
        
        print(f"\n📊 After Reset:")
        print(f"   Total locations: {total_locations}")
        print(f"   Available: {final_available}")
        print(f"   Occupied: {final_occupied}")
        
        # Show sample of reset locations
        sample_locations = list(location_collection.find(
            {"resetAt": {"$exists": True}}, 
            {"locationID": 1, "type": 1, "available": 1}
        ).limit(10))
        
        if sample_locations:
            print(f"\n📍 Sample reset locations:")
            for loc in sample_locations:
                print(f"   {loc['locationID']} ({loc['type']}) - Available: {loc['available']}")
        
        client.close()
        print(f"\n🎉 Location reset completed successfully!")
        
    except Exception as e:
        print(f"❌ Error resetting locations: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("🔧 Starting location reset...")
    print("=" * 60)
    reset_all_locations()
    print("=" * 60)
    print("✅ Script completed!")
    
    input("\nPress Enter to exit...")  # Keep window open to see results