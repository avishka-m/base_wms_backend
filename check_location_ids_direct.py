import pymongo
from pymongo import MongoClient

def check_location_ids():
    print("🔍 Checking location IDs in location_inventory...")
    
    # Connect to MongoDB
    client = MongoClient("mongodb://localhost:27017/")
    db = client["warehouse_management"]
    
    # Get location_inventory collection
    location_inventory = db["location_inventory"]
    
    # Get unique location IDs
    locations = list(location_inventory.find({}, {'locationID': 1}).limit(20))
    
    print("Location IDs in location_inventory:")
    unique_locations = set()
    for loc in locations:
        location_id = loc.get('locationID')
        unique_locations.add(location_id)
        print(f"  {location_id}")
    
    print(f"\nUnique location formats found: {len(unique_locations)}")
    print("Sample locations:")
    for loc in sorted(list(unique_locations))[:10]:
        print(f"  {loc}")
    
    # Check inventory collection too
    print("\n🔍 Checking location IDs in inventory...")
    inventory = db["inventory"]
    inventory_locations = list(inventory.find({}, {'locationID': 1}).limit(10))
    
    print("Location IDs in inventory:")
    for inv in inventory_locations:
        location_id = inv.get('locationID')
        if location_id:
            print(f"  {location_id}")

if __name__ == "__main__":
    check_location_ids()
