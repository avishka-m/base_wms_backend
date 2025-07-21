from app.utils.database import get_collection

def check_location_ids():
    print("🔍 Checking location IDs in location_inventory...")
    
    # Get location_inventory collection
    db = get_collection('location_inventory')
    
    # Get unique location IDs
    locations = list(db.find({}, {'locationID': 1}).limit(20))
    
    print("Location IDs in database:")
    for loc in locations:
        location_id = loc.get('locationID')
        print(f"  {location_id}")
    
    # Check inventory collection too
    print("\n🔍 Checking location IDs in inventory...")
    inventory_db = get_collection('inventory')
    inventory_locations = list(inventory_db.find({}, {'locationID': 1}).limit(10))
    
    print("Location IDs in inventory:")
    for inv in inventory_locations:
        location_id = inv.get('locationID')
        print(f"  {location_id}")

if __name__ == "__main__":
    check_location_ids()
