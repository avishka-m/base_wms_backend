from pymongo import MongoClient
from datetime import datetime

def initialize_location_inventory():
    """Initialize warehouse location inventory in localhost MongoDB"""
    
    # Connect to localhost MongoDB
    client = MongoClient("mongodb://localhost:27017/")
    db = client["warehouse_management"]  # Replace with your database name
    location_collection = db["location_inventory"]
    
    print("🔍 Starting location initialization...")
    
    # Check if already initialized
    existing_count = location_collection.count_documents({})
    if existing_count > 0:
        print(f"⚠️ Found {existing_count} existing locations, clearing them...")
        location_collection.delete_many({})
    
    # Define all_locations list
    all_locations = []
    
    # B slots: B01-B21, each with 4 floors
    for i in range(1, 22):
        slot_code = f"B{str(i).zfill(2)}"
        for floor in range(1, 5):
            location_id = f"{slot_code}.{floor}"
            all_locations.append({
                "locationID": location_id,
                "slotCode": slot_code,
                "floor": floor,
                "type": "M",  # Medium/Bin
                "available": True,
                "itemID": None,
                "itemName": None,
                "quantity": 0,
                "lastUpdated": datetime.utcnow().isoformat()
            })
    
    # P slots: P01-P14, each with 4 floors  
    for i in range(1, 15):
        slot_code = f"P{str(i).zfill(2)}"
        for floor in range(1, 5):
            location_id = f"{slot_code}.{floor}"
            all_locations.append({
                "locationID": location_id,
                "slotCode": slot_code,
                "floor": floor,
                "type": "S",  # Small/Pellet
                "available": True,
                "itemID": None,
                "itemName": None,
                "quantity": 0,
                "lastUpdated": datetime.utcnow().isoformat()
            })
    
    # D slots: D01-D14, each with 4 floors
    for i in range(1, 15):
        slot_code = f"D{str(i).zfill(2)}"
        for floor in range(1, 5):
            location_id = f"{slot_code}.{floor}"
            all_locations.append({
                "locationID": location_id,
                "slotCode": slot_code,
                "floor": floor,
                "type": "D",  # Large
                "available": True,
                "itemID": None,
                "itemName": None,
                "quantity": 0,
                "lastUpdated": datetime.utcnow().isoformat()
            })
    
    # Insert all locations and verify
    result = location_collection.insert_many(all_locations)
    final_count = location_collection.count_documents({})
    
    print(f"✅ Inserted {len(result.inserted_ids)} locations")
    print(f"✅ Final count in database: {final_count}")
    
    # Create useful indexes
    location_collection.create_index("locationID", unique=True)
    location_collection.create_index("available")
    location_collection.create_index("type")
    location_collection.create_index("slotCode")
    
    print("📊 Summary:")
    print(f"  - Total locations: {final_count}")
    print(f"  - B slots (Medium): {21 * 4} locations")
    print(f"  - P slots (Small): {14 * 4} locations")
    print(f"  - D slots (Large): {14 * 4} locations")
    
    client.close()
    return final_count

if __name__ == "__main__":
    try:
        total_locations = initialize_location_inventory()
        print(f"\n🎉 Successfully initialized {total_locations} warehouse locations!")
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()