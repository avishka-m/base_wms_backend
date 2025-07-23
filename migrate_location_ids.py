import pymongo
from pymongo import MongoClient

def migrate_location_ids():
    print("🔄 Migrating location IDs to correct format...")
    
    # Connect to MongoDB
    client = MongoClient("mongodb://localhost:27017/")
    db = client["warehouse_management"]
    
    # Location ID mapping from old format to new format
    location_mapping = {
        # Electronics -> Bulk
        "E1.1": "B01.1",
        "E1.2": "B01.2", 
        "E2.1": "B02.1",
        "E2.2": "B02.2",
        "E3.1": "B03.1",
        "E3.2": "B03.2",
        
        # General -> Dangerous/Delicate  
        "G1.1": "D01.1",
        "G1.2": "D01.2",
        "G2.1": "D02.1", 
        "G2.2": "D02.2",
        "G3.1": "D03.1",
        "G3.2": "D03.2",
        
        # Perishables stays P but with zero-padded format
        "P1.1": "P01.1",
        "P1.2": "P01.2", 
        "P2.1": "P02.1",
        "P2.2": "P02.2",
        "P3.1": "P03.1",
        "P3.2": "P03.2",
    }
    
    # Update inventory collection
    print("Updating inventory collection...")
    inventory = db["inventory"]
    for old_id, new_id in location_mapping.items():
        result = inventory.update_many(
            {"locationID": old_id},
            {"$set": {"locationID": new_id}}
        )
        if result.modified_count > 0:
            print(f"  Updated {result.modified_count} inventory records: {old_id} -> {new_id}")
    
    # Update location_inventory collection
    print("Updating location_inventory collection...")
    location_inventory = db["location_inventory"]
    for old_id, new_id in location_mapping.items():
        result = location_inventory.update_many(
            {"locationID": old_id},
            {"$set": {"locationID": new_id}}
        )
        if result.modified_count > 0:
            print(f"  Updated {result.modified_count} location_inventory records: {old_id} -> {new_id}")
    
    # Update any other collections that might have location references
    collections_to_check = ["orders", "returns", "job_queue", "picking_tasks", "storage_history"]
    
    for collection_name in collections_to_check:
        if collection_name in db.list_collection_names():
            collection = db[collection_name]
            print(f"Checking {collection_name} collection...")
            
            for old_id, new_id in location_mapping.items():
                # Check various field names that might contain location IDs
                location_fields = ["locationID", "location_id", "location", "actual_location", "predicted_location"]
                
                for field in location_fields:
                    result = collection.update_many(
                        {field: old_id},
                        {"$set": {field: new_id}}
                    )
                    if result.modified_count > 0:
                        print(f"  Updated {result.modified_count} {collection_name} records: {field} {old_id} -> {new_id}")
                
                # Check nested fields in items arrays
                if collection_name in ["orders", "returns"]:
                    result = collection.update_many(
                        {"items.locationID": old_id},
                        {"$set": {"items.$.locationID": new_id}}
                    )
                    if result.modified_count > 0:
                        print(f"  Updated {result.modified_count} {collection_name} item records: {old_id} -> {new_id}")
    
    print("\n✅ Location ID migration completed!")
    
    # Verify the migration
    print("\n🔍 Verifying migration...")
    inventory_locations = list(inventory.find({}, {'locationID': 1}).limit(10))
    print("Sample inventory locations after migration:")
    for inv in inventory_locations:
        location_id = inv.get('locationID')
        if location_id:
            print(f"  {location_id}")

if __name__ == "__main__":
    migrate_location_ids()
