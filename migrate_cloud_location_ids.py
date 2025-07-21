import pymongo
from pymongo import MongoClient
import os

def migrate_cloud_location_ids():
    print("🔄 Migrating cloud MongoDB location IDs to correct format...")
    
    # Cloud MongoDB connection string
    CLOUD_MONGO_URI = "mongodb+srv://judithfdo2002:kTCN07mlhHmtgrt0@cluster0.9wwflqj.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
    
    if not CLOUD_MONGO_URI:
        print("❌ No connection string provided. Exiting.")
        return
    
    try:
        # Connect to cloud MongoDB
        client = MongoClient(CLOUD_MONGO_URI)
        db = client["warehouse_management"]  # Update database name if different
        
        # Test connection
        client.server_info()
        print("✅ Connected to cloud MongoDB successfully!")
        
    except Exception as e:
        print(f"❌ Failed to connect to cloud MongoDB: {e}")
        return
    
    # Location ID mapping from numeric IDs to proper format
    # Based on the pattern: B (Bulk), D (Dangerous/Delicate), P (Perishables)
    location_mapping = {
        # Numeric ID -> Proper Location Format
        "1": "B01.1",   # Bulk zone 1, shelf 1
        "2": "B01.2",   # Bulk zone 1, shelf 2
        "3": "B02.1",   # Bulk zone 2, shelf 1
        "4": "B02.2",   # Bulk zone 2, shelf 2
        "5": "B03.1",   # Bulk zone 3, shelf 1
        "6": "B03.2",   # Bulk zone 3, shelf 2
        "7": "D01.1",   # Dangerous/Delicate zone 1, shelf 1
        "8": "D01.2",   # Dangerous/Delicate zone 1, shelf 2
        "9": "D02.1",   # Dangerous/Delicate zone 2, shelf 1
        "10": "D02.2",  # Dangerous/Delicate zone 2, shelf 2
        "11": "D03.1",  # Dangerous/Delicate zone 3, shelf 1
        "12": "D03.2",  # Dangerous/Delicate zone 3, shelf 2
        "13": "P01.1",  # Perishables zone 1, shelf 1
        "14": "P01.2",  # Perishables zone 1, shelf 2
        "15": "P02.1",  # Perishables zone 2, shelf 1
        "16": "P02.2",  # Perishables zone 2, shelf 2
        "17": "P03.1",  # Perishables zone 3, shelf 1
        "18": "P03.2",  # Perishables zone 3, shelf 2
        
        # Integer versions (in case stored as numbers)
        1: "B01.1",
        2: "B01.2", 
        3: "B02.1",
        4: "B02.2",
        5: "B03.1",
        6: "B03.2",
        7: "D01.1",
        8: "D01.2",
        9: "D02.1",
        10: "D02.2",
        11: "D03.1",
        12: "D03.2",
        13: "P01.1",
        14: "P01.2",
        15: "P02.1",
        16: "P02.2",
        17: "P03.1",
        18: "P03.2",
    }
    
    # First, let's check what collections exist
    collections = db.list_collection_names()
    print(f"Available collections: {collections}")
    
    # Check current location IDs in the database
    print("\n🔍 Checking current location IDs in cloud database...")
    if "inventory" in collections:
        sample_inventory = list(db["inventory"].find({}, {"locationID": 1}).limit(10))
        print("Sample inventory locationIDs:")
        for item in sample_inventory:
            print(f"  {item.get('locationID', 'None')}")
    
    if "location_inventory" in collections:
        sample_locations = list(db["location_inventory"].find({}, {"locationID": 1}).limit(10))
        print("Sample location_inventory locationIDs:")
        for item in sample_locations:
            print(f"  {item.get('locationID', 'None')}")
    
    # Ask for confirmation
    confirm = input("\n⚠️  Do you want to proceed with the migration? (yes/no): ").strip().lower()
    if confirm != 'yes':
        print("Migration cancelled.")
        return
    
    # Update inventory collection
    if "inventory" in collections:
        print("\nUpdating inventory collection...")
        inventory = db["inventory"]
        total_updated = 0
        
        for old_id, new_id in location_mapping.items():
            result = inventory.update_many(
                {"locationID": old_id},
                {"$set": {"locationID": new_id}}
            )
            if result.modified_count > 0:
                print(f"  Updated {result.modified_count} inventory records: {old_id} -> {new_id}")
                total_updated += result.modified_count
        
        print(f"Total inventory records updated: {total_updated}")
    
    # Update location_inventory collection
    if "location_inventory" in collections:
        print("\nUpdating location_inventory collection...")
        location_inventory = db["location_inventory"]
        total_updated = 0
        
        for old_id, new_id in location_mapping.items():
            result = location_inventory.update_many(
                {"locationID": old_id},
                {"$set": {"locationID": new_id}}
            )
            if result.modified_count > 0:
                print(f"  Updated {result.modified_count} location_inventory records: {old_id} -> {new_id}")
                total_updated += result.modified_count
        
        print(f"Total location_inventory records updated: {total_updated}")
    
    # Update other collections that might have location references
    collections_to_check = ["orders", "returns", "job_queue", "picking_tasks", "storage_history"]
    
    for collection_name in collections_to_check:
        if collection_name in collections:
            collection = db[collection_name]
            print(f"\nChecking {collection_name} collection...")
            collection_total = 0
            
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
                        collection_total += result.modified_count
                
                # Check nested fields in items arrays
                if collection_name in ["orders", "returns"]:
                    result = collection.update_many(
                        {"items.locationID": old_id},
                        {"$set": {"items.$.locationID": new_id}}
                    )
                    if result.modified_count > 0:
                        print(f"  Updated {result.modified_count} {collection_name} item records: {old_id} -> {new_id}")
                        collection_total += result.modified_count
            
            if collection_total > 0:
                print(f"Total {collection_name} records updated: {collection_total}")
    
    print("\n✅ Cloud location ID migration completed!")
    
    # Verify the migration
    print("\n🔍 Verifying cloud migration...")
    if "inventory" in collections:
        inventory_locations = list(db["inventory"].find({}, {'locationID': 1}).limit(10))
        print("Sample inventory locations after migration:")
        for inv in inventory_locations:
            location_id = inv.get('locationID')
            if location_id:
                print(f"  {location_id}")
    
    if "location_inventory" in collections:
        location_locations = list(db["location_inventory"].find({}, {'locationID': 1}).limit(10))
        print("Sample location_inventory locations after migration:")
        for loc in location_locations:
            location_id = loc.get('locationID')
            if location_id:
                print(f"  {location_id}")
    
    client.close()
    print("\n🎉 Migration verification completed!")

def create_location_mapping_reference():
    """Helper function to show the location mapping reference"""
    print("\n📋 Location ID Mapping Reference:")
    print("=" * 50)
    print("Numeric ID -> Proper Location Code")
    print("=" * 50)
    
    mappings = [
        ("1-6", "B01.1 to B03.2", "Bulk Storage (Electronics, General Items)"),
        ("7-12", "D01.1 to D03.2", "Dangerous/Delicate Storage (Fragile, Hazardous)"),
        ("13-18", "P01.1 to P03.2", "Perishables Storage (Food, Time-sensitive)")
    ]
    
    for numeric_range, location_range, description in mappings:
        print(f"{numeric_range:8} -> {location_range:15} ({description})")
    
    print("\nFormat: [Type][Zone].[Shelf]")
    print("Types: B=Bulk, D=Dangerous/Delicate, P=Perishables")
    print("Zones: 01, 02, 03 (zero-padded)")
    print("Shelves: 1, 2")

if __name__ == "__main__":
    print("🚀 Cloud MongoDB Location ID Migration Script")
    print("=" * 50)
    
    # Show mapping reference
    create_location_mapping_reference()
    
    print("\n" + "=" * 50)
    
    # Run migration
    migrate_cloud_location_ids()
