#!/usr/bin/env python3
"""
Migrate inventory items from old numeric locationID to new location_inventory format
"""

import pymongo
from datetime import datetime

def migrate_inventory_locations():
    """Update inventory items to use new location_inventory IDs instead of old numeric ones"""
    try:
        # Connect to MongoDB
        client = pymongo.MongoClient('mongodb://localhost:27017/')
        db = client['warehouse_management']
        
        print("🔄 Starting inventory location migration...")
        
        # Get collections
        inventory_collection = db['inventory']
        old_locations_collection = db['locations']
        location_inventory_collection = db['location_inventory']
        
        # Create mapping from old location IDs to new location IDs
        print("🗺️ Creating location mapping...")
        
        # Get all old locations
        old_locations = list(old_locations_collection.find())
        location_mapping = {}
        
        for old_loc in old_locations:
            old_id = old_loc.get('locationID')
            section = old_loc.get('section', 'B')
            row = old_loc.get('row', '1')
            shelf = old_loc.get('shelf', '1')
            bin_num = old_loc.get('bin', '1')
            
            # Create new location ID format: {Section}{Row:02d}.{Shelf}
            # For simplicity, we'll map to available locations in location_inventory
            new_location_id = f"{section}{row.zfill(2)}.{shelf}"
            location_mapping[old_id] = new_location_id
            
        print(f"📋 Created mapping for {len(location_mapping)} locations")
        
        # Get all inventory items with old locationID
        inventory_items = list(inventory_collection.find({"locationID": {"$exists": True}}))
        print(f"📦 Found {len(inventory_items)} inventory items with locationID")
        
        updated_count = 0
        
        for item in inventory_items:
            old_location_id = item.get('locationID')
            
            if isinstance(old_location_id, int) and old_location_id in location_mapping:
                new_location_id = location_mapping[old_location_id]
                
                # Check if the new location exists in location_inventory
                new_location = location_inventory_collection.find_one({"locationID": new_location_id})
                
                if new_location:
                    # Update the inventory item
                    result = inventory_collection.update_one(
                        {"_id": item["_id"]},
                        {
                            "$set": {
                                "locationID": new_location_id,
                                "location_updated_at": datetime.utcnow(),
                                "migrated_from_old_location": old_location_id
                            }
                        }
                    )
                    
                    if result.modified_count > 0:
                        updated_count += 1
                        item_name = item.get('item_name', 'Unknown')
                        print(f"✅ Updated {item_name}: {old_location_id} → {new_location_id}")
                    
                else:
                    # If the mapped location doesn't exist, find any available location of same type
                    # Determine type based on old section
                    section = location_mapping[old_location_id][0]  # First character
                    location_type = 'M' if section == 'B' else ('S' if section == 'P' else 'D')
                    
                    # Find an available location of the same type
                    available_location = location_inventory_collection.find_one({
                        "type": location_type,
                        "available": True
                    })
                    
                    if available_location:
                        new_location_id = available_location['locationID']
                        
                        # Update inventory and mark location as occupied
                        inventory_collection.update_one(
                            {"_id": item["_id"]},
                            {
                                "$set": {
                                    "locationID": new_location_id,
                                    "location_updated_at": datetime.utcnow(),
                                    "migrated_from_old_location": old_location_id
                                }
                            }
                        )
                        
                        location_inventory_collection.update_one(
                            {"_id": available_location["_id"]},
                            {"$set": {"available": False}}
                        )
                        
                        updated_count += 1
                        item_name = item.get('item_name', 'Unknown')
                        print(f"✅ Updated {item_name}: {old_location_id} → {new_location_id} (auto-assigned)")
                    else:
                        item_name = item.get('item_name', 'Unknown')
                        print(f"⚠️ No available {location_type} type location for {item_name} (old ID: {old_location_id})")
            
            else:
                # Already in new format or not found in mapping
                if not isinstance(old_location_id, int):
                    item_name = item.get('item_name', 'Unknown')
                    print(f"ℹ️ {item_name} already has new format: {old_location_id}")
        
        print(f"\n🎉 Migration complete! Updated {updated_count} inventory items")
        
        # Show sample results
        print("\n📊 Sample updated inventory items:")
        updated_items = list(inventory_collection.find({"migrated_from_old_location": {"$exists": True}}).limit(5))
        for item in updated_items:
            item_name = item.get('item_name', 'Unknown')
            old_id = item.get('migrated_from_old_location')
            new_id = item.get('locationID')
            print(f"  📦 {item_name}: {old_id} → {new_id}")
        
    except Exception as e:
        print(f"❌ Error during migration: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("🔄 Inventory Location Migration Script")
    print("Updating inventory items from old numeric locationIDs to new format...\n")
    migrate_inventory_locations()
