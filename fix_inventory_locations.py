#!/usr/bin/env python3
# fix_inventory_locations.py - Fix inventory locations to use proper location codes

import pymongo

def fix_inventory_locations():
    """Update inventory collection to use proper location codes instead of numbers"""
    try:
        client = pymongo.MongoClient('mongodb://localhost:27017/')
        db = client['warehouse_management']
        
        print('🔧 FIXING INVENTORY LOCATIONS')
        print('Converting numeric location IDs to proper location codes...\n')
        
        # Mapping of numeric IDs to proper location codes
        location_mapping = {
            1: {"locationID": "B01.1", "coordinates": {"x": 1, "y": 2, "floor": 1}},
            2: {"locationID": "B02.1", "coordinates": {"x": 1, "y": 3, "floor": 1}},
            3: {"locationID": "B03.1", "coordinates": {"x": 1, "y": 4, "floor": 1}},
            4: {"locationID": "B04.1", "coordinates": {"x": 1, "y": 5, "floor": 1}},
            5: {"locationID": "B05.1", "coordinates": {"x": 1, "y": 6, "floor": 1}},
            6: {"locationID": "D01.1", "coordinates": {"x": 3, "y": 10, "floor": 1}},
            7: {"locationID": "D02.1", "coordinates": {"x": 3, "y": 11, "floor": 1}},
            8: {"locationID": "D03.1", "coordinates": {"x": 3, "y": 12, "floor": 1}},
            101: {"locationID": "B01.1", "coordinates": {"x": 1, "y": 2, "floor": 1}},
            103: {"locationID": "D01.1", "coordinates": {"x": 3, "y": 10, "floor": 1}},
            105: {"locationID": "B03.1", "coordinates": {"x": 1, "y": 4, "floor": 1}}
        }
        
        # Update inventory items
        inventory_collection = db.inventory
        updated_count = 0
        
        for item_id, location_data in location_mapping.items():
            result = inventory_collection.update_one(
                {"itemID": item_id},
                {
                    "$set": {
                        "locationID": location_data["locationID"],
                        "coordinates": location_data["coordinates"]
                    }
                }
            )
            
            if result.matched_count > 0:
                print(f'✅ Updated Item {item_id} → {location_data["locationID"]} at {location_data["coordinates"]}')
                updated_count += 1
            else:
                print(f'⚠️ Item {item_id} not found in inventory')
        
        print(f'\n🎉 Updated {updated_count} inventory items with proper location codes!')
        
        # Verify the changes
        print('\n🔍 VERIFICATION - Updated inventory locations:')
        updated_items = list(inventory_collection.find(
            {"itemID": {"$in": list(location_mapping.keys())}},
            {"itemID": 1, "item_name": 1, "locationID": 1, "coordinates": 1}
        ))
        
        for item in updated_items:
            location = item.get('locationID', 'NO LOCATION')
            coords = item.get('coordinates', {})
            name = item.get('item_name', 'Unknown')
            print(f'Item {item["itemID"]}: {name} → {location} at ({coords.get("x", "?")}, {coords.get("y", "?")})')
            
    except Exception as e:
        print(f'❌ Error: {e}')
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    fix_inventory_locations()
