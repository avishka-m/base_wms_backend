#!/usr/bin/env python3
# sync_location_inventory.py - Initialize and sync location_inventory with current inventory

import pymongo
from datetime import datetime

def sync_location_inventory():
    """Initialize location_inventory and sync with current inventory data"""
    try:
        client = pymongo.MongoClient('mongodb://localhost:27017/')
        db = client['warehouse_management']
        
        print('🔄 SYNCING LOCATION_INVENTORY WITH INVENTORY')
        
        # Clear existing location_inventory
        location_collection = db.location_inventory
        existing_count = location_collection.count_documents({})
        if existing_count > 0:
            print(f'🗑️ Clearing {existing_count} existing location_inventory records...')
            location_collection.delete_many({})
        
        # Initialize all locations as available
        all_locations = []
        
        # Create coordinate mapping for locations
        def get_coordinates(slot_code, floor):
            if slot_code.startswith('B'):
                slot_num = int(slot_code[1:])
                if slot_num <= 7:
                    return {"x": 1, "y": 1 + slot_num, "floor": floor}
                elif slot_num <= 14:
                    return {"x": 3, "y": slot_num - 6, "floor": floor}
                else:
                    return {"x": 5, "y": slot_num - 13, "floor": floor}
            elif slot_code.startswith('P'):
                slot_num = int(slot_code[1:])
                return {"x": 7, "y": 1 + slot_num, "floor": floor}
            elif slot_code.startswith('D'):
                slot_num = int(slot_code[1:])
                return {"x": 3, "y": 9 + slot_num, "floor": floor}
            return {"x": 0, "y": 0, "floor": floor}
        
        # B slots: B01-B21, each with 4 floors
        for i in range(1, 22):
            slot_code = f"B{str(i).zfill(2)}"
            for floor in range(1, 5):
                location_id = f"{slot_code}.{floor}"
                coordinates = get_coordinates(slot_code, floor)
                all_locations.append({
                    "locationID": location_id,
                    "slotCode": slot_code,
                    "floor": floor,
                    "type": "M",  # Medium/Bin
                    "available": True,
                    "itemID": None,
                    "itemName": None,
                    "quantity": 0,
                    "coordinates": coordinates,
                    "lastUpdated": datetime.utcnow().isoformat()
                })
        
        # P slots: P01-P14, each with 4 floors  
        for i in range(1, 15):
            slot_code = f"P{str(i).zfill(2)}"
            for floor in range(1, 5):
                location_id = f"{slot_code}.{floor}"
                coordinates = get_coordinates(slot_code, floor)
                all_locations.append({
                    "locationID": location_id,
                    "slotCode": slot_code,
                    "floor": floor,
                    "type": "S",  # Small/Pellet
                    "available": True,
                    "itemID": None,
                    "itemName": None,
                    "quantity": 0,
                    "coordinates": coordinates,
                    "lastUpdated": datetime.utcnow().isoformat()
                })
        
        # D slots: D01-D14, each with 4 floors
        for i in range(1, 15):
            slot_code = f"D{str(i).zfill(2)}"
            for floor in range(1, 5):
                location_id = f"{slot_code}.{floor}"
                coordinates = get_coordinates(slot_code, floor)
                all_locations.append({
                    "locationID": location_id,
                    "slotCode": slot_code,
                    "floor": floor,
                    "type": "D",  # Large
                    "available": True,
                    "itemID": None,
                    "itemName": None,
                    "quantity": 0,
                    "coordinates": coordinates,
                    "lastUpdated": datetime.utcnow().isoformat()
                })
        
        # Insert all locations
        result = location_collection.insert_many(all_locations)
        print(f'✅ Inserted {len(result.inserted_ids)} location records')
        
        # Now sync with current inventory
        inventory_collection = db.inventory
        items_with_locations = list(inventory_collection.find({
            "locationID": {"$exists": True, "$ne": None}
        }))
        
        print(f'\n🔄 Syncing {len(items_with_locations)} inventory items...')
        
        synced_count = 0
        for item in items_with_locations:
            item_id = item.get('itemID')
            location_id = item.get('locationID')
            item_name = item.get('item_name', f'Item {item_id}')
            quantity = item.get('quantity_available', 0)
            
            # Update the location as occupied
            update_result = location_collection.update_one(
                {"locationID": location_id},
                {
                    "$set": {
                        "available": False,
                        "itemID": item_id,
                        "itemName": item_name,
                        "quantity": quantity,
                        "syncedFromInventory": True,
                        "lastUpdated": datetime.utcnow().isoformat()
                    }
                }
            )
            
            if update_result.matched_count > 0:
                print(f'  ✅ {location_id}: Item {item_id} - {item_name}')
                synced_count += 1
            else:
                print(f'  ⚠️ Location {location_id} not found for item {item_id}')
        
        print(f'\n🎉 Sync completed!')
        print(f'📍 Total locations: {len(all_locations)}')
        print(f'📦 Synced items: {synced_count}')
        print(f'🆓 Available locations: {len(all_locations) - synced_count}')
        
        # Verify
        print(f'\n🔍 Verification:')
        total_count = location_collection.count_documents({})
        available_count = location_collection.count_documents({"available": True})
        occupied_count = location_collection.count_documents({"available": False})
        
        print(f'Total location records: {total_count}')
        print(f'Available locations: {available_count}')
        print(f'Occupied locations: {occupied_count}')
        
    except Exception as e:
        print(f'❌ Error: {e}')
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    sync_location_inventory()
