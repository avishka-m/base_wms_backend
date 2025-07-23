#!/usr/bin/env python3
# check_location_inventory.py - Check what's in the location_inventory collection

import pymongo

def check_location_inventory():
    try:
        client = pymongo.MongoClient('mongodb://localhost:27017/')
        db = client['warehouse_management']
        
        print('🔍 CHECKING LOCATION_INVENTORY COLLECTION')
        
        # Check total count
        total_count = db.location_inventory.count_documents({})
        print(f'Total location_inventory records: {total_count}')
        
        if total_count == 0:
            print('❌ location_inventory collection is EMPTY!')
            print('You need to initialize it first via the API endpoint or a script.')
            return
        
        # Check if any locations have items assigned
        occupied_count = db.location_inventory.count_documents({'available': False})
        available_count = db.location_inventory.count_documents({'available': True})
        
        print(f'Available locations: {available_count}')
        print(f'Occupied locations: {occupied_count}')
        
        # Show occupied locations if any
        if occupied_count > 0:
            print('\nOccupied locations:')
            occupied_locations = list(db.location_inventory.find({'available': False}).limit(5))
            for loc in occupied_locations:
                location_id = loc.get('locationID', 'Unknown')
                item_id = loc.get('itemID', 'None')
                item_name = loc.get('itemName', 'None')
                print(f'  {location_id}: Item {item_id} - {item_name}')
        
        # Show sample available locations
        print('\nSample available locations:')
        sample_locations = list(db.location_inventory.find({'available': True}).limit(5))
        for loc in sample_locations:
            location_id = loc.get('locationID', 'Unknown')
            location_type = loc.get('type', 'Unknown')
            print(f'  {location_id} (Type: {location_type})')
        
        # Check for items 101, 103, 105 in location_inventory
        print('\nChecking for items 101, 103, 105 in location_inventory:')
        for item_id in [101, 103, 105]:
            item_location = db.location_inventory.find_one({'itemID': item_id})
            if item_location:
                location_id = item_location.get('locationID', 'Unknown')
                print(f'  Item {item_id}: Found at {location_id}')
            else:
                print(f'  Item {item_id}: NOT FOUND in location_inventory')
                
    except Exception as e:
        print(f'❌ Error: {e}')
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    check_location_inventory()
