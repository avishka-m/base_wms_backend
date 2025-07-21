#!/usr/bin/env python3
# check_locations.py - Check location data consistency

import pymongo

def check_locations():
    try:
        client = pymongo.MongoClient('mongodb://localhost:27017/')
        db = client['warehouse_management']
        
        print('=== CHECKING LOCATION DATA SOURCES ===')
        
        # Check inventory locations
        print('\n📍 INVENTORY LOCATIONS:')
        inventory_items = list(db.inventory.find({}, {'itemID': 1, 'item_name': 1, 'locationID': 1, 'coordinates': 1}).limit(10))
        for item in inventory_items:
            location = item.get('locationID', 'NO LOCATION')
            name = item.get('item_name', 'Unknown')
            print(f'Item {item["itemID"]}: {name} → Location: {location}')
        
        # Check recent orders
        print('\n📦 RECENT ORDER LOCATIONS:')
        recent_orders = list(db.orders.find({}, {'orderID': 1, 'items': 1}).sort('orderID', -1).limit(3))
        for order in recent_orders:
            print(f'\nOrder #{order["orderID"]}:')
            for item in order.get('items', []):
                location = item.get('locationID', 'NO LOCATION')
                name = item.get('item_name', 'Unknown')
                item_id = item.get('itemID', 'Unknown')
                print(f'  - Item {item_id}: {name} → {location}')
        
        # Check if there are location_inventory records
        print('\n🗺️ LOCATION_INVENTORY RECORDS:')
        location_inv_count = db.location_inventory.count_documents({})
        print(f'Total location_inventory records: {location_inv_count}')
        
        if location_inv_count > 0:
            location_samples = list(db.location_inventory.find({}, {'itemID': 1, 'locationID': 1}).limit(5))
            for loc in location_samples:
                item_id = loc.get('itemID', 'Unknown')
                location = loc.get('locationID', 'Unknown')
                print(f'  - Item {item_id}: {location}')
        
        # Check collection names
        print('\n📋 COLLECTIONS IN DATABASE:')
        collections = db.list_collection_names()
        for collection in collections:
            print(f'  - {collection}')
            
    except Exception as e:
        print(f'Error: {e}')
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    check_locations()
