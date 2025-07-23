#!/usr/bin/env python3
# add_missing_inventory.py - Add missing inventory items that orders expect

import pymongo
from datetime import datetime, timezone

def add_missing_inventory():
    """Add missing inventory items 101, 103, 105 with proper location data"""
    try:
        client = pymongo.MongoClient('mongodb://localhost:27017/')
        db = client['warehouse_management']
        inventory_collection = db.inventory
        
        print('🔧 ADDING MISSING INVENTORY ITEMS')
        print('Adding items 101, 103, 105 with proper location codes...\n')
        
        # Items to add
        missing_items = [
            {
                "itemID": 101,
                "item_name": "Wireless Bluetooth Headphones",
                "category": "Electronics",
                "locationID": "B01.1",
                "coordinates": {"x": 1, "y": 2, "floor": 1},
                "quantity_available": 15,
                "unit_price": 79.99,
                "supplier": "TechCorp",
                "created_at": datetime.now(timezone.utc),
                "updated_at": datetime.now(timezone.utc)
            },
            {
                "itemID": 103,
                "item_name": "Premium Coffee Beans",
                "category": "Food & Beverage",
                "locationID": "D01.1",
                "coordinates": {"x": 3, "y": 10, "floor": 1},
                "quantity_available": 25,
                "unit_price": 18.50,
                "supplier": "CoffeeCo",
                "created_at": datetime.now(timezone.utc),
                "updated_at": datetime.now(timezone.utc)
            },
            {
                "itemID": 105,
                "item_name": "Stainless Steel Water Bottle",
                "category": "Home & Garden",
                "locationID": "B03.1",
                "coordinates": {"x": 1, "y": 4, "floor": 1},
                "quantity_available": 30,
                "unit_price": 24.99,
                "supplier": "HomeCorp",
                "created_at": datetime.now(timezone.utc),
                "updated_at": datetime.now(timezone.utc)
            }
        ]
        
        added_count = 0
        for item in missing_items:
            # Check if item already exists
            existing = inventory_collection.find_one({"itemID": item["itemID"]})
            
            if existing:
                # Update existing item with proper location data
                result = inventory_collection.update_one(
                    {"itemID": item["itemID"]},
                    {"$set": {
                        "item_name": item["item_name"],
                        "category": item["category"],
                        "locationID": item["locationID"],
                        "coordinates": item["coordinates"],
                        "quantity_available": item["quantity_available"],
                        "updated_at": item["updated_at"]
                    }}
                )
                print(f'✅ Updated Item {item["itemID"]}: {item["item_name"]} → {item["locationID"]}')
            else:
                # Insert new item
                inventory_collection.insert_one(item)
                print(f'✅ Added Item {item["itemID"]}: {item["item_name"]} → {item["locationID"]}')
            
            added_count += 1
        
        print(f'\n🎉 Processed {added_count} inventory items!')
        
        # Verify the changes
        print('\n🔍 VERIFICATION - Items 101, 103, 105:')
        target_items = list(inventory_collection.find(
            {"itemID": {"$in": [101, 103, 105]}},
            {"itemID": 1, "item_name": 1, "locationID": 1, "coordinates": 1, "quantity_available": 1}
        ))
        
        for item in target_items:
            location = item.get('locationID', 'NO LOCATION')
            coords = item.get('coordinates', {})
            name = item.get('item_name', 'Unknown')
            qty = item.get('quantity_available', 0)
            print(f'Item {item["itemID"]}: {name} → {location} at ({coords.get("x", "?")}, {coords.get("y", "?")}) | Qty: {qty}')
            
    except Exception as e:
        print(f'❌ Error: {e}')
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    add_missing_inventory()
