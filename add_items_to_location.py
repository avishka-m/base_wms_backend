#!/usr/bin/env python3
"""
Script to add 2 new items to location_inventory collection
"""

from pymongo import MongoClient

def add_items_to_location_inventory():
    """Add 2 new items to location_inventory"""
    client = MongoClient("mongodb://localhost:27017/")
    db = client["warehouse_management"]
    
    print("🔍 Adding 2 new items to location_inventory...")
    
    # Check current highest itemID
    existing_items = list(db.location_inventory.find({}, {"itemID": 1}).sort("itemID", -1).limit(1))
    next_item_id = 1
    if existing_items:
        next_item_id = existing_items[0].get("itemID", 0) + 1
    
    print(f"Next available itemID: {next_item_id}")
    
    # Define 2 new items to add
    new_items = [
        {
            "itemID": next_item_id,
            "itemName": "Wireless Keyboard",
            "quantity": 35,
            "locationID": "E3.1",  # Electronics zone
            "capacity": 60,
            "zone": "E",
            "aisle": "3",
            "shelf": "1"
        },
        {
            "itemID": next_item_id + 1,
            "itemName": "Portable Hard Drive",
            "quantity": 28,
            "locationID": "E3.2",  # Electronics zone
            "capacity": 50,
            "zone": "E",
            "aisle": "3",
            "shelf": "2"
        }
    ]
    
    # Insert the new items
    added_items = []
    
    for item in new_items:
        # Check if item already exists
        existing = db.location_inventory.find_one({"itemID": item["itemID"]})
        if existing:
            print(f"⚠️  Item ID {item['itemID']} already exists, skipping...")
            continue
        
        # Insert the item
        result = db.location_inventory.insert_one(item)
        added_items.append(item)
        
        print(f"✅ Added: {item['itemName']} (ID: {item['itemID']}) - Qty: {item['quantity']} at {item['locationID']}")
    
    if len(added_items) == 0:
        print("❌ No new items were added (all already existed)")
    else:
        print(f"\n🎉 Successfully added {len(added_items)} new items to location_inventory!")
        
        # Show summary
        print("\n📊 Added items summary:")
        for item in added_items:
            print(f"  - {item['itemName']} (ID: {item['itemID']})")
            print(f"    Location: {item['locationID']} (Zone: {item['zone']}, Aisle: {item['aisle']}, Shelf: {item['shelf']})")
            print(f"    Quantity: {item['quantity']}/{item['capacity']}")
    
    # Show total items in location_inventory
    total_items = db.location_inventory.count_documents({})
    unique_items = len(db.location_inventory.distinct("itemID"))
    
    print(f"\n📈 Total location_inventory entries: {total_items}")
    print(f"📈 Unique items: {unique_items}")
    
    client.close()
    return added_items

if __name__ == "__main__":
    try:
        added = add_items_to_location_inventory()
        if added:
            print(f"\n✨ All {len(added)} items added successfully!")
        else:
            print("\n✨ Operation completed.")
    except Exception as e:
        print(f"\n💥 Error adding items: {str(e)}")
        import traceback
        traceback.print_exc()
