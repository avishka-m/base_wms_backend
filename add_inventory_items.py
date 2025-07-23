#!/usr/bin/env python3
"""
Script to add more items to the inventory and location inventory to support order creation
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from pymongo import MongoClient
from datetime import datetime
import random

def add_inventory_items():
    """Add more items to inventory and place them in locations"""
    
    # Connect to MongoDB
    client = MongoClient("mongodb://localhost:27017/")
    db = client["warehouse_management"]
    inventory_collection = db["inventory"]
    location_collection = db["location_inventory"]
    
    print("🔍 Adding more inventory items...")
    
    # Check current max itemID
    last_item = inventory_collection.find_one({}, sort=[("itemID", -1)])
    next_item_id = (last_item["itemID"] + 1) if last_item else 1
    
    # New items to add
    new_items = [
        {
            "itemID": next_item_id,
            "itemName": "Wireless Headphones",
            "category": "Electronics",
            "description": "Bluetooth wireless headphones with noise cancellation",
            "price": 129.99,
            "supplier": "TechCorp",
            "reorderLevel": 10,
            "reorderQuantity": 50,
            "unit": "piece",
            "weight": 0.3,
            "size": "S"
        },
        {
            "itemID": next_item_id + 1,
            "itemName": "Coffee Beans Premium",
            "category": "Food",
            "description": "Premium arabica coffee beans 1kg bag",
            "price": 24.99,
            "supplier": "CoffeeWorld",
            "reorderLevel": 20,
            "reorderQuantity": 100,
            "unit": "bag",
            "weight": 1.0,
            "size": "M"
        },
        {
            "itemID": next_item_id + 2,
            "itemName": "Office Chair Ergonomic",
            "category": "Furniture",
            "description": "Ergonomic office chair with lumbar support",
            "price": 299.99,
            "supplier": "OfficePlus",
            "reorderLevel": 5,
            "reorderQuantity": 20,
            "unit": "piece",
            "weight": 15.0,
            "size": "D"
        },
        {
            "itemID": next_item_id + 3,
            "itemName": "Smartphone Case",
            "category": "Electronics",
            "description": "Protective smartphone case with screen protector",
            "price": 19.99,
            "supplier": "TechCorp",
            "reorderLevel": 30,
            "reorderQuantity": 200,
            "unit": "piece",
            "weight": 0.1,
            "size": "S"
        },
        {
            "itemID": next_item_id + 4,
            "itemName": "Kitchen Knife Set",
            "category": "Kitchenware",
            "description": "Professional kitchen knife set with wooden block",
            "price": 89.99,
            "supplier": "KitchenPro",
            "reorderLevel": 8,
            "reorderQuantity": 30,
            "unit": "set",
            "weight": 2.5,
            "size": "M"
        },
        {
            "itemID": next_item_id + 5,
            "itemName": "Yoga Mat Premium",
            "category": "Sports",
            "description": "Premium non-slip yoga mat with carrying strap",
            "price": 45.99,
            "supplier": "FitLife",
            "reorderLevel": 15,
            "reorderQuantity": 75,
            "unit": "piece",
            "weight": 1.2,
            "size": "M"
        }
    ]
    
    # Add creation timestamps
    for item in new_items:
        item["created_at"] = datetime.utcnow()
        item["updated_at"] = datetime.utcnow()
    
    # Insert items into inventory
    result = inventory_collection.insert_many(new_items)
    print(f"✅ Added {len(result.inserted_ids)} new items to inventory")
    
    # Now place these items in various locations
    print("📦 Placing items in warehouse locations...")
    
    # Get available locations grouped by type
    available_locations = {
        "S": list(location_collection.find({"type": "S", "available": True}).limit(20)),
        "M": list(location_collection.find({"type": "M", "available": True}).limit(20)),
        "D": list(location_collection.find({"type": "D", "available": True}).limit(10))
    }
    
    # Place items strategically
    item_placements = [
        # Small items (S locations)
        {"itemID": next_item_id, "itemName": "Wireless Headphones", "quantities": [25, 30, 15], "type": "S"},
        {"itemID": next_item_id + 3, "itemName": "Smartphone Case", "quantities": [45, 60, 40, 35], "type": "S"},
        
        # Medium items (M locations)
        {"itemID": next_item_id + 1, "itemName": "Coffee Beans Premium", "quantities": [20, 25, 30, 15], "type": "M"},
        {"itemID": next_item_id + 4, "itemName": "Kitchen Knife Set", "quantities": [12, 18, 15], "type": "M"},
        {"itemID": next_item_id + 5, "itemName": "Yoga Mat Premium", "quantities": [10, 15, 20], "type": "M"},
        
        # Large items (D locations)
        {"itemID": next_item_id + 2, "itemName": "Office Chair Ergonomic", "quantities": [5, 8, 6], "type": "D"}
    ]
    
    updates = []
    for placement in item_placements:
        item_type = placement["type"]
        locations = available_locations[item_type]
        
        for i, quantity in enumerate(placement["quantities"]):
            if i < len(locations):
                location = locations[i]
                
                update = {
                    "filter": {"locationID": location["locationID"]},
                    "update": {
                        "$set": {
                            "available": False,
                            "itemID": placement["itemID"],
                            "itemName": placement["itemName"],
                            "quantity": quantity,
                            "lastUpdated": datetime.utcnow().isoformat()
                        }
                    }
                }
                updates.append(update)
                
                print(f"  📍 {location['locationID']}: {placement['itemName']} x{quantity}")
    
    # Execute all updates
    for update in updates:
        location_collection.update_one(update["filter"], update["update"])
    
    print(f"✅ Placed {len(updates)} item batches in warehouse locations")
    
    # Summary
    print(f"\n📊 Summary:")
    total_items = inventory_collection.count_documents({})
    occupied_locations = location_collection.count_documents({"available": False})
    
    print(f"  - Total items in inventory: {total_items}")
    print(f"  - Occupied warehouse locations: {occupied_locations}")
    
    # Show available items for orders
    stored_items = list(location_collection.aggregate([
        {"$match": {"quantity": {"$gt": 0}}},
        {"$group": {
            "_id": "$itemID",
            "itemName": {"$first": "$itemName"},
            "totalQuantity": {"$sum": "$quantity"}
        }},
        {"$sort": {"_id": 1}}
    ]))
    
    print(f"\n🛒 Items available for orders:")
    for item in stored_items:
        print(f"  - {item['itemName']} (ID: {item['_id']}) - Total: {item['totalQuantity']} units")
    
    client.close()
    return len(stored_items)

if __name__ == "__main__":
    try:
        item_count = add_inventory_items()
        print(f"\n🎉 Successfully prepared {item_count} different items for order creation!")
    except Exception as e:
        print(f"\n💥 Error adding inventory: {str(e)}")
        import traceback
        traceback.print_exc()
