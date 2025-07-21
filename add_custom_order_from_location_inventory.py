#!/usr/bin/env python3
# add_custom_order_from_location_inventory.py - Create orders using location_inventory collection

import pymongo
from datetime import datetime, timezone

def add_custom_order_from_location_inventory():
    """Add a custom order with location data from location_inventory collection"""
    try:
        # Connect to MongoDB
        MONGO_URL = "mongodb://localhost:27017/"
        DATABASE_NAME = "warehouse_management"
        
        print("🔄 Connecting to MongoDB...")
        client = pymongo.MongoClient(MONGO_URL)
        db = client[DATABASE_NAME]
        
        # Test connection
        db.command('ping')
        print("✅ Connected to MongoDB successfully")
        
        # Collections
        orders_collection = db["orders"]
        location_inventory_collection = db["location_inventory"]  # ✨ NEW: Using location_inventory
        inventory_collection = db["inventory"]  # For fallback item details
        
        # Get next order ID
        last_order = orders_collection.find_one({}, sort=[("orderID", -1)])
        next_order_id = (last_order.get("orderID", 0) + 1) if last_order else 1
        
        print(f"📝 Creating custom order #{next_order_id} using location_inventory")
        
        # ✨ CUSTOMIZABLE ORDER DATA
        customer_id = 1
        customer_name = "John Smith"
        shipping_address = "123 Main Street, New York, NY 10001"
        priority = 2  # 1=high, 2=medium, 3=low
        notes = "Please handle electronics with care - Using location_inventory lookup"
        
        # ✨ CUSTOMIZABLE ITEMS
        custom_items = [
            {
                "itemID": 101,
                "quantity": 2,
                "price": 79.99
            },
            {
                "itemID": 103,
                "quantity": 3,
                "price": 18.50
            },
            {
                "itemID": 105,
                "quantity": 1,
                "price": 24.99
            }
        ]
        
        print("🔍 Looking up item details from location_inventory...")
        
        # Create order items with location_inventory lookup
        order_items = []
        total_amount = 0
        
        for i, item_request in enumerate(custom_items):
            item_id = item_request["itemID"]
            requested_qty = item_request["quantity"]
            requested_price = item_request["price"]
            
            # ✨ NEW: Look up item in location_inventory first
            location_item = location_inventory_collection.find_one({"itemID": item_id})
            
            if location_item:
                # Found in location_inventory
                item_name = location_item.get("itemName", f"Item {item_id}")
                location_id = location_item.get("locationID")
                coordinates = location_item.get("coordinates")
                available_qty = location_item.get("quantity", 0)
                category = "General"  # Default category
                
                # Try to get more details from inventory collection
                inventory_item = inventory_collection.find_one({"itemID": item_id})
                if inventory_item:
                    item_name = inventory_item.get("item_name", item_name)
                    category = inventory_item.get("category", category)
                    # Don't override location data from location_inventory
                
                print(f"✅ Found in location_inventory: {item_name} at {location_id} - Available: {available_qty}")
            else:
                # Fallback to inventory collection
                inventory_item = inventory_collection.find_one({"itemID": item_id})
                if inventory_item:
                    item_name = inventory_item.get("item_name", f"Item {item_id}")
                    category = inventory_item.get("category", "General")
                    location_id = inventory_item.get("locationID")
                    coordinates = inventory_item.get("coordinates")
                    available_qty = inventory_item.get("quantity_available", 0)
                    print(f"⚠️ Not in location_inventory, using inventory: {item_name} at {location_id}")
                else:
                    # Ultimate fallback
                    item_name = f"Custom Item {item_id}"
                    category = "General"
                    location_id = None
                    coordinates = None
                    available_qty = 0
                    print(f"❌ Item {item_id} not found in either collection, using fallback data")
            
            if available_qty < requested_qty:
                print(f"⚠️ Warning: Only {available_qty} available for {item_name}, but {requested_qty} requested")
            
            item_total = requested_qty * requested_price
            total_amount += item_total
            
            order_detail = {
                "orderDetailID": i + 1,
                "itemID": item_id,
                "item_name": item_name,
                "quantity": requested_qty,
                "price": requested_price,
                "fulfilled_quantity": 0,
                "category": category,
                "locationID": location_id,
                "coordinates": coordinates,
                "source": "location_inventory" if location_item else "inventory",  # ✨ NEW: Track source
                "created_at": datetime.now(timezone.utc),
                "updated_at": datetime.now(timezone.utc)
            }
            
            order_items.append(order_detail)
            location_info = f" at {location_id}" if location_id else " (no location)"
            print(f"📦 Item {i+1}: {item_name} (ID: {item_id}) - Qty: {requested_qty}, Price: ${requested_price}, Total: ${item_total:.2f}{location_info}")
        
        # Create the order document
        current_time = datetime.now(timezone.utc)
        
        order_document = {
            "orderID": next_order_id,
            "customerID": customer_id,
            "customer_name": customer_name,
            "order_date": current_time,
            "shipping_address": shipping_address,
            "order_status": "pending",
            "priority": priority,
            "notes": notes,
            "items": order_items,
            "total_amount": round(total_amount, 2),
            "assigned_worker": None,
            "created_at": current_time,
            "updated_at": current_time,
            "location_source": "location_inventory"  # ✨ NEW: Track which system was used
        }
        
        # Insert the order
        result = orders_collection.insert_one(order_document)
        order_id = str(result.inserted_id)
        
        print("=" * 60)
        print("🎉 CUSTOM ORDER CREATED SUCCESSFULLY!")
        print("=" * 60)
        print(f"📋 Order ID: #{next_order_id}")
        print(f"🆔 MongoDB ID: {order_id}")
        print(f"👤 Customer: {customer_name} (ID: {customer_id})")
        print(f"📍 Shipping: {shipping_address}")
        print(f"📅 Order Date: {current_time.strftime('%Y-%m-%d %H:%M:%S UTC')}")
        print(f"⚡ Priority: {priority} ({'High' if priority == 1 else 'Medium' if priority == 2 else 'Low'})")
        print(f"💰 Total Amount: ${total_amount:.2f}")
        print(f"📝 Status: pending")
        print(f"📋 Notes: {notes}")
        print(f"🗂️ Location Source: location_inventory")
        
        print("\n📦 ITEMS IN ORDER:")
        print("-" * 60)
        for i, item in enumerate(order_items, 1):
            location_info = f" at {item['locationID']}" if item.get('locationID') else " (no location)"
            coords_info = f" ({item['coordinates']['x']}, {item['coordinates']['y']})" if item.get('coordinates') else ""
            source_info = f" [from {item.get('source', 'unknown')}]"
            print(f"{i}. {item['item_name']} (ID: {item['itemID']}){location_info}{coords_info}{source_info}")
            print(f"   Quantity: {item['quantity']} × ${item['price']} = ${item['quantity'] * item['price']:.2f}")
        
        print("\n✅ Order is now available in the Picker Dashboard for processing!")
        print("🗺️ Items with locations will show proper picking paths!")
        
        # Show current order statistics
        total_orders = orders_collection.count_documents({})
        pending_orders = orders_collection.count_documents({"order_status": "pending"})
        print(f"\n📊 Total Orders: {total_orders} | Pending Orders: {pending_orders}")
        
    except Exception as e:
        print(f"❌ Error creating custom order: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("🛒 Custom Order Creation Script (Using location_inventory)")
    print("Creating a custom order with location data from location_inventory...\n")
    add_custom_order_from_location_inventory()
