#!/usr/bin/env python3
"""
Script to create 5 orders with different numbers of items (2, 3, 4 items each)
using items from the location_inventory collection with proper warehouse locations.
"""

from pymongo import MongoClient
from datetime import datetime, timezone
import random

def create_orders_from_location_inventory():
    """Create 5 orders using items from location_inventory table"""
    client = MongoClient("mongodb://localhost:27017/")
    db = client["warehouse_management"]
    
    print("🔍 Starting order creation using location_inventory...")
    
    # Get available items from location_inventory with stock > 0
    available_items = list(db.location_inventory.find({
        "quantity": {"$gt": 0},
        "itemID": {"$ne": None},
        "itemName": {"$ne": None}
    }))
    
    if len(available_items) == 0:
        print("❌ Error: No items with stock found in location_inventory")
        client.close()
        return
    
    # Group by itemID to get unique items with their total available quantities and locations
    items_dict = {}
    for location in available_items:
        item_id = location["itemID"]
        if item_id not in items_dict:
            items_dict[item_id] = {
                "itemID": item_id,
                "itemName": location["itemName"],
                "total_available": 0,
                "locations": []
            }
        
        items_dict[item_id]["total_available"] += location["quantity"]
        items_dict[item_id]["locations"].append({
            "locationID": location["locationID"],
            "quantity": location["quantity"],
            "zone": location.get("zone", ""),
            "aisle": location.get("aisle", ""),
            "shelf": location.get("shelf", "")
        })
    
    unique_items = list(items_dict.values())
    
    if len(unique_items) < 4:
        print(f"❌ Error: Need at least 4 different items in location_inventory, found only {len(unique_items)}")
        print("Available items:")
        for item in unique_items:
            print(f"  - {item['itemName']} (ID: {item['itemID']}) - Available: {item['total_available']}")
        client.close()
        return
    
    print(f"✅ Found {len(unique_items)} different items in location_inventory")
    
    # Show some sample items with their locations
    print("\nSample available items:")
    for item in unique_items[:5]:
        locations_str = ", ".join([f"{loc['locationID']}({loc['quantity']})" for loc in item['locations'][:3]])
        print(f"  - {item['itemName']} (ID: {item['itemID']}) - Total: {item['total_available']} - Locations: {locations_str}")
    
    # Get existing customers
    customers = list(db.customers.find({}))
    if len(customers) == 0:
        # Create a sample customer if none exist
        sample_customer = {
            "customerID": 1,
            "name": "John Doe",
            "email": "john.doe@example.com",
            "phone": "555-0123",
            "address": "123 Main St, Anytown, USA"
        }
        db.customers.insert_one(sample_customer)
        customers = [sample_customer]
        print("✅ Created sample customer")
    
    print(f"✅ Found {len(customers)} customers")
    
    # Get next order ID
    last_order = db.orders.find_one({}, sort=[("orderID", -1)])
    next_order_id = 1 if not last_order else last_order.get("orderID", 0) + 1
    
    # Get next order detail ID
    max_detail_id = 0
    for order in db.orders.find({}):
        for item in order.get("items", []):
            if "orderDetailID" in item:
                max_detail_id = max(max_detail_id, item["orderDetailID"])
    next_detail_id = max_detail_id + 1
    
    # Sample shipping addresses
    shipping_addresses = [
        "123 Main St, New York, NY 10001",
        "456 Oak Ave, Los Angeles, CA 90210", 
        "789 Pine Rd, Chicago, IL 60601",
        "321 Elm St, Houston, TX 77001",
        "654 Maple Dr, Phoenix, AZ 85001"
    ]
    
    # Order configurations: [number_of_items, priority, status]
    order_configs = [
        (2, 1, "pending"),       # Order 1: 2 items, high priority
        (3, 2, "pending"),       # Order 2: 3 items, medium priority
        (4, 3, "pending"),       # Order 3: 4 items, low priority
        (2, 1, "processing"),    # Order 4: 2 items, high priority, processing
        (3, 2, "pending")        # Order 5: 3 items, medium priority
    ]
    
    created_orders = []
    
    for i, (num_items, priority, status) in enumerate(order_configs, 1):
        print(f"\n📦 Creating Order {i} with {num_items} items...")
        
        # Select random customer
        customer = random.choice(customers)
        customer_id = customer.get("customerID", customer.get("_id"))
        
        # Select random items for this order
        selected_items = random.sample(unique_items, min(num_items, len(unique_items)))
        
        # Create order items
        order_items = []
        total_amount = 0
        
        for item in selected_items:
            # Determine reasonable order quantity (not more than available)
            max_qty = min(item["total_available"], 5)  # Cap at 5 units per item
            order_qty = random.randint(1, max_qty)
            
            # Generate realistic price
            price = round(random.uniform(15.00, 89.99), 2)
            
            # Select primary location for this item (location with most stock)
            primary_location = max(item["locations"], key=lambda x: x["quantity"])
            
            order_item = {
                "itemID": item["itemID"],
                "itemName": item["itemName"],
                "quantity": order_qty,
                "price": price,
                "orderDetailID": next_detail_id,
                "fulfilled_quantity": 0,
                "primary_location": primary_location["locationID"],
                "zone": primary_location.get("zone", ""),
                "available_locations": [loc["locationID"] for loc in item["locations"]]
            }
            
            order_items.append(order_item)
            total_amount += price * order_qty
            next_detail_id += 1
            
            print(f"  - {item['itemName']} (ID: {item['itemID']}) x{order_qty} @ ${price:.2f} from {primary_location['locationID']}")
        
        # Create the order
        order = {
            "orderID": next_order_id,
            "customerID": customer_id,
            "order_date": datetime.now(timezone.utc),
            "shipping_address": shipping_addresses[i-1],
            "order_status": status,
            "priority": priority,
            "notes": f"Sample order {i} created for testing - {num_items} items with warehouse locations",
            "items": order_items,
            "total_amount": round(total_amount, 2),
            "assigned_worker": None,
            "created_at": datetime.now(timezone.utc),
            "updated_at": datetime.now(timezone.utc)
        }
        
        # Insert order
        result = db.orders.insert_one(order)
        created_orders.append({
            "orderID": next_order_id,
            "items_count": len(order_items),
            "total_amount": order["total_amount"],
            "status": status,
            "priority": priority,
            "locations": [item["primary_location"] for item in order_items]
        })
        
        print(f"  ✅ Created Order {next_order_id} - Total: ${order['total_amount']:.2f}")
        next_order_id += 1
    
    # Summary
    print(f"\n🎉 Successfully created {len(created_orders)} orders!")
    print("\n📊 Summary:")
    for order in created_orders:
        priority_text = {1: "High", 2: "Medium", 3: "Low"}[order["priority"]]
        locations_str = ", ".join(order["locations"])
        print(f"  Order {order['orderID']}: {order['items_count']} items, ${order['total_amount']:.2f}, {priority_text} priority, {order['status']}")
        print(f"    Warehouse locations: {locations_str}")
    
    print(f"\n💡 Orders are ready for picking workflow with proper warehouse locations!")
    
    client.close()
    return created_orders

if __name__ == "__main__":
    try:
        orders = create_orders_from_location_inventory()
        if orders:
            print(f"\n✨ All {len(orders)} orders created successfully using location_inventory!")
        else:
            print("\n❌ Failed to create orders")
    except Exception as e:
        print(f"\n💥 Error creating orders: {str(e)}")
        import traceback
        traceback.print_exc()
