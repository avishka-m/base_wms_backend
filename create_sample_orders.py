#!/usr/bin/env python3
"""
Script to create 5 sample orders with different numbers of items (2, 3, 4 items each)
using items from the existing location_inventory with stored items.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from pymongo import MongoClient
from datetime import datetime
import random
from typing import List, Dict, Any

def get_available_items(db) -> List[Dict[str, Any]]:
    """Get items that are currently stored in location inventory"""
    location_collection = db["location_inventory"]
    
    # Find all locations that have items stored (quantity > 0)
    stored_items = list(location_collection.find({
        "quantity": {"$gt": 0},
        "itemID": {"$ne": None},
        "itemName": {"$ne": None}
    }))
    
    # Group by itemID to get unique items with their total available quantities
    items_dict = {}
    for location in stored_items:
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
            "quantity": location["quantity"]
        })
    
    return list(items_dict.values())

def get_next_order_id(db) -> int:
    """Get the next available order ID"""
    orders_collection = db["orders"]
    
    # Find the highest existing orderID
    last_order = orders_collection.find_one(
        {},
        sort=[("orderID", -1)]
    )
    
    if last_order and "orderID" in last_order:
        return last_order["orderID"] + 1
    else:
        return 1

def get_next_order_detail_id(db) -> int:
    """Get the next available order detail ID"""
    orders_collection = db["orders"]
    
    # Find all order details and get the highest orderDetailID
    max_detail_id = 0
    for order in orders_collection.find({}):
        for item in order.get("items", []):
            if "orderDetailID" in item:
                max_detail_id = max(max_detail_id, item["orderDetailID"])
    
    return max_detail_id + 1

def create_sample_orders():
    """Create 5 sample orders with varying numbers of items"""
    
    # Connect to MongoDB
    client = MongoClient("mongodb://localhost:27017/")
    db = client["warehouse_management"]
    orders_collection = db["orders"]
    customers_collection = db["customers"]
    
    print("🔍 Starting order creation...")
    
    # Get available items from location inventory
    available_items = get_available_items(db)
    
    if len(available_items) < 4:
        print(f"❌ Error: Need at least 4 different items in inventory, found only {len(available_items)}")
        print("Available items:")
        for item in available_items:
            print(f"  - {item['itemName']} (ID: {item['itemID']}) - Available: {item['total_available']}")
        client.close()
        return
    
    print(f"✅ Found {len(available_items)} different items in inventory")
    
    # Get existing customers
    customers = list(customers_collection.find({}))
    if len(customers) == 0:
        print("❌ Error: No customers found in database")
        client.close()
        return
    
    print(f"✅ Found {len(customers)} customers")
    
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
    next_order_id = get_next_order_id(db)
    next_detail_id = get_next_order_detail_id(db)
    
    for i, (num_items, priority, status) in enumerate(order_configs, 1):
        print(f"\n📦 Creating Order {i} with {num_items} items...")
        
        # Select random customer
        customer = random.choice(customers)
        customer_id = customer.get("customerID", customer.get("_id"))
        
        # Select random items for this order
        selected_items = random.sample(available_items, min(num_items, len(available_items)))
        
        # Create order items
        order_items = []
        total_amount = 0
        
        for item in selected_items:
            # Determine reasonable order quantity (not more than available)
            max_qty = min(item["total_available"], 10)  # Cap at 10 units per item
            order_qty = random.randint(1, max_qty)
            
            # Generate realistic price based on item name
            base_price = random.uniform(5.99, 89.99)
            price = round(base_price, 2)
            
            order_item = {
                "itemID": item["itemID"],
                "quantity": order_qty,
                "price": price,
                "orderDetailID": next_detail_id,
                "fulfilled_quantity": 0
            }
            
            order_items.append(order_item)
            total_amount += price * order_qty
            next_detail_id += 1
            
            print(f"  - {item['itemName']} (ID: {item['itemID']}) x{order_qty} @ ${price:.2f}")
        
        # Create the order
        order = {
            "orderID": next_order_id,
            "customerID": customer_id,
            "order_date": datetime.utcnow(),
            "shipping_address": shipping_addresses[i-1],
            "order_status": status,
            "priority": priority,
            "notes": f"Sample order {i} created for testing - {num_items} items",
            "items": order_items,
            "total_amount": round(total_amount, 2),
            "assigned_worker": None,
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow()
        }
        
        # Insert order
        result = orders_collection.insert_one(order)
        created_orders.append({
            "orderID": next_order_id,
            "items_count": len(order_items),
            "total_amount": order["total_amount"],
            "status": status,
            "priority": priority
        })
        
        print(f"  ✅ Created Order {next_order_id} - Total: ${order['total_amount']:.2f}")
        next_order_id += 1
    
    # Summary
    print(f"\n🎉 Successfully created {len(created_orders)} orders!")
    print("\n📊 Summary:")
    for order in created_orders:
        priority_text = {1: "High", 2: "Medium", 3: "Low"}[order["priority"]]
        print(f"  Order {order['orderID']}: {order['items_count']} items, ${order['total_amount']:.2f}, {priority_text} priority, {order['status']}")
    
    print(f"\n💡 Orders are ready for picking workflow!")
    
    client.close()
    return created_orders

if __name__ == "__main__":
    try:
        orders = create_sample_orders()
        if orders:
            print(f"\n✨ All {len(orders)} orders created successfully!")
        else:
            print("\n❌ Failed to create orders")
    except Exception as e:
        print(f"\n💥 Error creating orders: {str(e)}")
        import traceback
        traceback.print_exc()
