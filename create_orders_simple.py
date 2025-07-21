#!/usr/bin/env python3
"""
Simple script to create 5 orders with different numbers of items (2, 3, 4 items each)
using items from the existing inventory collection.
"""

from pymongo import MongoClient
from datetime import datetime
import random

def create_sample_orders():
    # Connect to MongoDB
    client = MongoClient("mongodb://localhost:27017/")
    db = client["warehouse_management"]
    
    print("🔍 Starting order creation...")
    
    # Get available items from inventory
    inventory_items = list(db.inventory.find({"quantity": {"$gt": 0}}))
    
    if len(inventory_items) < 4:
        print(f"❌ Error: Need at least 4 different items in inventory, found only {len(inventory_items)}")
        client.close()
        return
    
    print(f"✅ Found {len(inventory_items)} items in inventory")
    
    # Get existing customers
    customers = list(db.customers.find({}))
    if len(customers) == 0:
        print("❌ Error: No customers found in database")
        client.close()
        return
    
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
        selected_items = random.sample(inventory_items, min(num_items, len(inventory_items)))
        
        # Create order items
        order_items = []
        total_amount = 0
        
        for item in selected_items:
            # Determine reasonable order quantity (not more than available)
            max_qty = min(item.get("quantity", 1), 5)  # Cap at 5 units per item
            order_qty = random.randint(1, max_qty)
            
            # Generate realistic price
            price = round(random.uniform(10.00, 99.99), 2)
            
            order_item = {
                "itemID": item.get("itemID"),
                "quantity": order_qty,
                "price": price,
                "orderDetailID": next_detail_id,
                "fulfilled_quantity": 0
            }
            
            order_items.append(order_item)
            total_amount += price * order_qty
            next_detail_id += 1
            
            item_name = item.get("itemName", f"Item {item.get('itemID')}")
            print(f"  - {item_name} (ID: {item.get('itemID')}) x{order_qty} @ ${price:.2f}")
        
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
        result = db.orders.insert_one(order)
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
