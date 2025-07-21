#!/usr/bin/env python3
"""
Script to create sample inventory items and then create 5 orders with different numbers of items.
"""

from pymongo import MongoClient
from datetime import datetime
import random

def setup_inventory():
    """Add sample inventory items if they don't exist"""
    client = MongoClient("mongodb://localhost:27017/")
    db = client["warehouse_management"]
    
    print("🔧 Setting up inventory...")
    
    # Sample inventory items
    sample_items = [
        {"itemID": 1, "itemName": "Wireless Headphones", "quantity": 25, "price": 79.99},
        {"itemID": 2, "itemName": "Bluetooth Speaker", "quantity": 15, "price": 49.99},
        {"itemID": 3, "itemName": "USB-C Cable", "quantity": 50, "price": 12.99},
        {"itemID": 4, "itemName": "Phone Case", "quantity": 30, "price": 24.99},
        {"itemID": 5, "itemName": "Wireless Charger", "quantity": 20, "price": 34.99},
        {"itemID": 6, "itemName": "Fresh Vegetables", "quantity": 75, "price": 5.99},
        {"itemID": 7, "itemName": "Laptop Stand", "quantity": 12, "price": 89.99},
        {"itemID": 8, "itemName": "Gaming Mouse", "quantity": 18, "price": 59.99}
    ]
    
    # Update existing items with proper data
    for item in sample_items:
        db.inventory.update_one(
            {"itemID": item["itemID"]},
            {"$set": item},
            upsert=True
        )
    
    print(f"✅ Updated {len(sample_items)} inventory items")
    client.close()

def create_sample_orders():
    """Create 5 orders with different numbers of items"""
    client = MongoClient("mongodb://localhost:27017/")
    db = client["warehouse_management"]
    
    print("📦 Creating sample orders...")
    
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
        selected_items = random.sample(inventory_items, min(num_items, len(inventory_items)))
        
        # Create order items
        order_items = []
        total_amount = 0
        
        for item in selected_items:
            # Determine reasonable order quantity (not more than available)
            max_qty = min(item.get("quantity", 1), 5)  # Cap at 5 units per item
            order_qty = random.randint(1, max_qty)
            
            # Use item price or generate one
            price = item.get("price", round(random.uniform(10.00, 99.99), 2))
            
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
        # First setup inventory
        setup_inventory()
        
        # Then create orders
        orders = create_sample_orders()
        if orders:
            print(f"\n✨ All {len(orders)} orders created successfully!")
        else:
            print("\n❌ Failed to create orders")
    except Exception as e:
        print(f"\n💥 Error: {str(e)}")
        import traceback
        traceback.print_exc()
