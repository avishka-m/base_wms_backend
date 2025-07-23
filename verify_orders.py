#!/usr/bin/env python3
"""
Script to verify the created orders and show their details
"""

from pymongo import MongoClient
from datetime import datetime

def verify_orders():
    """Verify the created orders and show their details"""
    
    # Connect to MongoDB
    client = MongoClient("mongodb://localhost:27017/")
    db = client["warehouse_management"]
    orders_collection = db["orders"]
    
    print("🔍 Verifying created orders...")
    
    # Get all orders
    orders = list(orders_collection.find({}).sort("orderID", 1))
    
    if not orders:
        print("❌ No orders found in database")
        client.close()
        return
    
    print(f"✅ Found {len(orders)} orders in database\n")
    
    # Show detailed information for each order
    for order in orders:
        order_id = order.get("orderID", "N/A")
        customer_id = order.get("customerID", "N/A")
        status = order.get("order_status", "N/A")
        priority = order.get("priority", "N/A")
        total = order.get("total_amount", 0)
        shipping_address = order.get("shipping_address", "N/A")
        items = order.get("items", [])
        
        priority_text = {1: "High", 2: "Medium", 3: "Low"}.get(priority, "Unknown")
        
        print(f"📦 Order {order_id}")
        print(f"   Customer: {customer_id}")
        print(f"   Status: {status}")
        print(f"   Priority: {priority_text}")
        print(f"   Total: ${total:.2f}")
        print(f"   Shipping: {shipping_address}")
        print(f"   Items ({len(items)}):")
        
        for item in items:
            item_id = item.get("itemID", "N/A")
            quantity = item.get("quantity", 0)
            price = item.get("price", 0)
            detail_id = item.get("orderDetailID", "N/A")
            fulfilled = item.get("fulfilled_quantity", 0)
            
            print(f"     - Item {item_id}: {quantity} units @ ${price:.2f} each (Detail ID: {detail_id}, Fulfilled: {fulfilled})")
        
        print()
    
    # Summary statistics
    total_orders = len(orders)
    pending_orders = len([o for o in orders if o.get("order_status") == "pending"])
    processing_orders = len([o for o in orders if o.get("order_status") == "processing"])
    total_value = sum(o.get("total_amount", 0) for o in orders)
    total_items = sum(len(o.get("items", [])) for o in orders)
    
    print("📊 Summary Statistics:")
    print(f"   Total Orders: {total_orders}")
    print(f"   Pending Orders: {pending_orders}")
    print(f"   Processing Orders: {processing_orders}")
    print(f"   Total Order Value: ${total_value:.2f}")
    print(f"   Total Order Items: {total_items}")
    
    # Show orders ready for picking
    pickable_orders = [o for o in orders if o.get("order_status") in ["pending", "processing"]]
    print(f"\n🎯 Orders Ready for Picking: {len(pickable_orders)}")
    for order in pickable_orders:
        items_count = len(order.get("items", []))
        priority_text = {1: "High", 2: "Medium", 3: "Low"}.get(order.get("priority"), "Unknown")
        print(f"   Order {order.get('orderID')}: {items_count} items, {priority_text} priority, ${order.get('total_amount', 0):.2f}")
    
    client.close()
    return total_orders

if __name__ == "__main__":
    try:
        count = verify_orders()
        print(f"\n✨ Verification complete! {count} orders are ready for the picking workflow.")
    except Exception as e:
        print(f"\n💥 Error verifying orders: {str(e)}")
        import traceback
        traceback.print_exc()
