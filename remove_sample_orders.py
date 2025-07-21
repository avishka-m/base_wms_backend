#!/usr/bin/env python3
"""
Script to remove orders created by create_complete_orders.py
"""

from pymongo import MongoClient

def remove_created_orders():
    """Remove orders created by create_complete_orders.py"""
    client = MongoClient("mongodb://localhost:27017/")
    db = client["warehouse_management"]
    
    print("🔍 Checking for orders to remove...")
    
    # Get all orders to see which ones were created
    all_orders = list(db.orders.find({}).sort("orderID", 1))
    
    print(f"Current orders in database: {len(all_orders)}")
    
    if len(all_orders) == 0:
        print("✅ No orders found in database")
        client.close()
        return
    
    print("\nCurrent orders:")
    for order in all_orders:
        order_id = order.get('orderID')
        items_count = len(order.get('items', []))
        total = order.get('total_amount', 0)
        status = order.get('order_status', 'unknown')
        notes = order.get('notes', '')
        print(f"  Order {order_id}: {items_count} items, ${total:.2f}, {status}")
        if 'Sample order' in notes:
            print(f"    Notes: {notes}")
    
    # Find orders created by create_complete_orders.py (they have specific notes)
    orders_to_remove = list(db.orders.find({
        "notes": {"$regex": "Sample order.*created for testing"}
    }))
    
    if len(orders_to_remove) == 0:
        print("\n✅ No orders from create_complete_orders.py found to remove")
        client.close()
        return
    
    print(f"\n🗑️  Found {len(orders_to_remove)} orders to remove:")
    
    order_ids_to_remove = []
    for order in orders_to_remove:
        order_id = order.get('orderID')
        items_count = len(order.get('items', []))
        total = order.get('total_amount', 0)
        order_ids_to_remove.append(order_id)
        print(f"  - Order {order_id}: {items_count} items, ${total:.2f}")
    
    # Confirm removal
    print(f"\n⚠️  About to remove {len(orders_to_remove)} orders with IDs: {order_ids_to_remove}")
    
    # Remove the orders
    result = db.orders.delete_many({
        "notes": {"$regex": "Sample order.*created for testing"}
    })
    
    print(f"✅ Removed {result.deleted_count} orders")
    
    # Show remaining orders
    remaining_orders = list(db.orders.find({}).sort("orderID", 1))
    print(f"\nRemaining orders in database: {len(remaining_orders)}")
    
    if len(remaining_orders) > 0:
        print("\nRemaining orders:")
        for order in remaining_orders:
            order_id = order.get('orderID')
            items_count = len(order.get('items', []))
            total = order.get('total_amount', 0)
            status = order.get('order_status', 'unknown')
            print(f"  Order {order_id}: {items_count} items, ${total:.2f}, {status}")
    
    client.close()
    return result.deleted_count

if __name__ == "__main__":
    try:
        removed_count = remove_created_orders()
        if removed_count and removed_count > 0:
            print(f"\n✨ Successfully removed {removed_count} orders created by create_complete_orders.py!")
        else:
            print("\n✨ No orders needed to be removed.")
    except Exception as e:
        print(f"\n💥 Error removing orders: {str(e)}")
        import traceback
        traceback.print_exc()
