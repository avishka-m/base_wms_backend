# cleanup_orders.py - Quick cleanup of orders without location data

import pymongo

def cleanup_orders():
    """Remove orders that don't have location information - fast version"""
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
        
        orders_collection = db["orders"]
        
        print("🔍 Finding orders without locations...")
        
        # Simple approach: check each order individually
        all_orders = list(orders_collection.find({}, {"orderID": 1, "customer_name": 1, "items.locationID": 1}))
        
        orders_to_delete = []
        orders_with_locations = []
        
        for order in all_orders:
            order_id = order.get("orderID")
            customer_name = order.get("customer_name", "Unknown")
            items = order.get("items", [])
            
            # Check if any item has a proper location
            has_location = False
            for item in items:
                location_id = item.get("locationID")
                if location_id and location_id not in [None, "NO LOCATION", ""]:
                    has_location = True
                    break
            
            if has_location:
                orders_with_locations.append(order_id)
                print(f"✅ Order #{order_id}: {customer_name} - HAS LOCATIONS")
            else:
                orders_to_delete.append(order_id)
                print(f"❌ Order #{order_id}: {customer_name} - NO LOCATIONS")
        
        print(f"\n📊 SUMMARY:")
        print(f"✅ Orders with locations: {len(orders_with_locations)}")
        print(f"❌ Orders without locations: {len(orders_to_delete)}")
        
        if orders_to_delete:
            print(f"\n�️ Orders to delete: {orders_to_delete}")
            
            # Auto-delete without prompting for speed
            print("⚡ Auto-deleting orders without locations...")
            
            result = orders_collection.delete_many({"orderID": {"$in": orders_to_delete}})
            
            print(f"🗑️ Deleted {result.deleted_count} orders")
            print("✅ Cleanup complete!")
        else:
            print("✅ No orders to delete - all orders have locations!")
        
        # Quick final count
        final_count = orders_collection.count_documents({})
        print(f"\n📊 Final order count: {final_count}")
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")

if __name__ == "__main__":
    print("🧹 Quick Order Cleanup Script")
    print("Removing orders without location data...\n")
    cleanup_orders()
