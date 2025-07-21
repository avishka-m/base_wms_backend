#!/usr/bin/env python3
"""
Debug the mark-as-stored 500 error by checking location_inventory data
"""

from pymongo import MongoClient

def debug_location_inventory():
    """Check location_inventory collection state"""
    
    client = MongoClient("mongodb://localhost:27017/")
    db = client["warehouse_management"]
    location_collection = db["location_inventory"]
    
    print("🔍 Debugging location_inventory collection...")
    
    # Check total records
    total_count = location_collection.count_documents({})
    print(f"📊 Total location records: {total_count}")
    
    # Check available locations
    available_count = location_collection.count_documents({"available": True})
    print(f"✅ Available locations: {available_count}")
    
    # Check occupied locations
    occupied_count = location_collection.count_documents({"available": False})
    print(f"🚫 Occupied locations: {occupied_count}")
    
    # Show some sample available locations
    print(f"\n📍 Sample available locations:")
    available_locations = list(location_collection.find({"available": True}).limit(5))
    for loc in available_locations:
        print(f"  - {loc.get('locationID', 'No ID')}: {loc.get('itemName', 'Empty')} (Qty: {loc.get('quantity', 0)})")
    
    # Show some sample occupied locations
    print(f"\n🔒 Sample occupied locations:")
    occupied_locations = list(location_collection.find({"available": False}).limit(5))
    for loc in occupied_locations:
        print(f"  - {loc.get('locationID', 'No ID')}: {loc.get('itemName', 'Empty')} (Qty: {loc.get('quantity', 0)})")
    
    # Check if there are any locations without the 'available' field
    no_available_field = location_collection.count_documents({"available": {"$exists": False}})
    print(f"\n⚠️  Locations missing 'available' field: {no_available_field}")
    
    if no_available_field > 0:
        print("🔧 Some locations might not have the 'available' field set!")
        sample_missing = list(location_collection.find({"available": {"$exists": False}}).limit(3))
        for loc in sample_missing:
            print(f"  - {loc.get('locationID', 'No ID')}: {dict(loc)}")
    
    print(f"\n🎯 LIKELY ISSUE:")
    print(f"   • Frontend sends locationID (e.g., 'B02.1')")
    print(f"   • Backend looks for location with available: true")
    print(f"   • If location doesn't exist or available != true, it fails")
    print(f"   • 500 error occurs during validation")
    
    client.close()

if __name__ == "__main__":
    debug_location_inventory()
