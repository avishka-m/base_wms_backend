#!/usr/bin/env python3

from pymongo import MongoClient
from datetime import datetime

def debug_location_inventory():
    """Debug the location_inventory collection to understand the storage issue"""
    
    client = MongoClient('mongodb://localhost:27017/')
    db = client['warehouse_management']
    location_collection = db['location_inventory']
    
    print('🔍 Debugging location_inventory collection...')
    
    # Check total counts
    total_count = location_collection.count_documents({})
    available_count = location_collection.count_documents({"available": True})
    occupied_count = location_collection.count_documents({"available": False})
    
    print(f'📊 Total location records: {total_count}')
    print(f'✅ Available locations: {available_count}')
    print(f'🚫 Occupied locations: {occupied_count}')
    
    # Check B-type locations specifically
    print('\n🔍 Checking B-type slots...')
    b_locations = list(location_collection.find({'locationID': {'$regex': '^B'}}).limit(10))
    for loc in b_locations:
        location_id = loc.get('locationID', 'Unknown')
        available = loc.get('available', 'Missing')
        item_id = loc.get('itemID', None)
        quantity = loc.get('quantity', 0)
        print(f'  {location_id}: available={available}, itemID={item_id}, quantity={quantity}')
    
    # Check for duplicate location IDs
    print('\n🔍 Checking for duplicate location IDs...')
    pipeline = [
        {'$group': {'_id': '$locationID', 'count': {'$sum': 1}}},
        {'$match': {'count': {'$gt': 1}}},
        {'$sort': {'count': -1}}
    ]
    duplicates = list(location_collection.aggregate(pipeline))
    print(f'Found {len(duplicates)} duplicate location IDs')
    for dup in duplicates[:5]:
        location_id = dup['_id']
        count = dup['count']
        print(f'  {location_id}: {count} records')
        
        # Show the duplicate records
        dupe_records = list(location_collection.find({'locationID': location_id}))
        for i, record in enumerate(dupe_records):
            available = record.get('available', 'Missing')
            item_id = record.get('itemID', None)
            quantity = record.get('quantity', 0)
            print(f'    Record {i+1}: available={available}, itemID={item_id}, qty={quantity}')
    
    # Check if there are locations without 'available' field
    print('\n🔍 Checking locations missing "available" field...')
    missing_available = list(location_collection.find({'available': {'$exists': False}}))
    print(f'Found {len(missing_available)} locations without "available" field')
    for loc in missing_available[:3]:
        location_id = loc.get('locationID', 'Unknown')
        item_id = loc.get('itemID', None)
        print(f'  {location_id}: itemID={item_id}')
    
    # Check what happens when we try to find a specific location
    print('\n🔍 Testing location lookup for B02.1...')
    test_location = location_collection.find_one({"locationID": "B02.1"})
    if test_location:
        print(f'  Found B02.1: available={test_location.get("available")}, itemID={test_location.get("itemID")}')
    else:
        print('  B02.1 not found!')
    
    # Check recent storage activity
    print('\n🔍 Checking recent storage activity...')
    recent_stored = list(location_collection.find({
        'available': False,
        'storedAt': {'$exists': True}
    }).sort([('storedAt', -1)]).limit(5))
    
    print(f'Found {len(recent_stored)} recently stored items:')
    for item in recent_stored:
        location_id = item.get('locationID', 'Unknown')
        item_name = item.get('itemName', 'Unknown')
        stored_at = item.get('storedAt', 'Unknown')
        quantity = item.get('quantity', 0)
        print(f'  {location_id}: {item_name} (qty: {quantity}) at {stored_at}')
    
    client.close()

if __name__ == "__main__":
    debug_location_inventory()
