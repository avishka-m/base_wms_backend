#!/usr/bin/env python3

from pymongo import MongoClient

def check_unprocessed_items():
    """Check for unprocessed items in receiving records"""
    
    client = MongoClient('mongodb://localhost:27017/')
    db = client['warehouse_management']
    receiving_collection = db['receiving']
    
    print('🔍 Checking for unprocessed items in receiving records...')
    
    # Find receiving records with unprocessed items
    unprocessed_records = list(receiving_collection.find({
        "items.processed": {"$ne": True}
    }))
    
    print(f'📊 Found {len(unprocessed_records)} receiving records with unprocessed items')
    
    unprocessed_items = []
    for record in unprocessed_records:
        for item in record.get("items", []):
            if not item.get("processed", False):
                unprocessed_items.append({
                    "receivingID": record.get("receivingID"),
                    "itemID": item.get("itemID"),
                    "quantity": item.get("quantity"),
                    "processed": item.get("processed", False),
                    "locationID": item.get("locationID")
                })
    
    print(f'📦 Total unprocessed items: {len(unprocessed_items)}')
    
    if unprocessed_items:
        print('🎯 Sample unprocessed items:')
        for item in unprocessed_items[:5]:
            print(f'  Receiving #{item["receivingID"]}: Item {item["itemID"]} (Qty: {item["quantity"]})')
    
    client.close()
    return unprocessed_items

if __name__ == "__main__":
    check_unprocessed_items()
