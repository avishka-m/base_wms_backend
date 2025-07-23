#!/usr/bin/env python3
"""
Examine location_inventory collection structure
"""

import os
import sys
from pymongo import MongoClient

def examine_location_inventory():
    # Connect to localhost MongoDB
    client = MongoClient('mongodb://localhost:27017')
    db = client['warehouse_management']
    
    print("=== Location Inventory Collection Analysis ===")
    
    # Get collection stats
    total_count = db.location_inventory.count_documents({})
    print(f"Total documents in location_inventory: {total_count}")
    
    # Sample documents
    print("\n=== Sample Documents ===")
    for i, doc in enumerate(db.location_inventory.find().limit(10)):
        print(f"\nDocument {i+1}:")
        print(doc)
    
    # Find unique field names
    print("\n=== Field Analysis ===")
    pipeline = [
        {"$project": {
            "fields": {"$objectToArray": "$$ROOT"}
        }},
        {"$unwind": "$fields"},
        {"$group": {
            "_id": "$fields.k"
        }}
    ]
    
    fields = list(db.location_inventory.aggregate(pipeline))
    print("Fields in collection:")
    for field in fields:
        print(f"  - {field['_id']}")
    
    # Check for location-related fields
    print("\n=== Location Fields Analysis ===")
    location_fields = ['location', 'location_id', 'locationId', 'location_code', 'shelf', 'rack']
    
    for field in location_fields:
        sample = list(db.location_inventory.find({field: {"$exists": True}}).limit(3))
        if sample:
            print(f"\nField '{field}' exists - Sample values:")
            for doc in sample:
                print(f"  {field}: {doc.get(field)}")
    
    client.close()

if __name__ == "__main__":
    examine_location_inventory()
