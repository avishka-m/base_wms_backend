#!/usr/bin/env python3
"""
Examine inventory collection structure
"""

import os
import sys
from pymongo import MongoClient

def examine_inventory():
    # Connect to localhost MongoDB
    client = MongoClient('mongodb+srv://wms:3cVnhHuj5caki0Ve@cluster0.99chyus.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0')
    db = client['warehouse_management']
    
    print("=== Inventory Collection Analysis ===")
    
    # Get collection stats
    total_count = db.inventory.count_documents({})
    print(f"Total documents in inventory: {total_count}")
    
    # Sample documents
    print("\n=== Sample Documents ===")
    for i, doc in enumerate(db.inventory.find().limit(10)):
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
    
    fields = list(db.inventory.aggregate(pipeline))
    print("Fields in collection:")
    for field in fields:
        print(f"  - {field['_id']}")
    
    # Check for location-related fields
    print("\n=== Location Fields Analysis ===")
    location_fields = ['location', 'location_id', 'locationId', 'locationID', 'location_code', 'shelf', 'rack', 'position', 'slot']
    
    for field in location_fields:
        sample = list(db.inventory.find({field: {"$exists": True}}).limit(3))
        if sample:
            print(f"\nField '{field}' exists - Sample values:")
            for doc in sample:
                print(f"  {field}: {doc.get(field)}")
    
    # Check for numeric location IDs that need conversion
    print("\n=== Numeric Location ID Analysis ===")
    numeric_location_samples = []
    
    # Check various possible numeric location fields
    possible_numeric_fields = ['location_id', 'locationId', 'locationID', 'location', 'position']
    
    for field in possible_numeric_fields:
        # Find documents where the field exists and is numeric
        pipeline = [
            {"$match": {field: {"$exists": True, "$type": "number"}}},
            {"$limit": 5}
        ]
        numeric_samples = list(db.inventory.aggregate(pipeline))
        
        if numeric_samples:
            print(f"\nNumeric values found in field '{field}':")
            for doc in numeric_samples:
                print(f"  {field}: {doc.get(field)} (Document ID: {doc['_id']})")
    
    # Also check for string numeric values
    print("\n=== String Numeric Location ID Analysis ===")
    for field in possible_numeric_fields:
        pipeline = [
            {"$match": {field: {"$exists": True, "$type": "string"}}},
            {"$match": {field: {"$regex": "^[0-9]+$"}}},  # Only numeric strings
            {"$limit": 5}
        ]
        string_numeric_samples = list(db.inventory.aggregate(pipeline))
        
        if string_numeric_samples:
            print(f"\nString numeric values found in field '{field}':")
            for doc in string_numeric_samples:
                print(f"  {field}: '{doc.get(field)}' (Document ID: {doc['_id']})")
    
    client.close()

if __name__ == "__main__":
    examine_inventory()
