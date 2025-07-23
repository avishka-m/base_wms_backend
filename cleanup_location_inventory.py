#!/usr/bin/env python3
"""
Cleanup script for location_inventory collection.
This script helps reset or fix location inventory data in MongoDB.
"""

import sys
import os
import argparse
from pymongo import MongoClient
from datetime import datetime

# Database configuration
DEFAULT_MONGODB_URL = "mongodb://localhost:27017/"
DEFAULT_DATABASE_NAME = "warehouse_management"

# Parse command line arguments
parser = argparse.ArgumentParser(description='Cleanup location_inventory collection')
parser.add_argument('--mongodb-url', default=DEFAULT_MONGODB_URL, 
                    help=f'MongoDB connection URL (default: {DEFAULT_MONGODB_URL})')
parser.add_argument('--database-name', default=DEFAULT_DATABASE_NAME, 
                    help=f'Database name (default: {DEFAULT_DATABASE_NAME})')
parser.add_argument('--reset-all', action='store_true',
                    help='Reset all locations to available')
parser.add_argument('--fix-inconsistent', action='store_true',
                    help='Fix inconsistent location states')
args = parser.parse_args()

def get_collection(collection_name):
    """Get MongoDB collection with direct connection"""
    client = MongoClient(args.mongodb_url)
    # Force a connection to check if it works
    try:
        client.admin.command('ping')
        print(f"✅ Connected to MongoDB at {args.mongodb_url}")
    except Exception as e:
        print(f"❌ Could not connect to MongoDB: {str(e)}")
    
    db = client[args.database_name]
    return db[collection_name]

def reset_all_locations():
    """Reset all locations to available state"""
    try:
        location_collection = get_collection("location_inventory")
        
        print("🔄 Resetting all locations to available state...")
        
        # Count total locations
        total_locations = location_collection.count_documents({})
        print(f"📊 Found {total_locations} locations in location_inventory collection")
        
        # Reset locations to available
        result = location_collection.update_many(
            {},  # Match all documents
            {
                "$set": {
                    "available": True,
                    "lastUpdated": datetime.utcnow().isoformat(),
                    "clearedAt": datetime.utcnow().isoformat(),
                    "clearedBy": "system"
                },
                "$unset": {
                    "itemID": "",
                    "itemName": "",
                    "quantity": "",
                    "storedAt": "",
                    "storedBy": "",
                    "receivingID": ""
                }
            }
        )
        
        print(f"✅ Reset {result.modified_count} locations to available state")
        return result.modified_count
        
    except Exception as e:
        print(f"❌ Error resetting locations: {str(e)}")
        return 0

def fix_inconsistent_locations():
    """Fix inconsistent locations (e.g. available=False but no itemID)"""
    try:
        location_collection = get_collection("location_inventory")
        inventory_collection = get_collection("inventory")
        
        print("🔍 Checking for inconsistent location states...")
        
        # Find locations marked as unavailable but with no item
        inconsistent_locations = list(location_collection.find({
            "available": False,
            "$or": [
                {"itemID": {"$exists": False}},
                {"itemID": None}
            ]
        }))
        
        print(f"📊 Found {len(inconsistent_locations)} inconsistent locations")
        
        # Fix each inconsistent location
        fixed_count = 0
        for location in inconsistent_locations:
            location_id = location.get("locationID")
            print(f"🔧 Fixing inconsistent location: {location_id}")
            
            # Mark as available
            location_collection.update_one(
                {"locationID": location_id},
                {
                    "$set": {
                        "available": True,
                        "lastUpdated": datetime.utcnow().isoformat(),
                        "clearedAt": datetime.utcnow().isoformat(),
                        "clearedBy": "system",
                        "cleanup_note": "Fixed inconsistent state"
                    },
                    "$unset": {
                        "itemID": "",
                        "itemName": "",
                        "quantity": "",
                        "storedAt": "",
                        "storedBy": "",
                        "receivingID": ""
                    }
                }
            )
            fixed_count += 1
            
        # Find locations with items that don't exist in inventory
        orphaned_locations = []
        occupied_locations = list(location_collection.find({
            "available": False,
            "itemID": {"$exists": True}
        }))
        
        for location in occupied_locations:
            item_id = location.get("itemID")
            if item_id is not None:
                # Check if item exists in inventory
                inventory_item = inventory_collection.find_one({"itemID": item_id})
                if not inventory_item:
                    orphaned_locations.append(location)
        
        print(f"📊 Found {len(orphaned_locations)} locations with non-existent items")
        
        # Fix orphaned locations
        for location in orphaned_locations:
            location_id = location.get("locationID")
            print(f"🔧 Fixing orphaned location: {location_id} (itemID={location.get('itemID')})")
            
            # Mark as available
            location_collection.update_one(
                {"locationID": location_id},
                {
                    "$set": {
                        "available": True,
                        "lastUpdated": datetime.utcnow().isoformat(),
                        "clearedAt": datetime.utcnow().isoformat(),
                        "clearedBy": "system",
                        "cleanup_note": "Fixed orphaned item reference"
                    },
                    "$unset": {
                        "itemID": "",
                        "itemName": "",
                        "quantity": "",
                        "storedAt": "",
                        "storedBy": "",
                        "receivingID": ""
                    }
                }
            )
            fixed_count += 1
            
        print(f"✅ Fixed {fixed_count} inconsistent locations")
        return fixed_count
        
    except Exception as e:
        print(f"❌ Error fixing inconsistent locations: {str(e)}")
        return 0

def verify_location_integrity():
    """Verify integrity of location inventory data"""
    try:
        location_collection = get_collection("location_inventory")
        
        print("🔍 Verifying location inventory integrity...")
        
        # Count locations by state
        total_locations = location_collection.count_documents({})
        available_locations = location_collection.count_documents({"available": True})
        occupied_locations = location_collection.count_documents({"available": False})
        
        print(f"📊 Location inventory summary:")
        print(f"   - Total locations: {total_locations}")
        print(f"   - Available locations: {available_locations}")
        print(f"   - Occupied locations: {occupied_locations}")
        
        # Check for inconsistencies
        inconsistent_count = location_collection.count_documents({
            "available": False,
            "$or": [
                {"itemID": {"$exists": False}},
                {"itemID": None}
            ]
        })
        
        print(f"   - Inconsistent locations: {inconsistent_count}")
        
        return {
            "total": total_locations,
            "available": available_locations,
            "occupied": occupied_locations,
            "inconsistent": inconsistent_count
        }
        
    except Exception as e:
        print(f"❌ Error verifying location inventory: {str(e)}")
        return None

def show_sample_locations():
    """Show sample locations from location_inventory"""
    try:
        location_collection = get_collection("location_inventory")
        
        print("\n📋 Sample locations:")
        sample_locations = list(location_collection.find().limit(3))
        
        if not sample_locations:
            print("   No locations found in the location_inventory collection!")
            return
            
        for location in sample_locations:
            print(f"   Location {location.get('locationID')}")
            print(f"      available: {location.get('available', 'N/A')}")
            
            if location.get('available') == False:
                print(f"      itemID: {location.get('itemID', 'N/A')}")
                print(f"      itemName: {location.get('itemName', 'N/A')}")
                print(f"      quantity: {location.get('quantity', 'N/A')}")
                
            print(f"      lastUpdated: {location.get('lastUpdated', 'N/A')}")
            print()
            
    except Exception as e:
        print(f"❌ Error showing sample locations: {str(e)}")

if __name__ == "__main__":
    print("🚀 Location Inventory Cleanup Tool")
    print("=" * 50)
    
    # Show connection details
    print(f"📡 MongoDB URL: {args.mongodb_url}")
    print(f"🗄️ Database: {args.database_name}")
    print(f"🔄 Reset all locations: {'Enabled' if args.reset_all else 'Disabled'}")
    print(f"🔧 Fix inconsistent locations: {'Enabled' if args.fix_inconsistent else 'Disabled'}")
    print("=" * 50)
    
    # Show current state
    print("📊 Current location inventory state:")
    show_sample_locations()
    initial_state = verify_location_integrity()
    
    # Perform requested operations
    if args.reset_all:
        print("\n🔄 Resetting all locations...")
        reset_count = reset_all_locations()
        print(f"✅ Reset {reset_count} locations to available state")
    
    if args.fix_inconsistent:
        print("\n🔧 Fixing inconsistent locations...")
        fixed_count = fix_inconsistent_locations()
        print(f"✅ Fixed {fixed_count} inconsistent locations")
    
    if not args.reset_all and not args.fix_inconsistent:
        print("\n⚠️ No cleanup operations selected. Use --reset-all or --fix-inconsistent")
        print("   For example: python cleanup_location_inventory.py --fix-inconsistent")
    
    # Show final state
    print("\n📊 Updated location inventory state:")
    show_sample_locations()
    final_state = verify_location_integrity()
    
    print("\n🎉 Location inventory cleanup completed!")
    if initial_state and final_state:
        print(f"💡 Available locations: {initial_state['available']} → {final_state['available']}")
        print(f"💡 Occupied locations: {initial_state['occupied']} → {final_state['occupied']}")
        print(f"💡 Inconsistent locations: {initial_state['inconsistent']} → {final_state['inconsistent']}")