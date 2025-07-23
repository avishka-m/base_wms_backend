#!/usr/bin/env python3
"""
Migration script to update inventory collection structure.
Changes current_stock to total_stock for clarity.
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
parser = argparse.ArgumentParser(description='Migrate inventory collection from current_stock to total_stock')
parser.add_argument('--mongodb-url', default=DEFAULT_MONGODB_URL, 
                    help=f'MongoDB connection URL (default: {DEFAULT_MONGODB_URL})')
parser.add_argument('--database-name', default=DEFAULT_DATABASE_NAME, 
                    help=f'Database name (default: {DEFAULT_DATABASE_NAME})')
parser.add_argument('--force', action='store_true',
                    help='Force update all records (even if they already have total_stock)')
parser.add_argument('--add-test', action='store_true',
                    help='Add a test item with current_stock to verify migration')
args = parser.parse_args()

# MongoDB connection
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

def add_test_item():
    """Add a test item with current_stock structure"""
    inventory_collection = get_collection("inventory")
    
    # Find highest item ID
    last_items = list(inventory_collection.find().sort("itemID", -1).limit(1))
    new_id = 1
    if last_items:
        new_id = last_items[0].get("itemID", 0) + 1
    
    # Create test item with current_stock (old structure)
    test_item = {
        "itemID": new_id,
        "name": "Test Migration Item",
        "category": "Test",
        "current_stock": 50,
        "created_at": datetime.utcnow().isoformat(),
        "updated_at": datetime.utcnow().isoformat()
    }
    
    result = inventory_collection.insert_one(test_item)
    if result.inserted_id:
        print(f"✅ Added test item with itemID={new_id}, current_stock=50")
        return new_id
    else:
        print("❌ Failed to add test item")
        return None

def migrate_inventory_collection():
    """
    Migrate inventory collection to use total_stock instead of current_stock
    """
    try:
        inventory_collection = get_collection("inventory")
        
        print("🔍 Starting inventory collection migration...")
        
        # Get all inventory items
        all_items = list(inventory_collection.find({}))
        print(f"📊 Found {len(all_items)} items in inventory collection")
        
        updated_count = 0
        
        for item in all_items:
            item_id = item.get("itemID")
            current_stock = item.get("current_stock")
            total_stock = item.get("total_stock")
            
            # Check if migration is needed
            if (current_stock is not None and total_stock is None) or args.force:
                # Determine what value to use for total_stock
                stock_value = current_stock
                if stock_value is None and total_stock is not None:
                    stock_value = total_stock  # Keep existing total_stock if no current_stock
                elif stock_value is None:
                    stock_value = 0  # Default to 0 if neither exists
                
                # Prepare update operation with new fields
                update_op = {
                    "$set": {
                        "total_stock": stock_value,
                        "last_updated": datetime.utcnow().isoformat(),
                        "migrated_at": datetime.utcnow().isoformat()
                    },
                    # Always remove old fields, regardless of force mode
                    "$unset": {
                        "current_stock": "",
                        "stock_level": "",  # Also remove other legacy field names
                        "quantity": "",     # Other possible legacy field names
                        "available_stock": ""
                    }
                }
                
                # Add migration note based on whether it's forced or normal
                if args.force:
                    update_op["$set"]["migration_note"] = "Forced update to total_stock structure (legacy fields removed)"
                else:
                    update_op["$set"]["migration_note"] = "Migrated current_stock to total_stock (legacy fields removed)"
                
                result = inventory_collection.update_one(
                    {"itemID": item_id},
                    update_op
                )
                
                if result.modified_count > 0:
                    updated_count += 1
                    if args.force:
                        print(f"✅ Force-updated item {item_id}: total_stock set to {stock_value}")
                    else:
                        print(f"✅ Migrated item {item_id}: current_stock={current_stock} → total_stock={stock_value}")
                else:
                    print(f"⚠️ No changes needed for item {item_id}")
                    
            elif total_stock is not None:
                print(f"✅ Item {item_id} already has total_stock={total_stock} (no migration needed)")
            else:
                # Initialize total_stock if both are missing
                result = inventory_collection.update_one(
                    {"itemID": item_id},
                    {
                        "$set": {
                            "total_stock": 0,
                            "last_updated": datetime.utcnow().isoformat(),
                            "initialized_at": datetime.utcnow().isoformat(),
                            "initialization_note": "Initialized total_stock to 0"
                        }
                    }
                )
                
                if result.modified_count > 0:
                    updated_count += 1
                    print(f"🔧 Initialized item {item_id}: total_stock=0")
        
        print(f"\n📊 Migration completed!")
        print(f"   - Total items processed: {len(all_items)}")
        print(f"   - Items updated: {updated_count}")
        print(f"   - Items already migrated: {len(all_items) - updated_count}")
        
        # Verify migration
        print(f"\n🔍 Verifying migration...")
        items_with_total_stock = inventory_collection.count_documents({"total_stock": {"$exists": True}})
        items_with_legacy_fields = inventory_collection.count_documents({
            "$or": [
                {"current_stock": {"$exists": True}},
                {"stock_level": {"$exists": True}},
                {"quantity": {"$exists": True}},
                {"available_stock": {"$exists": True}}
            ]
        })
        
        print(f"   - Items with total_stock: {items_with_total_stock}")
        print(f"   - Items with legacy fields: {items_with_legacy_fields}")
        
        if items_with_legacy_fields == 0 and items_with_total_stock == len(all_items):
            print("✅ Migration successful! All items now use total_stock exclusively")
        else:
            print("⚠️ Migration incomplete - some items may need manual review")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during migration: {str(e)}")
        return False

def show_sample_items():
    """Show sample items before and after migration"""
    try:
        inventory_collection = get_collection("inventory")
        
        print("\n📋 Sample inventory items:")
        sample_items = list(inventory_collection.find({}).limit(3))
        
        if not sample_items:
            print("   No items found in the inventory collection!")
            return
            
        for item in sample_items:
            print(f"   Item {item.get('itemID')}: {item.get('name', 'Unknown')}")
            
            # Show all possible stock-related fields for transparency
            print(f"      current_stock: {item.get('current_stock', 'N/A')}")
            print(f"      total_stock: {item.get('total_stock', 'N/A')}")
            print(f"      stock_level: {item.get('stock_level', 'N/A')}")
            print(f"      quantity: {item.get('quantity', 'N/A')}")
            
            # Other fields
            print(f"      category: {item.get('category', 'N/A')}")
            print(f"      last_updated: {item.get('last_updated', 'N/A')}")
            print()
            
    except Exception as e:
        print(f"❌ Error showing sample items: {str(e)}")

if __name__ == "__main__":
    print("🚀 Inventory Collection Migration Tool")
    print("=" * 50)
    
    # Show connection details
    print(f"📡 MongoDB URL: {args.mongodb_url}")
    print(f"🗄️ Database: {args.database_name}")
    print(f"🔧 Force mode: {'Enabled' if args.force else 'Disabled'}")
    print(f"🧪 Test mode: {'Enabled' if args.add_test else 'Disabled'}")
    print("=" * 50)
    
    # Add test item if requested
    test_item_id = None
    if args.add_test:
        print("🧪 Adding a test item with current_stock structure...")
        test_item_id = add_test_item()
    
    # Show current state
    print("📊 Current inventory structure:")
    show_sample_items()
    
    # Run migration
    success = migrate_inventory_collection()
    
    if success:
        print("\n📊 Updated inventory structure:")
        show_sample_items()
        
        # Check test item if one was added
        if test_item_id:
            print("\n🧪 Checking migration of test item...")
            inventory_collection = get_collection("inventory")
            test_item = inventory_collection.find_one({"itemID": test_item_id})
            if test_item:
                # Check if all legacy fields are removed and total_stock is set
                legacy_fields_exist = any(field in test_item for field in ["current_stock", "stock_level", "quantity", "available_stock"])
                
                if test_item.get("total_stock") == 50 and not legacy_fields_exist:
                    print(f"✅ Test successful! Test item correctly migrated to total_stock=50 with all legacy fields removed")
                else:
                    print(f"⚠️ Test results:")
                    print(f"   - total_stock: {test_item.get('total_stock')}")
                    
                    # Show any legacy fields that still exist
                    for field in ["current_stock", "stock_level", "quantity", "available_stock"]:
                        if field in test_item:
                            print(f"   - {field}: {test_item.get(field)} (should be removed)")
            else:
                print("❌ Test item not found after migration")
        
        print("\n🎉 Migration completed successfully!")
        print("💡 The inventory collection now uses 'total_stock' to track warehouse totals")
        print("💡 The location_inventory collection remains unchanged (tracks per-location quantities)")
    else:
        print("\n❌ Migration failed - please check the error messages above")
