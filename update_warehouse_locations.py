#!/usr/bin/env python3
"""
Script to update inventory items with realistic warehouse location IDs
"""

from pymongo import MongoClient
import random

def check_current_locations():
    """Check what locations currently exist"""
    client = MongoClient("mongodb://localhost:27017/")
    db = client["warehouse_management"]
    
    print("=== Current Location Inventory ===")
    locations = list(db.location_inventory.find({}).limit(10))
    print(f"Total locations: {len(list(db.location_inventory.find({})))}")
    
    print("\nFirst 10 locations:")
    for i, loc in enumerate(locations):
        print(f"{i+1}. LocationID: {loc.get('locationID')}, Item: {loc.get('itemName', 'N/A')}, Qty: {loc.get('quantity', 0)}")
    
    print("\n=== Current Inventory ===")
    inventory = list(db.inventory.find({}))
    print(f"Total inventory items: {len(inventory)}")
    
    print("\nInventory items:")
    for item in inventory:
        print(f"ItemID: {item.get('itemID')}, Name: {item.get('itemName')}, Qty: {item.get('quantity', 0)}")
    
    client.close()

def update_inventory_locations():
    """Update inventory items with realistic warehouse location IDs"""
    client = MongoClient("mongodb://localhost:27017/")
    db = client["warehouse_management"]
    
    print("\n🔧 Updating inventory with realistic location IDs...")
    
    # Realistic warehouse location IDs
    # Format: [ZONE][AISLE].[SHELF] 
    # Zones: B=Bulk, D=Dry goods, P=Perishables, E=Electronics, G=General
    realistic_locations = [
        "E1.1", "E1.2", "E1.3", "E2.1", "E2.2",  # Electronics zone
        "G3.1", "G3.2", "G4.1", "G4.2",          # General zone
        "P1.1", "P1.2", "P2.1",                  # Perishables zone
        "B5.1", "B5.2", "B6.1",                  # Bulk zone
        "D7.1", "D7.2", "D8.1"                   # Dry goods zone
    ]
    
    # Get current inventory items
    inventory_items = list(db.inventory.find({}))
    
    # Update each inventory item with a realistic location
    updated_items = []
    
    for i, item in enumerate(inventory_items):
        # Assign location based on item type
        item_name = item.get('itemName', '')
        item_id = item.get('itemID')
        
        # Choose location zone based on item type
        if any(word in item_name.lower() for word in ['headphones', 'speaker', 'usb', 'charger', 'mouse', 'laptop']):
            # Electronics items go to E zone
            location_options = [loc for loc in realistic_locations if loc.startswith('E')]
        elif 'vegetables' in item_name.lower():
            # Perishables go to P zone
            location_options = [loc for loc in realistic_locations if loc.startswith('P')]
        elif 'case' in item_name.lower():
            # General items go to G zone
            location_options = [loc for loc in realistic_locations if loc.startswith('G')]
        else:
            # Everything else can go anywhere
            location_options = realistic_locations
        
        # Pick a random location from the appropriate zone
        new_location = random.choice(location_options)
        
        # Update the inventory item with location
        db.inventory.update_one(
            {"itemID": item_id},
            {"$set": {"locationID": new_location}}
        )
        
        updated_items.append({
            "itemID": item_id,
            "itemName": item_name,
            "locationID": new_location,
            "quantity": item.get('quantity', 0)
        })
        
        print(f"  ✅ {item_name} (ID: {item_id}) → Location: {new_location}")
    
    print(f"\n✅ Updated {len(updated_items)} inventory items with realistic locations")
    
    # Also update location_inventory if it exists
    print("\n🔧 Updating location_inventory collection...")
    
    # Clear existing location_inventory and rebuild with new structure
    db.location_inventory.delete_many({})
    
    # Create location_inventory entries for each item
    for item in updated_items:
        location_entry = {
            "locationID": item["locationID"],
            "itemID": item["itemID"],
            "itemName": item["itemName"],
            "quantity": item["quantity"],
            "capacity": item["quantity"] + random.randint(10, 50),  # Add some extra capacity
            "zone": item["locationID"][0],  # First character is the zone
            "aisle": item["locationID"][1],  # Second character is the aisle
            "shelf": item["locationID"].split(".")[1]  # After the dot is the shelf
        }
        
        db.location_inventory.insert_one(location_entry)
    
    print(f"✅ Created {len(updated_items)} location_inventory entries")
    
    # Show summary
    print("\n📊 Summary of Updated Locations:")
    print("Zone Distribution:")
    zones = {}
    for item in updated_items:
        zone = item["locationID"][0]
        zones[zone] = zones.get(zone, 0) + 1
    
    zone_names = {
        'E': 'Electronics',
        'G': 'General',
        'P': 'Perishables', 
        'B': 'Bulk',
        'D': 'Dry Goods'
    }
    
    for zone, count in zones.items():
        zone_name = zone_names.get(zone, 'Unknown')
        print(f"  {zone} ({zone_name}): {count} items")
    
    client.close()
    return updated_items

if __name__ == "__main__":
    try:
        # First check current state
        check_current_locations()
        
        # Then update with realistic locations
        updated = update_inventory_locations()
        
        print(f"\n✨ Successfully updated {len(updated)} items with realistic warehouse locations!")
        
    except Exception as e:
        print(f"\n💥 Error: {str(e)}")
        import traceback
        traceback.print_exc()
