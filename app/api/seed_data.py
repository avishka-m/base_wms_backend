from fastapi import APIRouter, Depends, HTTPException
from datetime import datetime, timedelta
import random
from typing import List, Dict, Any

from ..auth.dependencies import get_current_active_user
from ..utils.database import get_collection

router = APIRouter()

# Dummy data templates
ITEM_NAMES = [
    {"name": "Widget A", "category": "small"},
    {"name": "Gadget B", "category": "small"},
    {"name": "Component C", "category": "small"},
    {"name": "Module D", "category": "medium"},
    {"name": "Assembly E", "category": "medium"},
    {"name": "Unit F", "category": "medium"},
    {"name": "System G", "category": "large"},
    {"name": "Machine H", "category": "large"},
    {"name": "Equipment I", "category": "large"},
    {"name": "Tool J", "category": "small"},
    {"name": "Device K", "category": "medium"},
    {"name": "Apparatus L", "category": "large"},
    {"name": "Instrument M", "category": "small"},
    {"name": "Mechanism N", "category": "medium"},
    {"name": "Fixture O", "category": "large"}
]

SUPPLIER_NAMES = ["TechCorp", "GlobalSupply", "MegaParts", "QuickShip", "ReliableGoods"]
CONDITIONS = ["good", "damaged", "acceptable"]

@router.post("/seed-all")
async def seed_all_data(
    current_user: Dict[str, Any] = Depends(get_current_active_user)
):
    """Seed all dummy data including receiving items, inventory, and occupied locations"""
    try:
        # Get collections
        receiving_items_collection = get_collection("receiving_items")
        inventory_collection = get_collection("inventory")
        location_occupancy_collection = get_collection("location_occupancy")
        storage_history_collection = get_collection("storage_history")
        receiving_collection = get_collection("receiving")
        
        # Clear existing data first (optional - comment out if you want to keep existing data)
        receiving_items_collection.delete_many({})
        inventory_collection.delete_many({})
        location_occupancy_collection.delete_many({})
        storage_history_collection.delete_many({})
        receiving_collection.delete_many({})
        
        # Seed receiving items (available for storing)
        receiving_items = seed_receiving_items()
        
        # Seed inventory items (available for picking) with occupied locations
        inventory_items = seed_inventory_items()
        
        return {
            "message": "Data seeded successfully",
            "receiving_items": len(receiving_items),
            "inventory_items": len(inventory_items)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error seeding data: {str(e)}")


def seed_receiving_items() -> List[dict]:
    """Seed items available for storing"""
    receiving_items_collection = get_collection("receiving_items")
    receiving_collection = get_collection("receiving")
    
    receiving_items = []
    
    # Create a dummy receiving record first
    receiving_record = {
        "_id": "REC001",
        "receivingID": "REC001",
        "supplierName": random.choice(SUPPLIER_NAMES),
        "receivedDate": datetime.utcnow(),
        "status": "processing",
        "totalItems": 8,
        "items": []
    }
    
    # Create 8 items available for storing
    for i in range(8):
        item_template = random.choice(ITEM_NAMES)
        item = {
            "itemID": f"ITEM{i+1:03d}",
            "itemName": item_template["name"],
            "category": item_template["category"],
            "quantity": random.randint(10, 200),
            "condition": random.choice(CONDITIONS),
            "status": "received",
            "receivingID": "REC001"
        }
        receiving_items.append(item)
        receiving_record["items"].append(item)
    
    # Insert receiving record
    receiving_collection.insert_one(receiving_record)
    
    # Insert individual items for the status query
    if receiving_items:
        receiving_items_collection.insert_many(receiving_items)
    
    return receiving_items


def seed_inventory_items() -> List[dict]:
    """Seed items available for picking with their locations"""
    inventory_collection = get_collection("inventory")
    location_occupancy_collection = get_collection("location_occupancy")
    
    inventory_items = []
    occupied_locations = []
    
    # Define some occupied locations
    occupied_slots = [
        {"rack": "B1", "x": 1, "y": 3, "floor": 1},
        {"rack": "B1", "x": 1, "y": 5, "floor": 1},
        {"rack": "B2", "x": 3, "y": 2, "floor": 1},
        {"rack": "B2", "x": 3, "y": 4, "floor": 1},
        {"rack": "B3", "x": 5, "y": 3, "floor": 1},
        {"rack": "P1", "x": 7, "y": 2, "floor": 1},
        {"rack": "P1", "x": 7, "y": 4, "floor": 1},
        {"rack": "P2", "x": 9, "y": 3, "floor": 1},
        {"rack": "D", "x": 5, "y": 10, "floor": 1},
        {"rack": "D", "x": 7, "y": 10, "floor": 1},
    ]
    
    # Create 10 items already stored (available for picking)
    for i in range(10):
        item_template = random.choice(ITEM_NAMES)
        location = occupied_slots[i]
        location_id = f"{location['rack']}-{location['x']}{location['y']-location['y']//2+1}-F{location['floor']}"
        
        item = {
            "itemID": f"INV{i+1:03d}",
            "itemName": item_template["name"],
            "category": item_template["category"],
            "stockLevel": random.randint(20, 150),
            "locationID": location_id,
            "status": "stored",
            "lastUpdated": datetime.utcnow()
        }
        inventory_items.append(item)
        
        # Create location occupancy record
        occupancy = {
            "locationID": location_id,
            "coordinates": {
                "x": location["x"],
                "y": location["y"],
                "floor": location["floor"]
            },
            "occupied": True,
            "itemID": item["itemID"],
            "itemName": item["itemName"],
            "quantity": item["stockLevel"],
            "category": item["category"],
            "storedAt": datetime.utcnow() - timedelta(days=random.randint(1, 30)),
            "lastUpdated": datetime.utcnow()
        }
        occupied_locations.append(occupancy)
    
    # Insert inventory items
    if inventory_items:
        inventory_collection.insert_many(inventory_items)
    
    # Insert location occupancy records
    if occupied_locations:
        location_occupancy_collection.insert_many(occupied_locations)
    
    return inventory_items


@router.delete("/clear-all")
async def clear_all_data(
    current_user: Dict[str, Any] = Depends(get_current_active_user)
):
    """Clear all dummy data"""
    try:
        receiving_items_collection = get_collection("receiving_items")
        inventory_collection = get_collection("inventory")
        location_occupancy_collection = get_collection("location_occupancy")
        storage_history_collection = get_collection("storage_history")
        receiving_collection = get_collection("receiving")
        
        receiving_items_collection.delete_many({})
        receiving_collection.delete_many({})
        inventory_collection.delete_many({})
        location_occupancy_collection.delete_many({})
        storage_history_collection.delete_many({})
        
        return {"message": "All data cleared successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error clearing data: {str(e)}")