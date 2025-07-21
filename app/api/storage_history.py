from fastapi import APIRouter, HTTPException, Depends, status
from typing import List, Optional, Dict, Any
from datetime import datetime

from ..auth.dependencies import get_current_active_user
from ..models.storage_history import StorageHistory, LocationOccupancy
from ..utils.database import get_collection

router = APIRouter()


@router.post("/store-item")
async def store_item(
    storage_data: dict,
    current_user: Dict[str, Any] = Depends(get_current_active_user)
):
    """Store an item and update location occupancy"""
    try:
        storage_history_collection = get_collection("storage_history")
        location_occupancy_collection = get_collection("location_occupancy")
        
        # Create storage history entry
        storage_history = {
            "itemID": storage_data["itemID"],
            "itemName": storage_data["itemName"],
            "quantity": storage_data["quantity"],
            "locationID": storage_data["locationID"],
            "locationCoordinates": storage_data["locationCoordinates"],
            "storedBy": current_user.get("username", "unknown"),
            "storedAt": datetime.utcnow(),
            "action": "stored",
            "category": storage_data.get("category"),
            "condition": storage_data.get("condition"),
            "receivingID": storage_data.get("receivingID")
        }
        
        # Save to storage history collection
        result = storage_history_collection.insert_one(storage_history)
        storage_history["_id"] = str(result.inserted_id)
        
        # Update location occupancy
        location_occupancy = {
            "locationID": storage_data["locationID"],
            "coordinates": storage_data["locationCoordinates"],
            "occupied": True,
            "itemID": storage_data["itemID"],
            "itemName": storage_data["itemName"],
            "quantity": storage_data["quantity"],
            "category": storage_data.get("category"),
            "storedAt": datetime.utcnow(),
            "lastUpdated": datetime.utcnow()
        }
        
        # Upsert location occupancy
        location_occupancy_collection.update_one(
            {"locationID": storage_data["locationID"]},
            {"$set": location_occupancy},
            upsert=True
        )
        
        return storage_history
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to store item: {str(e)}"
        )


@router.post("/collect-item")
async def collect_item(
    collection_data: dict,
    current_user: Dict[str, Any] = Depends(get_current_active_user)
):
    """Collect an item and update location occupancy"""
    try:
        storage_history_collection = get_collection("storage_history")
        location_occupancy_collection = get_collection("location_occupancy")
        
        # Create collection history entry
        collection_history = {
            "itemID": collection_data["itemID"],
            "itemName": collection_data["itemName"],
            "quantity": collection_data["quantity"],
            "locationID": collection_data["locationID"],
            "locationCoordinates": collection_data["locationCoordinates"],
            "storedBy": current_user.get("username", "unknown"),
            "storedAt": datetime.utcnow(),
            "category": collection_data.get("category"),
            "action": "collected"
        }
        
        # Save to storage history collection
        result = storage_history_collection.insert_one(collection_history)
        collection_history["_id"] = str(result.inserted_id)
        
        # Clear location occupancy
        location_occupancy_collection.update_one(
            {"locationID": collection_data["locationID"]},
            {"$set": {
                "occupied": False,
                "itemID": None,
                "itemName": None,
                "quantity": None,
                "category": None,
                "storedAt": None,
                "lastUpdated": datetime.utcnow()
            }}
        )
        
        return collection_history
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to collect item: {str(e)}"
        )


@router.get("/history")
async def get_storage_history(
    skip: int = 0,
    limit: int = 100,
    action: Optional[str] = None,
    current_user: Dict[str, Any] = Depends(get_current_active_user)
):
    """Get storage history with optional filtering"""
    storage_history_collection = get_collection("storage_history")
    
    query = {}
    if action:
        query["action"] = action
    
    history = list(storage_history_collection.find(query).sort("storedAt", -1).skip(skip).limit(limit))
    
    # Convert ObjectId to string
    for item in history:
        item["_id"] = str(item["_id"])
    
    return history


@router.get("/occupied-locations")
async def get_occupied_locations(
    floor: Optional[int] = None
):
    """Get all occupied locations from location_inventory collection"""
    # ✨ FIXED: Use location_inventory instead of location_occupancy
    location_inventory_collection = get_collection("location_inventory")
    
    query = {"available": False}  # occupied locations have available=False
    
    locations = list(location_inventory_collection.find(query))
    
    # Convert to format expected by frontend
    formatted_locations = []
    for loc in locations:
        # Extract coordinates from locationID (e.g., "B02.1" -> x=1, y=3, floor=1)
        location_id = loc.get("locationID", "")
        if '.' in location_id:
            slot_code, floor_str = location_id.split('.')
            floor_num = int(floor_str)
            
            # Skip if floor filter is specified and doesn't match
            if floor is not None and floor_num != floor:
                continue
            
            # Calculate coordinates based on slot code
            coordinates = calculate_coordinates_from_location_id(location_id)
            
            formatted_location = {
                "_id": str(loc.get("_id", "")),
                "locationID": location_id,
                "coordinates": coordinates,
                "occupied": True,  # All these locations are occupied (available=False)
                "itemID": loc.get("itemID"),
                "itemName": loc.get("itemName"),
                "quantity": loc.get("quantity", 0),
                "category": loc.get("category", "General"),
                "storedAt": loc.get("storedAt"),
                "lastUpdated": loc.get("lastUpdated")
            }
            formatted_locations.append(formatted_location)
    
    return formatted_locations

def calculate_coordinates_from_location_id(location_id: str) -> dict:
    """Calculate map coordinates from location ID (e.g., B02.1 -> {x: 1, y: 3, floor: 1})"""
    try:
        if '.' not in location_id:
            return {"x": 0, "y": 0, "floor": 1}
        
        slot_code, floor_str = location_id.split('.')
        floor = int(floor_str)
        
        # Extract slot type and number
        slot_type = slot_code[0]  # B, P, or D
        slot_num = int(slot_code[1:])  # 01, 02, etc.
        
        if slot_type == 'B':
            # B slots: B01-B07 at x=1, B08-B14 at x=3, B15-B21 at x=5
            if 1 <= slot_num <= 7:
                x = 1
                y = 2 + (slot_num - 1)  # B01=y2, B02=y3, ..., B07=y8
            elif 8 <= slot_num <= 14:
                x = 3
                y = 2 + (slot_num - 8)  # B08=y2, B09=y3, ..., B14=y8
            elif 15 <= slot_num <= 21:
                x = 5
                y = 2 + (slot_num - 15)  # B15=y2, B16=y3, ..., B21=y8
            else:
                x, y = 1, 2
        elif slot_type == 'P':
            # P slots: P01-P07 at x=7, P08-P14 at x=9
            if 1 <= slot_num <= 7:
                x = 7
                y = 2 + (slot_num - 1)  # P01=y2, P02=y3, ..., P07=y8
            elif 8 <= slot_num <= 14:
                x = 9
                y = 2 + (slot_num - 8)  # P08=y2, P09=y3, ..., P14=y8
            else:
                x, y = 7, 2
        elif slot_type == 'D':
            # D slots: D01-D07 at y=10, D08-D14 at y=11
            if 1 <= slot_num <= 7:
                x = 3 + (slot_num - 1)  # D01=x3, D02=x4, ..., D07=x9
                y = 10
            elif 8 <= slot_num <= 14:
                x = 3 + (slot_num - 8)  # D08=x3, D09=x4, ..., D14=x9
                y = 11
            else:
                x, y = 3, 10
        else:
            x, y = 0, 0
        
        return {"x": x, "y": y, "floor": floor}
    
    except Exception as e:
        print(f"Error calculating coordinates for {location_id}: {e}")
        return {"x": 0, "y": 0, "floor": 1}


@router.get("/location/{location_id}/details")
async def get_location_details(
    location_id: str
):
    """Get details of a specific location"""
    location_occupancy_collection = get_collection("location_occupancy")
    
    location = location_occupancy_collection.find_one({"locationID": location_id})
    
    if not location:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Location {location_id} not found"
        )
    
    location["_id"] = str(location["_id"])
    return location