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
    """Get all occupied locations"""
    location_occupancy_collection = get_collection("location_occupancy")
    
    query = {"occupied": True}
    if floor is not None:
        query["coordinates.floor"] = floor
    
    locations = list(location_occupancy_collection.find(query))
    
    # Convert ObjectId to string
    for loc in locations:
        loc["_id"] = str(loc["_id"])
    
    return locations


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