from fastapi import APIRouter, Depends, HTTPException, status
from typing import Dict, List, Any, Optional
from datetime import datetime

from ..auth.dependencies import get_current_active_user, has_role
from ..utils.database import get_collection

router = APIRouter()

@router.post("/initialize-locations")
async def initialize_all_locations(
    current_user: Dict[str, Any] = Depends(has_role(["Manager", "ReceivingClerk"]))
) -> Dict[str, Any]:
    """Initialize all warehouse locations as available"""
    try:
        location_collection = get_collection("location_inventory")
        
        print("🔍 Starting location initialization...")
        
        # Check if already initialized
        existing_count = location_collection.count_documents({})
        if existing_count > 0:
            print(f"⚠️ Found {existing_count} existing locations, clearing them...")
            location_collection.delete_many({})
        
        # ✨ FIXED: Define all_locations list
        all_locations = []
        
        # B slots: B01-B21, each with 4 floors
        for i in range(1, 22):
            slot_code = f"B{str(i).zfill(2)}"
            for floor in range(1, 5):
                location_id = f"{slot_code}.{floor}"
                all_locations.append({
                    "locationID": location_id,
                    "slotCode": slot_code,
                    "floor": floor,
                    "type": "M",  # Medium/Bin
                    "available": True,
                    "itemID": None,
                    "itemName": None,
                    "quantity": 0,
                    "lastUpdated": datetime.utcnow().isoformat()
                })
        
        # P slots: P01-P14, each with 4 floors  
        for i in range(1, 15):
            slot_code = f"P{str(i).zfill(2)}"
            for floor in range(1, 5):
                location_id = f"{slot_code}.{floor}"
                all_locations.append({
                    "locationID": location_id,
                    "slotCode": slot_code,
                    "floor": floor,
                    "type": "S",  # Small/Pellet
                    "available": True,
                    "itemID": None,
                    "itemName": None,
                    "quantity": 0,
                    "lastUpdated": datetime.utcnow().isoformat()
                })
        
        # D slots: D01-D14, each with 4 floors
        for i in range(1, 15):
            slot_code = f"D{str(i).zfill(2)}"
            for floor in range(1, 5):
                location_id = f"{slot_code}.{floor}"
                all_locations.append({
                    "locationID": location_id,
                    "slotCode": slot_code,
                    "floor": floor,
                    "type": "D",  # Large
                    "available": True,
                    "itemID": None,
                    "itemName": None,
                    "quantity": 0,
                    "lastUpdated": datetime.utcnow().isoformat()
                })
        
        # ✨ FIXED: Insert all locations and verify
        result = location_collection.insert_many(all_locations)
        final_count = location_collection.count_documents({})
        
        print(f"✅ Inserted {len(result.inserted_ids)} locations")
        print(f"✅ Final count in database: {final_count}")
        
        return {
            "message": f"Initialized {final_count} warehouse locations",
            "total_locations": final_count,
            "inserted_ids": len(result.inserted_ids),
            "b_slots": 21 * 4,  # 84 locations
            "p_slots": 14 * 4,  # 56 locations  
            "d_slots": 14 * 4   # 56 locations
        }
        
    except Exception as e:
        print(f"❌ Error in initialize_all_locations: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to initialize locations: {str(e)}"
        )
    
@router.get("/available/{location_type}")
async def get_available_locations(
    location_type: str,  # "B", "P", "D", or "all"
    current_user: Dict[str, Any] = Depends(get_current_active_user)
) -> List[Dict[str, Any]]:
    """Get all available locations of a specific type"""
    try:
        location_collection = get_collection("location_inventory")
        
        # ✨ FIX: Check if collection exists and has data
        total_count = location_collection.count_documents({})
        print(f"🔍 Total locations in collection: {total_count}")
        
        if total_count == 0:
            return {
                "error": "Location inventory is empty. Run initialize-locations first.",
                "available_locations": []
            }
        
        # Build query
        query = {"available": True}
        if location_type.upper() != "ALL":
            query["type"] = location_type.upper()
        
        print(f"🔍 Query: {query}")
        
        # Execute query with error handling
        available_locations = list(location_collection.find(query))
        
        # Convert ObjectId to string for JSON serialization
        for location in available_locations:
            if "_id" in location:
                location["_id"] = str(location["_id"])
        
        print(f"✅ Found {len(available_locations)} available locations")
        return available_locations
        
    except Exception as e:
        print(f"❌ Error in get_available_locations: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get available locations: {str(e)}"
        )
    
@router.post("/occupy/{location_id}")
async def occupy_location(
    location_id: str,
    item_data: Dict[str, Any],
    current_user: Dict[str, Any] = Depends(has_role(["Manager", "Picker", "ReceivingClerk"]))
) -> Dict[str, Any]:
    """Mark a location as occupied by an item"""
    try:
        location_collection = get_collection("location_inventory")
        
        # Check if location exists and is available
        location = location_collection.find_one({"locationID": location_id})
        if not location:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Location {location_id} not found"
            )
            
        if not location.get("available", False):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Location {location_id} is already occupied"
            )
        
        # Update location as occupied
        update_data = {
            "available": False,
            "itemID": item_data.get("itemID"),
            "itemName": item_data.get("itemName"),
            "quantity": item_data.get("quantity"),
            "storedAt": datetime.utcnow().isoformat(),
            "storedBy": current_user.get("username"),
            "receivingID": item_data.get("receivingID"),
            "lastUpdated": datetime.utcnow().isoformat()
        }
        
        location_collection.update_one(
            {"locationID": location_id},
            {"$set": update_data}
        )
        
        return {
            "message": f"Location {location_id} marked as occupied",
            "locationID": location_id,
            "itemID": item_data.get("itemID"),
            "itemName": item_data.get("itemName")
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to occupy location: {str(e)}"
        )

@router.post("/free/{location_id}")
async def free_location(
    location_id: str,
    current_user: Dict[str, Any] = Depends(has_role(["Manager", "Picker"]))
) -> Dict[str, Any]:
    """Mark a location as available (free up the location)"""
    try:
        location_collection = get_collection("location_inventory")
        
        update_data = {
            "available": True,
            "itemID": None,
            "itemName": None,
            "quantity": 0,
            "freedAt": datetime.utcnow().isoformat(),
            "freedBy": current_user.get("username"),
            "lastUpdated": datetime.utcnow().isoformat()
        }
        
        result = location_collection.update_one(
            {"locationID": location_id},
            {"$set": update_data}
        )
        
        if result.matched_count == 0:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Location {location_id} not found"
            )
        
        return {
            "message": f"Location {location_id} marked as available",
            "locationID": location_id
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to free location: {str(e)}"
        )