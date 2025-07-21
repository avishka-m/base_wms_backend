from fastapi import APIRouter, Depends, HTTPException, status, Body
from typing import Dict, List, Any, Optional
from datetime import datetime

from ..auth.dependencies import get_current_active_user, has_role
from ..utils.database import get_collection

router = APIRouter()

@router.post("/update-location")
async def update_worker_location(
    location_data: Dict[str, Any] = Body(...),
    current_user: Dict[str, Any] = Depends(get_current_active_user)
) -> Dict[str, Any]:
    """
    Update worker's current location in the warehouse
    """
    try:
        workers_collection = get_collection("workers")
        
        # Get location data
        x = location_data.get("x")
        y = location_data.get("y")
        floor = location_data.get("floor", 1)
        status = location_data.get("status", "online")  # online, offline, working
        
        if x is None or y is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="x and y coordinates are required"
            )
        
        # Get worker info
        worker_id = current_user.get("workerID") or current_user.get("id")
        username = current_user.get("username")
        
        if not worker_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Worker ID not found in user data"
            )
        
        # Get current location for history
        current_worker = workers_collection.find_one({"workerID": worker_id})
        if not current_worker:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Worker not found"
            )
        
        current_time = datetime.utcnow().isoformat()
        
        # Prepare location history entry
        location_history_entry = {
            "x": x,
            "y": y,
            "floor": floor,
            "timestamp": current_time,
            "status": status
        }
        
        # Update worker location
        update_data = {
            "worker_location": {
                "x": x,
                "y": y,
                "floor": floor,
                "last_updated": current_time,
                "updated_by": username,
                "status": status
            },
            "last_location_update": current_time
        }
        
        # Add to location history (keep last 50 entries)
        workers_collection.update_one(
            {"workerID": worker_id},
            {
                "$set": update_data,
                "$push": {
                    "location_history": {
                        "$each": [location_history_entry],
                        "$slice": -50  # Keep only last 50 entries
                    }
                }
            }
        )
        
        print(f"✅ Updated location for worker {username}: ({x}, {y}, F{floor})")
        
        return {
            "success": True,
            "message": f"Location updated for worker {username}",
            "worker_id": worker_id,
            "location": {
                "x": x,
                "y": y,
                "floor": floor,
                "status": status
            },
            "updated_at": current_time
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error updating worker location: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update location: {str(e)}"
        )

@router.get("/current-location")
async def get_worker_current_location(
    current_user: Dict[str, Any] = Depends(get_current_active_user)
) -> Dict[str, Any]:
    """
    Get worker's current location
    """
    try:
        workers_collection = get_collection("workers")
        
        worker_id = current_user.get("workerID") or current_user.get("id")
        
        if not worker_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Worker ID not found"
            )
        
        worker = workers_collection.find_one({"workerID": worker_id})
        
        if not worker:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Worker not found"
            )
        
        location = worker.get("worker_location", {
            "x": 0,
            "y": 0,
            "floor": 1,
            "status": "offline"
        })
        
        return {
            "success": True,
            "worker_id": worker_id,
            "username": worker.get("username"),
            "location": location
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error getting worker location: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get location: {str(e)}"
        )

@router.get("/all-workers-locations")
async def get_all_workers_locations(
    current_user: Dict[str, Any] = Depends(get_current_active_user)
) -> List[Dict[str, Any]]:
    """
    Get locations of all workers (for managers/supervisors)
    """
    try:
        workers_collection = get_collection("workers")
        
        # Get all workers with their locations
        workers = list(workers_collection.find(
            {},
            {
                "workerID": 1,
                "username": 1,
                "worker_location": 1,
                "last_location_update": 1
            }
        ))
        
        worker_locations = []
        for worker in workers:
            location = worker.get("worker_location", {})
            worker_locations.append({
                "worker_id": worker.get("workerID"),
                "username": worker.get("username"),
                "location": location,
                "last_updated": worker.get("last_location_update")
            })
        
        return worker_locations
        
    except Exception as e:
        print(f"❌ Error getting all worker locations: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get worker locations: {str(e)}"
        )

@router.post("/set-status")
async def set_worker_status(
    status_data: Dict[str, Any] = Body(...),
    current_user: Dict[str, Any] = Depends(get_current_active_user)
) -> Dict[str, Any]:
    """
    Set worker status (online, offline, working)
    """
    try:
        workers_collection = get_collection("workers")
        
        worker_id = current_user.get("workerID") or current_user.get("id")
        new_status = status_data.get("status")
        
        if not new_status or new_status not in ["online", "offline", "working"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Valid status required: online, offline, or working"
            )
        
        # Update worker status
        result = workers_collection.update_one(
            {"workerID": worker_id},
            {
                "$set": {
                    "worker_location.status": new_status,
                    "worker_location.last_updated": datetime.utcnow().isoformat()
                }
            }
        )
        
        if result.matched_count == 0:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Worker not found"
            )
        
        return {
            "success": True,
            "message": f"Status updated to {new_status}",
            "worker_id": worker_id,
            "status": new_status
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error setting worker status: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to set status: {str(e)}"
        )
    
@router.post("/set-dummy-location")
async def set_dummy_worker_location(
    location_data: Dict[str, Any] = Body(...),
    current_user: Dict[str, Any] = Depends(get_current_active_user)
) -> Dict[str, Any]:
    """Set a dummy location for a worker (for development/testing)"""
    try:
        workers_collection = get_collection("workers")
        
        worker_id = current_user.get("workerID") or current_user.get("id")
        x = location_data.get("x", 0)  # Default to receiving area
        y = location_data.get("y", 0)
        floor = location_data.get("floor", 1)
        
        # Update worker with dummy location
        current_time = datetime.utcnow().isoformat()
        update_data = {
            "worker_location": {
                "x": x,
                "y": y,
                "floor": floor,
                "last_updated": current_time,
                "updated_by": current_user.get("username"),
                "status": "online",
                "is_dummy": True  # Flag to indicate this is dummy data
            },
            "last_location_update": current_time
        }
        
        result = workers_collection.update_one(
            {"workerID": worker_id},
            {"$set": update_data}
        )
        
        if result.matched_count == 0:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Worker not found"
            )
        
        print(f"✅ Set dummy location for worker {current_user.get('username')}: ({x}, {y}, F{floor})")
        
        return {
            "success": True,
            "message": f"Dummy location set for worker {current_user.get('username')}",
            "worker_id": worker_id,
            "location": {
                "x": x,
                "y": y,
                "floor": floor
            },
            "note": "This is dummy location data for development"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error setting dummy location: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to set dummy location: {str(e)}"
        )

@router.post("/initialize-dummy-locations")
async def initialize_dummy_worker_locations(
    current_user: Dict[str, Any] = Depends(has_role(["Manager"]))
) -> Dict[str, Any]:
    """Initialize dummy locations for all workers"""
    try:
        workers_collection = get_collection("workers")
        
        # Get all workers
        workers = list(workers_collection.find({}))
        
        updated_count = 0
        # ✨ FREE PATH COORDINATES (avoiding rack locations)
        # Based on 10x12 warehouse grid with racks at specific locations
        dummy_locations = [
            {"x": 2, "y": 1, "floor": 1},   # Aisle between B Rack 1 & 2
            {"x": 4, "y": 1, "floor": 1},   # Aisle between B Rack 2 & 3  
            {"x": 6, "y": 1, "floor": 1},   # Aisle between B Rack 3 & P Rack 1
            {"x": 8, "y": 1, "floor": 1},   # Aisle between P Rack 1 & 2
            {"x": 2, "y": 9, "floor": 1},   # Main pathway (north of D racks)
            {"x": 4, "y": 9, "floor": 1},   # Main pathway (north of D racks)
            {"x": 6, "y": 9, "floor": 1},   # Main pathway (north of D racks)
            {"x": 8, "y": 9, "floor": 1},   # Main pathway (north of D racks)
            {"x": 0, "y": 5, "floor": 1},   # West pathway (along warehouse edge)
            {"x": 9, "y": 5, "floor": 1},   # East pathway (along warehouse edge)
        ]
        
        current_time = datetime.utcnow().isoformat()
        
        for i, worker in enumerate(workers):
            # Assign dummy location (cycle through available locations)
            location = dummy_locations[i % len(dummy_locations)]
            
            update_data = {
                "worker_location": {
                    "x": location["x"],
                    "y": location["y"], 
                    "floor": location["floor"],
                    "last_updated": current_time,
                    "status": "online",
                    "is_dummy": True
                },
                "last_location_update": current_time
            }
            
            workers_collection.update_one(
                {"_id": worker["_id"]},
                {"$set": update_data}
            )
            updated_count += 1
        
        return {
            "message": f"Initialized dummy locations for {updated_count} workers",
            "workers_updated": updated_count,
            "available_locations": dummy_locations
        }
        
    except Exception as e:
        print(f"❌ Error initializing dummy locations: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to initialize dummy locations: {str(e)}"
        )


@router.post("/redistribute-workers-from-receiving")
async def redistribute_workers_from_receiving(
    current_user: Dict[str, Any] = Depends(has_role(["Manager"]))
) -> Dict[str, Any]:
    """Move all workers away from receiving counter (0,0) to different free path locations"""
    try:
        workers_collection = get_collection("workers")
        
        # Get workers currently at receiving counter (0,0)
        workers_at_receiving = list(workers_collection.find({
            "$or": [
                {"worker_location.x": 0, "worker_location.y": 0},
                {"worker_location": {"$exists": False}}  # Workers without location
            ]
        }))
        
        if not workers_at_receiving:
            return {
                "message": "No workers found at receiving counter",
                "workers_moved": 0
            }
        
        # Free path coordinates (excluding receiving area and rack locations)
        free_path_locations = [
            {"x": 2, "y": 1, "floor": 1},   # Aisle between B Rack 1 & 2
            {"x": 4, "y": 1, "floor": 1},   # Aisle between B Rack 2 & 3  
            {"x": 6, "y": 1, "floor": 1},   # Aisle between B Rack 3 & P Rack 1
            {"x": 8, "y": 1, "floor": 1},   # Aisle between P Rack 1 & 2
            {"x": 2, "y": 9, "floor": 1},   # Main pathway (north of D racks)
            {"x": 4, "y": 9, "floor": 1},   # Main pathway (north of D racks)
            {"x": 6, "y": 9, "floor": 1},   # Main pathway (north of D racks)
            {"x": 8, "y": 9, "floor": 1},   # Main pathway (north of D racks)
            {"x": 0, "y": 5, "floor": 1},   # West pathway (along warehouse edge)
            {"x": 9, "y": 5, "floor": 1},   # East pathway (along warehouse edge)
            {"x": 2, "y": 5, "floor": 1},   # Central aisle
            {"x": 4, "y": 5, "floor": 1},   # Central aisle
            {"x": 6, "y": 5, "floor": 1},   # Central aisle
            {"x": 8, "y": 5, "floor": 1},   # Central aisle
        ]
        
        current_time = datetime.utcnow().isoformat()
        moved_count = 0
        
        for i, worker in enumerate(workers_at_receiving):
            # Assign free path location (cycle through available locations)
            location = free_path_locations[i % len(free_path_locations)]
            
            update_data = {
                "worker_location": {
                    "x": location["x"],
                    "y": location["y"], 
                    "floor": location["floor"],
                    "last_updated": current_time,
                    "status": "online",
                    "redistributed_from_receiving": True
                },
                "last_location_update": current_time
            }
            
            workers_collection.update_one(
                {"_id": worker["_id"]},
                {"$set": update_data}
            )
            moved_count += 1
            
            worker_name = worker.get("name", "Unknown")
            print(f"✅ Moved worker {worker_name} from receiving to ({location['x']}, {location['y']})")
        
        return {
            "message": f"Successfully moved {moved_count} workers from receiving counter to free path locations",
            "workers_moved": moved_count,
            "new_locations": free_path_locations[:moved_count]
        }
        
    except Exception as e:
        print(f"❌ Error redistributing workers: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to redistribute workers: {str(e)}"
        )