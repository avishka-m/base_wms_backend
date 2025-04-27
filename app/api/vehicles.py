from fastapi import APIRouter, Depends, HTTPException, Query, status
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta

from ..auth.dependencies import get_current_active_user, has_role
from ..models.vehicle import VehicleCreate, VehicleUpdate, VehicleResponse
from ..utils.database import get_collection

router = APIRouter()

# Get all vehicles
@router.get("/", response_model=List[VehicleResponse])
async def get_vehicles(
    current_user: Dict[str, Any] = Depends(get_current_active_user),
    skip: int = 0,
    limit: int = 100,
    status: Optional[str] = None,
    vehicle_type: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Get all vehicles with optional filtering.
    
    You can filter by status and vehicle type.
    """
    vehicle_collection = get_collection("vehicles")
    
    # Build query
    query = {}
    if status:
        query["status"] = status
    if vehicle_type:
        query["vehicle_type"] = vehicle_type
    
    # Execute query
    vehicles = list(vehicle_collection.find(query).skip(skip).limit(limit))
    return vehicles

# Get available vehicles
@router.get("/available", response_model=List[VehicleResponse])
async def get_available_vehicles(
    current_user: Dict[str, Any] = Depends(get_current_active_user),
    vehicle_type: Optional[str] = None,
    min_capacity: Optional[float] = None
) -> List[Dict[str, Any]]:
    """
    Get all available vehicles with optional filtering.
    """
    vehicle_collection = get_collection("vehicles")
    
    # Build query
    query = {"status": "available"}
    if vehicle_type:
        query["vehicle_type"] = vehicle_type
    if min_capacity:
        query["capacity"] = {"$gte": min_capacity}
    
    # Execute query
    vehicles = list(vehicle_collection.find(query))
    return vehicles

# Get vehicle by ID
@router.get("/{vehicle_id}", response_model=VehicleResponse)
async def get_vehicle(
    vehicle_id: int,
    current_user: Dict[str, Any] = Depends(get_current_active_user)
) -> Dict[str, Any]:
    """
    Get a specific vehicle by ID.
    """
    vehicle_collection = get_collection("vehicles")
    vehicle = vehicle_collection.find_one({"vehicleID": vehicle_id})
    
    if not vehicle:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Vehicle with ID {vehicle_id} not found"
        )
    
    return vehicle

# Create new vehicle
@router.post("/", response_model=VehicleResponse, status_code=status.HTTP_201_CREATED)
async def create_vehicle(
    vehicle: VehicleCreate,
    current_user: Dict[str, Any] = Depends(has_role(["Manager"]))
) -> Dict[str, Any]:
    """
    Create a new vehicle.
    
    Only managers can create vehicles.
    """
    vehicle_collection = get_collection("vehicles")
    
    # Find the next available vehicleID
    last_vehicle = vehicle_collection.find_one(
        sort=[("vehicleID", -1)]
    )
    next_id = 1
    if last_vehicle:
        next_id = last_vehicle.get("vehicleID", 0) + 1
    
    # Prepare vehicle document
    vehicle_data = vehicle.model_dump()
    vehicle_data.update({
        "vehicleID": next_id,
        "status": "available",
        "last_maintenance_date": None,
        "next_maintenance_date": datetime.utcnow() + timedelta(days=90),  # Default 90 days
        "created_at": datetime.utcnow(),
        "updated_at": datetime.utcnow()
    })
    
    # Insert vehicle to database
    result = vehicle_collection.insert_one(vehicle_data)
    
    # Return the created vehicle
    created_vehicle = vehicle_collection.find_one({"_id": result.inserted_id})
    return created_vehicle

# Update vehicle
@router.put("/{vehicle_id}", response_model=VehicleResponse)
async def update_vehicle(
    vehicle_id: int,
    vehicle_update: VehicleUpdate,
    current_user: Dict[str, Any] = Depends(has_role(["Manager"]))
) -> Dict[str, Any]:
    """
    Update a vehicle.
    
    Only managers can update vehicles.
    """
    vehicle_collection = get_collection("vehicles")
    
    # Check if vehicle exists
    vehicle = vehicle_collection.find_one({"vehicleID": vehicle_id})
    if not vehicle:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Vehicle with ID {vehicle_id} not found"
        )
    
    # Prepare update data
    update_data = vehicle_update.model_dump(exclude_unset=True)
    update_data["updated_at"] = datetime.utcnow()
    
    # Update vehicle
    vehicle_collection.update_one(
        {"vehicleID": vehicle_id},
        {"$set": update_data}
    )
    
    # Return updated vehicle
    updated_vehicle = vehicle_collection.find_one({"vehicleID": vehicle_id})
    return updated_vehicle

# Delete vehicle
@router.delete("/{vehicle_id}", response_model=Dict[str, Any])
async def delete_vehicle(
    vehicle_id: int,
    current_user: Dict[str, Any] = Depends(has_role(["Manager"]))
) -> Dict[str, Any]:
    """
    Delete a vehicle.
    
    Only managers can delete vehicles.
    """
    vehicle_collection = get_collection("vehicles")
    shipping_collection = get_collection("shipping")
    
    # Check if vehicle exists
    vehicle = vehicle_collection.find_one({"vehicleID": vehicle_id})
    if not vehicle:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Vehicle with ID {vehicle_id} not found"
        )
    
    # Check if vehicle is in use
    if vehicle.get("status") == "in_use":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Cannot delete vehicle with ID {vehicle_id} because it is currently in use"
        )
    
    # Check if vehicle is associated with any shipping records
    shipping = shipping_collection.find_one({"vehicleID": vehicle_id})
    if shipping:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Cannot delete vehicle with ID {vehicle_id} because it is associated with shipping records"
        )
    
    # Delete vehicle
    vehicle_collection.delete_one({"vehicleID": vehicle_id})
    
    return {"message": f"Vehicle with ID {vehicle_id} has been deleted"}

# Update vehicle status
@router.put("/{vehicle_id}/status", response_model=VehicleResponse)
async def update_vehicle_status(
    vehicle_id: int,
    status: str = Query(..., description="New status for the vehicle (available, in_use, maintenance)"),
    current_user: Dict[str, Any] = Depends(has_role(["Manager", "Driver"]))
) -> Dict[str, Any]:
    """
    Update the status of a vehicle.
    
    Managers can update any vehicle status. Drivers can only update their assigned vehicles.
    """
    vehicle_collection = get_collection("vehicles")
    shipping_collection = get_collection("shipping")
    
    # Check if vehicle exists
    vehicle = vehicle_collection.find_one({"vehicleID": vehicle_id})
    if not vehicle:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Vehicle with ID {vehicle_id} not found"
        )
    
    # Restrict drivers to only update vehicles assigned to them
    if current_user.get("role") == "Driver":
        # Check if driver is assigned to this vehicle
        shipping = shipping_collection.find_one({
            "vehicleID": vehicle_id,
            "workerID": current_user.get("workerID"),
            "status": "in_transit"
        })
        if not shipping:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="You do not have permission to update this vehicle's status"
            )
    
    # Update status
    vehicle_collection.update_one(
        {"vehicleID": vehicle_id},
        {"$set": {"status": status, "updated_at": datetime.utcnow()}}
    )
    
    # Return updated vehicle
    updated_vehicle = vehicle_collection.find_one({"vehicleID": vehicle_id})
    return updated_vehicle

# Record vehicle maintenance
@router.post("/{vehicle_id}/maintenance", response_model=VehicleResponse)
async def record_maintenance(
    vehicle_id: int,
    maintenance_notes: str = Query(..., description="Notes about the maintenance performed"),
    next_maintenance_days: int = Query(90, description="Days until next maintenance is due"),
    current_user: Dict[str, Any] = Depends(has_role(["Manager"]))
) -> Dict[str, Any]:
    """
    Record maintenance for a vehicle and update maintenance schedule.
    
    Only managers can record vehicle maintenance.
    """
    vehicle_collection = get_collection("vehicles")
    
    # Check if vehicle exists
    vehicle = vehicle_collection.find_one({"vehicleID": vehicle_id})
    if not vehicle:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Vehicle with ID {vehicle_id} not found"
        )
    
    # Update maintenance records
    now = datetime.utcnow()
    next_maintenance = now + timedelta(days=next_maintenance_days)
    
    # In a real system, you would store the maintenance records in a separate collection
    
    # Update vehicle
    vehicle_collection.update_one(
        {"vehicleID": vehicle_id},
        {
            "$set": {
                "status": "available",
                "last_maintenance_date": now,
                "next_maintenance_date": next_maintenance,
                "updated_at": now
            }
        }
    )
    
    # Return updated vehicle
    updated_vehicle = vehicle_collection.find_one({"vehicleID": vehicle_id})
    return updated_vehicle