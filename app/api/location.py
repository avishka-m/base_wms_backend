from fastapi import APIRouter, Depends, HTTPException, Query, status
from typing import List, Dict, Any, Optional

from ..auth.dependencies import get_current_active_user, has_role
from ..models.location import LocationCreate, LocationUpdate, LocationResponse
from ..models.warehouse import WarehouseCreate, WarehouseUpdate, WarehouseResponse
from ..services.warehouse_service import WarehouseService

router = APIRouter()

# --- WAREHOUSE ENDPOINTS ---

# Get all warehouses
@router.get("/warehouses", response_model=List[WarehouseResponse])
async def get_warehouses(
    current_user: Dict[str, Any] = Depends(get_current_active_user),
    skip: int = 0,
    limit: int = 100
) -> List[Dict[str, Any]]:
    """
    Get all warehouses.
    """
    warehouses = await WarehouseService.get_warehouses(skip=skip, limit=limit)
    return warehouses

# Get warehouse by ID
@router.get("/warehouses/{warehouse_id}", response_model=WarehouseResponse)
async def get_warehouse(
    warehouse_id: int,
    current_user: Dict[str, Any] = Depends(get_current_active_user)
) -> Dict[str, Any]:
    """
    Get a specific warehouse by ID.
    """
    warehouse = await WarehouseService.get_warehouse(warehouse_id)
    
    if not warehouse:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Warehouse with ID {warehouse_id} not found"
        )
    
    return warehouse

# Create new warehouse
@router.post("/warehouses", response_model=WarehouseResponse, status_code=status.HTTP_201_CREATED)
async def create_warehouse(
    warehouse: WarehouseCreate,
    current_user: Dict[str, Any] = Depends(has_role(["Manager"]))
) -> Dict[str, Any]:
    """
    Create a new warehouse.
    
    Only managers can create warehouses.
    """
    created_warehouse = await WarehouseService.create_warehouse(warehouse)
    
    if "error" in created_warehouse:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=created_warehouse["error"]
        )
    
    return created_warehouse

# Update warehouse
@router.put("/warehouses/{warehouse_id}", response_model=WarehouseResponse)
async def update_warehouse(
    warehouse_id: int,
    warehouse_update: WarehouseUpdate,
    current_user: Dict[str, Any] = Depends(has_role(["Manager"]))
) -> Dict[str, Any]:
    """
    Update a warehouse.
    
    Only managers can update warehouses.
    """
    # Check if warehouse exists
    warehouse = await WarehouseService.get_warehouse(warehouse_id)
    if not warehouse:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Warehouse with ID {warehouse_id} not found"
        )
    
    updated_warehouse = await WarehouseService.update_warehouse(warehouse_id, warehouse_update)
    return updated_warehouse

# Delete warehouse
@router.delete("/warehouses/{warehouse_id}", response_model=Dict[str, Any])
async def delete_warehouse(
    warehouse_id: int,
    current_user: Dict[str, Any] = Depends(has_role(["Manager"]))
) -> Dict[str, Any]:
    """
    Delete a warehouse.
    
    Only managers can delete warehouses.
    """
    result = await WarehouseService.delete_warehouse(warehouse_id)
    
    if "error" in result:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=result["error"]
        )
    
    return result

# Calculate warehouse utilization
@router.get("/warehouses/{warehouse_id}/utilization", response_model=Dict[str, Any])
async def calculate_warehouse_utilization(
    warehouse_id: int,
    current_user: Dict[str, Any] = Depends(get_current_active_user)
) -> Dict[str, Any]:
    """
    Calculate the storage utilization of a warehouse.
    """
    result = await WarehouseService.calculate_warehouse_utilization(warehouse_id)
    
    if "error" in result:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=result["error"]
        )
    
    return result

# --- LOCATION ENDPOINTS ---

# Get all locations
@router.get("/", response_model=List[LocationResponse])
async def get_locations(
    current_user: Dict[str, Any] = Depends(get_current_active_user),
    warehouse_id: Optional[int] = None,
    is_occupied: Optional[bool] = None,
    skip: int = 0,
    limit: int = 100
) -> List[Dict[str, Any]]:
    """
    Get all storage locations with optional filtering.
    
    You can filter by warehouse ID and occupancy status.
    """
    locations = await WarehouseService.get_locations(
        warehouse_id=warehouse_id,
        is_occupied=is_occupied,
        skip=skip,
        limit=limit
    )
    return locations

# Get location by ID
@router.get("/{location_id}", response_model=LocationResponse)
async def get_location(
    location_id: int,
    current_user: Dict[str, Any] = Depends(get_current_active_user)
) -> Dict[str, Any]:
    """
    Get a specific storage location by ID.
    """
    location = await WarehouseService.get_location(location_id)
    
    if not location:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Location with ID {location_id} not found"
        )
    
    return location

# Create new location
@router.post("/", response_model=LocationResponse, status_code=status.HTTP_201_CREATED)
async def create_location(
    location: LocationCreate,
    current_user: Dict[str, Any] = Depends(has_role(["Manager"]))
) -> Dict[str, Any]:
    """
    Create a new storage location.
    
    Only managers can create storage locations.
    """
    created_location = await WarehouseService.create_location(location)
    
    if "error" in created_location:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=created_location["error"]
        )
    
    return created_location

# Update location
@router.put("/{location_id}", response_model=LocationResponse)
async def update_location(
    location_id: int,
    location_update: LocationUpdate,
    current_user: Dict[str, Any] = Depends(has_role(["Manager", "ReceivingClerk"]))
) -> Dict[str, Any]:
    """
    Update a storage location.
    
    Only managers and receiving clerks can update storage locations.
    """
    # Check if location exists
    location = await WarehouseService.get_location(location_id)
    if not location:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Location with ID {location_id} not found"
        )
    
    updated_location = await WarehouseService.update_location(location_id, location_update)
    return updated_location

# Delete location
@router.delete("/{location_id}", response_model=Dict[str, Any])
async def delete_location(
    location_id: int,
    current_user: Dict[str, Any] = Depends(has_role(["Manager"]))
) -> Dict[str, Any]:
    """
    Delete a storage location.
    
    Only managers can delete storage locations.
    """
    result = await WarehouseService.delete_location(location_id)
    
    if "error" in result:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=result["error"]
        )
    
    return result

# Find optimal location
@router.get("/optimal/{item_id}", response_model=Dict[str, Any])
async def find_optimal_location(
    item_id: int,
    warehouse_id: Optional[int] = None,
    current_user: Dict[str, Any] = Depends(has_role(["Manager", "ReceivingClerk", "Picker"]))
) -> Dict[str, Any]:
    """
    Find the optimal storage location for an item.
    
    This endpoint uses algorithms to determine the best storage location based on
    item properties and available space.
    """
    result = await WarehouseService.find_optimal_location(item_id, warehouse_id)
    
    if "error" in result:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=result["error"]
        )
    
    return result

# Optimize picking path
@router.post("/optimize-path", response_model=List[Dict[str, Any]])
async def optimize_picking_path(
    picking_locations: List[Dict[str, Any]],
    current_user: Dict[str, Any] = Depends(has_role(["Manager", "Picker"]))
) -> List[Dict[str, Any]]:
    """
    Optimize the path for picking items from multiple locations.
    
    This endpoint uses algorithms to determine the most efficient path
    for visiting multiple storage locations.
    """
    optimized_path = await WarehouseService.optimize_picking_path(picking_locations)
    return optimized_path