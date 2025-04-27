from fastapi import APIRouter, Depends, HTTPException, Query, Body, status
from typing import List, Dict, Any, Optional

from ..auth.dependencies import get_current_active_user, has_role
from ..models.picking import PickingCreate, PickingUpdate, PickingResponse
from ..services.workflow_service import WorkflowService
from ..utils.database import get_collection

router = APIRouter()

# Get all picking records
@router.get("/", response_model=List[PickingResponse])
async def get_picking_records(
    current_user: Dict[str, Any] = Depends(get_current_active_user),
    skip: int = 0,
    limit: int = 100,
    status: Optional[str] = None,
    order_id: Optional[int] = None,
    worker_id: Optional[int] = None
) -> List[Dict[str, Any]]:
    """
    Get all picking records with optional filtering.
    
    You can filter by status, order ID, and worker ID.
    """
    picking_collection = get_collection("picking")
    
    # Build query
    query = {}
    if status:
        query["status"] = status
    if order_id:
        query["orderID"] = order_id
    if worker_id:
        query["workerID"] = worker_id
    
    # Restrict pickers to only see their own picking records
    if current_user.get("role") == "Picker":
        query["workerID"] = current_user.get("workerID")
    
    # Execute query with sorting by priority
    picking_records = list(picking_collection.find(query).sort([("priority", 1)]).skip(skip).limit(limit))
    return picking_records

# Get picking record by ID
@router.get("/{picking_id}", response_model=PickingResponse)
async def get_picking_record(
    picking_id: int,
    current_user: Dict[str, Any] = Depends(get_current_active_user)
) -> Dict[str, Any]:
    """
    Get a specific picking record by ID.
    """
    picking_collection = get_collection("picking")
    picking = picking_collection.find_one({"pickingID": picking_id})
    
    if not picking:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Picking record with ID {picking_id} not found"
        )
    
    # Restrict pickers to only see their own picking records
    if current_user.get("role") == "Picker" and picking.get("workerID") != current_user.get("workerID"):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You do not have permission to view this picking record"
        )
    
    return picking

# Create new picking record
@router.post("/", response_model=PickingResponse, status_code=status.HTTP_201_CREATED)
async def create_picking_record(
    picking: PickingCreate,
    current_user: Dict[str, Any] = Depends(has_role(["Manager"]))
) -> Dict[str, Any]:
    """
    Create a new picking record.
    
    Only managers can create picking records.
    """
    picking_collection = get_collection("picking")
    
    # Find the next available pickingID
    last_picking = picking_collection.find_one(
        sort=[("pickingID", -1)]
    )
    next_id = 1
    if last_picking:
        next_id = last_picking.get("pickingID", 0) + 1
    
    # Prepare picking document
    picking_data = picking.model_dump()
    picking_data.update({
        "pickingID": next_id,
        "created_at": picking_data.get("pick_date"),
        "updated_at": picking_data.get("pick_date")
    })
    
    # Initialize items with picked=False
    for item in picking_data.get("items", []):
        item["picked"] = False
        item["actual_quantity"] = None
    
    # Insert picking to database
    result = picking_collection.insert_one(picking_data)
    
    # Return the created picking record
    created_picking = picking_collection.find_one({"_id": result.inserted_id})
    return created_picking

# Update picking record
@router.put("/{picking_id}", response_model=PickingResponse)
async def update_picking_record(
    picking_id: int,
    picking_update: PickingUpdate,
    current_user: Dict[str, Any] = Depends(has_role(["Manager", "Picker"]))
) -> Dict[str, Any]:
    """
    Update a picking record.
    
    Managers can update any picking record. Pickers can only update their own records.
    """
    picking_collection = get_collection("picking")
    
    # Check if picking record exists
    picking = picking_collection.find_one({"pickingID": picking_id})
    if not picking:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Picking record with ID {picking_id} not found"
        )
    
    # Restrict pickers to only update their own picking records
    if current_user.get("role") == "Picker" and picking.get("workerID") != current_user.get("workerID"):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You do not have permission to update this picking record"
        )
    
    # Check if already completed
    if picking.get("status") == "completed":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Cannot update completed picking record"
        )
    
    # Prepare update data
    update_data = picking_update.model_dump(exclude_unset=True)
    update_data["updated_at"] = picking_update.model_dump().get("updated_at")
    
    # Update picking record
    picking_collection.update_one(
        {"pickingID": picking_id},
        {"$set": update_data}
    )
    
    # Return updated picking record
    updated_picking = picking_collection.find_one({"pickingID": picking_id})
    return updated_picking

# Start picking process
@router.post("/{picking_id}/start", response_model=PickingResponse)
async def start_picking(
    picking_id: int,
    current_user: Dict[str, Any] = Depends(has_role(["Picker"]))
) -> Dict[str, Any]:
    """
    Start a picking process.
    
    This updates the picking record status to "in_progress" and sets the start time.
    """
    result = await WorkflowService.process_picking(
        picking_id=picking_id,
        worker_id=current_user.get("workerID")
    )
    
    if "error" in result:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=result["error"]
        )
    
    return result

# Complete picking process
@router.post("/{picking_id}/complete", response_model=PickingResponse)
async def complete_picking(
    picking_id: int,
    picked_items: List[Dict[str, Any]] = Body(..., description="List of items picked with their quantities"),
    current_user: Dict[str, Any] = Depends(has_role(["Picker"]))
) -> Dict[str, Any]:
    """
    Complete a picking process.
    
    This updates the inventory and marks the picking as completed.
    """
    result = await WorkflowService.complete_picking(
        picking_id=picking_id,
        worker_id=current_user.get("workerID"),
        picked_items=picked_items
    )
    
    if "error" in result:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=result["error"]
        )
    
    return result