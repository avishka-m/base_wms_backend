from fastapi import APIRouter, Depends, HTTPException, Query, Body, status
from typing import List, Dict, Any, Optional

from ..auth.dependencies import get_current_active_user, has_role
from ..models.packing import PackingCreate, PackingUpdate, PackingResponse
from ..services.workflow_service import WorkflowService
from ..utils.database import get_collection

router = APIRouter()

# Get all packing records
@router.get("/", response_model=List[PackingResponse])
async def get_packing_records(
    current_user: Dict[str, Any] = Depends(get_current_active_user),
    skip: int = 0,
    limit: int = 100,
    status: Optional[str] = None,
    order_id: Optional[int] = None,
    worker_id: Optional[int] = None
) -> List[Dict[str, Any]]:
    """
    Get all packing records with optional filtering.
    
    You can filter by status, order ID, and worker ID.
    """
    packing_collection = get_collection("packing")
    
    # Build query
    query = {}
    if status:
        query["status"] = status
    if order_id:
        query["orderID"] = order_id
    if worker_id:
        query["workerID"] = worker_id
    
    # Restrict packers to only see their own packing records
    if current_user.get("role") == "Packer":
        query["workerID"] = current_user.get("workerID")
    
    # Execute query
    packing_records = list(packing_collection.find(query).skip(skip).limit(limit))
    return packing_records

# Get packing record by ID
@router.get("/{packing_id}", response_model=PackingResponse)
async def get_packing_record(
    packing_id: int,
    current_user: Dict[str, Any] = Depends(get_current_active_user)
) -> Dict[str, Any]:
    """
    Get a specific packing record by ID.
    """
    packing_collection = get_collection("packing")
    packing = packing_collection.find_one({"packingID": packing_id})
    
    if not packing:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Packing record with ID {packing_id} not found"
        )
    
    # Restrict packers to only see their own packing records
    if current_user.get("role") == "Packer" and packing.get("workerID") != current_user.get("workerID"):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You do not have permission to view this packing record"
        )
    
    return packing

# Create new packing record
@router.post("/", response_model=PackingResponse, status_code=status.HTTP_201_CREATED)
async def create_packing_record(
    packing: PackingCreate,
    current_user: Dict[str, Any] = Depends(has_role(["Manager"]))
) -> Dict[str, Any]:
    """
    Create a new packing record.
    
    Only managers can create packing records.
    """
    packing_collection = get_collection("packing")
    
    # Find the next available packingID
    last_packing = packing_collection.find_one(
        sort=[("packingID", -1)]
    )
    next_id = 1
    if last_packing:
        next_id = last_packing.get("packingID", 0) + 1
    
    # Prepare packing document
    packing_data = packing.model_dump()
    packing_data.update({
        "packingID": next_id,
        "created_at": packing_data.get("pack_date"),
        "updated_at": packing_data.get("pack_date")
    })
    
    # Initialize items with packed=False
    for item in packing_data.get("items", []):
        item["packed"] = False
        item["actual_quantity"] = None
    
    # Insert packing to database
    result = packing_collection.insert_one(packing_data)
    
    # Return the created packing record
    created_packing = packing_collection.find_one({"_id": result.inserted_id})
    return created_packing

# Update packing record
@router.put("/{packing_id}", response_model=PackingResponse)
async def update_packing_record(
    packing_id: int,
    packing_update: PackingUpdate,
    current_user: Dict[str, Any] = Depends(has_role(["Manager", "Packer"]))
) -> Dict[str, Any]:
    """
    Update a packing record.
    
    Managers can update any packing record. Packers can only update their own records.
    """
    packing_collection = get_collection("packing")
    
    # Check if packing record exists
    packing = packing_collection.find_one({"packingID": packing_id})
    if not packing:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Packing record with ID {packing_id} not found"
        )
    
    # Restrict packers to only update their own packing records
    if current_user.get("role") == "Packer" and packing.get("workerID") != current_user.get("workerID"):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You do not have permission to update this packing record"
        )
    
    # Check if already completed
    if packing.get("status") == "completed":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Cannot update completed packing record"
        )
    
    # Prepare update data
    update_data = packing_update.model_dump(exclude_unset=True)
    update_data["updated_at"] = packing_update.model_dump().get("updated_at")
    
    # Update packing record
    packing_collection.update_one(
        {"packingID": packing_id},
        {"$set": update_data}
    )
    
    # Return updated packing record
    updated_packing = packing_collection.find_one({"packingID": packing_id})
    return updated_packing

# Start packing process
@router.post("/{packing_id}/start", response_model=PackingResponse)
async def start_packing(
    packing_id: int,
    current_user: Dict[str, Any] = Depends(has_role(["Packer"]))
) -> Dict[str, Any]:
    """
    Start a packing process.
    
    This updates the packing record status to "in_progress" and sets the start time.
    """
    result = await WorkflowService.process_packing(
        packing_id=packing_id,
        worker_id=current_user.get("workerID")
    )
    
    if "error" in result:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=result["error"]
        )
    
    return result

# Complete packing process
@router.post("/{packing_id}/complete", response_model=PackingResponse)
async def complete_packing(
    packing_id: int,
    packed_items: List[Dict[str, Any]] = Body(..., description="List of items packed with their quantities"),
    package_details: Dict[str, Any] = Body(..., description="Details about the package"),
    current_user: Dict[str, Any] = Depends(has_role(["Packer"]))
) -> Dict[str, Any]:
    """
    Complete a packing process.
    
    This finalizes the packing with the actual packed quantities and package details.
    """
    result = await WorkflowService.complete_packing(
        packing_id=packing_id,
        worker_id=current_user.get("workerID"),
        packed_items=packed_items,
        package_details=package_details
    )
    
    if "error" in result:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=result["error"]
        )
    
    return result

# Print shipping label
@router.post("/{packing_id}/label", response_model=Dict[str, Any])
async def print_shipping_label(
    packing_id: int,
    current_user: Dict[str, Any] = Depends(has_role(["Packer", "Manager"]))
) -> Dict[str, Any]:
    """
    Print a shipping label for a packed order.
    
    This endpoint would typically integrate with a label printing system.
    """
    packing_collection = get_collection("packing")
    
    # Check if packing record exists
    packing = packing_collection.find_one({"packingID": packing_id})
    if not packing:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Packing record with ID {packing_id} not found"
        )
    
    # Check if already completed
    if packing.get("status") != "completed":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Cannot print label for incomplete packing"
        )
    
    # Update label printed status
    packing_collection.update_one(
        {"packingID": packing_id},
        {"$set": {"label_printed": True, "updated_at": packing.get("updated_at")}}
    )
    
    # In a real system, you would integrate with a label printing service here
    
    return {
        "message": f"Shipping label for packing ID {packing_id} has been printed",
        "packingID": packing_id,
        "orderID": packing.get("orderID"),
        "status": "success"
    }