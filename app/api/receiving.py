from fastapi import APIRouter, Depends, HTTPException, Query, Body, status
from typing import List, Dict, Any, Optional

from ..auth.dependencies import get_current_active_user, has_role
from ..models.receiving import ReceivingCreate, ReceivingUpdate, ReceivingResponse
from ..services.workflow_service import WorkflowService
from ..utils.database import get_collection

router = APIRouter()

# Get all receiving records
@router.get("/", response_model=List[ReceivingResponse])
async def get_receiving_records(
    current_user: Dict[str, Any] = Depends(get_current_active_user),
    skip: int = 0,
    limit: int = 100,
    status: Optional[str] = None,
    supplier_id: Optional[int] = None
) -> List[Dict[str, Any]]:
    """
    Get all receiving records with optional filtering.
    
    You can filter by status and supplier ID.
    """
    receiving_collection = get_collection("receiving")
    
    # Build query
    query = {}
    if status:
        query["status"] = status
    if supplier_id:
        query["supplierID"] = supplier_id
    
    # Execute query
    receiving_records = list(receiving_collection.find(query).skip(skip).limit(limit))
    return receiving_records

# Get receiving record by ID
@router.get("/{receiving_id}", response_model=ReceivingResponse)
async def get_receiving_record(
    receiving_id: int,
    current_user: Dict[str, Any] = Depends(get_current_active_user)
) -> Dict[str, Any]:
    """
    Get a specific receiving record by ID.
    """
    receiving_collection = get_collection("receiving")
    receiving = receiving_collection.find_one({"receivingID": receiving_id})
    
    if not receiving:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Receiving record with ID {receiving_id} not found"
        )
    
    return receiving

# Create new receiving record
@router.post("/", response_model=ReceivingResponse, status_code=status.HTTP_201_CREATED)
async def create_receiving_record(
    receiving: ReceivingCreate,
    current_user: Dict[str, Any] = Depends(has_role(["Manager", "ReceivingClerk"]))
) -> Dict[str, Any]:
    """
    Create a new receiving record.
    
    Only managers and receiving clerks can create receiving records.
    """
    receiving_collection = get_collection("receiving")
    
    # Find the next available receivingID
    last_receiving = receiving_collection.find_one(
        sort=[("receivingID", -1)]
    )
    next_id = 1
    if last_receiving:
        next_id = last_receiving.get("receivingID", 0) + 1
    
    # Prepare receiving document
    receiving_data = receiving.model_dump()
    receiving_data.update({
        "receivingID": next_id,
        "created_at": receiving_data.get("received_date"),
        "updated_at": receiving_data.get("received_date")
    })
    
    # Initialize items with processed=False
    for item in receiving_data.get("items", []):
        item["processed"] = False
    
    # Insert receiving to database
    result = receiving_collection.insert_one(receiving_data)
    
    # Return the created receiving record
    created_receiving = receiving_collection.find_one({"_id": result.inserted_id})
    return created_receiving

# Update receiving record
@router.put("/{receiving_id}", response_model=ReceivingResponse)
async def update_receiving_record(
    receiving_id: int,
    receiving_update: ReceivingUpdate,
    current_user: Dict[str, Any] = Depends(has_role(["Manager", "ReceivingClerk"]))
) -> Dict[str, Any]:
    """
    Update a receiving record.
    
    Only managers and receiving clerks can update receiving records.
    """
    receiving_collection = get_collection("receiving")
    
    # Check if receiving record exists
    receiving = receiving_collection.find_one({"receivingID": receiving_id})
    if not receiving:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Receiving record with ID {receiving_id} not found"
        )
    
    # Check if already completed
    if receiving.get("status") == "completed" and receiving_update.status != "completed":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Cannot update completed receiving record"
        )
    
    # Prepare update data
    update_data = receiving_update.model_dump(exclude_unset=True)
    update_data["updated_at"] = receiving_update.model_dump().get("updated_at")
    
    # Update receiving record
    receiving_collection.update_one(
        {"receivingID": receiving_id},
        {"$set": update_data}
    )
    
    # Return updated receiving record
    updated_receiving = receiving_collection.find_one({"receivingID": receiving_id})
    return updated_receiving

# Process receiving
@router.post("/{receiving_id}/process", response_model=ReceivingResponse)
async def process_receiving(
    receiving_id: int,
    current_user: Dict[str, Any] = Depends(has_role(["Manager", "ReceivingClerk"]))
) -> Dict[str, Any]:
    """
    Process a receiving record and update inventory.
    
    This endpoint handles the complete receiving workflow:
    1. Validate the receiving request
    2. Update inventory with received items
    3. Complete the receiving process
    """
    result = await WorkflowService.process_receiving(
        receiving_id=receiving_id,
        worker_id=current_user.get("workerID")
    )
    
    if "error" in result:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=result["error"]
        )
    
    return result