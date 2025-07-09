from fastapi import APIRouter, Depends, HTTPException, Query, Body, status
from typing import List, Dict, Any, Optional

from ..auth.dependencies import get_current_active_user, has_role
from ..models.receiving import ReceivingCreate, ReceivingUpdate, ReceivingResponse
from ..services.workflow_service import WorkflowService
from ..utils.database import get_collection

router = APIRouter()

# Get items by status endpoint
@router.get("/items/by-status")
async def get_items_by_status(
    current_user: Dict[str, Any] = Depends(get_current_active_user)
) -> Dict[str, Any]:
    """
    Get items grouped by their status for picker dashboard.
    Returns items available for storing and items available for picking.
    """
    try:
        receiving_collection = get_collection("receiving_items")
        inventory_collection = get_collection("inventory")
        
        # Get items available for storing (status = 'received')
        storing_items = list(receiving_collection.find({"status": "received"}))
        
        # Get items available for picking (from inventory with status = 'stored')
        picking_items = list(inventory_collection.find({"status": "stored"}))
        
        # Convert ObjectId to string for JSON serialization
        for item in storing_items:
            if "_id" in item:
                item["_id"] = str(item["_id"])
        
        for item in picking_items:
            if "_id" in item:
                item["_id"] = str(item["_id"])
        
        return {
            "available_for_storing": storing_items,
            "available_for_picking": picking_items
        }
    except Exception as e:
        print(f"Error in get_items_by_status: {str(e)}")
        return {
            "available_for_storing": [],
            "available_for_picking": []
        }

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

# Get items by storage status
@router.get("/items/by-status", response_model=Dict[str, List[Dict[str, Any]]])
async def get_items_by_storage_status(
    current_user: Dict[str, Any] = Depends(has_role(["Manager", "Picker", "ReceivingClerk"]))
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Get items grouped by their storage status.
    
    Returns:
    - available_for_storing: Items that have been received but not yet stored (processed=False, locationID=None)
    - available_for_picking: Items that have been stored and are ready for picking (processed=True, locationID exists)
    """
    receiving_collection = get_collection("receiving")
    inventory_collection = get_collection("inventory")
    
    # Get all receiving records with pending or processing status
    receiving_records = list(receiving_collection.find({
        "status": {"$in": ["pending", "processing"]}
    }))
    
    available_for_storing = []
    available_for_picking = []
    
    # Collect items available for storing
    for record in receiving_records:
        for item in record.get("items", []):
            if not item.get("processed", False) and item.get("locationID") is None:
                # Get item details from inventory
                inventory_item = inventory_collection.find_one({"itemID": item["itemID"]})
                if inventory_item:
                    available_for_storing.append({
                        "receivingID": record["receivingID"],
                        "itemID": item["itemID"],
                        "itemName": inventory_item.get("name", "Unknown"),
                        "quantity": item["quantity"],
                        "condition": item.get("condition", "good"),
                        "receivedDate": record.get("received_date"),
                        "supplierID": record.get("supplierID"),
                        "notes": item.get("notes", "")
                    })
    
    # Get completed receiving records for items available for picking
    completed_records = list(receiving_collection.find({
        "status": "completed"
    }))
    
    for record in completed_records:
        for item in record.get("items", []):
            if item.get("processed", False) and item.get("locationID") is not None:
                # Get item details from inventory
                inventory_item = inventory_collection.find_one({"itemID": item["itemID"]})
                if inventory_item and inventory_item.get("stock_level", 0) > 0:
                    available_for_picking.append({
                        "itemID": item["itemID"],
                        "itemName": inventory_item.get("name", "Unknown"),
                        "stockLevel": inventory_item.get("stock_level", 0),
                        "locationID": item["locationID"],
                        "category": inventory_item.get("category", ""),
                        "size": inventory_item.get("size", "")
                    })
    
    # Also add items from inventory that have stock and location
    inventory_items = list(inventory_collection.find({
        "stock_level": {"$gt": 0},
        "locationID": {"$ne": None}
    }))
    
    for item in inventory_items:
        # Check if not already in the list
        if not any(p["itemID"] == item["itemID"] for p in available_for_picking):
            available_for_picking.append({
                "itemID": item["itemID"],
                "itemName": item.get("name", "Unknown"),
                "stockLevel": item.get("stock_level", 0),
                "locationID": item.get("locationID"),
                "category": item.get("category", ""),
                "size": item.get("size", "")
            })
    
    return {
        "available_for_storing": available_for_storing,
        "available_for_picking": available_for_picking
    }

# Mark item as stored
@router.post("/{receiving_id}/items/{item_id}/store")
async def mark_item_as_stored(
    receiving_id: int,
    item_id: int,
    location_id: str = Body(..., embed=True),
    current_user: Dict[str, Any] = Depends(has_role(["Manager", "Picker", "ReceivingClerk"]))
) -> Dict[str, Any]:
    """
    Mark a received item as stored by assigning it a location.
    """
    receiving_collection = get_collection("receiving")
    
    # Find the receiving record
    receiving = receiving_collection.find_one({"receivingID": receiving_id})
    if not receiving:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Receiving record with ID {receiving_id} not found"
        )
    
    # Find and update the specific item
    item_found = False
    for idx, item in enumerate(receiving.get("items", [])):
        if item["itemID"] == item_id:
            item_found = True
            # Update the item
            receiving_collection.update_one(
                {"receivingID": receiving_id},
                {
                    "$set": {
                        f"items.{idx}.processed": True,
                        f"items.{idx}.locationID": location_id
                    }
                }
            )
            break
    
    if not item_found:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Item with ID {item_id} not found in receiving record {receiving_id}"
        )
    
    # Check if all items are processed and update status if needed
    updated_receiving = receiving_collection.find_one({"receivingID": receiving_id})
    all_processed = all(item.get("processed", False) for item in updated_receiving.get("items", []))
    
    if all_processed and updated_receiving.get("status") != "completed":
        receiving_collection.update_one(
            {"receivingID": receiving_id},
            {"$set": {"status": "completed"}}
        )
    
    return {"message": f"Item {item_id} marked as stored in location {location_id}"}