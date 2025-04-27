from fastapi import APIRouter, Depends, HTTPException, Query, Body, status
from typing import List, Dict, Any, Optional

from ..auth.dependencies import get_current_active_user, has_role
from ..models.returns import ReturnsCreate, ReturnsUpdate, ReturnsResponse
from ..services.workflow_service import WorkflowService
from ..utils.database import get_collection

router = APIRouter()

# Get all returns records
@router.get("/", response_model=List[ReturnsResponse])
async def get_returns_records(
    current_user: Dict[str, Any] = Depends(get_current_active_user),
    skip: int = 0,
    limit: int = 100,
    status: Optional[str] = None,
    order_id: Optional[int] = None,
    customer_id: Optional[int] = None
) -> List[Dict[str, Any]]:
    """
    Get all returns records with optional filtering.
    
    You can filter by status, order ID, and customer ID.
    """
    returns_collection = get_collection("returns")
    
    # Build query
    query = {}
    if status:
        query["status"] = status
    if order_id:
        query["orderID"] = order_id
    if customer_id:
        query["customerID"] = customer_id
    
    # Restrict customers to only see their own returns
    if current_user.get("role") == "Customer":
        query["customerID"] = current_user.get("customerID")
    
    # Execute query
    returns_records = list(returns_collection.find(query).skip(skip).limit(limit))
    return returns_records

# Get returns record by ID
@router.get("/{return_id}", response_model=ReturnsResponse)
async def get_returns_record(
    return_id: int,
    current_user: Dict[str, Any] = Depends(get_current_active_user)
) -> Dict[str, Any]:
    """
    Get a specific returns record by ID.
    """
    returns_collection = get_collection("returns")
    returns = returns_collection.find_one({"returnID": return_id})
    
    if not returns:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Returns record with ID {return_id} not found"
        )
    
    # Restrict customers to only see their own returns
    if current_user.get("role") == "Customer" and returns.get("customerID") != current_user.get("customerID"):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You do not have permission to view this returns record"
        )
    
    return returns

# Create new returns record
@router.post("/", response_model=ReturnsResponse, status_code=status.HTTP_201_CREATED)
async def create_returns_record(
    returns: ReturnsCreate,
    current_user: Dict[str, Any] = Depends(get_current_active_user)
) -> Dict[str, Any]:
    """
    Create a new returns record.
    
    Customers can create returns for their own orders. Staff can create returns for any order.
    """
    returns_collection = get_collection("returns")
    orders_collection = get_collection("orders")
    
    # Check if order exists
    order = orders_collection.find_one({"orderID": returns.orderID})
    if not order:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Order with ID {returns.orderID} not found"
        )
    
    # Restrict customers to only create returns for their own orders
    if current_user.get("role") == "Customer" and order.get("customerID") != current_user.get("customerID"):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You can only create returns for your own orders"
        )
    
    # Check order status - can only return delivered orders
    if order.get("order_status") != "delivered":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Order with ID {returns.orderID} is not eligible for return (status: {order.get('order_status')})"
        )
    
    # Find the next available returnID
    last_return = returns_collection.find_one(
        sort=[("returnID", -1)]
    )
    next_id = 1
    if last_return:
        next_id = last_return.get("returnID", 0) + 1
    
    # Prepare returns document
    returns_data = returns.model_dump()
    returns_data.update({
        "returnID": next_id,
        "created_at": returns_data.get("return_date"),
        "updated_at": returns_data.get("return_date")
    })
    
    # Initialize items with processed=False, resellable=False
    for item in returns_data.get("items", []):
        item["processed"] = False
        item["resellable"] = False
        item["locationID"] = None
    
    # Insert returns to database
    result = returns_collection.insert_one(returns_data)
    
    # Return the created returns record
    created_returns = returns_collection.find_one({"_id": result.inserted_id})
    return created_returns

# Update returns record
@router.put("/{return_id}", response_model=ReturnsResponse)
async def update_returns_record(
    return_id: int,
    returns_update: ReturnsUpdate,
    current_user: Dict[str, Any] = Depends(has_role(["Manager", "ReceivingClerk"]))
) -> Dict[str, Any]:
    """
    Update a returns record.
    
    Only managers and receiving clerks can update returns records.
    """
    returns_collection = get_collection("returns")
    
    # Check if returns record exists
    returns = returns_collection.find_one({"returnID": return_id})
    if not returns:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Returns record with ID {return_id} not found"
        )
    
    # Check if already completed
    if returns.get("status") == "completed":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Cannot update completed returns record"
        )
    
    # Prepare update data
    update_data = returns_update.model_dump(exclude_unset=True)
    update_data["updated_at"] = returns_update.model_dump().get("updated_at")
    
    # Update returns record
    returns_collection.update_one(
        {"returnID": return_id},
        {"$set": update_data}
    )
    
    # Return updated returns record
    updated_returns = returns_collection.find_one({"returnID": return_id})
    return updated_returns

# Process returns
@router.post("/{return_id}/process", response_model=ReturnsResponse)
async def process_returns(
    return_id: int,
    current_user: Dict[str, Any] = Depends(has_role(["ReceivingClerk"]))
) -> Dict[str, Any]:
    """
    Start processing a returns request.
    
    This updates the returns record status to "processing".
    """
    result = await WorkflowService.process_return(
        return_id=return_id,
        worker_id=current_user.get("workerID")
    )
    
    if "error" in result:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=result["error"]
        )
    
    return result

# Complete returns processing
@router.post("/{return_id}/complete", response_model=ReturnsResponse)
async def complete_returns(
    return_id: int,
    processed_items: List[Dict[str, Any]] = Body(..., description="List of processed return items with their status"),
    refund_details: Optional[Dict[str, Any]] = Body(None, description="Optional refund details"),
    current_user: Dict[str, Any] = Depends(has_role(["ReceivingClerk"]))
) -> Dict[str, Any]:
    """
    Complete a returns process.
    
    This finalizes the returns processing, updates inventory for resellable items,
    and processes refunds if applicable.
    """
    result = await WorkflowService.complete_return(
        return_id=return_id,
        worker_id=current_user.get("workerID"),
        processed_items=processed_items,
        refund_details=refund_details
    )
    
    if "error" in result:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=result["error"]
        )
    
    return result

# Process refund
@router.post("/{return_id}/refund", response_model=Dict[str, Any])
async def process_refund(
    return_id: int,
    refund_amount: float = Query(..., description="Amount to refund"),
    refund_status: str = Query("processed", description="Status of the refund"),
    current_user: Dict[str, Any] = Depends(has_role(["Manager"]))
) -> Dict[str, Any]:
    """
    Process a refund for a return.
    
    Only managers can process refunds.
    """
    returns_collection = get_collection("returns")
    
    # Check if returns record exists
    returns = returns_collection.find_one({"returnID": return_id})
    if not returns:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Returns record with ID {return_id} not found"
        )
    
    # Check if already refunded
    if returns.get("refund_status") == "processed":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Refund has already been processed for return with ID {return_id}"
        )
    
    # Update refund information
    returns_collection.update_one(
        {"returnID": return_id},
        {
            "$set": {
                "refund_amount": refund_amount,
                "refund_status": refund_status,
                "refund_date": returns.get("updated_at"),
                "updated_at": returns.get("updated_at")
            }
        }
    )
    
    # In a real system, you would integrate with a payment gateway here
    
    return {
        "message": f"Refund of {refund_amount} has been processed for return with ID {return_id}",
        "returnID": return_id,
        "refund_amount": refund_amount,
        "refund_status": refund_status,
        "status": "success"
    }