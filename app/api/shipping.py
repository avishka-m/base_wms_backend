from fastapi import APIRouter, Depends, HTTPException, Query, Body, status
from typing import List, Dict, Any, Optional
from datetime import datetime

from ..auth.dependencies import get_current_active_user, has_role
from ..models.shipping import ShippingCreate, ShippingUpdate, ShippingResponse
from ..services.workflow_service import WorkflowService
from ..utils.database import get_collection

router = APIRouter()

# Get all shipping records
@router.get("/", response_model=List[ShippingResponse])
async def get_shipping_records(
    current_user: Dict[str, Any] = Depends(get_current_active_user),
    skip: int = 0,
    limit: int = 100,
    status: Optional[str] = None,
    order_id: Optional[int] = None,
    worker_id: Optional[int] = None
) -> List[Dict[str, Any]]:
    """
    Get all shipping records with optional filtering.
    
    You can filter by status, order ID, and worker ID.
    """
    shipping_collection = get_collection("shipping")
    
    # Build query
    query = {}
    if status:
        query["status"] = status
    if order_id:
        query["orderID"] = order_id
    if worker_id:
        query["workerID"] = worker_id
    
    # Restrict drivers to only see their own shipping records
    if current_user.get("role") == "Driver":
        query["workerID"] = current_user.get("workerID")
    
    # Execute query
    shipping_records = list(shipping_collection.find(query).skip(skip).limit(limit))
    return shipping_records

# Get shipping record by ID
@router.get("/{shipping_id}", response_model=ShippingResponse)
async def get_shipping_record(
    shipping_id: int,
    current_user: Dict[str, Any] = Depends(get_current_active_user)
) -> Dict[str, Any]:
    """
    Get a specific shipping record by ID.
    """
    shipping_collection = get_collection("shipping")
    shipping = shipping_collection.find_one({"shippingID": shipping_id})
    
    if not shipping:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Shipping record with ID {shipping_id} not found"
        )
    
    # Restrict drivers to only see their own shipping records
    if current_user.get("role") == "Driver" and shipping.get("workerID") != current_user.get("workerID"):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You do not have permission to view this shipping record"
        )
    
    return shipping

# Create new shipping record
@router.post("/", response_model=ShippingResponse, status_code=status.HTTP_201_CREATED)
async def create_shipping_record(
    shipping: ShippingCreate,
    current_user: Dict[str, Any] = Depends(has_role(["Manager"]))
) -> Dict[str, Any]:
    """
    Create a new shipping record.
    
    Only managers can create shipping records.
    """
    shipping_collection = get_collection("shipping")
    orders_collection = get_collection("orders")
    
    # Check if order exists
    order = orders_collection.find_one({"orderID": shipping.orderID})
    if not order:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Order with ID {shipping.orderID} not found"
        )
    
    # Check if order is ready for shipping
    if order.get("order_status") != "packed":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Order with ID {shipping.orderID} is not ready for shipping (status: {order.get('order_status')})"
        )
    
    # Check if shipping record already exists for this order
    existing_shipping = shipping_collection.find_one({"orderID": shipping.orderID})
    if existing_shipping:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Shipping record already exists for order with ID {shipping.orderID}"
        )
    
    # Find the next available shippingID
    last_shipping = shipping_collection.find_one(
        sort=[("shippingID", -1)]
    )
    next_id = 1
    if last_shipping:
        next_id = last_shipping.get("shippingID", 0) + 1
    
    # Prepare shipping document
    shipping_data = shipping.model_dump()
    shipping_data.update({
        "shippingID": next_id,
        "created_at": shipping_data.get("ship_date"),
        "updated_at": shipping_data.get("ship_date"),
        "delivery_address": order.get("shipping_address"),
        "recipient_name": order.get("customerID"),  # In a real system, you would get the customer name
        "recipient_phone": None  # In a real system, you would get the customer phone
    })
    
    # Insert shipping to database
    result = shipping_collection.insert_one(shipping_data)
    
    # Return the created shipping record
    created_shipping = shipping_collection.find_one({"_id": result.inserted_id})
    return created_shipping

# Update shipping record
@router.put("/{shipping_id}", response_model=ShippingResponse)
async def update_shipping_record(
    shipping_id: int,
    shipping_update: ShippingUpdate,
    current_user: Dict[str, Any] = Depends(has_role(["Manager", "Driver"]))
) -> Dict[str, Any]:
    """
    Update a shipping record.
    
    Managers can update any shipping record. Drivers can only update their own records.
    """
    shipping_collection = get_collection("shipping")
    
    # Check if shipping record exists
    shipping = shipping_collection.find_one({"shippingID": shipping_id})
    if not shipping:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Shipping record with ID {shipping_id} not found"
        )
    
    # Restrict drivers to only update their own shipping records
    if current_user.get("role") == "Driver" and shipping.get("workerID") != current_user.get("workerID"):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You do not have permission to update this shipping record"
        )
    
    # Check if already delivered
    if shipping.get("status") == "delivered":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Cannot update delivered shipping record"
        )
    
    # Prepare update data
    update_data = shipping_update.model_dump(exclude_unset=True)
    update_data["updated_at"] = shipping_update.model_dump().get("updated_at")
    
    # Update shipping record
    shipping_collection.update_one(
        {"shippingID": shipping_id},
        {"$set": update_data}
    )
    
    # Return updated shipping record
    updated_shipping = shipping_collection.find_one({"shippingID": shipping_id})
    return updated_shipping

# Process shipping (Assign vehicle and dispatch)
@router.post("/{shipping_id}/dispatch", response_model=ShippingResponse)
async def dispatch_shipping(
    shipping_id: int,
    vehicle_id: int = Query(..., description="ID of the vehicle to use for shipping"),
    tracking_info: Dict[str, Any] = Body(..., description="Tracking information for the shipment"),
    current_user: Dict[str, Any] = Depends(has_role(["Driver"]))
) -> Dict[str, Any]:
    """
    Dispatch a shipment.
    
    This assigns a vehicle, updates tracking info, and changes status to "in_transit".
    """
    result = await WorkflowService.process_shipping(
        shipping_id=shipping_id,
        worker_id=current_user.get("workerID"),
        vehicle_id=vehicle_id,
        tracking_info=tracking_info
    )
    
    if "error" in result:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=result["error"]
        )
    
    return result

# Mark shipping as delivered
@router.post("/{shipping_id}/deliver", response_model=ShippingResponse)
async def deliver_shipping(
    shipping_id: int,
    delivery_proof: str = Query(..., description="Proof of delivery (signature, photo, etc.)"),
    notes: Optional[str] = Query(None, description="Optional notes about the delivery"),
    current_user: Dict[str, Any] = Depends(has_role(["Driver"]))
) -> Dict[str, Any]:
    """
    Mark a shipment as delivered.
    
    This updates the shipping record with delivery details and changes status to "delivered".
    """
    result = await WorkflowService.complete_shipping(
        shipping_id=shipping_id,
        worker_id=current_user.get("workerID"),
        delivery_proof=delivery_proof,
        notes=notes
    )
    
    if "error" in result:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=result["error"]
        )
    
    return result

# Get tracking information
@router.get("/{shipping_id}/tracking", response_model=Dict[str, Any])
async def get_tracking_info(
    shipping_id: int,
    current_user: Dict[str, Any] = Depends(get_current_active_user)
) -> Dict[str, Any]:
    """
    Get tracking information for a shipment.
    
    This endpoint provides detailed tracking information including status updates and ETA.
    """
    shipping_collection = get_collection("shipping")
    shipping = shipping_collection.find_one({"shippingID": shipping_id})
    
    if not shipping:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Shipping record with ID {shipping_id} not found"
        )
    
    # Compile tracking information
    tracking_info = {
        "shippingID": shipping_id,
        "orderID": shipping.get("orderID"),
        "status": shipping.get("status"),
        "tracking_number": shipping.get("tracking_number"),
        "shipped_at": shipping.get("departure_time"),
        "estimated_delivery": shipping.get("estimated_delivery"),
        "delivery_address": shipping.get("delivery_address"),
        "tracking_history": []  # In a real system, this would include status updates
    }
    
    # If delivered, add delivery information
    if shipping.get("status") == "delivered":
        tracking_info["delivered_at"] = shipping.get("actual_delivery")
        tracking_info["delivery_proof"] = shipping.get("delivery_proof")
    
    return tracking_info