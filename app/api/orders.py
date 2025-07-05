from fastapi import APIRouter, Depends, HTTPException, Query, status
from typing import List, Dict, Any, Optional

from ..auth.dependencies import get_current_active_user, has_role
from ..models.order import OrderCreate, OrderUpdate, OrderResponse
from ..services.orders_service import OrdersService
from ..services.workflow_service import WorkflowService
from ..services.websocket_service import websocket_manager

router = APIRouter()

# Get all orders
@router.get("/", response_model=List[OrderResponse])
async def get_orders(
    current_user: Dict[str, Any] = Depends(get_current_active_user),
    skip: int = 0,
    limit: int = 100,
    status: Optional[str] = None,
    customer_id: Optional[int] = None
) -> List[Dict[str, Any]]:
    """
    Get all orders with optional filtering.
    
    You can filter by status and customer ID.
    """
    # Restrict customers to only see their own orders
    if current_user.get("role") == "Customer":
        customer_id = current_user.get("customerID")
    
    orders = await OrdersService.get_orders(
        skip=skip,
        limit=limit,
        status=status,
        customer_id=customer_id
    )
    return orders

# Get order by ID
@router.get("/{order_id}", response_model=OrderResponse)
async def get_order(
    order_id: int,
    current_user: Dict[str, Any] = Depends(get_current_active_user)
) -> Dict[str, Any]:
    """
    Get a specific order by ID.
    """
    order = await OrdersService.get_order(order_id)
    
    if not order:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Order with ID {order_id} not found"
        )
    
    # Restrict customers to only see their own orders
    if current_user.get("role") == "Customer" and order.get("customerID") != current_user.get("customerID"):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You do not have permission to view this order"
        )
    
    return order

# Create new order
@router.post("/", response_model=OrderResponse, status_code=status.HTTP_201_CREATED)
async def create_order(
    order: OrderCreate,
    current_user: Dict[str, Any] = Depends(get_current_active_user)
) -> Dict[str, Any]:
    """
    Create a new order.
    
    Customers can create orders for themselves, while staff can create orders for any customer.
    """
    # Restrict customers to only create orders for themselves
    if current_user.get("role") == "Customer" and order.customerID != current_user.get("customerID"):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You can only create orders for yourself"
        )
    
    created_order = await OrdersService.create_order(order)
    
    if "error" in created_order:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=created_order["error"]
        )
    
    return created_order

# Update order
@router.put("/{order_id}", response_model=OrderResponse)
async def update_order(
    order_id: int,
    order_update: OrderUpdate,
    current_user: Dict[str, Any] = Depends(has_role(["Manager", "Picker", "Packer", "Driver"]))
) -> Dict[str, Any]:
    """
    Update an order.
    
    Only warehouse staff can update orders.
    """
    # Check if order exists
    order = await OrdersService.get_order(order_id)
    if not order:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Order with ID {order_id} not found"
        )
    
    updated_order = await OrdersService.update_order(order_id, order_update)
    return updated_order

# Delete order
@router.delete("/{order_id}", response_model=Dict[str, Any])
async def delete_order(
    order_id: int,
    current_user: Dict[str, Any] = Depends(has_role(["Manager"]))
) -> Dict[str, Any]:
    """
    Delete an order.
    
    Only managers can delete orders.
    """
    # Check if order exists
    order = await OrdersService.get_order(order_id)
    if not order:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Order with ID {order_id} not found"
        )
    
    result = await OrdersService.delete_order(order_id)
    return result

# Update order status
@router.put("/{order_id}/status", response_model=OrderResponse)
async def update_order_status(
    order_id: int,
    new_status: str = Query(..., description="New status for the order"),
    worker_id: Optional[int] = Query(None, description="ID of the worker to assign to the order"),
    current_user: Dict[str, Any] = Depends(has_role(["Manager", "ReceivingClerk", "receiving_clerk", "Picker", "Packer", "Driver"]))
) -> Dict[str, Any]:
    """
    Update the status of an order.
    
    Only warehouse staff can update order status.
    """
    # Check if order exists
    order = await OrdersService.get_order(order_id)
    if not order:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Order with ID {order_id} not found"
        )
    
    updated_order = await OrdersService.update_order_status(order_id, new_status, worker_id)
    
    # Emit WebSocket notification for real-time updates
    try:
        await websocket_manager.notify_order_update(
            order_id=order_id,
            order_status=new_status,
            user_roles=["Manager", "ReceivingClerk", "receiving_clerk", "Picker", "Packer", "Driver"]
        )
    except Exception as e:
        # Log the error but don't fail the request
        import logging
        logging.getLogger(__name__).error(f"Failed to send WebSocket notification: {e}")
    
    return updated_order

# Generate picking list
@router.get("/{order_id}/picking-list", response_model=Dict[str, Any])
async def generate_picking_list(
    order_id: int,
    current_user: Dict[str, Any] = Depends(has_role(["Manager", "Picker"]))
) -> Dict[str, Any]:
    """
    Generate a picking list for an order.
    
    This creates a structured list of items to pick, organized by warehouse location for efficiency.
    """
    result = await OrdersService.generate_picking_list(order_id)
    
    if "error" in result:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=result["error"]
        )
    
    return result

# Check order availability
@router.get("/{order_id}/availability", response_model=Dict[str, Any])
async def check_order_availability(
    order_id: int,
    current_user: Dict[str, Any] = Depends(get_current_active_user)
) -> Dict[str, Any]:
    """
    Check if all items in an order are available in inventory.
    """
    result = await OrdersService.check_order_availability(order_id)
    
    if "error" in result:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=result["error"]
        )
    
    return result

# Allocate inventory for order
@router.post("/{order_id}/allocate", response_model=Dict[str, Any])
async def allocate_inventory_for_order(
    order_id: int,
    current_user: Dict[str, Any] = Depends(has_role(["Manager", "Picker"]))
) -> Dict[str, Any]:
    """
    Allocate inventory for an order.
    
    This reserves inventory items for an order by creating inventory allocations.
    """
    result = await OrdersService.allocate_inventory_for_order(order_id)
    
    if "error" in result:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=result["error"]
        )
    
    return result

# Optimize order fulfillment
@router.get("/optimize-fulfillment", response_model=Dict[str, Any])
async def optimize_order_fulfillment(
    current_user: Dict[str, Any] = Depends(has_role(["Manager"]))
) -> Dict[str, Any]:
    """
    Optimize the order fulfillment process.
    
    This endpoint analyzes current workload and resources to suggest
    workflow optimizations and a recommended order processing sequence.
    """
    result = await WorkflowService.optimize_order_fulfillment()
    return result