from fastapi import APIRouter, Depends, HTTPException, status
from typing import List, Dict, Any
from pydantic import BaseModel

from ..auth.dependencies import get_current_active_user
from ..services.role_based_service import RoleBasedService
from ..services.orders_service import OrdersService
from ..utils.database import serialize_doc

router = APIRouter()

# Request models
class AssignOrderRequest(BaseModel):
    order_id: str
    worker_id: str

class UpdateOrderStatusRequest(BaseModel):
    order_id: str
    new_status: str
    worker_id: str

# Role-based order endpoints
@router.get("/receiving-clerk/orders")
async def get_orders_for_receiving_clerk(
    current_user: Dict[str, Any] = Depends(get_current_active_user)
) -> Dict[str, Any]:
    """
    Get orders available for receiving clerk.
    Only shows orders in initial phase: 'pending', 'confirmed', or 'receiving'.
    """
    if current_user.get("role") not in ["receiving_clerk", "ReceivingClerk", "Manager"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied: Receiving clerk role required"
        )
    
    orders = RoleBasedService.get_orders_for_receiving_clerk()
    return {"success": True, "data": orders}

@router.get("/picker/orders")
async def get_orders_for_picker(
    current_user: Dict[str, Any] = Depends(get_current_active_user)
) -> Dict[str, Any]:
    """
    Get orders available for picker.
    Only shows orders in 'picking' status.
    """
    if current_user.get("role") not in ["picker", "Picker", "Manager"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied: Picker role required"
        )
    
    orders = RoleBasedService.get_orders_for_picker()
    return {"success": True, "data": orders}

@router.get("/packer/orders")
async def get_orders_for_packer(
    current_user: Dict[str, Any] = Depends(get_current_active_user)
) -> Dict[str, Any]:
    """
    Get orders available for packer.
    Only shows orders in 'packing' status.
    """
    if current_user.get("role") not in ["packer", "Packer", "Manager"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied: Packer role required"
        )
    
    orders = RoleBasedService.get_orders_for_packer()
    return {"success": True, "data": orders}

@router.post("/receiving-clerk/start-receiving")
async def start_receiving_order(
    request: UpdateOrderStatusRequest,
    current_user: Dict[str, Any] = Depends(get_current_active_user)
) -> Dict[str, Any]:
    """
    Start receiving an order. Changes status from 'pending' or 'confirmed' to 'receiving'.
    """
    if current_user.get("role") not in ["receiving_clerk", "ReceivingClerk", "Manager"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied: Receiving clerk role required"
        )
    
    # Update order status to 'receiving'
    result = RoleBasedService.update_order_status(
        order_id=request.order_id,
        new_status="receiving",
        worker_id=request.worker_id,
        current_user=current_user
    )
    
    if not result.get("success"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=result.get("error", "Failed to start receiving order")
        )
    
    return result

@router.post("/receiving-clerk/complete-receiving")
async def complete_receiving_order(
    request: UpdateOrderStatusRequest,
    current_user: Dict[str, Any] = Depends(get_current_active_user)
) -> Dict[str, Any]:
    """
    Complete receiving an order. Changes status from 'receiving' to 'picking'.
    """
    if current_user.get("role") not in ["receiving_clerk", "ReceivingClerk", "Manager"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied: Receiving clerk role required"
        )
    
    # Update order status to 'picking' when receiving is complete
    result = RoleBasedService.update_order_status(
        order_id=request.order_id,
        new_status="picking",
        worker_id=request.worker_id,
        current_user=current_user
    )
    
    if not result.get("success"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=result.get("error", "Failed to complete receiving order")
        )
    
    return result

@router.post("/picker/start-picking")
async def start_picking_order(
    request: UpdateOrderStatusRequest,
    current_user: Dict[str, Any] = Depends(get_current_active_user)
) -> Dict[str, Any]:
    """
    Start picking an order. Changes status from 'picking' to 'picking_in_progress'.
    """
    if current_user.get("role") not in ["picker", "Picker", "Manager"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied: Picker role required"
        )
    
    result = RoleBasedService.update_order_status(
        order_id=request.order_id,
        new_status="picking_in_progress",
        worker_id=request.worker_id,
        current_user=current_user
    )
    
    if not result.get("success"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=result.get("error", "Failed to start picking order")
        )
    
    return result

@router.post("/picker/complete-picking")
async def complete_picking_order(
    request: UpdateOrderStatusRequest,
    current_user: Dict[str, Any] = Depends(get_current_active_user)
) -> Dict[str, Any]:
    """
    Complete picking an order. Changes status from 'picking_in_progress' to 'packing'.
    """
    if current_user.get("role") not in ["picker", "Picker", "Manager"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied: Picker role required"
        )
    
    result = RoleBasedService.update_order_status(
        order_id=request.order_id,
        new_status="packing",
        worker_id=request.worker_id,
        current_user=current_user
    )
    
    if not result.get("success"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=result.get("error", "Failed to complete picking order")
        )
    
    return result

@router.get("/order/{order_id}")
async def get_order_details(
    order_id: str,
    current_user: Dict[str, Any] = Depends(get_current_active_user)
) -> Dict[str, Any]:
    """
    Get detailed information about a specific order.
    """
    order = OrdersService.get_order_by_id(order_id)
    
    if not order:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Order not found"
        )
    
    return {"success": True, "data": serialize_doc(order)}

@router.get("/receiving-clerk/processed-orders/{worker_id}")
async def get_processed_orders_for_receiving_clerk(
    worker_id: str,
    current_user: Dict[str, Any] = Depends(get_current_active_user)
) -> Dict[str, Any]:
    """
    Get orders that were processed by this receiving clerk.
    """
    if current_user.get("role") not in ["receiving_clerk", "ReceivingClerk", "Manager"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied: Receiving clerk role required"
        )
    
    # Ensure users can only see their own processed orders (unless manager)
    if current_user.get("role") != "Manager" and current_user.get("username") != worker_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You can only view your own processed orders"
        )
    
    orders = RoleBasedService.get_processed_orders_by_worker(worker_id, "receiving")
    return {"success": True, "data": orders}

@router.get("/receiving-clerk/stats")
async def get_receiving_clerk_stats(
    current_user: Dict[str, Any] = Depends(get_current_active_user)
) -> Dict[str, Any]:
    """
    Get statistics for receiving clerk dashboard.
    """
    if current_user.get("role") not in ["receiving_clerk", "ReceivingClerk", "Manager"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied: Receiving clerk role required"
        )
    
    stats = RoleBasedService.get_receiving_clerk_stats()
    return {"success": True, "data": stats}

@router.get("/picker/stats")
async def get_picker_stats(
    current_user: Dict[str, Any] = Depends(get_current_active_user)
) -> Dict[str, Any]:
    """
    Get statistics for picker dashboard.
    """
    if current_user.get("role") not in ["picker", "Picker", "Manager"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied: Picker role required"
        )
    
    stats = RoleBasedService.get_picker_stats()
    return {"success": True, "data": stats}

@router.get("/packer/stats")
async def get_packer_stats(
    current_user: Dict[str, Any] = Depends(get_current_active_user)
) -> Dict[str, Any]:
    """
    Get statistics for packer dashboard.
    """
    if current_user.get("role") not in ["packer", "Packer", "Manager"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied: Packer role required"
        )
    
    stats = RoleBasedService.get_packer_stats()
    return {"success": True, "data": stats}
