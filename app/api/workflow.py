from fastapi import APIRouter, Depends, HTTPException, status
from typing import List, Dict, Any
from pydantic import BaseModel

from ..auth.dependencies import get_current_active_user
from ..services.role_based_service import RoleBasedService
from ..utils.database import serialize_doc

router = APIRouter(prefix="/workflow", tags=["Workflow"])

# Request models
class WorkflowActionRequest(BaseModel):
    order_id: str
    action: str
    worker_id: str
    notes: str = None

class WorkflowStatusRequest(BaseModel):
    order_id: str
    status: str

# Workflow endpoints
@router.get("/status-transitions")
async def get_valid_status_transitions(
    current_user: Dict[str, Any] = Depends(get_current_active_user)
) -> Dict[str, Any]:
    """
    Get valid status transitions for workflow management.
    """
    transitions = {
        "pending": ["confirmed", "receiving"],
        "confirmed": ["receiving"],
        "receiving": ["picking"],
        "picking": ["picking_in_progress"],
        "picking_in_progress": ["packing"],
        "packing": ["packed"],
        "packed": ["shipped"],
        "shipped": ["delivered"],
        "delivered": ["completed"]
    }
    
    return {"success": True, "data": transitions}

@router.get("/order-workflow/{order_id}")
async def get_order_workflow_status(
    order_id: str,
    current_user: Dict[str, Any] = Depends(get_current_active_user)
) -> Dict[str, Any]:
    """
    Get the current workflow status and history for a specific order.
    """
    try:
        from ..services.orders_service import OrdersService
        
        order = OrdersService.get_order_by_id(order_id)
        if not order:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Order not found"
            )
        
        workflow_info = {
            "order_id": order_id,
            "current_status": order.get("order_status"),
            "status_history": order.get("status_history", {}),
            "assigned_worker": order.get("assigned_worker"),
            "created_at": order.get("created_at"),
            "updated_at": order.get("updated_at")
        }
        
        return {"success": True, "data": workflow_info}
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving workflow status: {str(e)}"
        )

@router.post("/execute-action")
async def execute_workflow_action(
    request: WorkflowActionRequest,
    current_user: Dict[str, Any] = Depends(get_current_active_user)
) -> Dict[str, Any]:
    """
    Execute a workflow action on an order.
    """
    try:
        # Map actions to status transitions
        action_to_status = {
            "start_receiving": "receiving",
            "complete_receiving": "picking",
            "start_picking": "picking_in_progress",
            "complete_picking": "packing",
            "start_packing": "packing",
            "complete_packing": "packed",
            "ship_order": "shipped",
            "deliver_order": "delivered"
        }
        
        if request.action not in action_to_status:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid action: {request.action}"
            )
        
        new_status = action_to_status[request.action]
        
        # Execute the status update
        result = RoleBasedService.update_order_status(
            order_id=request.order_id,
            new_status=new_status,
            worker_id=request.worker_id,
            current_user=current_user
        )
        
        return result
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error executing workflow action: {str(e)}"
        )

@router.get("/worker-assignments")
async def get_worker_assignments(
    current_user: Dict[str, Any] = Depends(get_current_active_user)
) -> Dict[str, Any]:
    """
    Get current worker assignments across all workflow stages.
    """
    try:
        from ..utils.database import get_collection
        
        orders_collection = get_collection("orders")
        
        # Get orders with assigned workers
        pipeline = [
            {"$match": {"assigned_worker": {"$exists": True, "$ne": None}}},
            {"$group": {
                "_id": "$assigned_worker",
                "orders": {"$push": {
                    "order_id": "$orderID",
                    "status": "$order_status",
                    "customer": "$customerID",
                    "priority": "$priority"
                }},
                "order_count": {"$sum": 1}
            }}
        ]
        
        assignments = list(orders_collection.aggregate(pipeline))
        
        return {"success": True, "data": assignments}
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving worker assignments: {str(e)}"
        )

@router.get("/workflow-metrics")
async def get_workflow_metrics(
    current_user: Dict[str, Any] = Depends(get_current_active_user)
) -> Dict[str, Any]:
    """
    Get workflow performance metrics.
    """
    try:
        from ..utils.database import get_collection
        from datetime import datetime, timedelta
        
        orders_collection = get_collection("orders")
        
        # Calculate metrics for the last 7 days
        seven_days_ago = datetime.utcnow() - timedelta(days=7)
        
        metrics = {
            "total_orders": orders_collection.count_documents({}),
            "orders_by_status": {},
            "completion_rate": 0,
            "average_processing_time": 0,
            "bottlenecks": []
        }
        
        # Count orders by status
        status_pipeline = [
            {"$group": {
                "_id": "$order_status",
                "count": {"$sum": 1}
            }}
        ]
        
        status_counts = list(orders_collection.aggregate(status_pipeline))
        for item in status_counts:
            metrics["orders_by_status"][item["_id"]] = item["count"]
        
        # Calculate completion rate
        total_orders = metrics["total_orders"]
        completed_orders = metrics["orders_by_status"].get("completed", 0) + \
                          metrics["orders_by_status"].get("delivered", 0)
        
        if total_orders > 0:
            metrics["completion_rate"] = (completed_orders / total_orders) * 100
        
        # Identify bottlenecks (statuses with high order counts)
        bottleneck_threshold = total_orders * 0.2  # 20% threshold
        for status, count in metrics["orders_by_status"].items():
            if count > bottleneck_threshold and status not in ["completed", "delivered"]:
                metrics["bottlenecks"].append({
                    "status": status,
                    "count": count,
                    "percentage": (count / total_orders) * 100
                })
        
        return {"success": True, "data": metrics}
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving workflow metrics: {str(e)}"
        )

@router.get("/next-actions/{order_id}")
async def get_next_actions(
    order_id: str,
    current_user: Dict[str, Any] = Depends(get_current_active_user)
) -> Dict[str, Any]:
    """
    Get available next actions for a specific order.
    """
    try:
        from ..services.orders_service import OrdersService
        
        order = OrdersService.get_order_by_id(order_id)
        if not order:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Order not found"
            )
        
        current_status = order.get("order_status")
        
        # Define next actions based on current status
        next_actions = {
            "pending": ["start_receiving"],
            "confirmed": ["start_receiving"],
            "receiving": ["complete_receiving"],
            "picking": ["start_picking"],
            "picking_in_progress": ["complete_picking"],
            "packing": ["complete_packing"],
            "packed": ["ship_order"],
            "shipped": ["deliver_order"],
            "delivered": []
        }
        
        available_actions = next_actions.get(current_status, [])
        
        return {
            "success": True,
            "data": {
                "order_id": order_id,
                "current_status": current_status,
                "available_actions": available_actions
            }
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving next actions: {str(e)}"
        )
