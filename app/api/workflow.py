from fastapi import APIRouter, Depends, HTTPException, status
from typing import List, Dict, Any
from pydantic import BaseModel
from datetime import datetime, timedelta

from ..auth.dependencies import get_current_active_user
from ..services.role_based_service import RoleBasedService
from ..utils.database import serialize_doc, get_collection

router = APIRouter(tags=["Workflow"])

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

@router.post("/optimization/analyze")
async def analyze_workflow_optimization(
    request: Dict[str, Any],
    current_user: Dict[str, Any] = Depends(get_current_active_user)
) -> Dict[str, Any]:
    """
    Analyze workflow optimization based on worker roles and current workload.
    """
    try:
        worker_roles = request.get("worker_roles", [])
        
        # Mock optimization analysis for now
        optimization_data = {
            "efficiency_score": 85.5,
            "bottlenecks": [
                {
                    "stage": "picking",
                    "severity": "medium",
                    "description": "Picking stage has 15% slower than average completion time",
                    "suggested_action": "Reassign 2 workers from packing to picking during peak hours"
                },
                {
                    "stage": "packing", 
                    "severity": "low",
                    "description": "Packing capacity underutilized by 8%",
                    "suggested_action": "Consider reducing packing staff during low-volume periods"
                }
            ],
            "recommendations": [
                {
                    "type": "resource_allocation",
                    "priority": "high",
                    "description": "Redistribute workers to balance workload",
                    "expected_improvement": "12% efficiency increase"
                },
                {
                    "type": "process_improvement",
                    "priority": "medium", 
                    "description": "Implement batch picking for small orders",
                    "expected_improvement": "8% time reduction"
                }
            ],
            "analyzed_at": datetime.utcnow().isoformat()
        }
        
        return {
            "success": True,
            "data": optimization_data
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error analyzing workflow optimization: {str(e)}"
        )

@router.get("/status/overview")
async def get_workflow_status_overview(
    current_user: Dict[str, Any] = Depends(get_current_active_user)
) -> Dict[str, Any]:
    """
    Get workflow status overview with current statistics.
    """
    try:
        # Mock status overview for now
        status_data = {
            "total_active_orders": 45,
            "orders_by_stage": {
                "pending": 8,
                "receiving": 5,
                "picking": 12,
                "packing": 10,
                "shipping": 7,
                "delivered": 3
            },
            "worker_utilization": {
                "receiving_clerks": {"active": 2, "total": 3, "utilization": 67},
                "pickers": {"active": 4, "total": 6, "utilization": 83},
                "packers": {"active": 3, "total": 4, "utilization": 75},
                "drivers": {"active": 2, "total": 3, "utilization": 67}
            },
            "completion_times": {
                "average_order_time": "4.2 hours",
                "picking_time": "45 minutes",
                "packing_time": "25 minutes",
                "shipping_time": "2.1 hours"
            },
            "alerts": [
                {
                    "type": "warning",
                    "message": "Picking queue above normal capacity",
                    "count": 12
                }
            ],
            "updated_at": datetime.utcnow().isoformat()
        }
        
        return {
            "success": True,
            "data": status_data
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting workflow status overview: {str(e)}"
        )

@router.get("/active-tasks")
async def get_active_workflow_tasks(
    current_user: Dict[str, Any] = Depends(get_current_active_user)
) -> Dict[str, Any]:
    """
    Get currently active workflow tasks across all stages.
    """
    try:
        # Mock active tasks for now
        active_tasks = [
            {
                "task_id": "task_001",
                "order_id": "12345",
                "stage": "picking",
                "worker_id": 3,
                "worker_name": "Alice Picker",
                "started_at": datetime.utcnow().isoformat(),
                "estimated_completion": "15 minutes",
                "status": "in_progress",
                "priority": "high"
            },
            {
                "task_id": "task_002", 
                "order_id": "12346",
                "stage": "packing",
                "worker_id": 4,
                "worker_name": "Bob Packer",
                "started_at": datetime.utcnow().isoformat(),
                "estimated_completion": "10 minutes",
                "status": "in_progress",
                "priority": "medium"
            },
            {
                "task_id": "task_003",
                "order_id": "12347", 
                "stage": "receiving",
                "worker_id": 2,
                "worker_name": "Carol Receiver",
                "started_at": datetime.utcnow().isoformat(),
                "estimated_completion": "30 minutes",
                "status": "in_progress",
                "priority": "low"
            }
        ]
        
        return {
            "success": True,
            "data": active_tasks
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting active workflow tasks: {str(e)}"
        )
