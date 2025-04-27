from fastapi import APIRouter, Depends, Query
from typing import Dict, Any

from ..auth.dependencies import get_current_active_user, has_role
from ..services.analytics_service import AnalyticsService

router = APIRouter()

@router.get("/inventory", response_model=Dict[str, Any])
async def get_inventory_metrics(
    current_user: Dict[str, Any] = Depends(has_role(["Manager"]))
) -> Dict[str, Any]:
    """
    Get inventory analytics for management dashboard.
    
    Returns metrics such as total inventory, low stock items, and inventory by category.
    """
    return await AnalyticsService.get_inventory_metrics()

@router.get("/orders", response_model=Dict[str, Any])
async def get_order_metrics(
    days: int = Query(30, description="Number of days to analyze"),
    current_user: Dict[str, Any] = Depends(has_role(["Manager"]))
) -> Dict[str, Any]:
    """
    Get order analytics for management dashboard.
    
    Returns metrics such as order counts, status breakdown, and revenue.
    """
    return await AnalyticsService.get_order_metrics(days)

@router.get("/operations", response_model=Dict[str, Any])
async def get_operations_metrics(
    days: int = Query(30, description="Number of days to analyze"),
    current_user: Dict[str, Any] = Depends(has_role(["Manager"]))
) -> Dict[str, Any]:
    """
    Get warehouse operations analytics.
    
    Returns metrics on picking, packing, and shipping operations.
    """
    return await AnalyticsService.get_operations_metrics(days)

@router.get("/warehouse-utilization", response_model=Dict[str, Any])
async def get_warehouse_utilization(
    current_user: Dict[str, Any] = Depends(has_role(["Manager"]))
) -> Dict[str, Any]:
    """
    Get warehouse space utilization metrics.
    
    Returns metrics on space usage across warehouses.
    """
    return await AnalyticsService.get_warehouse_utilization()

@router.get("/dashboard", response_model=Dict[str, Any])
async def get_dashboard_metrics(
    days: int = Query(30, description="Number of days to analyze"),
    current_user: Dict[str, Any] = Depends(has_role(["Manager"]))
) -> Dict[str, Any]:
    """
    Get all dashboard metrics in a single request.
    
    Returns metrics for inventory, orders, operations, and warehouse utilization.
    """
    inventory_metrics = await AnalyticsService.get_inventory_metrics()
    order_metrics = await AnalyticsService.get_order_metrics(days)
    operations_metrics = await AnalyticsService.get_operations_metrics(days)
    warehouse_metrics = await AnalyticsService.get_warehouse_utilization()
    
    return {
        "inventory": inventory_metrics,
        "orders": order_metrics,
        "operations": operations_metrics,
        "warehouse": warehouse_metrics
    }