from fastapi import APIRouter, Depends, Query, HTTPException, status
from typing import Dict, Any, Optional
from datetime import datetime, timedelta

from ..auth.dependencies import get_current_active_user, has_role
from ..services.analytics_service import AnalyticsService
from ..services.inventory_service import InventoryService
from ..services.orders_service import OrdersService
from ..utils.database import get_collection

router = APIRouter()

@router.get("/stats", response_model=Dict[str, Any])
async def get_dashboard_stats(
    role: str,
    current_user: Dict[str, Any] = Depends(get_current_active_user)
) -> Dict[str, Any]:
    """
    Get role-specific dashboard statistics
    """
    try:
        # Get database collections
        orders_collection = get_collection("orders")
        inventory_collection = get_collection("inventory")
        picking_collection = get_collection("picking")
        packing_collection = get_collection("packing")
        shipping_collection = get_collection("shipping")
        returns_collection = get_collection("returns")

        # Get today's date range
        today_start = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        today_end = today_start + timedelta(days=1)
        today_query = {"created_at": {"$gte": today_start, "$lt": today_end}}

        # Base stats that all roles can see
        base_stats = {
            "totalOrdersToday": orders_collection.count_documents(today_query),
            "warehouseEfficiency": 92,  # This should be calculated based on various metrics
            "workerAttendance": 98,  # This should come from attendance tracking
        }

        # Role-specific stats
        if role.lower() == "picker":
            picking_stats = {
                "ordersPickedToday": picking_collection.count_documents({
                    **today_query,
                    "status": "completed"
                }),
                "pickRate": 65,  # items per hour, should be calculated
                "accuracyRate": 99.5,  # percentage, should be calculated
                "pendingOrders": picking_collection.count_documents({
                    "status": "pending"
                })
            }
            return {**base_stats, **picking_stats}

        elif role.lower() == "packer":
            packing_stats = {
                "ordersPackedToday": packing_collection.count_documents({
                    **today_query,
                    "status": "completed"
                }),
                "packingRate": 22,  # orders per hour, should be calculated
                "qualityScore": 97.8,  # percentage, should be calculated
                "packingQueue": packing_collection.count_documents({
                    "status": "pending"
                })
            }
            return {**base_stats, **packing_stats}

        elif role.lower() == "driver":
            shipping_stats = {
                "deliveriesToday": shipping_collection.count_documents({
                    **today_query,
                    "status": "delivered"
                }),
                "onTimeRate": 94.5,  # percentage, should be calculated
                "routeEfficiency": 88,  # percentage, should be calculated
                "remainingStops": shipping_collection.count_documents({
                    "status": "in_transit"
                })
            }
            return {**base_stats, **shipping_stats}

        elif role.lower() == "clerk":
            clerk_stats = {
                "returnsProcessed": returns_collection.count_documents({
                    **today_query,
                    "status": "processed"
                }),
                "inventoryUpdates": 112,  # should be tracked in inventory history
                "accuracyRate": 99.2,  # percentage, should be calculated
                "pendingReturns": returns_collection.count_documents({
                    "status": "pending"
                })
            }
            return {**base_stats, **clerk_stats}

        elif role.lower() in ["manager", "admin"]:
            # Get analytics service instance
            analytics_service = AnalyticsService()
            
            # Get inventory metrics
            inventory_metrics = await AnalyticsService.get_inventory_metrics()
            
            manager_stats = {
                "totalOrdersToday": base_stats["totalOrdersToday"],
                "warehouseEfficiency": base_stats["warehouseEfficiency"],
                "workerAttendance": base_stats["workerAttendance"],
                "lowStockItems": inventory_metrics.get("low_stock_count", 0)
            }
            return manager_stats

        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid role: {role}"
            )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error fetching dashboard stats: {str(e)}"
        )

@router.get("/activities", response_model=Dict[str, Any])
async def get_activity_feed(
    limit: int = Query(5, description="Number of activities to return"),
    current_user: Dict[str, Any] = Depends(get_current_active_user)
) -> Dict[str, Any]:
    """
    Get recent activities for the dashboard activity feed
    """
    try:
        # This should query an activity log collection
        # For now, return mock data
        activities = [
            {
                "id": "ACT-1001",
                "type": "order_completed",
                "description": "Order #1234 completed",
                "timestamp": datetime.now() - timedelta(minutes=5)
            },
            {
                "id": "ACT-1002",
                "type": "inventory_update",
                "description": "Inventory count updated for SKU-789",
                "timestamp": datetime.now() - timedelta(minutes=15)
            }
        ]
        return {"activities": activities[:limit]}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error fetching activity feed: {str(e)}"
        )