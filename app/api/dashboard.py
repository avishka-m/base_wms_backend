from fastapi import APIRouter, Depends, Query, HTTPException, status
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta

from ..auth.dependencies import get_current_active_user, has_role
from ..services.analytics_service import AnalyticsService
from ..services.inventory_service import InventoryService
from ..services.orders_service import OrdersService
from ..utils.database import get_collection

router = APIRouter()

# @router.get("/summary", response_model=Dict[str, Any])
# async def get_dashboard_summary(
#     current_user: Dict[str, Any] = Depends(get_current_active_user)
# ) -> Dict[str, Any]:
#     """
#     Get overall dashboard summary statistics
    
#     Returns:
#         Dict containing:
#         - total_orders: Total number of orders
#         - total_inventory: Total inventory items
#         - active_workers: Number of active workers
#         - available_vehicles: Number of available vehicles
#         - monthly_revenue: Total revenue this month
#         - pending_orders: Number of pending orders
#     """
#     try:
#         # Get database collections
#         orders_collection = get_collection("orders")
#         inventory_collection = get_collection("inventory")
#         workers_collection = get_collection("workers")
#         vehicles_collection = get_collection("vehicles")

#         # Get current month's date range
#         today = datetime.now()
#         month_start = today.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
#         next_month = month_start.replace(month=month_start.month + 1) if month_start.month < 12 else month_start.replace(year=month_start.year + 1, month=1)

#         # Calculate monthly revenue
#         monthly_orders = orders_collection.find({
#             "created_at": {
#                 "$gte": month_start,
#                 "$lt": next_month
#             },
#             "status": "completed"
#         })
#         monthly_revenue = sum(order.get("total", 0) for order in monthly_orders)

#         # Get summary statistics
#         summary = {
#             "total_orders": orders_collection.count_documents({}),
#             "total_inventory": inventory_collection.count_documents({}),
#             "active_workers": workers_collection.count_documents({"status": "active"}),
#             "available_vehicles": vehicles_collection.count_documents({"status": "available"}),
#             "monthly_revenue": monthly_revenue,
#             "pending_orders": orders_collection.count_documents({"status": "pending"})
#         }

#         return summary

#     except Exception as e:
#         raise HTTPException(
#             status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
#             detail=f"Error fetching dashboard summary: {str(e)}"
#         )

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

# @router.get("/recent-orders", response_model=List[Dict[str, Any]])
# async def get_recent_orders(
#     limit: int = Query(5, description="Number of recent orders to return"),
#     current_user: Dict[str, Any] = Depends(get_current_active_user)
# ) -> List[Dict[str, Any]]:
#     """
#     Get recent orders for dashboard display
    
#     Args:
#         limit: Number of orders to return (default: 5)
#         current_user: Current authenticated user
        
#     Returns:
#         List of recent orders with their details
#     """
#     try:
#         # Get orders collection
#         orders_collection = get_collection("orders")
        
#         # Get recent orders sorted by creation date
#         recent_orders = list(orders_collection
#             .find({})
#             .sort("created_at", -1)
#             .limit(limit)
#         )
        
#         # Format orders for frontend
#         formatted_orders = []
#         for order in recent_orders:
#             formatted_orders.append({
#                 "order_id": str(order.get("_id")),
#                 "customer_name": order.get("customer_name", "Unknown"),
#                 "order_date": order.get("created_at").isoformat(),
#                 "status": order.get("status", "Unknown"),
#                 "total": float(order.get("total", 0))
#             })
            
#         return formatted_orders

#     except Exception as e:
#         raise HTTPException(
#             status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
#             detail=f"Error fetching recent orders: {str(e)}"
#         )

# @router.get("/low-stock", response_model=List[Dict[str, Any]])
# async def get_low_stock_items(
#     threshold: int = Query(10, description="Low stock threshold"),
#     limit: int = Query(5, description="Number of items to return"),
#     current_user: Dict[str, Any] = Depends(get_current_active_user)
# ) -> List[Dict[str, Any]]:
#     """
#     Get items with low stock for dashboard display
    
#     Args:
#         threshold: Quantity threshold for low stock (default: 10)
#         limit: Number of items to return (default: 5)
#         current_user: Current authenticated user
        
#     Returns:
#         List of low stock items
#     """
#     try:
#         # Get inventory collection
#         inventory_collection = get_collection("inventory")
        
#         # Get items with quantity below threshold
#         low_stock_items = list(inventory_collection
#             .find({"quantity": {"$lte": threshold}})
#             .sort("quantity", 1)
#             .limit(limit)
#         )
        
#         # Format items for frontend
#         formatted_items = []
#         for item in low_stock_items:
#             formatted_items.append({
#                 "id": str(item.get("_id")),
#                 "name": item.get("name", "Unknown Item"),
#                 "sku": item.get("sku", "N/A"),
#                 "quantity": int(item.get("quantity", 0))
#             })
            
#         return formatted_items

#     except Exception as e:
#         raise HTTPException(
#             status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
#             detail=f"Error fetching low stock items: {str(e)}"
#         )

# @router.get("/pending-tasks", response_model=List[Dict[str, Any]])
# async def get_pending_tasks(
#     limit: int = Query(5, description="Number of tasks to return"),
#     current_user: Dict[str, Any] = Depends(get_current_active_user)
# ) -> List[Dict[str, Any]]:
#     """
#     Get pending tasks for dashboard display
    
#     Args:
#         limit: Number of tasks to return (default: 5)
#         current_user: Current authenticated user
        
#     Returns:
#         List of pending tasks
#     """
#     try:
#         # Get tasks collection
#         tasks_collection = get_collection("tasks")
        
#         # Get pending tasks sorted by due date
#         pending_tasks = list(tasks_collection
#             .find({"status": "pending"})
#             .sort("due_date", 1)
#             .limit(limit)
#         )
        
#         # Format tasks for frontend
#         formatted_tasks = []
#         for task in pending_tasks:
#             formatted_tasks.append({
#                 "id": str(task.get("_id")),
#                 "name": task.get("name", "Unknown Task"),
#                 "due_date": task.get("due_date").isoformat().split('T')[0],  # Format as YYYY-MM-DD
#                 "priority": task.get("priority", "Medium")
#             })
            
#         return formatted_tasks

#     except Exception as e:
#         raise HTTPException(
#             status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
#             detail=f"Error fetching pending tasks: {str(e)}"
#         )