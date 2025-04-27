from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta

from ..utils.database import get_collection

class AnalyticsService:
    """
    Service for providing analytics and reporting for the warehouse management system.
    
    This service provides metrics, KPIs, and reports for management dashboards.
    """
    
    @staticmethod
    async def get_inventory_metrics() -> Dict[str, Any]:
        """
        Get key inventory metrics for management dashboard.
        """
        inventory_collection = get_collection("inventory")
        
        # Get total inventory count
        total_items = inventory_collection.count_documents({})
        
        # Get low stock items count
        low_stock_query = {"$expr": {"$lte": ["$stock_level", "$min_stock_level"]}}
        low_stock_count = inventory_collection.count_documents(low_stock_query)
        
        # Get items by category
        pipeline = [
            {"$group": {"_id": "$category", "count": {"$sum": 1}}}
        ]
        categories = list(inventory_collection.aggregate(pipeline))
        
        # Calculate total stock value (assuming each item has a price field, or add it if needed)
        total_value = 0
        for item in inventory_collection.find({}, {"stock_level": 1, "price": 1}):
            if "price" in item:
                total_value += item.get("stock_level", 0) * item.get("price", 0)
        
        return {
            "total_items": total_items,
            "low_stock_count": low_stock_count,
            "low_stock_percentage": round((low_stock_count / total_items * 100), 1) if total_items > 0 else 0,
            "categories": categories,
            "total_value": total_value
        }
    
    @staticmethod
    async def get_order_metrics(days: int = 30) -> Dict[str, Any]:
        """
        Get key order metrics for management dashboard.
        
        Args:
            days: Number of days to look back for trends
        """
        orders_collection = get_collection("orders")
        
        # Calculate date range
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=days)
        
        # Get total orders in period
        date_query = {"order_date": {"$gte": start_date, "$lte": end_date}}
        total_orders = orders_collection.count_documents(date_query)
        
        # Get orders by status
        pipeline = [
            {"$match": date_query},
            {"$group": {"_id": "$order_status", "count": {"$sum": 1}}}
        ]
        status_breakdown = list(orders_collection.aggregate(pipeline))
        
        # Get total revenue
        revenue = 0
        for order in orders_collection.find(date_query, {"total_amount": 1}):
            revenue += order.get("total_amount", 0)
        
        # Get order trend by day
        pipeline = [
            {"$match": date_query},
            {"$group": {
                "_id": {"$dateToString": {"format": "%Y-%m-%d", "date": "$order_date"}},
                "count": {"$sum": 1},
                "revenue": {"$sum": "$total_amount"}
            }},
            {"$sort": {"_id": 1}}
        ]
        daily_trend = list(orders_collection.aggregate(pipeline))
        
        return {
            "total_orders": total_orders,
            "status_breakdown": status_breakdown,
            "total_revenue": revenue,
            "average_order_value": round(revenue / total_orders, 2) if total_orders > 0 else 0,
            "daily_trend": daily_trend
        }
    
    @staticmethod
    async def get_operations_metrics(days: int = 30) -> Dict[str, Any]:
        """
        Get warehouse operations metrics.
        
        Args:
            days: Number of days to look back for trends
        """
        picking_collection = get_collection("picking")
        packing_collection = get_collection("packing")
        shipping_collection = get_collection("shipping")
        
        # Calculate date range
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=days)
        date_query = {"created_at": {"$gte": start_date, "$lte": end_date}}
        
        # Get picking metrics
        total_picking = picking_collection.count_documents(date_query)
        completed_picking = picking_collection.count_documents({**date_query, "status": "completed"})
        
        # Get average picking time
        picking_times = []
        for pick in picking_collection.find({
            **date_query,
            "status": "completed",
            "start_time": {"$exists": True},
            "complete_time": {"$exists": True}
        }):
            if pick.get("start_time") and pick.get("complete_time"):
                picking_time = (pick["complete_time"] - pick["start_time"]).total_seconds() / 60  # in minutes
                picking_times.append(picking_time)
        
        avg_picking_time = sum(picking_times) / len(picking_times) if picking_times else 0
        
        # Similar calculations for packing and shipping
        total_packing = packing_collection.count_documents(date_query)
        completed_packing = packing_collection.count_documents({**date_query, "status": "completed"})
        
        packing_times = []
        for pack in packing_collection.find({
            **date_query,
            "status": "completed",
            "start_time": {"$exists": True},
            "complete_time": {"$exists": True}
        }):
            if pack.get("start_time") and pack.get("complete_time"):
                packing_time = (pack["complete_time"] - pack["start_time"]).total_seconds() / 60
                packing_times.append(packing_time)
        
        avg_packing_time = sum(packing_times) / len(packing_times) if packing_times else 0
        
        total_shipping = shipping_collection.count_documents(date_query)
        delivered_shipping = shipping_collection.count_documents({**date_query, "status": "delivered"})
        
        # Calculate on-time delivery rate
        on_time_count = 0
        for ship in shipping_collection.find({
            **date_query,
            "status": "delivered",
            "estimated_delivery": {"$exists": True},
            "actual_delivery": {"$exists": True}
        }):
            if ship.get("actual_delivery") <= ship.get("estimated_delivery"):
                on_time_count += 1
                
        on_time_rate = on_time_count / delivered_shipping if delivered_shipping > 0 else 0
        
        return {
            "picking": {
                "total": total_picking,
                "completed": completed_picking,
                "completion_rate": round(completed_picking / total_picking * 100, 1) if total_picking > 0 else 0,
                "avg_time_minutes": round(avg_picking_time, 1)
            },
            "packing": {
                "total": total_packing,
                "completed": completed_packing,
                "completion_rate": round(completed_packing / total_packing * 100, 1) if total_packing > 0 else 0,
                "avg_time_minutes": round(avg_packing_time, 1)
            },
            "shipping": {
                "total": total_shipping,
                "delivered": delivered_shipping,
                "delivery_rate": round(delivered_shipping / total_shipping * 100, 1) if total_shipping > 0 else 0,
                "on_time_rate": round(on_time_rate * 100, 1)
            }
        }
        
    @staticmethod
    async def get_warehouse_utilization() -> Dict[str, Any]:
        """
        Get warehouse space utilization metrics.
        """
        warehouses_collection = get_collection("warehouses")
        locations_collection = get_collection("locations")
        
        # Get warehouse utilization
        warehouses = list(warehouses_collection.find())
        warehouse_stats = []
        
        for warehouse in warehouses:
            warehouse_id = warehouse.get("warehouseID")
            capacity = warehouse.get("capacity", 0)
            
            # Count occupied locations
            occupied_count = locations_collection.count_documents({
                "warehouseID": warehouse_id,
                "is_occupied": True
            })
            
            # Count total locations
            total_locations = locations_collection.count_documents({
                "warehouseID": warehouse_id
            })
            
            # Calculate utilization
            location_utilization = occupied_count / total_locations if total_locations > 0 else 0
            capacity_utilization = (capacity - warehouse.get("available_storage", 0)) / capacity if capacity > 0 else 0
            
            warehouse_stats.append({
                "warehouseID": warehouse_id,
                "name": warehouse.get("name"),
                "location_utilization": round(location_utilization * 100, 1),
                "capacity_utilization": round(capacity_utilization * 100, 1),
                "occupied_locations": occupied_count,
                "total_locations": total_locations
            })
        
        return {
            "warehouses": warehouse_stats,
            "overall_location_utilization": round(
                sum(w["occupied_locations"] for w in warehouse_stats) / 
                sum(w["total_locations"] for w in warehouse_stats) * 100, 1
            ) if sum(w["total_locations"] for w in warehouse_stats) > 0 else 0
        }