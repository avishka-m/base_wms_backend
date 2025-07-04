from typing import List, Dict, Any, Optional
from datetime import datetime
from ..utils.database import get_collection, serialize_doc
from ..services.orders_service import OrdersService

class RoleBasedService:
    """Service for role-based order management and workflow operations."""
    
    @staticmethod
    def get_orders_for_receiving_clerk() -> List[Dict[str, Any]]:
        """
        Get orders available for receiving clerk.
        Returns orders in 'pending', 'confirmed', or 'receiving' status.
        """
        try:
            orders_collection = get_collection("orders")
            
            # Query for orders in receiving phases
            query = {
                "order_status": {"$in": ["pending", "confirmed", "receiving"]}
            }
            
            orders = list(orders_collection.find(query))
            
            # Sort by priority (1 = highest priority) and then by date
            orders.sort(key=lambda x: (x.get('priority', 999), x.get('order_date', '')))
            
            return serialize_doc(orders)
        except Exception as e:
            print(f"Error getting orders for receiving clerk: {e}")
            return []
    
    @staticmethod
    def get_orders_for_picker() -> List[Dict[str, Any]]:
        """
        Get orders available for picker.
        Returns orders in 'picking' or 'picking_in_progress' status.
        """
        try:
            orders_collection = get_collection("orders")
            
            # Query for orders in picking phase
            query = {
                "order_status": {"$in": ["picking", "picking_in_progress"]}
            }
            
            orders = list(orders_collection.find(query))
            
            # Sort by priority and date
            orders.sort(key=lambda x: (x.get('priority', 999), x.get('order_date', '')))
            
            return serialize_doc(orders)
        except Exception as e:
            print(f"Error getting orders for picker: {e}")
            return []
    
    @staticmethod
    def get_orders_for_packer() -> List[Dict[str, Any]]:
        """
        Get orders available for packer.
        Returns orders in 'packing' status.
        """
        try:
            orders_collection = get_collection("orders")
            
            # Query for orders in packing phase
            query = {
                "order_status": "packing"
            }
            
            orders = list(orders_collection.find(query))
            
            # Sort by priority and date
            orders.sort(key=lambda x: (x.get('priority', 999), x.get('order_date', '')))
            
            return serialize_doc(orders)
        except Exception as e:
            print(f"Error getting orders for packer: {e}")
            return []
    
    @staticmethod
    def update_order_status(
        order_id: str,
        new_status: str,
        worker_id: str,
        current_user: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Update order status with workflow validation.
        """
        try:
            orders_collection = get_collection("orders")
            
            # Find the order
            order = orders_collection.find_one({"orderID": order_id})
            if not order:
                return {"success": False, "error": "Order not found"}
            
            current_status = order.get("order_status")
            
            # Validate status transitions
            valid_transitions = {
                "pending": ["receiving", "confirmed"],
                "confirmed": ["receiving"],
                "receiving": ["picking"],
                "picking": ["picking_in_progress"],
                "picking_in_progress": ["packing"],
                "packing": ["packed"],
                "packed": ["shipped"],
                "shipped": ["delivered"]
            }
            
            if current_status not in valid_transitions:
                return {"success": False, "error": f"Invalid current status: {current_status}"}
            
            if new_status not in valid_transitions[current_status]:
                return {"success": False, "error": f"Invalid status transition from {current_status} to {new_status}"}
            
            # Update order
            update_data = {
                "order_status": new_status,
                "updated_at": datetime.utcnow().isoformat(),
                "assigned_worker": worker_id
            }
            
            # Add status history
            status_history = order.get("status_history", {})
            status_history[new_status] = {
                "timestamp": datetime.utcnow().isoformat(),
                "worker_id": worker_id,
                "previous_status": current_status
            }
            update_data["status_history"] = status_history
            
            result = orders_collection.update_one(
                {"orderID": order_id},
                {"$set": update_data}
            )
            
            if result.modified_count > 0:
                return {
                    "success": True,
                    "message": f"Order {order_id} status updated to {new_status}",
                    "data": {
                        "orderID": order_id,
                        "old_status": current_status,
                        "new_status": new_status,
                        "worker_id": worker_id
                    }
                }
            else:
                return {"success": False, "error": "Failed to update order status"}
                
        except Exception as e:
            print(f"Error updating order status: {e}")
            return {"success": False, "error": str(e)}
    
    @staticmethod
    def get_processed_orders_by_worker(worker_id: str, stage: str) -> List[Dict[str, Any]]:
        """
        Get orders that were processed by a specific worker in a specific stage.
        """
        try:
            orders_collection = get_collection("orders")
            
            # Query for orders processed by this worker
            query = {
                "$or": [
                    {"assigned_worker": worker_id},
                    {f"status_history.{stage}": {"$exists": True}}
                ]
            }
            
            orders = list(orders_collection.find(query))
            
            # Filter for orders where this worker was involved in the specified stage
            filtered_orders = []
            for order in orders:
                status_history = order.get("status_history", {})
                for status, history in status_history.items():
                    if history.get("worker_id") == worker_id and stage in status.lower():
                        filtered_orders.append(order)
                        break
            
            return serialize_doc(filtered_orders)
        except Exception as e:
            print(f"Error getting processed orders by worker: {e}")
            return []
    
    @staticmethod
    def get_receiving_clerk_stats() -> Dict[str, Any]:
        """
        Get statistics for receiving clerk dashboard.
        """
        try:
            orders_collection = get_collection("orders")
            
            # Get counts for different statuses
            total_orders = orders_collection.count_documents({
                "order_status": {"$in": ["pending", "confirmed", "receiving"]}
            })
            
            pending_orders = orders_collection.count_documents({"order_status": "pending"})
            confirmed_orders = orders_collection.count_documents({"order_status": "confirmed"})
            receiving_orders = orders_collection.count_documents({"order_status": "receiving"})
            
            # Get today's completed orders (moved to picking)
            today = datetime.utcnow().strftime("%Y-%m-%d")
            completed_today = orders_collection.count_documents({
                "order_status": "picking",
                "updated_at": {"$regex": f"^{today}"}
            })
            
            return {
                "total_orders": total_orders,
                "pending_orders": pending_orders,
                "confirmed_orders": confirmed_orders,
                "receiving_orders": receiving_orders,
                "completed_today": completed_today
            }
        except Exception as e:
            print(f"Error getting receiving clerk stats: {e}")
            return {
                "total_orders": 0,
                "pending_orders": 0,
                "confirmed_orders": 0,
                "receiving_orders": 0,
                "completed_today": 0
            }
    
    @staticmethod
    def get_picker_stats() -> Dict[str, Any]:
        """
        Get statistics for picker dashboard.
        """
        try:
            orders_collection = get_collection("orders")
            
            # Get counts for picking statuses
            total_orders = orders_collection.count_documents({
                "order_status": {"$in": ["picking", "picking_in_progress"]}
            })
            
            picking_orders = orders_collection.count_documents({"order_status": "picking"})
            picking_in_progress = orders_collection.count_documents({"order_status": "picking_in_progress"})
            
            # Get today's completed orders (moved to packing)
            today = datetime.utcnow().strftime("%Y-%m-%d")
            completed_today = orders_collection.count_documents({
                "order_status": "packing",
                "updated_at": {"$regex": f"^{today}"}
            })
            
            return {
                "total_orders": total_orders,
                "picking_orders": picking_orders,
                "picking_in_progress": picking_in_progress,
                "completed_today": completed_today
            }
        except Exception as e:
            print(f"Error getting picker stats: {e}")
            return {
                "total_orders": 0,
                "picking_orders": 0,
                "picking_in_progress": 0,
                "completed_today": 0
            }
    
    @staticmethod
    def get_packer_stats() -> Dict[str, Any]:
        """
        Get statistics for packer dashboard.
        """
        try:
            orders_collection = get_collection("orders")
            
            # Get counts for packing statuses
            total_orders = orders_collection.count_documents({"order_status": "packing"})
            
            # Get today's completed orders (moved to packed)
            today = datetime.utcnow().strftime("%Y-%m-%d")
            completed_today = orders_collection.count_documents({
                "order_status": "packed",
                "updated_at": {"$regex": f"^{today}"}
            })
            
            return {
                "total_orders": total_orders,
                "packing_orders": total_orders,
                "completed_today": completed_today
            }
        except Exception as e:
            print(f"Error getting packer stats: {e}")
            return {
                "total_orders": 0,
                "packing_orders": 0,
                "completed_today": 0
            }
