from typing import List, Dict, Any, Optional
from datetime import datetime

from ..utils.database import get_collection
from ..models.order import OrderCreate, OrderUpdate
from .inventory_service import InventoryService

class OrdersService:
    """
    Service for order management operations.
    
    This service handles business logic for order processing, including
    creating orders, updating status, and generating picking lists.
    """
    
    @staticmethod
    def _normalize_order_data(order: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalize order data to match the expected API response format.
        
        This handles inconsistent data structures in the database.
        """
        if not order:
            return order
            
        # Create a copy to avoid modifying the original
        normalized = order.copy()
        
        # Convert string IDs to integers where possible, otherwise use defaults
        try:
            if isinstance(normalized.get('customerID'), str):
                # For string customerIDs like 'CUST001', extract number or use default
                customer_id = normalized.get('customerID', '')
                if customer_id.startswith('CUST'):
                    normalized['customerID'] = int(customer_id[4:]) if customer_id[4:].isdigit() else 1
                else:
                    normalized['customerID'] = 1
        except (ValueError, TypeError):
            normalized['customerID'] = 1
            
        try:
            if isinstance(normalized.get('orderID'), str):
                # For UUID orderIDs, we'll need to generate a sequential ID
                # For now, use the hash of the string
                normalized['orderID'] = abs(hash(normalized['orderID'])) % 1000000
        except (ValueError, TypeError):
            normalized['orderID'] = 1
            
        try:
            if isinstance(normalized.get('assigned_worker'), str):
                # For string worker IDs like 'clerk1', extract number or use default
                worker_id = normalized.get('assigned_worker', '')
                if worker_id.startswith('clerk'):
                    normalized['assigned_worker'] = int(worker_id[5:]) if worker_id[5:].isdigit() else 1
                else:
                    normalized['assigned_worker'] = 1
        except (ValueError, TypeError):
            if 'assigned_worker' in normalized:
                normalized['assigned_worker'] = 1
        
        # Ensure shipping_address exists
        if 'shipping_address' not in normalized or not normalized['shipping_address']:
            normalized['shipping_address'] = 'Address not specified'
            
        # Normalize items structure
        if 'items' in normalized:
            normalized_items = []
            for i, item in enumerate(normalized['items']):
                normalized_item = item.copy()
                
                # Convert item_id to itemID if needed
                if 'item_id' in normalized_item and 'itemID' not in normalized_item:
                    try:
                        item_id = normalized_item['item_id']
                        if isinstance(item_id, str) and item_id.startswith('ITEM'):
                            normalized_item['itemID'] = int(item_id[4:]) if item_id[4:].isdigit() else 1
                        else:
                            normalized_item['itemID'] = 1
                    except (ValueError, TypeError):
                        normalized_item['itemID'] = 1
                    # Remove old field
                    normalized_item.pop('item_id', None)
                
                # Convert unit_price to price if needed
                if 'unit_price' in normalized_item and 'price' not in normalized_item:
                    normalized_item['price'] = normalized_item['unit_price']
                    normalized_item.pop('unit_price', None)
                
                # Ensure orderDetailID exists
                if 'orderDetailID' not in normalized_item:
                    normalized_item['orderDetailID'] = i + 1
                
                # Ensure required fields exist
                if 'fulfilled_quantity' not in normalized_item:
                    normalized_item['fulfilled_quantity'] = 0
                    
                normalized_items.append(normalized_item)
            
            normalized['items'] = normalized_items
        
        return normalized
    
    @staticmethod
    async def get_orders(skip: int = 0, limit: int = 100, 
                         status: Optional[str] = None,
                         customer_id: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get orders with optional filtering.
        """
        orders_collection = get_collection("orders")
        
        # Build query
        query = {}
        if status:
            query["order_status"] = status
        if customer_id:
            query["customerID"] = customer_id
        
        # Execute query with sorting by priority and date
        orders = list(orders_collection.find(query)
                     .sort([("priority", 1), ("order_date", 1)])
                     .skip(skip).limit(limit))
        
        # Normalize all orders
        normalized_orders = [OrdersService._normalize_order_data(order) for order in orders]
        return normalized_orders
    
    @staticmethod
    async def get_order(order_id: int) -> Dict[str, Any]:
        """
        Get a specific order by ID.
        """
        orders_collection = get_collection("orders")
        order = orders_collection.find_one({"orderID": order_id})
        return OrdersService._normalize_order_data(order) if order else order
    
    @staticmethod
    def get_order_by_id(order_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a specific order by ID (synchronous version for string IDs).
        """
        orders_collection = get_collection("orders")
        # Try to find by orderID as string first, then as int
        order = orders_collection.find_one({"orderID": order_id})
        if not order:
            try:
                # Try to find by orderID as integer
                order = orders_collection.find_one({"orderID": int(order_id)})
            except ValueError:
                pass
        return OrdersService._normalize_order_data(order) if order else order
    
    @staticmethod
    async def create_order(order: OrderCreate) -> Dict[str, Any]:
        """
        Create a new order.
        """
        orders_collection = get_collection("orders")
        inventory_collection = get_collection("inventory")
        
        # Find the next available orderID
        last_order = orders_collection.find_one(
            sort=[("orderID", -1)]
        )
        next_id = 1
        if last_order:
            next_id = last_order.get("orderID", 0) + 1
        
        # Calculate total amount and prepare order_details
        total_amount = 0
        order_details = []
        
        # Assign IDs to order details and validate inventory
        for i, item in enumerate(order.items):
            # Check if item exists and has sufficient stock
            inventory_item = inventory_collection.find_one({"itemID": item.itemID})
            if not inventory_item:
                return {"error": f"Item with ID {item.itemID} not found"}
            
            if inventory_item.get("stock_level", 0) < item.quantity:
                return {"error": f"Insufficient stock for item {item.itemID}. Available: {inventory_item.get('stock_level')}, Requested: {item.quantity}"}
            
            # Create order detail with ID
            order_detail = item.model_dump()
            order_detail["orderDetailID"] = i + 1
            order_detail["fulfilled_quantity"] = 0
            order_detail["created_at"] = datetime.utcnow()
            order_detail["updated_at"] = datetime.utcnow()
            
            # Add to total amount
            total_amount += item.quantity * item.price
            
            order_details.append(order_detail)
        
        # Prepare order document
        order_data = order.model_dump(exclude={"items"})
        order_data.update({
            "orderID": next_id,
            "items": order_details,
            "total_amount": total_amount,
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow()
        })
        
        # Insert order to database
        result = orders_collection.insert_one(order_data)
        
        # Return the created order
        created_order = orders_collection.find_one({"_id": result.inserted_id})
        return created_order
    
    @staticmethod
    async def update_order(order_id: int, order_update: OrderUpdate) -> Dict[str, Any]:
        """
        Update an order.
        """
        orders_collection = get_collection("orders")
        
        # Prepare update data
        update_data = order_update.model_dump(exclude_unset=True)
        update_data["updated_at"] = datetime.utcnow()
        
        # Update order
        orders_collection.update_one(
            {"orderID": order_id},
            {"$set": update_data}
        )
        
        # Return updated order
        updated_order = orders_collection.find_one({"orderID": order_id})
        return updated_order
    
    @staticmethod
    async def delete_order(order_id: int) -> Dict[str, Any]:
        """
        Delete an order.
        """
        orders_collection = get_collection("orders")
        
        # Delete order
        orders_collection.delete_one({"orderID": order_id})
        
        return {"message": f"Order with ID {order_id} has been deleted"}
    
    @staticmethod
    async def update_order_status(order_id: int, status: str, worker_id: Optional[int] = None) -> Dict[str, Any]:
        """
        Update the status of an order and optionally assign a worker.
        """
        orders_collection = get_collection("orders")
        
        # Prepare update data
        update_data = {
            "order_status": status,
            "updated_at": datetime.utcnow()
        }
        
        if worker_id:
            update_data["assigned_worker"] = worker_id
        
        # Update order
        orders_collection.update_one(
            {"orderID": order_id},
            {"$set": update_data}
        )
        
        # Return updated order
        updated_order = orders_collection.find_one({"orderID": order_id})
        return updated_order
    
    @staticmethod
    async def generate_picking_list(order_id: int) -> Dict[str, Any]:
        """
        Generate a picking list for an order.
        
        This creates a structured list of items to pick,
        organized by warehouse location for efficiency.
        """
        orders_collection = get_collection("orders")
        inventory_collection = get_collection("inventory")
        location_collection = get_collection("locations")
        
        # Get order
        order = orders_collection.find_one({"orderID": order_id})
        if not order:
            return {"error": f"Order with ID {order_id} not found"}
        
        # Check order status
        if order.get("order_status") not in ["processing", "pending"]:
            return {"error": f"Cannot generate picking list for order with status {order.get('order_status')}"}
        
        picking_list = []
        
        # Process each order item
        for item in order.get("items", []):
            item_id = item.get("itemID")
            quantity = item.get("quantity")
            
            # Find inventory locations for this item
            inventory_locations = list(inventory_collection.find({
                "itemID": item_id,
                "stock_level": {"$gt": 0}
            }))
            
            # If no inventory found, add to list with no location
            if not inventory_locations:
                picking_list.append({
                    "itemID": item_id,
                    "orderDetailID": item.get("orderDetailID"),
                    "name": "Item not found",
                    "quantity": quantity,
                    "location": None,
                    "status": "not_found"
                })
                continue
            
            # Get the location details for the inventory
            remaining_quantity = quantity
            for inv_item in inventory_locations:
                location_id = inv_item.get("locationID")
                available_quantity = inv_item.get("stock_level", 0)
                
                # Skip if no location assigned
                if not location_id:
                    continue
                
                # Get location details
                location = location_collection.find_one({"locationID": location_id})
                if not location:
                    continue
                
                # Calculate how much to pick from this location
                pick_quantity = min(remaining_quantity, available_quantity)
                
                # Add to picking list
                picking_list.append({
                    "itemID": item_id,
                    "orderDetailID": item.get("orderDetailID"),
                    "name": inv_item.get("name"),
                    "quantity": pick_quantity,
                    "location": {
                        "locationID": location_id,
                        "section": location.get("section"),
                        "row": location.get("row"),
                        "shelf": location.get("shelf"),
                        "bin": location.get("bin")
                    },
                    "status": "to_pick"
                })
                
                remaining_quantity -= pick_quantity
                if remaining_quantity <= 0:
                    break
            
            # If still have remaining quantity, add to list as unavailable
            if remaining_quantity > 0:
                picking_list.append({
                    "itemID": item_id,
                    "orderDetailID": item.get("orderDetailID"),
                    "name": inventory_locations[0].get("name"),
                    "quantity": remaining_quantity,
                    "location": None,
                    "status": "unavailable"
                })
        
        # Sort picking list by location for efficiency
        # This is a simple sort - in real applications, you would use a path optimization algorithm
        picking_list.sort(key=lambda x: (
            x.get("location", {}).get("section", ""),
            x.get("location", {}).get("row", ""),
            x.get("location", {}).get("shelf", ""),
            x.get("location", {}).get("bin", "")
        ))
        
        return {
            "orderID": order_id,
            "customer": order.get("customerID"),
            "picking_list": picking_list,
            "generated_at": datetime.utcnow()
        }
    
    @staticmethod
    async def check_order_availability(order_id: int) -> Dict[str, Any]:
        """
        Check if all items in an order are available.
        """
        orders_collection = get_collection("orders")
        inventory_collection = get_collection("inventory")
        
        # Get order
        order = orders_collection.find_one({"orderID": order_id})
        if not order:
            return {"error": f"Order with ID {order_id} not found"}
        
        availability = []
        all_available = True
        
        # Check each item
        for item in order.get("items", []):
            item_id = item.get("itemID")
            quantity = item.get("quantity")
            
            # Get total stock for this item
            inventory_items = list(inventory_collection.find({"itemID": item_id}))
            total_stock = sum(inv.get("stock_level", 0) for inv in inventory_items)
            
            is_available = total_stock >= quantity
            if not is_available:
                all_available = False
            
            availability.append({
                "itemID": item_id,
                "required_quantity": quantity,
                "available_quantity": total_stock,
                "is_available": is_available
            })
        
        return {
            "orderID": order_id,
            "all_available": all_available,
            "items": availability,
            "checked_at": datetime.utcnow()
        }
    
    @staticmethod
    async def allocate_inventory_for_order(order_id: int) -> Dict[str, Any]:
        """
        Allocate inventory for an order.
        
        This method reserves inventory items for an order by creating
        inventory allocations but does not reduce the actual stock level.
        """
        orders_collection = get_collection("orders")
        inventory_collection = get_collection("inventory")
        allocations_collection = get_collection("inventory_allocations")
        
        # Get order
        order = orders_collection.find_one({"orderID": order_id})
        if not order:
            return {"error": f"Order with ID {order_id} not found"}
        
        # Check if already allocated
        existing_allocation = allocations_collection.find_one({"orderID": order_id})
        if existing_allocation:
            return {"error": f"Inventory already allocated for order {order_id}"}
        
        # Check availability first
        availability = await OrdersService.check_order_availability(order_id)
        if not availability.get("all_available", False):
            return {"error": "Not all items are available for allocation", "details": availability}
        
        allocations = []
        
        # Process each order item
        for item in order.get("items", []):
            item_id = item.get("itemID")
            quantity = item.get("quantity")
            
            # Find inventory locations for this item
            inventory_locations = list(inventory_collection.find({
                "itemID": item_id,
                "stock_level": {"$gt": 0}
            }))
            
            # Allocate from each location until fulfilled
            remaining_quantity = quantity
            for inv_item in inventory_locations:
                location_id = inv_item.get("locationID")
                available_quantity = inv_item.get("stock_level", 0)
                
                # Skip if no location assigned
                if not location_id:
                    continue
                
                # Calculate how much to allocate from this location
                allocate_quantity = min(remaining_quantity, available_quantity)
                
                # Create allocation record
                allocation = {
                    "orderID": order_id,
                    "orderDetailID": item.get("orderDetailID"),
                    "itemID": item_id,
                    "locationID": location_id,
                    "quantity": allocate_quantity,
                    "allocated_at": datetime.utcnow()
                }
                allocations.append(allocation)
                
                # Update inventory with allocation
                inventory_collection.update_one(
                    {"itemID": item_id, "locationID": location_id},
                    {"$inc": {"allocated": allocate_quantity}}
                )
                
                remaining_quantity -= allocate_quantity
                if remaining_quantity <= 0:
                    break
        
        # Insert all allocations
        if allocations:
            allocations_collection.insert_many(allocations)
        
        # Update order status
        orders_collection.update_one(
            {"orderID": order_id},
            {"$set": {"order_status": "allocated", "updated_at": datetime.utcnow()}}
        )
        
        return {
            "message": f"Inventory allocated for order {order_id}",
            "orderID": order_id,
            "allocations": len(allocations),
            "allocated_at": datetime.utcnow()
        }