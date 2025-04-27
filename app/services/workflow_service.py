from typing import List, Dict, Any, Optional
from datetime import datetime

from ..utils.database import get_collection
from .inventory_service import InventoryService
from .orders_service import OrdersService

class WorkflowService:
    """
    Service for managing workflow processes in the warehouse.
    
    This service handles the overall workflow of orders from creation to delivery,
    coordinating between different warehouse operations like receiving, picking,
    packing and shipping.
    """
    
    @staticmethod
    async def process_receiving(receiving_id: int, worker_id: int) -> Dict[str, Any]:
        """
        Process a receiving request and update inventory.
        
        This method handles the complete receiving workflow:
        1. Validate the receiving request
        2. Update inventory with received items
        3. Assign storage locations if needed
        4. Complete the receiving process
        """
        receiving_collection = get_collection("receiving")
        inventory_collection = get_collection("inventory")
        
        # Get receiving request
        receiving = receiving_collection.find_one({"receivingID": receiving_id})
        if not receiving:
            return {"error": f"Receiving request with ID {receiving_id} not found"}
        
        # Check if already processed
        if receiving.get("status") == "completed":
            return {"error": f"Receiving request with ID {receiving_id} has already been processed"}
        
        # Check worker assignment
        if receiving.get("workerID") != worker_id:
            return {"error": f"Worker {worker_id} is not assigned to this receiving request"}
        
        # Process each item
        processed_items = []
        for item in receiving.get("items", []):
            item_id = item.get("itemID")
            quantity = item.get("quantity")
            expected_quantity = item.get("expected_quantity")
            condition = item.get("condition")
            
            # Skip already processed items
            if item.get("processed"):
                processed_items.append(item)
                continue
            
            # Check item condition - only process if good
            if condition != "good":
                processed_items.append({
                    **item,
                    "processed": True,
                    "notes": f"Not processed due to condition: {condition}"
                })
                continue
            
            # Update inventory
            inventory_update = await InventoryService.update_stock_level(
                item_id=item_id,
                quantity_change=quantity,
                reason=f"Receiving {receiving_id}"
            )
            
            # Check for errors
            if "error" in inventory_update:
                processed_items.append({
                    **item,
                    "processed": False,
                    "notes": inventory_update.get("error")
                })
                continue
            
            # Mark as processed
            processed_items.append({
                **item,
                "processed": True,
                "notes": f"Processed {quantity} units"
            })
        
        # Update receiving status
        all_processed = all(item.get("processed") for item in processed_items)
        status = "completed" if all_processed else "processing"
        
        receiving_collection.update_one(
            {"receivingID": receiving_id},
            {
                "$set": {
                    "status": status,
                    "items": processed_items,
                    "updated_at": datetime.utcnow()
                }
            }
        )
        
        # Get updated receiving
        updated_receiving = receiving_collection.find_one({"receivingID": receiving_id})
        return updated_receiving
    
    @staticmethod
    async def process_picking(picking_id: int, worker_id: int) -> Dict[str, Any]:
        """
        Process a picking request and update inventory.
        
        This method handles the picking workflow:
        1. Validate the picking request
        2. Update inventory as items are picked
        3. Complete the picking process
        """
        picking_collection = get_collection("picking")
        inventory_collection = get_collection("inventory")
        orders_collection = get_collection("orders")
        
        # Get picking request
        picking = picking_collection.find_one({"pickingID": picking_id})
        if not picking:
            return {"error": f"Picking request with ID {picking_id} not found"}
        
        # Check if already completed
        if picking.get("status") == "completed":
            return {"error": f"Picking request with ID {picking_id} has already been completed"}
        
        # Check worker assignment
        if picking.get("workerID") != worker_id:
            return {"error": f"Worker {worker_id} is not assigned to this picking request"}
        
        # Update picking status to in_progress if it's pending
        if picking.get("status") == "pending":
            picking_collection.update_one(
                {"pickingID": picking_id},
                {
                    "$set": {
                        "status": "in_progress",
                        "start_time": datetime.utcnow(),
                        "updated_at": datetime.utcnow()
                    }
                }
            )
        
        # Get updated picking
        picking = picking_collection.find_one({"pickingID": picking_id})
        
        return picking
    
    @staticmethod
    async def complete_picking(picking_id: int, worker_id: int, 
                               picked_items: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Complete a picking operation with the actual picked quantities.
        
        Args:
            picking_id: ID of the picking request
            worker_id: ID of the worker performing the picking
            picked_items: List of items picked with their quantities
                [{"itemID": 1, "locationID": 2, "orderDetailID": 3, "actual_quantity": 5}]
        """
        picking_collection = get_collection("picking")
        inventory_collection = get_collection("inventory")
        orders_collection = get_collection("orders")
        
        # Get picking request
        picking = picking_collection.find_one({"pickingID": picking_id})
        if not picking:
            return {"error": f"Picking request with ID {picking_id} not found"}
        
        # Check if already completed
        if picking.get("status") == "completed":
            return {"error": f"Picking request with ID {picking_id} has already been completed"}
        
        # Check worker assignment
        if picking.get("workerID") != worker_id:
            return {"error": f"Worker {worker_id} is not assigned to this picking request"}
        
        # Get order
        order_id = picking.get("orderID")
        order = orders_collection.find_one({"orderID": order_id})
        if not order:
            return {"error": f"Order with ID {order_id} not found"}
        
        # Process each picked item
        updated_items = []
        for item in picking.get("items", []):
            item_id = item.get("itemID")
            location_id = item.get("locationID")
            order_detail_id = item.get("orderDetailID")
            requested_quantity = item.get("quantity")
            
            # Find the picked item data
            picked_item = next((p for p in picked_items if 
                           p.get("itemID") == item_id and 
                           p.get("locationID") == location_id and
                           p.get("orderDetailID") == order_detail_id), None)
            
            if not picked_item:
                # Item not picked
                updated_items.append(item)
                continue
            
            actual_quantity = picked_item.get("actual_quantity", 0)
            
            # Update inventory
            if actual_quantity > 0:
                inventory_update = await InventoryService.update_stock_level(
                    item_id=item_id,
                    quantity_change=-actual_quantity,
                    reason=f"Picking for order {order_id}"
                )
                
                # Check for errors
                if "error" in inventory_update:
                    updated_items.append({
                        **item,
                        "picked": False,
                        "actual_quantity": 0,
                        "notes": inventory_update.get("error")
                    })
                    continue
            
            # Update the item
            updated_items.append({
                **item,
                "picked": True,
                "actual_quantity": actual_quantity,
                "pick_time": datetime.utcnow(),
                "notes": picked_item.get("notes", "")
            })
            
            # Update order detail with fulfilled quantity
            orders_collection.update_one(
                {"orderID": order_id, "items.orderDetailID": order_detail_id},
                {"$inc": {"items.$.fulfilled_quantity": actual_quantity}}
            )
        
        # Check if all items are picked
        all_picked = all(item.get("picked") for item in updated_items)
        
        # Update picking request
        picking_collection.update_one(
            {"pickingID": picking_id},
            {
                "$set": {
                    "status": "completed" if all_picked else "partial",
                    "items": updated_items,
                    "complete_time": datetime.utcnow(),
                    "updated_at": datetime.utcnow()
                }
            }
        )
        
        # Update order status if all picked
        if all_picked:
            orders_collection.update_one(
                {"orderID": order_id},
                {"$set": {"order_status": "picked", "updated_at": datetime.utcnow()}}
            )
        
        # Get updated picking
        updated_picking = picking_collection.find_one({"pickingID": picking_id})
        return updated_picking
    
    @staticmethod
    async def process_packing(packing_id: int, worker_id: int) -> Dict[str, Any]:
        """
        Process a packing request.
        
        This method handles the packing workflow:
        1. Validate the packing request
        2. Mark items as packed
        3. Complete the packing process
        """
        packing_collection = get_collection("packing")
        orders_collection = get_collection("orders")
        
        # Get packing request
        packing = packing_collection.find_one({"packingID": packing_id})
        if not packing:
            return {"error": f"Packing request with ID {packing_id} not found"}
        
        # Check if already completed
        if packing.get("status") == "completed":
            return {"error": f"Packing request with ID {packing_id} has already been completed"}
        
        # Check worker assignment
        if packing.get("workerID") != worker_id:
            return {"error": f"Worker {worker_id} is not assigned to this packing request"}
        
        # Update packing status to in_progress if it's pending
        if packing.get("status") == "pending":
            packing_collection.update_one(
                {"packingID": packing_id},
                {
                    "$set": {
                        "status": "in_progress",
                        "start_time": datetime.utcnow(),
                        "updated_at": datetime.utcnow()
                    }
                }
            )
        
        # Get updated packing
        packing = packing_collection.find_one({"packingID": packing_id})
        
        return packing
    
    @staticmethod
    async def complete_packing(packing_id: int, worker_id: int, 
                               packed_items: List[Dict[str, Any]],
                               package_details: Dict[str, Any]) -> Dict[str, Any]:
        """
        Complete a packing operation with the actual packed quantities.
        
        Args:
            packing_id: ID of the packing request
            worker_id: ID of the worker performing the packing
            packed_items: List of items packed with their quantities
                [{"itemID": 1, "pickingID": 2, "orderDetailID": 3, "actual_quantity": 5}]
            package_details: Details about the package
                {"weight": 2.5, "dimensions": "30x20x15", "package_type": "box"}
        """
        packing_collection = get_collection("packing")
        orders_collection = get_collection("orders")
        
        # Get packing request
        packing = packing_collection.find_one({"packingID": packing_id})
        if not packing:
            return {"error": f"Packing request with ID {packing_id} not found"}
        
        # Check if already completed
        if packing.get("status") == "completed":
            return {"error": f"Packing request with ID {packing_id} has already been completed"}
        
        # Check worker assignment
        if packing.get("workerID") != worker_id:
            return {"error": f"Worker {worker_id} is not assigned to this packing request"}
        
        # Get order
        order_id = packing.get("orderID")
        order = orders_collection.find_one({"orderID": order_id})
        if not order:
            return {"error": f"Order with ID {order_id} not found"}
        
        # Process each packed item
        updated_items = []
        for item in packing.get("items", []):
            item_id = item.get("itemID")
            picking_id = item.get("pickingID")
            order_detail_id = item.get("orderDetailID")
            requested_quantity = item.get("quantity")
            
            # Find the packed item data
            packed_item = next((p for p in packed_items if 
                           p.get("itemID") == item_id and 
                           p.get("pickingID") == picking_id and
                           p.get("orderDetailID") == order_detail_id), None)
            
            if not packed_item:
                # Item not packed
                updated_items.append(item)
                continue
            
            actual_quantity = packed_item.get("actual_quantity", 0)
            
            # Update the item
            updated_items.append({
                **item,
                "packed": True,
                "actual_quantity": actual_quantity,
                "pack_time": datetime.utcnow(),
                "notes": packed_item.get("notes", "")
            })
        
        # Check if all items are packed
        all_packed = all(item.get("packed") for item in updated_items)
        
        # Update packing request
        packing_collection.update_one(
            {"packingID": packing_id},
            {
                "$set": {
                    "status": "completed" if all_packed else "partial",
                    "items": updated_items,
                    "weight": package_details.get("weight"),
                    "dimensions": package_details.get("dimensions"),
                    "package_type": package_details.get("package_type", packing.get("package_type")),
                    "complete_time": datetime.utcnow(),
                    "updated_at": datetime.utcnow()
                }
            }
        )
        
        # Update order status if all packed
        if all_packed:
            orders_collection.update_one(
                {"orderID": order_id},
                {"$set": {"order_status": "packed", "updated_at": datetime.utcnow()}}
            )
        
        # Get updated packing
        updated_packing = packing_collection.find_one({"packingID": packing_id})
        return updated_packing
    
    @staticmethod
    async def process_shipping(shipping_id: int, worker_id: int, 
                              vehicle_id: int, tracking_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a shipping request.
        
        This method handles the shipping workflow:
        1. Validate the shipping request
        2. Assign vehicle and tracking info
        3. Update shipping status
        """
        shipping_collection = get_collection("shipping")
        orders_collection = get_collection("orders")
        vehicle_collection = get_collection("vehicles")
        
        # Get shipping request
        shipping = shipping_collection.find_one({"shippingID": shipping_id})
        if not shipping:
            return {"error": f"Shipping request with ID {shipping_id} not found"}
        
        # Check if already in transit or delivered
        if shipping.get("status") in ["in_transit", "delivered"]:
            return {"error": f"Shipping request with ID {shipping_id} is already {shipping.get('status')}"}
        
        # Check worker assignment
        if shipping.get("workerID") != worker_id:
            return {"error": f"Worker {worker_id} is not assigned to this shipping request"}
        
        # Check vehicle availability
        vehicle = vehicle_collection.find_one({"vehicleID": vehicle_id})
        if not vehicle:
            return {"error": f"Vehicle with ID {vehicle_id} not found"}
        
        if vehicle.get("status") != "available":
            return {"error": f"Vehicle with ID {vehicle_id} is not available (status: {vehicle.get('status')})"}
        
        # Update vehicle status
        vehicle_collection.update_one(
            {"vehicleID": vehicle_id},
            {"$set": {"status": "in_use", "updated_at": datetime.utcnow()}}
        )
        
        # Update shipping with vehicle, tracking info and change status to in_transit
        shipping_collection.update_one(
            {"shippingID": shipping_id},
            {
                "$set": {
                    "vehicleID": vehicle_id,
                    "tracking_number": tracking_info.get("tracking_number"),
                    "estimated_delivery": tracking_info.get("estimated_delivery"),
                    "status": "in_transit",
                    "departure_time": datetime.utcnow(),
                    "updated_at": datetime.utcnow()
                }
            }
        )
        
        # Update order status
        order_id = shipping.get("orderID")
        orders_collection.update_one(
            {"orderID": order_id},
            {"$set": {"order_status": "shipped", "updated_at": datetime.utcnow()}}
        )
        
        # Get updated shipping
        updated_shipping = shipping_collection.find_one({"shippingID": shipping_id})
        return updated_shipping
    
    @staticmethod
    async def complete_shipping(shipping_id: int, worker_id: int, 
                              delivery_proof: str, notes: Optional[str] = None) -> Dict[str, Any]:
        """
        Complete a shipping operation.
        
        Args:
            shipping_id: ID of the shipping request
            worker_id: ID of the worker performing the delivery
            delivery_proof: Proof of delivery (signature, photo, etc.)
            notes: Optional notes about the delivery
        """
        shipping_collection = get_collection("shipping")
        orders_collection = get_collection("orders")
        vehicle_collection = get_collection("vehicles")
        
        # Get shipping request
        shipping = shipping_collection.find_one({"shippingID": shipping_id})
        if not shipping:
            return {"error": f"Shipping request with ID {shipping_id} not found"}
        
        # Check if already delivered
        if shipping.get("status") == "delivered":
            return {"error": f"Shipping request with ID {shipping_id} has already been delivered"}
        
        # Check worker assignment
        if shipping.get("workerID") != worker_id:
            return {"error": f"Worker {worker_id} is not assigned to this shipping request"}
        
        # Update shipping to delivered
        shipping_collection.update_one(
            {"shippingID": shipping_id},
            {
                "$set": {
                    "status": "delivered",
                    "actual_delivery": datetime.utcnow(),
                    "delivery_proof": delivery_proof,
                    "notes": notes,
                    "updated_at": datetime.utcnow()
                }
            }
        )
        
        # Update order status
        order_id = shipping.get("orderID")
        orders_collection.update_one(
            {"orderID": order_id},
            {"$set": {"order_status": "delivered", "updated_at": datetime.utcnow()}}
        )
        
        # Update vehicle status to available
        vehicle_id = shipping.get("vehicleID")
        if vehicle_id:
            vehicle_collection.update_one(
                {"vehicleID": vehicle_id},
                {"$set": {"status": "available", "updated_at": datetime.utcnow()}}
            )
        
        # Get updated shipping
        updated_shipping = shipping_collection.find_one({"shippingID": shipping_id})
        return updated_shipping
    
    @staticmethod
    async def process_return(return_id: int, worker_id: int) -> Dict[str, Any]:
        """
        Process a return request.
        
        This method handles the return workflow:
        1. Validate the return request
        2. Inspect returned items
        3. Update inventory if items are resellable
        4. Process refund if applicable
        """
        returns_collection = get_collection("returns")
        inventory_collection = get_collection("inventory")
        orders_collection = get_collection("orders")
        
        # Get return request
        return_req = returns_collection.find_one({"returnID": return_id})
        if not return_req:
            return {"error": f"Return request with ID {return_id} not found"}
        
        # Check if already processed
        if return_req.get("status") == "completed":
            return {"error": f"Return request with ID {return_id} has already been processed"}
        
        # Check worker assignment
        if return_req.get("workerID") != worker_id:
            return {"error": f"Worker {worker_id} is not assigned to this return request"}
        
        # Update return status to processing if it's pending
        if return_req.get("status") == "pending":
            returns_collection.update_one(
                {"returnID": return_id},
                {
                    "$set": {
                        "status": "processing",
                        "updated_at": datetime.utcnow()
                    }
                }
            )
        
        # Get updated return
        return_req = returns_collection.find_one({"returnID": return_id})
        
        return return_req
    
    @staticmethod
    async def complete_return(return_id: int, worker_id: int, 
                             processed_items: List[Dict[str, Any]],
                             refund_details: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Complete a return operation.
        
        Args:
            return_id: ID of the return request
            worker_id: ID of the worker processing the return
            processed_items: List of processed return items with their status
                [{"itemID": 1, "orderDetailID": 2, "resellable": true, "locationID": 3}]
            refund_details: Optional refund details
                {"refund_amount": 49.99, "refund_status": "processed"}
        """
        returns_collection = get_collection("returns")
        inventory_collection = get_collection("inventory")
        
        # Get return request
        return_req = returns_collection.find_one({"returnID": return_id})
        if not return_req:
            return {"error": f"Return request with ID {return_id} not found"}
        
        # Check if already completed
        if return_req.get("status") == "completed":
            return {"error": f"Return request with ID {return_id} has already been completed"}
        
        # Check worker assignment
        if return_req.get("workerID") != worker_id:
            return {"error": f"Worker {worker_id} is not assigned to this return request"}
        
        # Process each return item
        updated_items = []
        for item in return_req.get("items", []):
            item_id = item.get("itemID")
            order_detail_id = item.get("orderDetailID")
            quantity = item.get("quantity")
            
            # Find the processed item data
            processed_item = next((p for p in processed_items if 
                               p.get("itemID") == item_id and 
                               p.get("orderDetailID") == order_detail_id), None)
            
            if not processed_item:
                # Item not processed
                updated_items.append(item)
                continue
            
            resellable = processed_item.get("resellable", False)
            location_id = processed_item.get("locationID") if resellable else None
            
            # Update inventory if resellable
            if resellable:
                inventory_update = await InventoryService.update_stock_level(
                    item_id=item_id,
                    quantity_change=quantity,
                    reason=f"Return {return_id}"
                )
                
                # Check for errors
                if "error" in inventory_update:
                    updated_items.append({
                        **item,
                        "processed": False,
                        "resellable": False,
                        "notes": inventory_update.get("error")
                    })
                    continue
            
            # Update the item
            updated_items.append({
                **item,
                "processed": True,
                "resellable": resellable,
                "locationID": location_id,
                "notes": processed_item.get("notes", "")
            })
        
        # Check if all items are processed
        all_processed = all(item.get("processed") for item in updated_items)
        
        # Prepare update data
        update_data = {
            "status": "completed" if all_processed else "partial",
            "items": updated_items,
            "updated_at": datetime.utcnow()
        }
        
        # Add refund details if provided
        if refund_details:
            update_data.update({
                "refund_amount": refund_details.get("refund_amount"),
                "refund_status": refund_details.get("refund_status"),
                "refund_date": datetime.utcnow()
            })
        
        # Update return request
        returns_collection.update_one(
            {"returnID": return_id},
            {"$set": update_data}
        )
        
        # Get updated return
        updated_return = returns_collection.find_one({"returnID": return_id})
        return updated_return
    
    @staticmethod
    async def optimize_order_fulfillment(worker_roles: Dict[str, int] = None) -> Dict[str, Any]:
        """
        Optimize the order fulfillment process based on current workload and available resources.
        
        This method analyzes pending orders and available resources to suggest the optimal
        sequence of operations to maximize efficiency and throughput.
        
        Args:
            worker_roles: Dict of available workers by role, e.g. {"Picker": 3, "Packer": 2}
                          If None, will use all available workers
        """
        orders_collection = get_collection("orders")
        picking_collection = get_collection("picking")
        packing_collection = get_collection("packing")
        shipping_collection = get_collection("shipping")
        workers_collection = get_collection("workers")
        
        # Get available workers by role if not provided
        if not worker_roles:
            worker_roles = {}
            for role in ["Picker", "Packer", "Driver", "ReceivingClerk"]:
                count = workers_collection.count_documents({
                    "role": role,
                    "disabled": False
                })
                worker_roles[role] = count
        
        # Get pending orders
        pending_orders = list(orders_collection.find({
            "order_status": {"$in": ["pending", "picking", "picked", "packing"]}
        }).sort([("priority", 1), ("order_date", 1)]))
        
        # Get current workload
        active_picking = picking_collection.count_documents({
            "status": {"$in": ["pending", "in_progress"]}
        })
        active_packing = packing_collection.count_documents({
            "status": {"$in": ["pending", "in_progress"]}
        })
        active_shipping = shipping_collection.count_documents({
            "status": {"$in": ["pending"]}
        })
        
        # Calculate capacity and backlog
        picking_capacity = worker_roles.get("Picker", 0) * 8  # Assuming 8 orders per picker per shift
        packing_capacity = worker_roles.get("Packer", 0) * 10  # Assuming 10 orders per packer per shift
        shipping_capacity = worker_roles.get("Driver", 0) * 12  # Assuming 12 deliveries per driver per shift
        
        picking_backlog = max(0, active_picking - picking_capacity)
        packing_backlog = max(0, active_packing - packing_capacity)
        shipping_backlog = max(0, active_shipping - shipping_capacity)
        
        # Identify bottlenecks
        bottlenecks = []
        if picking_backlog > 0 and picking_backlog >= packing_backlog and picking_backlog >= shipping_backlog:
            bottlenecks.append("Picking")
        if packing_backlog > 0 and packing_backlog >= picking_backlog and packing_backlog >= shipping_backlog:
            bottlenecks.append("Packing")
        if shipping_backlog > 0 and shipping_backlog >= picking_backlog and shipping_backlog >= packing_backlog:
            bottlenecks.append("Shipping")
        
        # Suggested workflow optimizations
        optimizations = []
        
        # If picking is bottleneck, prioritize high-value/urgent orders for picking
        if "Picking" in bottlenecks:
            optimizations.append({
                "area": "Picking",
                "action": "Prioritize high-value/urgent orders",
                "recommendation": "Assign additional temporary pickers or overtime"
            })
        
        # If packing is bottleneck, batch similar orders for efficient packing
        if "Packing" in bottlenecks:
            optimizations.append({
                "area": "Packing",
                "action": "Batch similar orders for efficient packing",
                "recommendation": "Group orders with similar packaging requirements"
            })
        
        # If shipping is bottleneck, optimize delivery routes
        if "Shipping" in bottlenecks:
            optimizations.append({
                "area": "Shipping",
                "action": "Optimize delivery routes",
                "recommendation": "Group deliveries by geographic area"
            })
        
        # Recommended order processing sequence
        recommended_sequence = []
        for order in pending_orders[:20]:  # Limit to top 20 orders
            order_id = order.get("orderID")
            priority = order.get("priority")
            status = order.get("order_status")
            
            # Determine next action based on current status
            next_action = "start_picking"
            if status == "picking":
                next_action = "complete_picking"
            elif status == "picked":
                next_action = "start_packing"
            elif status == "packing":
                next_action = "complete_packing"
            
            recommended_sequence.append({
                "orderID": order_id,
                "priority": priority,
                "status": status,
                "next_action": next_action
            })
        
        return {
            "current_workload": {
                "active_picking": active_picking,
                "active_packing": active_packing,
                "active_shipping": active_shipping
            },
            "capacity": {
                "picking": picking_capacity,
                "packing": packing_capacity,
                "shipping": shipping_capacity
            },
            "bottlenecks": bottlenecks,
            "optimizations": optimizations,
            "recommended_sequence": recommended_sequence
        }