from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field
from datetime import datetime

# Use absolute imports instead of relative imports
from app.tools.chatbot.base_tool import WMSBaseTool, create_tool
from app.utils.chatbot.mongodb_client import chatbot_mongodb_client
from app.utils.chatbot.knowledge_base import knowledge_base
from app.utils.chatbot.demo_data import get_demo_orders, is_api_error

# Define order query tool
async def order_query_func(order_id: Optional[int] = None,
                     customer_id: Optional[int] = None,
                     status: Optional[str] = None,
                     priority: Optional[str] = None,
                     date_from: Optional[str] = None,
                     date_to: Optional[str] = None) -> str:
    """Query orders based on various criteria using direct MongoDB access."""
    
    try:
        # If order_id is provided, do a direct lookup
        if order_id is not None:
            order = await chatbot_mongodb_client.get_order_by_id(order_id)
            
            if not order:
                return f"Order with ID {order_id} not found in the database."
            
            # Format single order response
            result = "Found order:\n\n"
            result += f"Order ID: {order.get('orderID')}\n"
            result += f"Customer ID: {order.get('customerID')}\n"
            result += f"Status: {order.get('order_status')}\n"
            result += f"Priority: {order.get('priority', 'Normal')}\n"
            result += f"Total Amount: ${order.get('total_amount', 'N/A')}\n"
            result += f"Order Date: {order.get('order_date', 'N/A')}\n"
            result += f"Shipping Address: {order.get('shipping_address', 'N/A')}\n"
            result += f"Assigned Worker: {order.get('assigned_worker', 'None')}\n"
            result += f"Notes: {order.get('notes', 'N/A')}\n"
            
            # Add order items if available
            items = order.get('items', [])
            if items:
                result += f"\nOrder Items ({len(items)}):\n"
                for item in items:
                    result += f"- Item ID: {item.get('itemID')} (Qty: {item.get('quantity', 0)}, Price: ${item.get('price', 'N/A')})\n"
                    result += f"  Fulfilled: {item.get('fulfilled_quantity', 0)}/{item.get('quantity', 0)}\n"
            
            return result
        
        # Build filter criteria for MongoDB query
        filter_criteria = {}
        
        if customer_id:
            filter_criteria["customerID"] = customer_id
        if status:
            filter_criteria["order_status"] = status
        if priority:
            filter_criteria["priority"] = priority
        
        # Handle date filtering (simplified for now)
        if date_from or date_to:
            date_filter = {}
            if date_from:
                # Note: In a real implementation, you'd parse the date string
                pass
            if date_to:
                # Note: In a real implementation, you'd parse the date string  
                pass
            if date_filter:
                filter_criteria["order_date"] = date_filter
        
        # Get orders based on criteria
        if customer_id:
            orders = await chatbot_mongodb_client.get_orders_by_customer(customer_id)
        elif status:
            orders = await chatbot_mongodb_client.get_orders_by_status(status)
        else:
            orders = await chatbot_mongodb_client.get_orders(filter_criteria)
        
        if not orders:
            return "No orders found matching your criteria."
            
        # Format the results
        result = f"Found {len(orders)} order(s):\n\n"
        for order in orders:
            result += f"Order ID: {order.get('orderID')}\n"
            result += f"Customer ID: {order.get('customerID')}\n"
            result += f"Status: {order.get('order_status')}\n"
            result += f"Priority: {order.get('priority', 'Normal')}\n"
            result += f"Total: ${order.get('total_amount', 'N/A')}\n"
            result += f"Order Date: {order.get('order_date', 'N/A')}\n"
            result += f"Assigned Worker: {order.get('assigned_worker', 'None')}\n"
            result += "-" * 40 + "\n"
            
        return result
        
    except Exception as e:
        # Fallback to demo data on database error
        try:
            orders = get_demo_orders(
                order_id=order_id, customer_id=customer_id, status=status, priority=priority
            )
            if not orders:
                return "No orders found matching your criteria."
            
            result = "Found the following orders (Demo data - Database error):\n\n"
            for order in orders:
                result += f"Order ID: {order.get('id')}\n"
                result += f"Customer ID: {order.get('customer_id')}\n"
                result += f"Status: {order.get('status')}\n"
                result += f"Priority: {order.get('priority', 'Normal')}\n"
                result += f"Total: ${order.get('total_amount', 'N/A')}\n"
                result += f"Created: {order.get('created_at', 'N/A')}\n"
                result += "-" * 40 + "\n"
            
            return result
        except:
            return f"Error querying orders: {str(e)}"

# Define order create tool
async def order_create_func(customer_id: int,
                      items: List[Dict[str, Any]],
                      shipping_address: str,
                      priority: Optional[str] = "normal",
                      notes: Optional[str] = None) -> str:
    """Create a new order using direct MongoDB access."""
    try:
        # Validate customer exists
        customer = await chatbot_mongodb_client.get_customer_by_id(customer_id)
        if not customer:
            return f"❌ Customer with ID {customer_id} not found."
        
        # Validate items exist in inventory
        total_amount = 0.0
        validated_items = []
        
        for item in items:
            item_id = item.get('item_id') or item.get('itemID')
            quantity = item.get('quantity', 1)
            
            if not item_id:
                return "❌ Each item must have an 'item_id' or 'itemID' field."
            
            # Check if inventory item exists
            inventory_item = await chatbot_mongodb_client.get_inventory_item_by_id(item_id)
            if not inventory_item:
                return f"❌ Inventory item with ID {item_id} not found."
            
            # Check stock availability
            available_stock = inventory_item.get('stock_level', 0)
            if available_stock < quantity:
                return f"❌ Insufficient stock for item {item_id}. Available: {available_stock}, Requested: {quantity}"
            
            # Calculate item total
            unit_price = inventory_item.get('unit_price', 0.0)
            item_total = unit_price * quantity
            total_amount += item_total
            
            validated_items.append({
                'itemID': item_id,
                'quantity': quantity,
                'unit_price': unit_price,
                'total_price': item_total,
                'fulfilled_quantity': 0
            })
        
        # Prepare order data
        order_data = {
            'customerID': customer_id,
            'items': validated_items,
            'shipping_address': shipping_address,
            'priority': priority.lower() if priority else 'normal',
            'total_amount': round(total_amount, 2),
            'order_date': datetime.utcnow(),
            'assigned_worker': None
        }
        
        if notes:
            order_data['notes'] = notes
        
        # Create the order
        result = await chatbot_mongodb_client.create_order(order_data)
        
        if result.get('success'):
            order = result.get('order', {})
            response = f"✅ Successfully created order!\n\n"
            response += f"Order ID: {result.get('order_id')}\n"
            response += f"Customer: {customer.get('name', 'Unknown')} (ID: {customer_id})\n"
            response += f"Total Amount: ${total_amount:.2f}\n"
            response += f"Priority: {priority}\n"
            response += f"Items: {len(validated_items)} item(s)\n"
            response += f"Shipping Address: {shipping_address}\n"
            if notes:
                response += f"Notes: {notes}\n"
            response += f"Status: {order.get('order_status', 'pending')}\n"
            
            return response
        else:
            return f"❌ Failed to create order: {result.get('message', 'Unknown error')}"
            
    except Exception as e:
        return f"❌ Error creating order: {str(e)}"

# Define order update tool
async def order_update_func(order_id: int,
                      status: Optional[str] = None,
                      priority: Optional[str] = None,
                      shipping_address: Optional[str] = None,
                      notes: Optional[str] = None) -> str:
    """Update an existing order using direct MongoDB access."""
    try:
        # First, check if the order exists
        existing_order = await chatbot_mongodb_client.get_order_by_id(order_id)
        if not existing_order:
            return f"❌ Order with ID {order_id} not found in the database."
        
        # Prepare update data - only include non-None values
        update_data = {}
        if status is not None:
            # Validate status
            valid_statuses = ['pending', 'approved', 'rejected', 'processing', 'picking', 'packing', 'shipped', 'delivered', 'cancelled']
            if status.lower() not in valid_statuses:
                return f"❌ Invalid status '{status}'. Valid options: {', '.join(valid_statuses)}"
            update_data["order_status"] = status.lower()
        
        if priority is not None:
            # Validate priority
            valid_priorities = ['low', 'normal', 'high', 'urgent']
            if priority.lower() not in valid_priorities:
                return f"❌ Invalid priority '{priority}'. Valid options: {', '.join(valid_priorities)}"
            update_data["priority"] = priority.lower()
        
        if shipping_address is not None:
            update_data["shipping_address"] = shipping_address
        
        if notes is not None:
            update_data["notes"] = notes
            
        if not update_data:
            return "❌ No update data provided. Please specify at least one field to update."
        
        # Update the order in the database
        result = await chatbot_mongodb_client.update_order(order_id, update_data)
        
        if result.get("success"):
            response = f"✅ Successfully updated order {order_id}!\n\n"
            response += f"Updated fields:\n"
            
            for field, value in update_data.items():
                if field == "order_status":
                    response += f"• Status: {value}\n"
                elif field == "shipping_address":
                    response += f"• Shipping Address: {value}\n"
                else:
                    response += f"• {field.replace('_', ' ').title()}: {value}\n"
            
            # Show current order details
            updated_order = result.get("order", {})
            if updated_order:
                response += f"\nCurrent order details:\n"
                response += f"Customer ID: {updated_order.get('customerID')}\n"
                response += f"Status: {updated_order.get('order_status')}\n"
                response += f"Priority: {updated_order.get('priority')}\n"
                response += f"Total Amount: ${updated_order.get('total_amount', 0):.2f}\n"
                response += f"Shipping Address: {updated_order.get('shipping_address', 'N/A')}\n"
                if updated_order.get('notes'):
                    response += f"Notes: {updated_order.get('notes')}\n"
            
            return response
        else:
            return f"❌ Failed to update order: {result.get('message', 'Unknown error')}"
            
    except Exception as e:
        return f"❌ Error updating order: {str(e)}"

# Define create sub-order tool
async def create_sub_order_func(parent_order_id: int, 
                         items: List[Dict[str, Any]],
                         reason: str) -> str:
    """Create a sub-order for partial fulfillment of a parent order using direct MongoDB access."""
    try:
        # Validate parent order exists
        parent_order = await chatbot_mongodb_client.get_order_by_id(parent_order_id)
        if not parent_order:
            return f"❌ Parent order with ID {parent_order_id} not found."
        
        # Validate parent order is approved
        if parent_order.get('order_status') not in ['approved', 'processing']:
            return f"❌ Parent order {parent_order_id} must be approved before creating sub-orders. Current status: {parent_order.get('order_status')}"
        
        # Validate items are part of the parent order
        parent_items = {item.get('itemID'): item for item in parent_order.get('items', [])}
        validated_items = []
        
        for sub_item in items:
            item_id = sub_item.get('item_id') or sub_item.get('itemID')
            quantity = sub_item.get('quantity', 1)
            
            if not item_id:
                return "❌ Each item must have an 'item_id' or 'itemID' field."
            
            if item_id not in parent_items:
                return f"❌ Item {item_id} is not part of parent order {parent_order_id}."
            
            parent_item = parent_items[item_id]
            original_quantity = parent_item.get('quantity', 0)
            fulfilled_quantity = parent_item.get('fulfilled_quantity', 0)
            available_quantity = original_quantity - fulfilled_quantity
            
            if quantity > available_quantity:
                return f"❌ Requested quantity ({quantity}) exceeds available quantity ({available_quantity}) for item {item_id}."
            
            validated_items.append({
                'itemID': item_id,
                'quantity': quantity,
                'unit_price': parent_item.get('unit_price', 0.0),
                'total_price': parent_item.get('unit_price', 0.0) * quantity,
                'fulfilled_quantity': 0
            })
        
        # Create the sub-order
        result = await chatbot_mongodb_client.create_sub_order(parent_order_id, validated_items, reason)
        
        if result.get('success'):
            sub_order = result.get('order', {})
            response = f"✅ Successfully created sub-order!\n\n"
            response += f"Sub-order ID: {result.get('order_id')}\n"
            response += f"Parent Order ID: {parent_order_id}\n"
            response += f"Customer ID: {sub_order.get('customerID')}\n"
            response += f"Reason: {reason}\n"
            response += f"Items: {len(validated_items)} item(s)\n"
            response += f"Status: {sub_order.get('order_status', 'pending')}\n"
            
            # Calculate sub-order total
            total_amount = sum(item.get('total_price', 0) for item in validated_items)
            response += f"Sub-order Total: ${total_amount:.2f}\n"
            
            # Show items
            response += f"\nSub-order Items:\n"
            for item in validated_items:
                response += f"- Item ID: {item.get('itemID')} (Qty: {item.get('quantity')}, Price: ${item.get('total_price', 0):.2f})\n"
            
            return response
        else:
            return f"❌ Failed to create sub-order: {result.get('message', 'Unknown error')}"
            
    except Exception as e:
        return f"❌ Error creating sub-order: {str(e)}"

# Define approve orders tool
async def approve_orders_func(order_id: int, approved: bool, notes: Optional[str] = None) -> str:
    """Approve or reject an order using direct MongoDB access."""
    try:
        # First, check if the order exists
        existing_order = await chatbot_mongodb_client.get_order_by_id(order_id)
        if not existing_order:
            return f"❌ Order with ID {order_id} not found in the database."
        
        # Check current status
        current_status = existing_order.get('order_status', '')
        if current_status in ['approved', 'rejected']:
            action = "approved" if existing_order.get('approved') else "rejected"
            return f"❌ Order {order_id} has already been {action}."
        
        # Approve or reject the order
        result = await chatbot_mongodb_client.approve_order(order_id, approved, notes)
        
        if result.get("success"):
            action = "approved" if approved else "rejected"
            response = f"✅ Successfully {action} order {order_id}!\n\n"
            
            # Show order details
            order = result.get("order", {})
            if order:
                response += f"Order Details:\n"
                response += f"Customer ID: {order.get('customerID')}\n"
                response += f"Status: {order.get('order_status')}\n"
                response += f"Priority: {order.get('priority', 'normal')}\n"
                response += f"Total Amount: ${order.get('total_amount', 0):.2f}\n"
                response += f"Order Date: {order.get('order_date', 'N/A')}\n"
                response += f"Approval Date: {order.get('approval_date', 'N/A')}\n"
                
                if notes:
                    response += f"Approval Notes: {notes}\n"
                elif order.get('approval_notes'):
                    response += f"Approval Notes: {order.get('approval_notes')}\n"
                
                # Show items if available
                items = order.get('items', [])
                if items:
                    response += f"\nOrder Items ({len(items)}):\n"
                    for item in items[:3]:  # Show first 3 items
                        response += f"- Item ID: {item.get('itemID')} (Qty: {item.get('quantity', 0)})\n"
                    if len(items) > 3:
                        response += f"... and {len(items) - 3} more items\n"
            
            return response
        else:
            return f"❌ Failed to approve/reject order: {result.get('message', 'Unknown error')}"
            
    except Exception as e:
        return f"❌ Error approving/rejecting order: {str(e)}"

# Define create picking task tool
async def create_picking_task_func(order_id: int, 
                            worker_id: Optional[int] = None,
                            priority: Optional[str] = None,
                            notes: Optional[str] = None) -> str:
    """Create a picking task for an order using direct MongoDB access."""
    try:
        # Validate order exists and is in appropriate status
        order = await chatbot_mongodb_client.get_order_by_id(order_id)
        if not order:
            return f"❌ Order with ID {order_id} not found."
        
        # Check order status
        order_status = order.get('order_status', '')
        if order_status not in ['approved', 'processing']:
            return f"❌ Order {order_id} must be approved before creating picking tasks. Current status: {order_status}"
        
        # Prepare task data
        task_data = {
            'orderID': order_id,
            'priority': priority.lower() if priority else order.get('priority', 'normal'),
            'assigned_worker': worker_id,
            'instructions': f"Pick items for order {order_id}",
            'estimated_duration': 30  # Default 30 minutes
        }
        
        if notes:
            task_data['notes'] = notes
        
        # Add order items to task
        items = order.get('items', [])
        if items:
            task_data['items'] = [
                {
                    'itemID': item.get('itemID'),
                    'quantity': item.get('quantity', 0),
                    'picked_quantity': 0
                }
                for item in items
            ]
        
        # Create the picking task
        result = await chatbot_mongodb_client.create_task('picking', task_data)
        
        if result.get('success'):
            task = result.get('task', {})
            response = f"✅ Successfully created picking task!\n\n"
            response += f"Task ID: {result.get('task_id')}\n"
            response += f"Order ID: {order_id}\n"
            response += f"Task Type: Picking\n"
            response += f"Priority: {task.get('priority', 'normal')}\n"
            response += f"Status: {task.get('status', 'pending')}\n"
            
            if worker_id:
                response += f"Assigned Worker: {worker_id}\n"
            else:
                response += f"Assigned Worker: Unassigned\n"
            
            if notes:
                response += f"Notes: {notes}\n"
            
            # Show items to pick
            task_items = task.get('items', [])
            if task_items:
                response += f"\nItems to Pick ({len(task_items)}):\n"
                for item in task_items:
                    response += f"- Item ID: {item.get('itemID')} (Qty: {item.get('quantity', 0)})\n"
            
            response += f"\nEstimated Duration: {task.get('estimated_duration', 30)} minutes"
            
            return response
        else:
            return f"❌ Failed to create picking task: {result.get('message', 'Unknown error')}"
            
    except Exception as e:
        return f"❌ Error creating picking task: {str(e)}"

# Define update picking task tool
async def update_picking_task_func(task_id: int, 
                            status: Optional[str] = None,
                            worker_id: Optional[int] = None,
                            notes: Optional[str] = None) -> str:
    """Update the status of a picking task using direct MongoDB access."""
    try:
        # Validate task exists
        task = await chatbot_mongodb_client.get_task_by_id(task_id)
        if not task:
            return f"❌ Task with ID {task_id} not found."
        
        # Verify it's a picking task
        if task.get('task_type') != 'picking':
            return f"❌ Task {task_id} is not a picking task. Type: {task.get('task_type')}"
        
        # Prepare update data
        update_data = {}
        
        if status is not None:
            # Validate status
            valid_statuses = ['pending', 'in_progress', 'completed', 'cancelled', 'on_hold']
            if status.lower() not in valid_statuses:
                return f"❌ Invalid status '{status}'. Valid options: {', '.join(valid_statuses)}"
            update_data['status'] = status.lower()
            
            # If completing, set completion timestamp
            if status.lower() == 'completed':
                update_data['completed_at'] = datetime.utcnow()
        
        if worker_id is not None:
            update_data['assigned_worker'] = worker_id
        
        if notes is not None:
            update_data['notes'] = notes
        
        if not update_data:
            return "❌ No update data provided. Please specify at least one field to update."
        
        # Update the task
        result = await chatbot_mongodb_client.update_task(task_id, update_data)
        
        if result.get('success'):
            updated_task = result.get('task', {})
            response = f"✅ Successfully updated picking task {task_id}!\n\n"
            
            # Show updated fields
            response += f"Updated fields:\n"
            for field, value in update_data.items():
                if field == 'assigned_worker':
                    response += f"• Assigned Worker: {value}\n"
                elif field == 'completed_at':
                    response += f"• Completed At: {value}\n"
                else:
                    response += f"• {field.replace('_', ' ').title()}: {value}\n"
            
            # Show current task status
            response += f"\nCurrent Task Status:\n"
            response += f"Order ID: {updated_task.get('orderID')}\n"
            response += f"Status: {updated_task.get('status')}\n"
            response += f"Priority: {updated_task.get('priority', 'normal')}\n"
            response += f"Assigned Worker: {updated_task.get('assigned_worker', 'Unassigned')}\n"
            
            if updated_task.get('notes'):
                response += f"Notes: {updated_task.get('notes')}\n"
            
            return response
        else:
            return f"❌ Failed to update picking task: {result.get('message', 'Unknown error')}"
            
    except Exception as e:
        return f"❌ Error updating picking task: {str(e)}"

# Define create packing task tool
async def create_packing_task_func(order_id: int, 
                            worker_id: Optional[int] = None,
                            priority: Optional[str] = None,
                            notes: Optional[str] = None) -> str:
    """Create a packing task for an order using direct MongoDB access."""
    try:
        # Validate order exists and is in appropriate status
        order = await chatbot_mongodb_client.get_order_by_id(order_id)
        if not order:
            return f"❌ Order with ID {order_id} not found."
        
        # Check order status - packing usually comes after picking
        order_status = order.get('order_status', '')
        if order_status not in ['approved', 'processing', 'picking']:
            return f"❌ Order {order_id} is not ready for packing. Current status: {order_status}"
        
        # Prepare task data
        task_data = {
            'orderID': order_id,
            'priority': priority.lower() if priority else order.get('priority', 'normal'),
            'assigned_worker': worker_id,
            'instructions': f"Pack items for order {order_id}",
            'estimated_duration': 20  # Default 20 minutes for packing
        }
        
        if notes:
            task_data['notes'] = notes
        
        # Add order items to task
        items = order.get('items', [])
        if items:
            task_data['items'] = [
                {
                    'itemID': item.get('itemID'),
                    'quantity': item.get('quantity', 0),
                    'packed_quantity': 0
                }
                for item in items
            ]
        
        # Create the packing task
        result = await chatbot_mongodb_client.create_task('packing', task_data)
        
        if result.get('success'):
            task = result.get('task', {})
            response = f"✅ Successfully created packing task!\n\n"
            response += f"Task ID: {result.get('task_id')}\n"
            response += f"Order ID: {order_id}\n"
            response += f"Task Type: Packing\n"
            response += f"Priority: {task.get('priority', 'normal')}\n"
            response += f"Status: {task.get('status', 'pending')}\n"
            
            if worker_id:
                response += f"Assigned Worker: {worker_id}\n"
            else:
                response += f"Assigned Worker: Unassigned\n"
            
            if notes:
                response += f"Notes: {notes}\n"
            
            # Show items to pack
            task_items = task.get('items', [])
            if task_items:
                response += f"\nItems to Pack ({len(task_items)}):\n"
                for item in task_items:
                    response += f"- Item ID: {item.get('itemID')} (Qty: {item.get('quantity', 0)})\n"
            
            response += f"\nEstimated Duration: {task.get('estimated_duration', 20)} minutes"
            
            return response
        else:
            return f"❌ Failed to create packing task: {result.get('message', 'Unknown error')}"
            
    except Exception as e:
        return f"❌ Error creating packing task: {str(e)}"

# Define update packing task tool
async def update_packing_task_func(task_id: int, 
                            status: Optional[str] = None,
                            worker_id: Optional[int] = None,
                            notes: Optional[str] = None) -> str:
    """Update the status of a packing task using direct MongoDB access."""
    try:
        # Validate task exists
        task = await chatbot_mongodb_client.get_task_by_id(task_id)
        if not task:
            return f"❌ Task with ID {task_id} not found."
        
        # Verify it's a packing task
        if task.get('task_type') != 'packing':
            return f"❌ Task {task_id} is not a packing task. Type: {task.get('task_type')}"
        
        # Prepare update data
        update_data = {}
        
        if status is not None:
            # Validate status
            valid_statuses = ['pending', 'in_progress', 'completed', 'cancelled', 'on_hold']
            if status.lower() not in valid_statuses:
                return f"❌ Invalid status '{status}'. Valid options: {', '.join(valid_statuses)}"
            update_data['status'] = status.lower()
            
            # If completing, set completion timestamp
            if status.lower() == 'completed':
                update_data['completed_at'] = datetime.utcnow()
        
        if worker_id is not None:
            update_data['assigned_worker'] = worker_id
        
        if notes is not None:
            update_data['notes'] = notes
        
        if not update_data:
            return "❌ No update data provided. Please specify at least one field to update."
        
        # Update the task
        result = await chatbot_mongodb_client.update_task(task_id, update_data)
        
        if result.get('success'):
            updated_task = result.get('task', {})
            response = f"✅ Successfully updated packing task {task_id}!\n\n"
            
            # Show updated fields
            response += f"Updated fields:\n"
            for field, value in update_data.items():
                if field == 'assigned_worker':
                    response += f"• Assigned Worker: {value}\n"
                elif field == 'completed_at':
                    response += f"• Completed At: {value}\n"
                else:
                    response += f"• {field.replace('_', ' ').title()}: {value}\n"
            
            # Show current task status
            response += f"\nCurrent Task Status:\n"
            response += f"Order ID: {updated_task.get('orderID')}\n"
            response += f"Status: {updated_task.get('status')}\n"
            response += f"Priority: {updated_task.get('priority', 'normal')}\n"
            response += f"Assigned Worker: {updated_task.get('assigned_worker', 'Unassigned')}\n"
            
            if updated_task.get('notes'):
                response += f"Notes: {updated_task.get('notes')}\n"
            
            return response
        else:
            return f"❌ Failed to update packing task: {result.get('message', 'Unknown error')}"
            
    except Exception as e:
        return f"❌ Error updating packing task: {str(e)}"

# Create the tools
check_order_tool = create_tool(
    name="check_order",
    description="Check the status and details of a specific order",
    function=order_query_func,
    arg_descriptions={
        "order_id": {
            "type": Optional[int], 
            "description": "ID of the order to check"
        },
        "customer_id": {
            "type": Optional[int], 
            "description": "ID of the customer associated with the order"
        },
        "status": {
            "type": Optional[str], 
            "description": "Status of the order"
        },
        "priority": {
            "type": Optional[str], 
            "description": "Priority level of the order"
        },
        "date_from": {
            "type": Optional[str], 
            "description": "Start date for order query"
        },
        "date_to": {
            "type": Optional[str], 
            "description": "End date for order query"
        }
    }
)

order_create_tool = create_tool(
    name="order_create",
    description="Create a new order with items and shipping details",
    function=order_create_func,
    arg_descriptions={
        "customer_id": {
            "type": int, 
            "description": "ID of the customer placing the order"
        },
        "items": {
            "type": List[Dict[str, Any]], 
            "description": "List of items to order, each with item_id and quantity"
        },
        "shipping_address": {
            "type": str, 
            "description": "Shipping address for the order"
        },
        "priority": {
            "type": Optional[str], 
            "description": "Priority level (low, normal, high, urgent) - defaults to normal"
        },
        "notes": {
            "type": Optional[str], 
            "description": "Optional notes for the order"
        }
    }
)

order_update_tool = create_tool(
    name="order_update",
    description="Update an existing order's status, priority, or other details",
    function=order_update_func,
    arg_descriptions={
        "order_id": {
            "type": int, 
            "description": "ID of the order to update"
        },
        "status": {
            "type": Optional[str], 
            "description": "New status (pending, approved, rejected, processing, picking, packing, shipped, delivered, cancelled)"
        },
        "priority": {
            "type": Optional[str], 
            "description": "New priority level (low, normal, high, urgent)"
        },
        "shipping_address": {
            "type": Optional[str], 
            "description": "Updated shipping address"
        },
        "notes": {
            "type": Optional[str], 
            "description": "Updated notes for the order"
        }
    }
)

create_sub_order_tool = create_tool(
    name="create_sub_order",
    description="Create a sub-order for partial fulfillment of a parent order",
    function=create_sub_order_func,
    arg_descriptions={
        "parent_order_id": {
            "type": int, 
            "description": "ID of the parent order"
        },
        "items": {
            "type": List[Dict[str, Any]], 
            "description": "List of items for the sub-order, each with item_id, quantity, and sku"
        },
        "reason": {
            "type": str, 
            "description": "Reason for creating the sub-order"
        }
    }
)

approve_orders_tool = create_tool(
    name="approve_orders",
    description="Approve or reject an order (manager only)",
    function=approve_orders_func,
    arg_descriptions={
        "order_id": {
            "type": int, 
            "description": "ID of the order to approve or reject"
        },
        "approved": {
            "type": bool, 
            "description": "Whether to approve (true) or reject (false) the order"
        },
        "notes": {
            "type": Optional[str], 
            "description": "Optional notes about the approval or rejection"
        }
    }
)

create_picking_task_tool = create_tool(
    name="create_picking_task",
    description="Create a picking task for an order",
    function=create_picking_task_func,
    arg_descriptions={
        "order_id": {
            "type": int, 
            "description": "ID of the order to create a picking task for"
        },
        "worker_id": {
            "type": Optional[int], 
            "description": "Optional ID of the worker to assign the task to"
        },
        "priority": {
            "type": Optional[str], 
            "description": "Optional priority level (low, medium, high, urgent)"
        },
        "notes": {
            "type": Optional[str], 
            "description": "Optional notes for the picking task"
        }
    }
)

update_picking_task_tool = create_tool(
    name="update_picking_task",
    description="Update the status of a picking task",
    function=update_picking_task_func,
    arg_descriptions={
        "task_id": {
            "type": int, 
            "description": "ID of the picking task to update"
        },
        "status": {
            "type": Optional[str], 
            "description": "New status (pending, in_progress, completed, cancelled)"
        },
        "worker_id": {
            "type": Optional[int], 
            "description": "Optional ID of the worker to reassign the task to"
        },
        "notes": {
            "type": Optional[str], 
            "description": "Optional notes to add to the task"
        }
    }
)

create_packing_task_tool = create_tool(
    name="create_packing_task",
    description="Create a packing task for an order",
    function=create_packing_task_func,
    arg_descriptions={
        "order_id": {
            "type": int, 
            "description": "ID of the order to create a packing task for"
        },
        "worker_id": {
            "type": Optional[int], 
            "description": "Optional ID of the worker to assign the task to"
        },
        "priority": {
            "type": Optional[str], 
            "description": "Optional priority level (low, medium, high, urgent)"
        },
        "notes": {
            "type": Optional[str], 
            "description": "Optional notes for the packing task"
        }
    }
)

update_packing_task_tool = create_tool(
    name="update_packing_task",
    description="Update the status of a packing task",
    function=update_packing_task_func,
    arg_descriptions={
        "task_id": {
            "type": int, 
            "description": "ID of the packing task to update"
        },
        "status": {
            "type": Optional[str], 
            "description": "New status (pending, in_progress, completed, cancelled)"
        },
        "worker_id": {
            "type": Optional[int], 
            "description": "Optional ID of the worker to reassign the task to"
        },
        "notes": {
            "type": Optional[str], 
            "description": "Optional notes to add to the task"
        }
    }
)

# # Import optimized versions for better performance
# try:
#     from optimized_order_tools import (
#         optimized_check_order_tool,
#         optimized_order_create_tool
#     )
#     # Use optimized versions if available
#     check_order_tool_optimized = optimized_check_order_tool
#     order_create_tool_optimized = optimized_order_create_tool
#     print("✅ Using optimized order tools for better performance!")
# except ImportError:
#     print("⚠️  Optimized tools not found - using standard versions")
#     check_order_tool_optimized = None
#     order_create_tool_optimized = None

# Export the tools
__all__ = [
    "check_order_tool",
    "order_create_tool",
    "order_update_tool",
    "create_sub_order_tool",
    "approve_orders_tool",
    "create_picking_task_tool",
    "update_picking_task_tool",
    "create_packing_task_tool",
    "update_packing_task_tool",
    # Optimized versions (if available)
    "check_order_tool_optimized",
    "order_create_tool_optimized"
]
