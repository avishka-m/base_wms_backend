from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field

# Use absolute imports instead of relative imports
from app.tools.chatbot.base_tool import WMSBaseTool, create_tool
from app.utils.chatbot.api_client import async_api_client
from app.utils.chatbot.knowledge_base import knowledge_base
from app.utils.chatbot.demo_data import get_demo_order_data, is_api_error

# Define order query tool
async def order_query_func(order_id: Optional[int] = None,
                     customer_id: Optional[int] = None,
                     status: Optional[str] = None,
                     priority: Optional[str] = None,
                     date_from: Optional[str] = None,
                     date_to: Optional[str] = None) -> str:
    """Query orders based on various criteria."""
    params = {}
    
    # Add non-None parameters to query
    if order_id is not None:
        # If order_id is provided, do a direct lookup
        try:
            order = await async_api_client.get_order(order_id)
            
            # Check if we got an error response - use demo data as fallback
            if is_api_error(order):
                demo_orders = get_demo_order_data(order_id=order_id)
                if demo_orders:
                    order = demo_orders[0]
                    note = " (Demo data - API not accessible)"
                else:
                    return f"Order with ID {order_id} not found in demo data."
            else:
                note = ""
            
            # Format single order response
            result = f"Found order{note}:\n\n"
            result += f"Order ID: {order.get('id')}\n"
            result += f"Customer ID: {order.get('customer_id')}\n"
            result += f"Status: {order.get('status')}\n"
            result += f"Priority: {order.get('priority', 'Normal')}\n"
            result += f"Total Amount: ${order.get('total_amount', 'N/A')}\n"
            result += f"Created: {order.get('created_at', 'N/A')}\n"
            result += f"Shipping Address: {order.get('shipping_address', 'N/A')}\n"
            
            # Add order items if available
            items = order.get('items', [])
            if items:
                result += f"\nOrder Items ({len(items)}):\n"
                for item in items:
                    result += f"- {item.get('name', 'Unknown')} (Qty: {item.get('quantity', 0)})\n"
            
            return result
        except Exception as e:
            # Fallback to demo data on exception
            demo_orders = get_demo_order_data(order_id=order_id)
            if demo_orders:
                order = demo_orders[0]
                result = "Found order (Demo data - API error):\n\n"
                result += f"Order ID: {order.get('id')}\n"
                result += f"Customer ID: {order.get('customer_id')}\n"
                result += f"Status: {order.get('status')}\n"
                result += f"Priority: {order.get('priority', 'Normal')}\n"
                result += f"Total Amount: ${order.get('total_amount', 'N/A')}\n"
                result += f"Created: {order.get('created_at', 'N/A')}\n"
                return result
            else:
                return f"Error retrieving order: {str(e)}"
            
    # Build filter parameters
    if customer_id:
        params["customer_id"] = customer_id
    if status:
        params["status"] = status
    if priority:
        params["priority"] = priority
    if date_from:
        params["date_from"] = date_from
    if date_to:
        params["date_to"] = date_to
        
    try:
        orders = await async_api_client.get_orders(params)
        
        # Check if we got an error response - use demo data as fallback
        if is_api_error(orders):
            orders = get_demo_order_data(
                customer_id=customer_id, status=status, priority=priority
            )
            note = " (Demo data - API not accessible)"
        else:
            note = ""
        
        if not orders:
            return "No orders found matching your criteria."
            
        # Format the results
        result = f"Found the following orders{note}:\n\n"
        for order in orders:
            result += f"Order ID: {order.get('id')}\n"
            result += f"Customer ID: {order.get('customer_id')}\n"
            result += f"Status: {order.get('status')}\n"
            result += f"Priority: {order.get('priority', 'Normal')}\n"
            result += f"Total: ${order.get('total_amount', 'N/A')}\n"
            result += f"Created: {order.get('created_at', 'N/A')}\n"
            result += "-" * 40 + "\n"
            
        return result
    except Exception as e:
        # Fallback to demo data on exception
        try:
            orders = get_demo_order_data(
                customer_id=customer_id, status=status, priority=priority
            )
            if not orders:
                return "No orders found matching your criteria."
            
            result = "Found the following orders (Demo data - API error):\n\n"
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
    """Create a new order."""
    # Validate required data
    if not items:
        return "Error: At least one item is required to create an order."
    
    # Prepare the order data
    order_data = {
        "customer_id": customer_id,
        "items": items,
        "shipping_address": shipping_address,
        "priority": priority,
        "status": "pending"
    }
    
    if notes:
        order_data["notes"] = notes
        
    try:
        # Create the order
        response = await async_api_client.post("orders", order_data)
        
        if is_api_error(response):
            return f"Error creating order: {response.get('error', 'Unknown error')}"
        
        return f"Successfully created order: {response.get('id')} for customer {customer_id}"
    except Exception as e:
        return f"Error creating order: {str(e)}"

# Define order update tool
async def order_update_func(order_id: int,
                      status: Optional[str] = None,
                      priority: Optional[str] = None,
                      shipping_address: Optional[str] = None,
                      notes: Optional[str] = None) -> str:
    """Update an existing order."""
    # First, verify the order exists
    try:
        existing_order = await async_api_client.get_order(order_id)
        if is_api_error(existing_order):
            return f"Error: Could not find order with ID {order_id}: {existing_order.get('error', 'Unknown error')}"
    except Exception as e:
        return f"Error: Could not find order with ID {order_id}: {str(e)}"
        
    # Prepare update data with only the fields that are provided
    update_data = {}
    if status:
        update_data["status"] = status
    if priority:
        update_data["priority"] = priority
    if shipping_address:
        update_data["shipping_address"] = shipping_address
    if notes:
        update_data["notes"] = notes
        
    if not update_data:
        return "No update data provided. Please specify at least one field to update."
        
    try:
        response = await async_api_client.put("orders", order_id, update_data)
        
        if is_api_error(response):
            return f"Error updating order: {response.get('error', 'Unknown error')}"
        
        return f"Successfully updated order {order_id}"
    except Exception as e:
        return f"Error updating order: {str(e)}"

# Define create sub-order tool
def create_sub_order_func(parent_order_id: int, 
                         items: List[Dict[str, Any]],
                         reason: str) -> str:
    """Create a sub-order for partial fulfillment of a parent order."""
    try:
        # First check if the parent order exists
        parent_order = async_api_client.get_order(parent_order_id)
        if not parent_order:
            return f"Error: Parent order {parent_order_id} does not exist."
        
        # Create the sub-order data
        sub_order_data = {
            "parent_order_id": parent_order_id,
            "items": items,
            "reason": reason,
            "is_sub_order": True
        }
        
        # Submit the sub-order
        response = async_api_client.post("orders", sub_order_data)
        
        return f"Successfully created sub-order {response.get('id')} for parent order {parent_order_id}."
        
    except Exception as e:
        return f"Error creating sub-order: {str(e)}"

# Define approve orders tool
def approve_orders_func(order_id: int, approved: bool, notes: Optional[str] = None) -> str:
    """Approve or reject an order."""
    try:
        # Check if the order exists
        order = async_api_client.get_order(order_id)
        if not order:
            return f"Error: Order {order_id} does not exist."
            
        # Prepare approval data
        approval_data = {
            "approved": approved,
            "status": "Processing" if approved else "Rejected"
        }
        
        if notes:
            approval_data["approval_notes"] = notes
            
        # Update the order
        response = async_api_client.put("orders", order_id, approval_data)
        
        if approved:
            return f"Successfully approved order {order_id}. New status: Processing."
        else:
            return f"Order {order_id} has been rejected. New status: Rejected."
            
    except Exception as e:
        return f"Error approving order: {str(e)}"

# Define create picking task tool
def create_picking_task_func(order_id: int, 
                            worker_id: Optional[int] = None,
                            priority: Optional[str] = None,
                            notes: Optional[str] = None) -> str:
    """Create a picking task for an order."""
    try:
        # Check if the order exists
        order = async_api_client.get_order(order_id)
        if not order:
            return f"Error: Order {order_id} does not exist."
            
        # Check if a picking task already exists
        existing_tasks = async_api_client.get("picking", {"order_id": order_id})
        if existing_tasks:
            return f"A picking task already exists for order {order_id}: Task ID {existing_tasks[0].get('id')}"
            
        # Prepare task data
        task_data = {
            "order_id": order_id,
            "status": "Pending"
        }
        
        if worker_id:
            # Verify the worker exists
            worker = async_api_client.get_by_id("workers", worker_id)
            if not worker:
                return f"Error: Worker {worker_id} does not exist."
            task_data["worker_id"] = worker_id
            
        if priority:
            valid_priorities = ["low", "medium", "high", "urgent"]
            if priority.lower() not in valid_priorities:
                return f"Error: Invalid priority. Must be one of {valid_priorities}"
            task_data["priority"] = priority.lower()
            
        if notes:
            task_data["notes"] = notes
            
        # Create the picking task
        response = async_api_client.post("picking", task_data)
        
        return f"Successfully created picking task {response.get('id')} for order {order_id}."
        
    except Exception as e:
        return f"Error creating picking task: {str(e)}"

# Define update picking task tool
def update_picking_task_func(task_id: int, 
                            status: Optional[str] = None,
                            worker_id: Optional[int] = None,
                            notes: Optional[str] = None) -> str:
    """Update the status of a picking task."""
    try:
        # Check if the task exists
        task = async_api_client.get_by_id("picking", task_id)
        if not task:
            return f"Error: Picking task {task_id} does not exist."
            
        # Prepare update data
        update_data = {}
        
        if status:
            valid_statuses = ["pending", "in_progress", "completed", "cancelled"]
            if status.lower() not in valid_statuses:
                return f"Error: Invalid status. Must be one of {valid_statuses}"
            update_data["status"] = status.lower()
            
        if worker_id:
            # Verify the worker exists
            worker = async_api_client.get_by_id("workers", worker_id)
            if not worker:
                return f"Error: Worker {worker_id} does not exist."
            update_data["worker_id"] = worker_id
            
        if notes:
            update_data["notes"] = notes
        
        # If no updates specified
        if not update_data:
            return "No update fields provided. Please specify at least one field to update."
            
        # Update the picking task
        response = async_api_client.put("picking", task_id, update_data)
        
        # If status is set to completed, trigger creation of a packing task
        if status and status.lower() == "completed":
            # Create a packing task automatically
            order_id = task.get("order_id")
            packing_data = {
                "order_id": order_id,
                "status": "Pending"
            }
            packing_response = async_api_client.post("packing", packing_data)
            return f"Successfully updated picking task {task_id}. Status changed to completed. Packing task {packing_response.get('id')} created automatically."
            
        return f"Successfully updated picking task {task_id}."
        
    except Exception as e:
        return f"Error updating picking task: {str(e)}"

# Define create packing task tool
def create_packing_task_func(order_id: int, 
                            worker_id: Optional[int] = None,
                            priority: Optional[str] = None,
                            notes: Optional[str] = None) -> str:
    """Create a packing task for an order."""
    try:
        # Check if the order exists
        order = async_api_client.get_order(order_id)
        if not order:
            return f"Error: Order {order_id} does not exist."
            
        # Check if a packing task already exists
        existing_tasks = async_api_client.get("packing", {"order_id": order_id})
        if existing_tasks:
            return f"A packing task already exists for order {order_id}: Task ID {existing_tasks[0].get('id')}"
            
        # Check if a picking task has been completed
        picking_tasks = async_api_client.get("picking", {"order_id": order_id})
        if not picking_tasks:
            return f"Error: No picking task exists for order {order_id}. Create a picking task first."
            
        if picking_tasks[0].get("status") != "completed":
            return f"Error: Picking task for order {order_id} has not been completed yet. Status: {picking_tasks[0].get('status')}"
            
        # Prepare task data
        task_data = {
            "order_id": order_id,
            "status": "Pending"
        }
        
        if worker_id:
            # Verify the worker exists
            worker = async_api_client.get_by_id("workers", worker_id)
            if not worker:
                return f"Error: Worker {worker_id} does not exist."
            task_data["worker_id"] = worker_id
            
        if priority:
            valid_priorities = ["low", "medium", "high", "urgent"]
            if priority.lower() not in valid_priorities:
                return f"Error: Invalid priority. Must be one of {valid_priorities}"
            task_data["priority"] = priority.lower()
            
        if notes:
            task_data["notes"] = notes
            
        # Create the packing task
        response = async_api_client.post("packing", task_data)
        
        return f"Successfully created packing task {response.get('id')} for order {order_id}."
        
    except Exception as e:
        return f"Error creating packing task: {str(e)}"

# Define update packing task tool
def update_packing_task_func(task_id: int, 
                            status: Optional[str] = None,
                            worker_id: Optional[int] = None,
                            notes: Optional[str] = None) -> str:
    """Update the status of a packing task."""
    try:
        # Check if the task exists
        task = async_api_client.get_by_id("packing", task_id)
        if not task:
            return f"Error: Packing task {task_id} does not exist."
            
        # Prepare update data
        update_data = {}
        
        if status:
            valid_statuses = ["pending", "in_progress", "completed", "cancelled"]
            if status.lower() not in valid_statuses:
                return f"Error: Invalid status. Must be one of {valid_statuses}"
            update_data["status"] = status.lower()
            
        if worker_id:
            # Verify the worker exists
            worker = async_api_client.get_by_id("workers", worker_id)
            if not worker:
                return f"Error: Worker {worker_id} does not exist."
            update_data["worker_id"] = worker_id
            
        if notes:
            update_data["notes"] = notes
        
        # If no updates specified
        if not update_data:
            return "No update fields provided. Please specify at least one field to update."
            
        # Update the packing task
        response = async_api_client.put("packing", task_id, update_data)
        
        # If status is set to completed, trigger creation of a shipping task
        if status and status.lower() == "completed":
            # Create a shipping task automatically
            order_id = task.get("order_id")
            shipping_data = {
                "order_id": order_id,
                "status": "Pending"
            }
            shipping_response = async_api_client.post("shipping", shipping_data)
            return f"Successfully updated packing task {task_id}. Status changed to completed. Shipping task {shipping_response.get('id')} created automatically."
            
        return f"Successfully updated packing task {task_id}."
        
    except Exception as e:
        return f"Error updating packing task: {str(e)}"

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

# Export the tools
__all__ = [
    "check_order_tool",
    "create_sub_order_tool",
    "approve_orders_tool",
    "create_picking_task_tool",
    "update_picking_task_tool",
    "create_packing_task_tool",
    "update_packing_task_tool"
]
