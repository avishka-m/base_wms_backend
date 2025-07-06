from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field

# Use absolute imports instead of relative imports
from app.tools.chatbot.base_tool import WMSBaseTool, create_tool
from app.utils.chatbot.api_client import api_client
from app.utils.chatbot.knowledge_base import knowledge_base
from app.utils.chatbot.demo_data import get_demo_orders, get_demo_workers, is_api_error

# Define order query tool
def check_order_func(order_id: int) -> str:
    """Check the status and details of a specific order."""
    try:
        order = api_client.get_order(order_id)
        
        # Check if we got an error response - use demo data as fallback
        if is_api_error(order):
            demo_orders = get_demo_orders(order_id=order_id)
            if demo_orders:
                order = demo_orders[0]
                note = " (Demo data - API not accessible)"
            else:
                return f"Order with ID {order_id} not found in demo data."
        else:
            note = ""
            
        # Format the response
        result = f"Order Details for Order #{order_id}{note}:\n\n"
        result += f"Status: {order.get('status')}\n"
        result += f"Customer: {order.get('customer_name')} (ID: {order.get('customer_id')})\n"
        result += f"Created: {order.get('created_at')}\n"
        result += f"Updated: {order.get('updated_at')}\n"
        result += f"Total Items: {len(order.get('items', []))}\n"
        result += f"Total Value: ${order.get('total_value')}\n\n"
        
        # Order items
        result += "Order Items:\n"
        result += "-" * 50 + "\n"
        
        for item in order.get('items', []):
            result += f"- {item.get('quantity')}x {item.get('item_name')} (SKU: {item.get('sku')})\n"
            result += f"  Price: ${item.get('unit_price')} each, Subtotal: ${item.get('subtotal')}\n"
            result += f"  Status: {item.get('status')}\n"
        
        result += "-" * 50 + "\n\n"
        
        # Related tasks
        if order.get('picking_tasks'):
            result += "Picking Tasks:\n"
            for task in order.get('picking_tasks'):
                result += f"- Task #{task.get('id')}: {task.get('status')}\n"
                result += f"  Assigned to: {task.get('worker_name')}\n"
                result += f"  Created: {task.get('created_at')}\n"
            result += "\n"
            
        if order.get('packing_tasks'):
            result += "Packing Tasks:\n"
            for task in order.get('packing_tasks'):
                result += f"- Task #{task.get('id')}: {task.get('status')}\n"
                result += f"  Assigned to: {task.get('worker_name')}\n"
                result += f"  Created: {task.get('created_at')}\n"
            result += "\n"
            
        if order.get('shipping_tasks'):
            result += "Shipping Tasks:\n"
            for task in order.get('shipping_tasks'):
                result += f"- Task #{task.get('id')}: {task.get('status')}\n"
                result += f"  Assigned to: {task.get('worker_name')}\n"
                result += f"  Vehicle: {task.get('vehicle_name')} (ID: {task.get('vehicle_id')})\n"
                result += f"  Created: {task.get('created_at')}\n"
            
        return result
    except Exception as e:
        # Fallback to demo data on exception
        demo_orders = get_demo_orders(order_id=order_id)
        if demo_orders:
            order = demo_orders[0]
            result = f"Order Details for Order #{order_id} (Demo data - API error):\n\n"
            result += f"Status: {order.get('status')}\n"
            result += f"Customer: {order.get('customer_name')} (ID: {order.get('customer_id')})\n"
            result += f"Created: {order.get('created_at')}\n"
            result += f"Updated: {order.get('updated_at')}\n"
            result += f"Total Items: {len(order.get('items', []))}\n"
            result += f"Total Value: ${order.get('total_value')}\n\n"
            
            # Order items
            result += "Order Items:\n"
            result += "-" * 50 + "\n"
            
            for item in order.get('items', []):
                result += f"- {item.get('quantity')}x {item.get('item_name')} (SKU: {item.get('sku')})\n"
                result += f"  Price: ${item.get('unit_price')} each, Subtotal: ${item.get('subtotal')}\n"
                result += f"  Status: {item.get('status')}\n"
            
            result += "-" * 50 + "\n\n"
            
            # Related tasks
            if order.get('picking_tasks'):
                result += "Picking Tasks:\n"
                for task in order.get('picking_tasks'):
                    result += f"- Task #{task.get('id')}: {task.get('status')}\n"
                    result += f"  Assigned to: {task.get('worker_name')}\n"
                    result += f"  Created: {task.get('created_at')}\n"
                result += "\n"
                
            if order.get('packing_tasks'):
                result += "Packing Tasks:\n"
                for task in order.get('packing_tasks'):
                    result += f"- Task #{task.get('id')}: {task.get('status')}\n"
                    result += f"  Assigned to: {task.get('worker_name')}\n"
                    result += f"  Created: {task.get('created_at')}\n"
                result += "\n"
                
            if order.get('shipping_tasks'):
                result += "Shipping Tasks:\n"
                for task in order.get('shipping_tasks'):
                    result += f"- Task #{task.get('id')}: {task.get('status')}\n"
                    result += f"  Assigned to: {task.get('worker_name')}\n"
                    result += f"  Vehicle: {task.get('vehicle_name')} (ID: {task.get('vehicle_id')})\n"
                    result += f"  Created: {task.get('created_at')}\n"
                
            return result
        else:
            return f"Error retrieving order: {str(e)}"

# Define create sub-order tool
def create_sub_order_func(parent_order_id: int, 
                         items: List[Dict[str, Any]],
                         reason: str) -> str:
    """Create a sub-order for partial fulfillment of a parent order."""
    try:
        # First check if the parent order exists
        parent_order = api_client.get_order(parent_order_id)
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
        response = api_client.post("orders", sub_order_data)
        
        return f"Successfully created sub-order {response.get('id')} for parent order {parent_order_id}."
        
    except Exception as e:
        return f"Error creating sub-order: {str(e)}"

# Define approve orders tool
def approve_orders_func(order_id: int, approved: bool, notes: Optional[str] = None) -> str:
    """Approve or reject an order."""
    try:
        # Check if the order exists
        order = api_client.get_order(order_id)
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
        response = api_client.put("orders", order_id, approval_data)
        
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
        order = api_client.get_order(order_id)
        if not order:
            return f"Error: Order {order_id} does not exist."
            
        # Check if a picking task already exists
        existing_tasks = api_client.get("picking", {"order_id": order_id})
        if existing_tasks:
            return f"A picking task already exists for order {order_id}: Task ID {existing_tasks[0].get('id')}"
            
        # Prepare task data
        task_data = {
            "order_id": order_id,
            "status": "Pending"
        }
        
        if worker_id:
            # Verify the worker exists
            worker = api_client.get_by_id("workers", worker_id)
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
        response = api_client.post("picking", task_data)
        
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
        task = api_client.get_by_id("picking", task_id)
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
            worker = api_client.get_by_id("workers", worker_id)
            if not worker:
                return f"Error: Worker {worker_id} does not exist."
            update_data["worker_id"] = worker_id
            
        if notes:
            update_data["notes"] = notes
        
        # If no updates specified
        if not update_data:
            return "No update fields provided. Please specify at least one field to update."
            
        # Update the picking task
        response = api_client.put("picking", task_id, update_data)
        
        # If status is set to completed, trigger creation of a packing task
        if status and status.lower() == "completed":
            # Create a packing task automatically
            order_id = task.get("order_id")
            packing_data = {
                "order_id": order_id,
                "status": "Pending"
            }
            packing_response = api_client.post("packing", packing_data)
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
        order = api_client.get_order(order_id)
        if not order:
            return f"Error: Order {order_id} does not exist."
            
        # Check if a packing task already exists
        existing_tasks = api_client.get("packing", {"order_id": order_id})
        if existing_tasks:
            return f"A packing task already exists for order {order_id}: Task ID {existing_tasks[0].get('id')}"
            
        # Check if a picking task has been completed
        picking_tasks = api_client.get("picking", {"order_id": order_id})
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
            worker = api_client.get_by_id("workers", worker_id)
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
        response = api_client.post("packing", task_data)
        
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
        task = api_client.get_by_id("packing", task_id)
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
            worker = api_client.get_by_id("workers", worker_id)
            if not worker:
                return f"Error: Worker {worker_id} does not exist."
            update_data["worker_id"] = worker_id
            
        if notes:
            update_data["notes"] = notes
        
        # If no updates specified
        if not update_data:
            return "No update fields provided. Please specify at least one field to update."
            
        # Update the packing task
        response = api_client.put("packing", task_id, update_data)
        
        # If status is set to completed, trigger creation of a shipping task
        if status and status.lower() == "completed":
            # Create a shipping task automatically
            order_id = task.get("order_id")
            shipping_data = {
                "order_id": order_id,
                "status": "Pending"
            }
            shipping_response = api_client.post("shipping", shipping_data)
            return f"Successfully updated packing task {task_id}. Status changed to completed. Shipping task {shipping_response.get('id')} created automatically."
            
        return f"Successfully updated packing task {task_id}."
        
    except Exception as e:
        return f"Error updating packing task: {str(e)}"

# Create the tools
check_order_tool = create_tool(
    name="check_order",
    description="Check the status and details of a specific order",
    function=check_order_func,
    arg_descriptions={
        "order_id": {
            "type": int, 
            "description": "ID of the order to check"
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
