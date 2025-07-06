from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field

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
    """Create a new order (currently not implemented with direct MongoDB access)."""
    return ("❌ Order creation operations are not yet implemented with direct database access. "
            "This feature will be available in a future update. "
            "For now, please use the web interface to create new orders.")

# Define order update tool
async def order_update_func(order_id: int,
                      status: Optional[str] = None,
                      priority: Optional[str] = None,
                      shipping_address: Optional[str] = None,
                      notes: Optional[str] = None) -> str:
    """Update an existing order (currently not implemented with direct MongoDB access)."""
    return ("❌ Order update operations are not yet implemented with direct database access. "
            "This feature will be available in a future update. "
            "For now, please use the web interface to update orders.")

# Define create sub-order tool
async def create_sub_order_func(parent_order_id: int, 
                         items: List[Dict[str, Any]],
                         reason: str) -> str:
    """Create a sub-order for partial fulfillment of a parent order (currently not implemented with direct MongoDB access)."""
    return ("❌ Sub-order operations are not yet implemented with direct database access. "
            "This feature will be available in a future update. "
            "For now, please use the web interface to create sub-orders.")

# Define approve orders tool
async def approve_orders_func(order_id: int, approved: bool, notes: Optional[str] = None) -> str:
    """Approve or reject an order (currently not implemented with direct MongoDB access)."""
    return ("❌ Order approval operations are not yet implemented with direct database access. "
            "This feature will be available in a future update. "
            "For now, please use the web interface to approve/reject orders.")

# Define create picking task tool
async def create_picking_task_func(order_id: int, 
                            worker_id: Optional[int] = None,
                            priority: Optional[str] = None,
                            notes: Optional[str] = None) -> str:
    """Create a picking task for an order (currently not implemented with direct MongoDB access)."""
    return ("❌ Picking task operations are not yet implemented with direct database access. "
            "This feature will be available in a future update. "
            "For now, please use the web interface to create picking tasks.")

# Define update picking task tool
async def update_picking_task_func(task_id: int, 
                            status: Optional[str] = None,
                            worker_id: Optional[int] = None,
                            notes: Optional[str] = None) -> str:
    """Update the status of a picking task (currently not implemented with direct MongoDB access)."""
    return ("❌ Picking task update operations are not yet implemented with direct database access. "
            "This feature will be available in a future update. "
            "For now, please use the web interface to update picking tasks.")

# Define create packing task tool
async def create_packing_task_func(order_id: int, 
                            worker_id: Optional[int] = None,
                            priority: Optional[str] = None,
                            notes: Optional[str] = None) -> str:
    """Create a packing task for an order (currently not implemented with direct MongoDB access)."""
    return ("❌ Packing task operations are not yet implemented with direct database access. "
            "This feature will be available in a future update. "
            "For now, please use the web interface to create packing tasks.")

# Define update packing task tool
async def update_packing_task_func(task_id: int, 
                            status: Optional[str] = None,
                            worker_id: Optional[int] = None,
                            notes: Optional[str] = None) -> str:
    """Update the status of a packing task (currently not implemented with direct MongoDB access)."""
    return ("❌ Packing task update operations are not yet implemented with direct database access. "
            "This feature will be available in a future update. "
            "For now, please use the web interface to update packing tasks.")

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
