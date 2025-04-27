from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field

# Use absolute imports instead of relative imports
from tools.base_tool import WMSBaseTool, create_tool
from utils.api_client import api_client
from utils.knowledge_base import knowledge_base

def process_return_func(order_id: int, 
                       items: List[Dict[str, Any]], 
                       reason: str, 
                       condition: str,
                       restock: Optional[bool] = True,
                       refund_amount: Optional[float] = None,
                       notes: Optional[str] = None) -> str:
    """
    Process a customer return.
    
    Args:
        order_id: Original order ID
        items: List of items being returned, each with item_id and quantity
        reason: Reason for the return
        condition: Condition of the returned items
        restock: Whether to return items to inventory
        refund_amount: Optional refund amount
        notes: Optional notes about the return
        
    Returns:
        Return confirmation and status
    """
    try:
        # Check if the order exists
        order = api_client.get_order(order_id)
        if not order:
            return f"Error: Order {order_id} not found. Cannot process return."
            
        # Verify the items being returned are part of the order
        order_items = {item.get('item_id'): item for item in order.get('items', [])}
        
        for return_item in items:
            item_id = return_item.get('item_id')
            quantity = return_item.get('quantity', 0)
            
            if item_id not in order_items:
                return f"Error: Item {item_id} was not part of order {order_id}."
                
            original_quantity = order_items[item_id].get('quantity', 0)
            if quantity > original_quantity:
                return f"Error: Return quantity ({quantity}) exceeds original order quantity ({original_quantity}) for item {item_id}."
        
        # Process the return
        return_data = {
            "order_id": order_id,
            "items": items,
            "reason": reason,
            "condition": condition,
            "restock": restock,
            "status": "Pending"
        }
        
        if refund_amount is not None:
            return_data["refund_amount"] = refund_amount
            
        if notes:
            return_data["notes"] = notes
        
        # Create the return in the system
        response = api_client.post("returns", return_data)
        
        # If restocking is requested, update inventory
        if restock:
            for return_item in items:
                item_id = return_item.get('item_id')
                quantity = return_item.get('quantity', 0)
                
                # Get the current inventory item
                try:
                    inventory_item = api_client.get_inventory_item(item_id)
                    current_quantity = inventory_item.get('quantity', 0)
                    
                    # Update the inventory quantity
                    update_data = {
                        "quantity": current_quantity + quantity
                    }
                    
                    api_client.put("inventory", item_id, update_data)
                except Exception as e:
                    return f"Return created but error updating inventory for item {item_id}: {str(e)}"
        
        return f"Successfully processed return for order {order_id}. Return ID: {response.get('id')}."
        
    except Exception as e:
        return f"Error processing return: {str(e)}"

# Create the process return tool
process_return_tool = create_tool(
    name="process_return",
    description="Process a customer return and optionally restock inventory",
    function=process_return_func,
    arg_descriptions={
        "order_id": {
            "type": int, 
            "description": "ID of the original order"
        },
        "items": {
            "type": List[Dict[str, Any]], 
            "description": "List of items being returned, each with item_id and quantity"
        },
        "reason": {
            "type": str, 
            "description": "Reason for the return (damaged, wrong item, unwanted, etc.)"
        },
        "condition": {
            "type": str, 
            "description": "Condition of the returned items (new, used, damaged, etc.)"
        },
        "restock": {
            "type": Optional[bool], 
            "description": "Whether to return items to inventory (default: true)"
        },
        "refund_amount": {
            "type": Optional[float], 
            "description": "Optional refund amount"
        },
        "notes": {
            "type": Optional[str], 
            "description": "Optional notes about the return"
        }
    }
)

# Export the tools
__all__ = ["process_return_tool"]