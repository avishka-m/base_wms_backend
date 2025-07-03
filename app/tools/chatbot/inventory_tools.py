from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field

# Use absolute imports instead of relative imports
from app.tools.chatbot.base_tool import WMSBaseTool, create_tool
from app.utils.chatbot.api_client import api_client
from app.utils.chatbot.knowledge_base import knowledge_base

# Define inventory query tool
def inventory_query_func(item_id: Optional[int] = None, 
                         sku: Optional[str] = None,
                         name: Optional[str] = None,
                         category: Optional[str] = None,
                         location_id: Optional[int] = None,
                         min_quantity: Optional[int] = None) -> str:
    """Query the inventory based on various parameters."""
    params = {}
    
    # Add non-None parameters to query
    if item_id is not None:
        # If item_id is provided, do a direct lookup
        try:
            item = api_client.get_inventory_item(item_id)
            return f"Found inventory item: {item}"
        except Exception as e:
            return f"Error retrieving inventory item: {str(e)}"
            
    # Build filter parameters
    if sku:
        params["sku"] = sku
    if name:
        params["name"] = name
    if category:
        params["category"] = category
    if location_id:
        params["location_id"] = location_id
    if min_quantity is not None:
        params["min_quantity"] = min_quantity
        
    try:
        inventory_items = api_client.get_inventory(params)
        
        if not inventory_items:
            return "No inventory items found matching your criteria."
            
        # Format the results
        result = "Found the following inventory items:\n\n"
        for item in inventory_items:
            result += f"ID: {item.get('id')}\n"
            result += f"SKU: {item.get('sku')}\n"
            result += f"Name: {item.get('name')}\n"
            result += f"Category: {item.get('category')}\n"
            result += f"Quantity: {item.get('quantity')}\n"
            result += f"Location: {item.get('location_id')}\n"
            result += "-" * 40 + "\n"
            
        return result
    except Exception as e:
        return f"Error querying inventory: {str(e)}"

# Define inventory add tool
def inventory_add_func(sku: str, 
                       name: str, 
                       category: str, 
                       quantity: int, 
                       location_id: int,
                       unit_price: float,
                       supplier_id: Optional[int] = None,
                       description: Optional[str] = None) -> str:
    """Add a new item to the inventory."""
    # Prepare the data for the new inventory item
    data = {
        "sku": sku,
        "name": name,
        "category": category,
        "quantity": quantity,
        "location_id": location_id,
        "unit_price": unit_price
    }
    
    # Add optional fields if provided
    if supplier_id:
        data["supplier_id"] = supplier_id
    if description:
        data["description"] = description
        
    try:
        # Check if location exists
        locations = api_client.get_locations({"id": location_id})
        if not locations:
            return f"Error: Location with ID {location_id} does not exist."
            
        # Create the inventory item
        response = api_client.post("inventory", data)
        
        return f"Successfully added inventory item: {response.get('id')} - {response.get('name')}"
    except Exception as e:
        return f"Error adding inventory item: {str(e)}"

# Define inventory update tool
def inventory_update_func(item_id: int,
                         sku: Optional[str] = None, 
                         name: Optional[str] = None, 
                         category: Optional[str] = None, 
                         quantity: Optional[int] = None, 
                         location_id: Optional[int] = None,
                         unit_price: Optional[float] = None,
                         supplier_id: Optional[int] = None,
                         description: Optional[str] = None) -> str:
    """Update an existing inventory item."""
    # First, verify the item exists
    try:
        existing_item = api_client.get_inventory_item(item_id)
    except Exception as e:
        return f"Error: Could not find inventory item with ID {item_id}: {str(e)}"
        
    # Prepare update data with only the fields that are provided
    update_data = {}
    if sku:
        update_data["sku"] = sku
    if name:
        update_data["name"] = name
    if category:
        update_data["category"] = category
    if quantity is not None:
        update_data["quantity"] = quantity
    if location_id:
        # Check if location exists
        try:
            locations = api_client.get_locations({"id": location_id})
            if not locations:
                return f"Error: Location with ID {location_id} does not exist."
            update_data["location_id"] = location_id
        except Exception as e:
            return f"Error verifying location: {str(e)}"
    if unit_price is not None:
        update_data["unit_price"] = unit_price
    if supplier_id:
        update_data["supplier_id"] = supplier_id
    if description:
        update_data["description"] = description
        
    # If no fields to update were provided
    if not update_data:
        return "No update fields provided. Please specify at least one field to update."
        
    try:
        # Update the inventory item
        response = api_client.put("inventory", item_id, update_data)
        
        return f"Successfully updated inventory item {item_id}. New details: {response}"
    except Exception as e:
        return f"Error updating inventory item: {str(e)}"

# Define locate item tool
def locate_item_func(item_id: Optional[int] = None, 
                    sku: Optional[str] = None,
                    name: Optional[str] = None) -> str:
    """Find the physical location of an item in the warehouse."""
    params = {}
    
    # Build query parameters
    if item_id is not None:
        # Direct lookup by ID
        try:
            item = api_client.get_inventory_item(item_id)
        except Exception as e:
            return f"Error retrieving inventory item: {str(e)}"
    elif sku:
        params["sku"] = sku
    elif name:
        params["name"] = name
    else:
        return "Error: Please provide either an item_id, SKU, or name to locate an item."
        
    try:
        if "item" not in locals():
            # If we didn't do a direct ID lookup earlier
            inventory_items = api_client.get_inventory(params)
            
            if not inventory_items:
                return "No inventory items found matching your criteria."
                
            # Use the first matching item
            item = inventory_items[0]
            
        # Get the location details
        location_id = item.get("location_id")
        if not location_id:
            return f"Item {item.get('name')} (ID: {item.get('id')}) does not have a location assigned."
            
        location = api_client.get_by_id("locations", location_id)
        
        # Format the response
        result = f"Item: {item.get('name')} (SKU: {item.get('sku')})\n"
        result += f"Location: {location.get('name')} (ID: {location.get('id')})\n"
        result += f"Zone: {location.get('zone')}\n"
        result += f"Aisle: {location.get('aisle')}\n"
        result += f"Shelf: {location.get('shelf')}\n"
        result += f"Bin: {location.get('bin')}\n"
        
        # Add quantity information
        result += f"Current quantity: {item.get('quantity')}\n"
        
        # Query knowledge base for any additional info about this location or item
        kb_results = knowledge_base.query(f"warehouse location {location.get('zone')} {location.get('aisle')}", n_results=1)
        if kb_results:
            result += "\nAdditional information:\n"
            result += kb_results[0].page_content

        return result
    except Exception as e:
        return f"Error locating item: {str(e)}"

# Create the tools
inventory_query_tool = create_tool(
    name="inventory_query",
    description="Query inventory items based on various criteria",
    function=inventory_query_func,
    arg_descriptions={
        "item_id": {
            "type": Optional[int], 
            "description": "Specific inventory item ID to look up"
        },
        "sku": {
            "type": Optional[str], 
            "description": "Stock Keeping Unit (SKU) to search for"
        },
        "name": {
            "type": Optional[str], 
            "description": "Name of the item to search for"
        },
        "category": {
            "type": Optional[str], 
            "description": "Category of items to search for"
        },
        "location_id": {
            "type": Optional[int], 
            "description": "Location ID to filter items by"
        },
        "min_quantity": {
            "type": Optional[int], 
            "description": "Minimum quantity threshold"
        }
    }
)

inventory_add_tool = create_tool(
    name="inventory_add",
    description="Add a new item to the inventory",
    function=inventory_add_func,
    arg_descriptions={
        "sku": {
            "type": str, 
            "description": "Stock Keeping Unit (SKU) for the new item"
        },
        "name": {
            "type": str, 
            "description": "Name of the new item"
        },
        "category": {
            "type": str, 
            "description": "Category of the new item"
        },
        "quantity": {
            "type": int, 
            "description": "Initial quantity of the new item"
        },
        "location_id": {
            "type": int, 
            "description": "Location ID where the item will be stored"
        },
        "unit_price": {
            "type": float, 
            "description": "Unit price of the item"
        },
        "supplier_id": {
            "type": Optional[int], 
            "description": "ID of the supplier for this item"
        },
        "description": {
            "type": Optional[str], 
            "description": "Detailed description of the item"
        }
    }
)

inventory_update_tool = create_tool(
    name="inventory_update",
    description="Update an existing inventory item",
    function=inventory_update_func,
    arg_descriptions={
        "item_id": {
            "type": int, 
            "description": "ID of the inventory item to update"
        },
        "sku": {
            "type": Optional[str], 
            "description": "Updated Stock Keeping Unit (SKU)"
        },
        "name": {
            "type": Optional[str], 
            "description": "Updated name of the item"
        },
        "category": {
            "type": Optional[str], 
            "description": "Updated category of the item"
        },
        "quantity": {
            "type": Optional[int], 
            "description": "Updated quantity of the item"
        },
        "location_id": {
            "type": Optional[int], 
            "description": "Updated location ID where the item is stored"
        },
        "unit_price": {
            "type": Optional[float], 
            "description": "Updated unit price of the item"
        },
        "supplier_id": {
            "type": Optional[int], 
            "description": "Updated ID of the supplier for this item"
        },
        "description": {
            "type": Optional[str], 
            "description": "Updated detailed description of the item"
        }
    }
)

locate_item_tool = create_tool(
    name="locate_item",
    description="Find the physical location of an item in the warehouse",
    function=locate_item_func,
    arg_descriptions={
        "item_id": {
            "type": Optional[int], 
            "description": "ID of the item to locate"
        },
        "sku": {
            "type": Optional[str], 
            "description": "SKU of the item to locate"
        },
        "name": {
            "type": Optional[str], 
            "description": "Name of the item to locate"
        }
    }
)

# Export the tools
__all__ = [
    "inventory_query_tool",
    "inventory_add_tool",
    "inventory_update_tool",
    "locate_item_tool"
]
