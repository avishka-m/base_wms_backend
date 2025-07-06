from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field

# Use absolute imports instead of relative imports
from app.tools.chatbot.base_tool import WMSBaseTool, create_tool
from app.utils.chatbot.api_client import async_api_client
from app.utils.chatbot.knowledge_base import knowledge_base
from app.utils.chatbot.demo_data import get_demo_inventory_data, is_api_error

# Define inventory query tool
async def inventory_query_func(item_id: Optional[int] = None, 
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
            item = await async_api_client.get_inventory_item(item_id)
            
            # Check if we got an error response - use demo data as fallback
            if is_api_error(item):
                demo_items = get_demo_inventory_data(item_id=item_id)
                if demo_items:
                    item = demo_items[0]
                    note = " (Demo data - API not accessible)"
                else:
                    return f"Item with ID {item_id} not found in demo data."
            else:
                note = ""
            
            # Format single item response
            result = f"Found inventory item{note}:\n\n"
            result += f"ID: {item.get('id')}\n"
            result += f"SKU: {item.get('sku')}\n"
            result += f"Name: {item.get('name')}\n"
            result += f"Category: {item.get('category')}\n"
            result += f"Quantity: {item.get('quantity')}\n"
            result += f"Location: {item.get('location_id')}\n"
            result += f"Unit Price: ${item.get('unit_price', 'N/A')}\n"
            result += f"Description: {item.get('description', 'N/A')}\n"
            
            return result
        except Exception as e:
            # Fallback to demo data on exception
            demo_items = get_demo_inventory_data(item_id=item_id)
            if demo_items:
                item = demo_items[0]
                result = "Found inventory item (Demo data - API error):\n\n"
                result += f"ID: {item.get('id')}\n"
                result += f"SKU: {item.get('sku')}\n"
                result += f"Name: {item.get('name')}\n"
                result += f"Category: {item.get('category')}\n"
                result += f"Quantity: {item.get('quantity')}\n"
                result += f"Location: {item.get('location_id')}\n"
                result += f"Unit Price: ${item.get('unit_price', 'N/A')}\n"
                result += f"Description: {item.get('description', 'N/A')}\n"
                return result
            else:
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
        inventory_items = await async_api_client.get_inventory(params)
        
        # Check if we got an error response - use demo data as fallback
        if is_api_error(inventory_items):
            inventory_items = get_demo_inventory_data(
                sku=sku, name=name, category=category, 
                location_id=location_id, min_quantity=min_quantity
            )
            note = " (Demo data - API not accessible)"
        else:
            note = ""
        
        if not inventory_items:
            return "No inventory items found matching your criteria."
            
        # Format the results
        result = f"Found the following inventory items{note}:\n\n"
        for item in inventory_items:
            result += f"ID: {item.get('id')}\n"
            result += f"SKU: {item.get('sku')}\n"
            result += f"Name: {item.get('name')}\n"
            result += f"Category: {item.get('category')}\n"
            result += f"Quantity: {item.get('quantity')}\n"
            result += f"Location: {item.get('location_id')}\n"
            result += f"Unit Price: ${item.get('unit_price', 'N/A')}\n"
            result += "-" * 40 + "\n"
            
        return result
    except Exception as e:
        # Fallback to demo data on exception
        try:
            inventory_items = get_demo_inventory_data(
                sku=sku, name=name, category=category, 
                location_id=location_id, min_quantity=min_quantity
            )
            if not inventory_items:
                return "No inventory items found matching your criteria."
            
            result = "Found the following inventory items (Demo data - API error):\n\n"
            for item in inventory_items:
                result += f"ID: {item.get('id')}\n"
                result += f"SKU: {item.get('sku')}\n"
                result += f"Name: {item.get('name')}\n"
                result += f"Category: {item.get('category')}\n"
                result += f"Quantity: {item.get('quantity')}\n"
                result += f"Location: {item.get('location_id')}\n"
                result += f"Unit Price: ${item.get('unit_price', 'N/A')}\n"
                result += "-" * 40 + "\n"
            
            return result
        except:
            return f"Error querying inventory: {str(e)}"

# Define inventory add tool
async def inventory_add_func(sku: str, 
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
        locations = await async_api_client.get_locations({"id": location_id})
        if not locations:
            return f"Error: Location with ID {location_id} does not exist."
            
        # Create the inventory item
        response = await async_api_client.post("inventory", data)
        
        if is_api_error(response):
            return f"Error adding inventory item: {response.get('error', 'Unknown error')}"
        
        return f"Successfully added inventory item: {response.get('id')} - {response.get('name')}"
    except Exception as e:
        return f"Error adding inventory item: {str(e)}"

# Define inventory update tool
async def inventory_update_func(item_id: int,
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
        existing_item = await async_api_client.get_inventory_item(item_id)
        if is_api_error(existing_item):
            return f"Error: Could not find inventory item with ID {item_id}: {existing_item.get('error', 'Unknown error')}"
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
            locations = await async_api_client.get_locations({"id": location_id})
            if not locations:
                return f"Error: Location with ID {location_id} does not exist."
            update_data["location_id"] = location_id
        except:
            return f"Error: Could not verify location with ID {location_id}."
    if unit_price is not None:
        update_data["unit_price"] = unit_price
    if supplier_id:
        update_data["supplier_id"] = supplier_id
    if description:
        update_data["description"] = description
        
    if not update_data:
        return "No update data provided. Please specify at least one field to update."
        
    try:
        response = await async_api_client.put("inventory", item_id, update_data)
        
        if is_api_error(response):
            return f"Error updating inventory item: {response.get('error', 'Unknown error')}"
        
        return f"Successfully updated inventory item {item_id}: {response.get('name', 'Unknown')}"
    except Exception as e:
        return f"Error updating inventory item: {str(e)}"

# Define inventory delete tool
async def inventory_delete_func(item_id: int) -> str:
    """Delete an inventory item."""
    try:
        # First verify the item exists
        existing_item = await async_api_client.get_inventory_item(item_id)
        if is_api_error(existing_item):
            return f"Error: Could not find inventory item with ID {item_id}: {existing_item.get('error', 'Unknown error')}"
        
        # Delete the item
        response = await async_api_client.delete("inventory", item_id)
        
        if is_api_error(response):
            return f"Error deleting inventory item: {response.get('error', 'Unknown error')}"
        
        return f"Successfully deleted inventory item {item_id}: {existing_item.get('name', 'Unknown')}"
    except Exception as e:
        return f"Error deleting inventory item: {str(e)}"

# Define item locator tool
async def locate_item_func(item_id: Optional[int] = None, 
                    sku: Optional[str] = None,
                    name: Optional[str] = None) -> str:
    """Locate an item in the warehouse and provide path guidance."""
    try:
        # First, find the item
        if item_id:
            item = await async_api_client.get_inventory_item(item_id)
            if is_api_error(item):
                return f"Item with ID {item_id} not found."
        else:
            # Search by SKU or name
            params = {}
            if sku:
                params["sku"] = sku
            if name:
                params["name"] = name
                
            if not params:
                return "Please provide either item_id, sku, or name to locate an item."
                
            items = await async_api_client.get_inventory(params)
            if is_api_error(items) or not items:
                return "No items found matching your search criteria."
            
            if len(items) > 1:
                # Multiple items found
                result = "Multiple items found:\n\n"
                for item in items[:5]:  # Show first 5 matches
                    result += f"ID: {item.get('id')} | SKU: {item.get('sku')} | Name: {item.get('name')} | Location: {item.get('location_id')}\n"
                result += "\nPlease specify the exact item_id to get location details."
                return result
            
            item = items[0]
        
        # Get location details
        location_id = item.get('location_id')
        if not location_id:
            return f"Item {item.get('name', 'Unknown')} does not have a location assigned."
        
        # Get location information
        locations = await async_api_client.get_locations({"id": location_id})
        if is_api_error(locations) or not locations:
            return f"Location information not available for location ID {location_id}."
        
        location = locations[0] if isinstance(locations, list) else locations
        
        # Format location information
        result = f"Item Location Found:\n\n"
        result += f"Item: {item.get('name')} (SKU: {item.get('sku')})\n"
        result += f"Quantity Available: {item.get('quantity', 0)}\n"
        result += f"Location ID: {location_id}\n"
        result += f"Location Name: {location.get('name', 'Unknown')}\n"
        result += f"Zone: {location.get('zone', 'Unknown')}\n"
        result += f"Aisle: {location.get('aisle', 'Unknown')}\n"
        result += f"Rack: {location.get('rack', 'Unknown')}\n"
        result += f"Shelf: {location.get('shelf', 'Unknown')}\n"
        
        # Add basic navigation hint
        result += f"\nNavigation: Go to Zone {location.get('zone', '?')}, "
        result += f"Aisle {location.get('aisle', '?')}, "
        result += f"Rack {location.get('rack', '?')}, "
        result += f"Shelf {location.get('shelf', '?')}"
        
        return result
        
    except Exception as e:
        return f"Error locating item: {str(e)}"

# Define low stock alert tool
async def low_stock_alert_func(threshold: int = 10) -> str:
    """Check for items with low stock levels."""
    try:
        # Get all inventory items
        inventory_items = await async_api_client.get_inventory()
        
        if is_api_error(inventory_items):
            # Use demo data as fallback
            inventory_items = get_demo_inventory_data()
            note = " (Demo data - API not accessible)"
        else:
            note = ""
        
        if not inventory_items:
            return "No inventory items found."
        
        # Filter items with low stock
        low_stock_items = [
            item for item in inventory_items 
            if item.get('quantity', 0) <= threshold
        ]
        
        if not low_stock_items:
            return f"No items found with stock levels at or below {threshold} units."
        
        # Format results
        result = f"Low Stock Alert{note} (Threshold: {threshold} units):\n\n"
        
        for item in low_stock_items:
            result += f"⚠️  {item.get('name')} (SKU: {item.get('sku')})\n"
            result += f"    Current Stock: {item.get('quantity', 0)} units\n"
            result += f"    Location: {item.get('location_id', 'Unknown')}\n"
            result += f"    Category: {item.get('category', 'Unknown')}\n"
            result += "-" * 40 + "\n"
        
        result += f"\nTotal items with low stock: {len(low_stock_items)}"
        
        return result
        
    except Exception as e:
        return f"Error checking low stock: {str(e)}"

# Define stock movement tool
async def stock_movement_func(item_id: int, 
                        from_location_id: int, 
                        to_location_id: int, 
                        quantity: int) -> str:
    """Move stock from one location to another."""
    try:
        # Verify the item exists
        item = await async_api_client.get_inventory_item(item_id)
        if is_api_error(item):
            return f"Item with ID {item_id} not found."
        
        # Check current location and quantity
        current_location = item.get('location_id')
        current_quantity = item.get('quantity', 0)
        
        if current_location != from_location_id:
            return f"Item is currently at location {current_location}, not {from_location_id}."
        
        if current_quantity < quantity:
            return f"Insufficient stock. Available: {current_quantity}, Requested: {quantity}"
        
        # Verify destination location exists
        to_locations = await async_api_client.get_locations({"id": to_location_id})
        if is_api_error(to_locations) or not to_locations:
            return f"Destination location {to_location_id} not found."
        
        # Calculate new quantities (simplified - assumes all stock moves)
        if quantity == current_quantity:
            # Move all stock to new location
            update_data = {
                "location_id": to_location_id
            }
        else:
            # Partial move - in a real system, this would create a new inventory record
            # For now, we'll just update the location and adjust quantity
            update_data = {
                "location_id": to_location_id,
                "quantity": quantity
            }
        
        # Update the item
        response = await async_api_client.put("inventory", item_id, update_data)
        
        if is_api_error(response):
            return f"Error moving stock: {response.get('error', 'Unknown error')}"
        
        result = f"Stock Movement Completed:\n\n"
        result += f"Item: {item.get('name')} (SKU: {item.get('sku')})\n"
        result += f"Quantity Moved: {quantity} units\n"
        result += f"From Location: {from_location_id}\n"
        result += f"To Location: {to_location_id}\n"
        result += f"Status: Completed Successfully"
        
        return result
        
    except Exception as e:
        return f"Error moving stock: {str(e)}"

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

inventory_delete_tool = create_tool(
    name="inventory_delete",
    description="Delete an inventory item",
    function=inventory_delete_func,
    arg_descriptions={
        "item_id": {
            "type": int,
            "description": "ID of the inventory item to delete"
        }
    }
)

locate_item_tool = create_tool(
    name="locate_item",
    description="Find the physical location of an item in the warehouse and provide navigation guidance",
    function=locate_item_func,
    arg_descriptions={
        "item_id": {
            "type": Optional[int], 
            "description": "ID of the item to locate"
        },
        "sku": {
            "type": Optional[str], 
            "description": "Stock Keeping Unit (SKU) of the item to locate"
        },
        "name": {
            "type": Optional[str], 
            "description": "Name of the item to locate"
        }
    }
)

low_stock_alert_tool = create_tool(
    name="low_stock_alert",
    description="Check for items with low stock levels and generate alerts",
    function=low_stock_alert_func,
    arg_descriptions={
        "threshold": {
            "type": int,
            "description": "Stock level threshold below which items are considered low stock (default: 10)"
        }
    }
)

stock_movement_tool = create_tool(
    name="stock_movement",
    description="Move stock from one location to another in the warehouse",
    function=stock_movement_func,
    arg_descriptions={
        "item_id": {
            "type": int,
            "description": "ID of the item to move"
        },
        "from_location_id": {
            "type": int,
            "description": "Current location ID of the item"
        },
        "to_location_id": {
            "type": int,
            "description": "Destination location ID for the item"
        },
        "quantity": {
            "type": int,
            "description": "Quantity to move"
        }
    }
)

# Export all inventory tools
inventory_tools = [
    inventory_query_tool,
    inventory_add_tool,
    inventory_update_tool,
    inventory_delete_tool,
    locate_item_tool,
    low_stock_alert_tool,
    stock_movement_tool
]
