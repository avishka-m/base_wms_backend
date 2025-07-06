from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field

# Use absolute imports instead of relative imports
from app.tools.chatbot.base_tool import WMSBaseTool, create_tool
from app.utils.chatbot.mongodb_client import chatbot_mongodb_client
from app.utils.chatbot.knowledge_base import knowledge_base
from app.utils.chatbot.demo_data import get_demo_inventory_data, is_api_error

# Define inventory query tool
async def inventory_query_func(item_id: Optional[int] = None, 
                         sku: Optional[str] = None,
                         name: Optional[str] = None,
                         category: Optional[str] = None,
                         location_id: Optional[int] = None,
                         min_quantity: Optional[int] = None) -> str:
    """Query the inventory based on various parameters using direct MongoDB access."""
    
    try:
        # If item_id is provided, do a direct lookup
        if item_id is not None:
            item = await chatbot_mongodb_client.get_inventory_item_by_id(item_id)
            
            if not item:
                return f"Item with ID {item_id} not found in the database."
            
            # Format single item response
            result = "Found inventory item:\n\n"
            result += f"ID: {item.get('itemID')}\n"
            result += f"Name: {item.get('name')}\n"
            result += f"Category: {item.get('category')}\n"
            result += f"Stock Level: {item.get('stock_level')}\n"
            result += f"Min Stock: {item.get('min_stock_level', 'N/A')}\n"
            result += f"Max Stock: {item.get('max_stock_level', 'N/A')}\n"
            result += f"Location ID: {item.get('locationID')}\n"
            result += f"Supplier ID: {item.get('supplierID', 'N/A')}\n"
            result += f"Size: {item.get('size', 'N/A')}\n"
            result += f"Storage Type: {item.get('storage_type', 'N/A')}\n"
            result += f"Created: {item.get('created_at', 'N/A')}\n"
            
            return result
        
        # Build filter criteria for MongoDB query
        filter_criteria = {}
        
        if name:
            # Search inventory by name (partial match)
            inventory_items = await chatbot_mongodb_client.search_inventory_by_name(name)
        elif category:
            # Search by category
            inventory_items = await chatbot_mongodb_client.get_inventory_by_category(category)
        else:
            # Build complex filter
            if location_id:
                filter_criteria["locationID"] = location_id
            if min_quantity is not None:
                filter_criteria["stock_level"] = {"$gte": min_quantity}
            
            inventory_items = await chatbot_mongodb_client.get_inventory_items(filter_criteria)
        
        if not inventory_items:
            return "No inventory items found matching your criteria."
            
        # Format the results
        result = f"Found {len(inventory_items)} inventory item(s):\n\n"
        for item in inventory_items:
            result += f"ID: {item.get('itemID')}\n"
            result += f"Name: {item.get('name')}\n"
            result += f"Category: {item.get('category')}\n"
            result += f"Stock Level: {item.get('stock_level')}\n"
            result += f"Location ID: {item.get('locationID')}\n"
            result += f"Storage Type: {item.get('storage_type', 'N/A')}\n"
            result += "-" * 40 + "\n"
            
        return result
        
    except Exception as e:
        # Fallback to demo data on database error
        try:
            inventory_items = get_demo_inventory_data(
                item_id=item_id, name=name, category=category, 
                location_id=location_id, min_quantity=min_quantity
            )
            if not inventory_items:
                return "No inventory items found matching your criteria."
            
            result = "Found the following inventory items (Demo data - Database error):\n\n"
            for item in inventory_items:
                result += f"ID: {item.get('id')}\n"
                result += f"Name: {item.get('name')}\n"
                result += f"Category: {item.get('category')}\n"
                result += f"Quantity: {item.get('quantity')}\n"
                result += f"Location: {item.get('location_id')}\n"
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
    """Add a new item to the inventory (currently not implemented with direct MongoDB access)."""
    return ("❌ Inventory add operations are not yet implemented with direct database access. "
            "This feature will be available in a future update. "
            "For now, please use the web interface to add new inventory items.")

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
    """Update an existing inventory item (currently not implemented with direct MongoDB access)."""
    return ("❌ Inventory update operations are not yet implemented with direct database access. "
            "This feature will be available in a future update. "
            "For now, please use the web interface to update inventory items.")

# Define inventory delete tool
async def inventory_delete_func(item_id: int) -> str:
    """Delete an inventory item (currently not implemented with direct MongoDB access)."""
    return ("❌ Inventory delete operations are not yet implemented with direct database access. "
            "This feature will be available in a future update. "
            "For now, please use the web interface to delete inventory items.")

# Define item locator tool
async def locate_item_func(item_id: Optional[int] = None, 
                    sku: Optional[str] = None,
                    name: Optional[str] = None) -> str:
    """Locate an item in the warehouse and provide path guidance using direct MongoDB access."""
    try:
        # First, find the item
        if item_id:
            item = await chatbot_mongodb_client.get_inventory_item_by_id(item_id)
            if not item:
                return f"Item with ID {item_id} not found."
        elif name:
            # Search by name
            items = await chatbot_mongodb_client.search_inventory_by_name(name)
            if not items:
                return "No items found matching your search criteria."
            
            if len(items) > 1:
                # Multiple items found
                result = "Multiple items found:\n\n"
                for item in items[:5]:  # Show first 5 matches
                    result += f"ID: {item.get('itemID')} | Name: {item.get('name')} | Location ID: {item.get('locationID')}\n"
                result += "\nPlease specify the exact item_id to get location details."
                return result
            
            item = items[0]
        else:
            return "Please provide either item_id or name to locate an item."
        
        # Get location details
        location_id = item.get('locationID')
        if not location_id:
            return f"Item {item.get('name', 'Unknown')} does not have a location assigned."
        
        # Get location information
        location = await chatbot_mongodb_client.get_location_by_id(location_id)
        if not location:
            return f"Location information not available for location ID {location_id}."
        
        # Format location information using actual database schema
        result = f"Item Location Found:\n\n"
        result += f"Item: {item.get('name')}\n"
        result += f"Item ID: {item.get('itemID')}\n"
        result += f"Stock Level: {item.get('stock_level', 0)} units\n"
        result += f"Category: {item.get('category', 'Unknown')}\n"
        result += f"Location ID: {location_id}\n"
        result += f"Section: {location.get('section', 'Unknown')}\n"
        result += f"Row: {location.get('row', 'Unknown')}\n"
        result += f"Shelf: {location.get('shelf', 'Unknown')}\n"
        result += f"Bin: {location.get('bin', 'Unknown')}\n"
        result += f"Warehouse ID: {location.get('warehouseID', 'Unknown')}\n"
        
        # Add basic navigation hint using actual schema
        result += f"\nNavigation: Go to Section {location.get('section', '?')}, "
        result += f"Row {location.get('row', '?')}, "
        result += f"Shelf {location.get('shelf', '?')}, "
        result += f"Bin {location.get('bin', '?')}"
        
        # Add occupancy status
        if location.get('is_occupied'):
            result += f"\nStatus: Location is occupied"
        else:
            result += f"\nStatus: Location is available"
        
        return result
        
    except Exception as e:
        return f"Error locating item: {str(e)}"

# Define low stock alert tool
async def low_stock_alert_func(threshold: int = 10) -> str:
    """Check for items with low stock levels using direct MongoDB access."""
    try:
        # Get low stock items directly from MongoDB
        low_stock_items = await chatbot_mongodb_client.get_low_stock_items(threshold)
        
        if not low_stock_items:
            return f"No items found with stock levels at or below {threshold} units."
        
        # Format results
        result = f"Low Stock Alert (Threshold: {threshold} units):\n\n"
        
        for item in low_stock_items:
            result += f"⚠️  {item.get('name')}\n"
            result += f"    Item ID: {item.get('itemID')}\n"
            result += f"    Current Stock: {item.get('stock_level', 0)} units\n"
            result += f"    Min Stock Level: {item.get('min_stock_level', 'N/A')} units\n"
            result += f"    Location ID: {item.get('locationID', 'Unknown')}\n"
            result += f"    Category: {item.get('category', 'Unknown')}\n"
            result += f"    Storage Type: {item.get('storage_type', 'N/A')}\n"
            result += "-" * 40 + "\n"
        
        result += f"\nTotal items with low stock: {len(low_stock_items)}"
        
        return result
        
    except Exception as e:
        # Fallback to demo data on database error
        try:
            inventory_items = get_demo_inventory_data()
            low_stock_items = [
                item for item in inventory_items 
                if item.get('quantity', 0) <= threshold
            ]
            
            if not low_stock_items:
                return f"No items found with stock levels at or below {threshold} units."
            
            result = f"Low Stock Alert (Demo data - Database error) (Threshold: {threshold} units):\n\n"
            
            for item in low_stock_items:
                result += f"⚠️  {item.get('name')}\n"
                result += f"    Current Stock: {item.get('quantity', 0)} units\n"
                result += f"    Location: {item.get('location_id', 'Unknown')}\n"
                result += f"    Category: {item.get('category', 'Unknown')}\n"
                result += "-" * 40 + "\n"
            
            result += f"\nTotal items with low stock: {len(low_stock_items)}"
            return result
        except:
            return f"Error checking low stock: {str(e)}"

# Define stock movement tool
async def stock_movement_func(item_id: int, 
                        from_location_id: int, 
                        to_location_id: int, 
                        quantity: int) -> str:
    """Move stock from one location to another (currently not implemented with direct MongoDB access)."""
    return ("❌ Stock movement operations are not yet implemented with direct database access. "
            "This feature will be available in a future update. "
            "For now, please use the web interface to move stock between locations.")

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
