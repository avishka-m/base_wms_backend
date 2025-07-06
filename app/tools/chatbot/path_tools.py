from typing import Dict, Any, List, Optional, Tuple
from pydantic import BaseModel, Field
import networkx as nx
import json

# Use absolute imports instead of relative imports
from app.tools.chatbot.base_tool import WMSBaseTool, create_tool
from app.utils.chatbot.api_client import api_client
from app.utils.chatbot.knowledge_base import knowledge_base
from app.utils.chatbot.demo_data import get_demo_locations, get_demo_inventory_data, is_api_error

class Location:
    """Represents a warehouse location for path finding."""
    def __init__(self, id: int, name: str, zone: str, aisle: str, 
                 shelf: str, bin: str, x: int, y: int, z: int):
        self.id = id
        self.name = name
        self.zone = zone
        self.aisle = aisle
        self.shelf = shelf
        self.bin = bin
        self.x = x
        self.y = y
        self.z = z
        
    def __repr__(self):
        return f"Location({self.name}, {self.zone}-{self.aisle}-{self.shelf}-{self.bin})"
        
    def distance_to(self, other: 'Location') -> float:
        """Calculate Manhattan distance to another location."""
        return abs(self.x - other.x) + abs(self.y - other.y) + abs(self.z - other.z)
        
    @classmethod
    def from_api_data(cls, data: Dict[str, Any]) -> 'Location':
        """Create a Location from API response data."""
        return cls(
            id=data.get('id'),
            name=data.get('name'),
            zone=data.get('zone'),
            aisle=data.get('aisle'),
            shelf=data.get('shelf', ''),
            bin=data.get('bin', ''),
            x=data.get('x', 0),
            y=data.get('y', 0),
            z=data.get('z', 0)
        )

def get_warehouse_layout() -> List[Location]:
    """Fetch all warehouse locations from the API."""
    try:
        locations_data = api_client.get_locations()
        
        # Check if we got an error response - use demo data as fallback
        if is_api_error(locations_data):
            locations_data = get_demo_locations()
        
        return [Location.from_api_data(loc) for loc in locations_data]
    except Exception as e:
        print(f"Error fetching warehouse layout: {e}")
        # Fallback to demo data
        locations_data = get_demo_locations()
        return [Location.from_api_data(loc) for loc in locations_data]

def get_item_locations(item_ids: List[int]) -> Dict[int, Location]:
    """Get the locations for a list of item IDs."""
    item_locations = {}
    
    for item_id in item_ids:
        try:
            item = api_client.get_inventory_item(item_id)
            
            # Check if we got an error response - use demo data as fallback
            if is_api_error(item):
                demo_items = get_demo_inventory_data(item_id=item_id)
                if demo_items:
                    item = demo_items[0]
                else:
                    print(f"Item {item_id} not found in demo data")
                    continue
            
            location_id = item.get('location_id')
            
            if location_id:
                location_data = api_client.get_by_id("locations", location_id)
                
                # Check if we got an error response - use demo data as fallback
                if is_api_error(location_data):
                    demo_locations = get_demo_locations(location_id=location_id)
                    if demo_locations:
                        location_data = demo_locations[0]
                    else:
                        print(f"Location {location_id} not found in demo data")
                        continue
                
                item_locations[item_id] = Location.from_api_data(location_data)
        except Exception as e:
            print(f"Error getting location for item {item_id}: {e}")
            # Try demo data as fallback
            demo_items = get_demo_inventory_data(item_id=item_id)
            if demo_items:
                item = demo_items[0]
                location_id = item.get('location_id')
                if location_id:
                    demo_locations = get_demo_locations(location_id=location_id)
                    if demo_locations:
                        location_data = demo_locations[0]
                        item_locations[item_id] = Location.from_api_data(location_data)
            
    return item_locations

def build_warehouse_graph(locations: List[Location]) -> nx.Graph:
    """Build a graph representing the warehouse layout."""
    G = nx.Graph()
    
    # Add all locations as nodes
    for loc in locations:
        G.add_node(loc.id, location=loc)
    
    # Connect adjacent locations based on aisle, shelf, and bin
    for loc1 in locations:
        for loc2 in locations:
            if loc1.id != loc2.id:
                # Same aisle connections
                if loc1.zone == loc2.zone and loc1.aisle == loc2.aisle:
                    # Adjacent shelves or bins
                    shelf1 = int(loc1.shelf) if loc1.shelf.isdigit() else 0
                    shelf2 = int(loc2.shelf) if loc2.shelf.isdigit() else 0
                    
                    bin1 = int(loc1.bin) if loc1.bin.isdigit() else 0
                    bin2 = int(loc2.bin) if loc2.bin.isdigit() else 0
                    
                    if abs(shelf1 - shelf2) <= 1 or abs(bin1 - bin2) <= 1:
                        weight = loc1.distance_to(loc2)
                        G.add_edge(loc1.id, loc2.id, weight=weight)
                
                # Connect different aisles at the ends
                elif loc1.zone == loc2.zone and abs(int(loc1.aisle) - int(loc2.aisle)) == 1:
                    # Connect the ends of aisles
                    if (loc1.bin == '1' and loc2.bin == '1') or \
                       (loc1.bin == '20' and loc2.bin == '20'):  # Assuming 20 bins per aisle
                        weight = loc1.distance_to(loc2)
                        G.add_edge(loc1.id, loc2.id, weight=weight)
    
    return G

def path_optimize_func(item_ids: List[int], start_location_id: Optional[int] = None) -> str:
    """
    Optimize the path to pick a list of items from the warehouse.
    
    Args:
        item_ids: List of inventory item IDs to pick
        start_location_id: Optional ID of starting location (defaults to warehouse entrance)
        
    Returns:
        Optimized path description
    """
    try:
        # Get the warehouse layout
        warehouse_locations = get_warehouse_layout()
        if not warehouse_locations:
            return "Error: Could not retrieve warehouse layout."
            
        # Get locations of all items
        item_locations = get_item_locations(item_ids)
        if not item_locations:
            return "Error: Could not find locations for any of the specified items."
            
        # Build warehouse graph
        warehouse_graph = build_warehouse_graph(warehouse_locations)
        
        # Define start location (warehouse entrance or specified location)
        start_location = None
        if start_location_id:
            for loc in warehouse_locations:
                if loc.id == start_location_id:
                    start_location = loc
                    break
                    
        if not start_location:
            # Use warehouse entrance as default (Zone A, Aisle 1, usually)
            for loc in warehouse_locations:
                if loc.zone == 'A' and loc.aisle == '1' and loc.bin == '1':
                    start_location = loc
                    break
                    
        if not start_location and warehouse_locations:
            # Fallback to first location
            start_location = warehouse_locations[0]
            
        # Run TSP-like algorithm to find optimized path
        current_location = start_location
        path = [current_location]
        remaining_items = set(item_ids)
        
        while remaining_items:
            # Find the nearest item location from current position
            min_distance = float('inf')
            next_item_id = None
            
            for item_id in remaining_items:
                if item_id in item_locations:
                    item_loc = item_locations[item_id]
                    
                    # Check if nodes exist in the graph
                    if current_location.id not in warehouse_graph or item_loc.id not in warehouse_graph:
                        distance = current_location.distance_to(item_loc)
                    else:
                        # Use graph shortest path if available
                        try:
                            distance = nx.shortest_path_length(
                                warehouse_graph, 
                                current_location.id, 
                                item_loc.id, 
                                weight='weight'
                            )
                        except nx.NetworkXNoPath:
                            distance = current_location.distance_to(item_loc)
                    
                    if distance < min_distance:
                        min_distance = distance
                        next_item_id = item_id
            
            if next_item_id:
                current_location = item_locations[next_item_id]
                path.append(current_location)
                remaining_items.remove(next_item_id)
            else:
                break
        
        # Format the response
        result = "Optimized Picking Path:\n\n"
        result += f"Start: {path[0].name} ({path[0].zone}-{path[0].aisle}-{path[0].shelf}-{path[0].bin})\n\n"
        
        for i, location in enumerate(path[1:], 1):
            # Find the item ID for this location
            item_id = None
            for iid, loc in item_locations.items():
                if loc.id == location.id:
                    item_id = iid
                    break
                    
            # Get item details if found
            item_name = "Unknown item"
            if item_id:
                try:
                    item = api_client.get_inventory_item(item_id)
                    item_name = f"{item.get('name')} (SKU: {item.get('sku')})"
                except:
                    pass
                    
            result += f"{i}. Go to {location.name} ({location.zone}-{location.aisle}-{location.shelf}-{location.bin})\n"
            result += f"   Pick: {item_name}\n\n"
        
        # Calculate total path length
        total_distance = 0
        for i in range(len(path) - 1):
            total_distance += path[i].distance_to(path[i + 1])
            
        result += f"Total path length: {total_distance} units\n"
        result += f"Total stops: {len(path) - 1}"
        
        return result
    except Exception as e:
        return f"Error optimizing path: {str(e)}"

def calculate_route_func(addresses: List[str], 
                         start_address: Optional[str] = None,
                         vehicle_id: Optional[int] = None) -> str:
    """
    Calculate an optimized delivery route for a set of addresses.
    
    Args:
        addresses: List of delivery addresses
        start_address: Optional starting warehouse address
        vehicle_id: Optional vehicle ID to use for delivery
        
    Returns:
        Optimized delivery route description
    """
    try:
        # Get warehouse address if not provided
        if not start_address:
            # Assume we have a warehouse location in the knowledge base
            kb_results = knowledge_base.query("warehouse address location", n_results=1)
            if kb_results:
                start_address = kb_results[0].page_content.strip()
            else:
                start_address = "Warehouse (123 Main St.)"
        
        # Get vehicle information if provided
        vehicle_info = "Default delivery vehicle"
        if vehicle_id:
            try:
                vehicle = api_client.get_by_id("vehicles", vehicle_id)
                vehicle_info = f"{vehicle.get('name')} ({vehicle.get('type')}, Capacity: {vehicle.get('capacity')} kg)"
            except:
                pass
        
        # This is a simplified implementation without actual route optimization
        # In a real implementation, we would use a routing API service
        # For now, we'll just create a plausible looking route
        
        # Simulate route optimization (greedy approach)
        current = start_address
        route = [current]
        remaining = addresses.copy()
        
        while remaining:
            # In a real implementation, we would calculate actual distances
            # For demonstration, just pick the next address alphabetically
            next_stop = min(remaining)
            route.append(next_stop)
            remaining.remove(next_stop)
        
        # Format the response
        result = "Optimized Delivery Route:\n\n"
        result += f"Vehicle: {vehicle_info}\n"
        result += f"Starting Point: {start_address}\n\n"
        
        for i, address in enumerate(route[1:], 1):
            result += f"Stop {i}: {address}\n"
        
        result += f"\nTotal stops: {len(route) - 1}"
        result += "\n\nNote: This is a simulated route. In a production environment, we would integrate with a routing API for accurate distance and time calculations."
        
        return result
    except Exception as e:
        return f"Error calculating delivery route: {str(e)}"

# Create the tools
path_optimize_tool = create_tool(
    name="path_optimize",
    description="Optimize the path to pick items in the warehouse",
    function=path_optimize_func,
    arg_descriptions={
        "item_ids": {
            "type": List[int], 
            "description": "List of inventory item IDs to pick"
        },
        "start_location_id": {
            "type": Optional[int], 
            "description": "Optional ID of starting location (defaults to warehouse entrance)"
        }
    }
)

calculate_route_tool = create_tool(
    name="calculate_route",
    description="Calculate an optimized delivery route for a set of addresses",
    function=calculate_route_func,
    arg_descriptions={
        "addresses": {
            "type": List[str], 
            "description": "List of delivery addresses"
        },
        "start_address": {
            "type": Optional[str], 
            "description": "Optional starting warehouse address"
        },
        "vehicle_id": {
            "type": Optional[int], 
            "description": "Optional vehicle ID to use for delivery"
        }
    }
)

# Export the tools
__all__ = [
    "path_optimize_tool",
    "calculate_route_tool"
]
